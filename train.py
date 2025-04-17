from accelerate import Accelerator
from transformers import (
    AutoModel,
    AutoTokenizer,
    MPNetModel,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader, Dataset
from typing import Union, Optional, List
from datasets import load_from_disk
import torch.nn.functional as F
from torch.optim import AdamW
from fire import Fire
from tqdm import tqdm
import itertools
import random
import pickle
import torch
import zstd
import os

# Self imports
from tokenizer import CodeRangeTokenizer
from common import (
    get_args,
    param_group,
    mean_pooling,
    cross_entropy_loss,
    recall_mrr,
    index_gather_with_grad,
    gather_with_grad,
)
from model import AsmEncoder, NewModel


def decompress_data(b_str):
    return pickle.loads(zstd.decompress(b_str))


def compress_data(obj):
    return zstd.compress(pickle.dumps(obj))


class BLADataset(Dataset):
    def __init__(self, path):
        self.ds = load_from_disk(path)

    def fetch_ds(self, idx, key: str = None):
        deserialize_methods = {
            "asm": decompress_data,
            "src": decompress_data,
            "no_strip_asm": decompress_data,
            "asm_range_list": decompress_data,
            "src_range_list": decompress_data,
            "src_asm_range_map": decompress_data,
            "description": decompress_data,
        }
        if key is None:
            result = self.ds[idx]
            for k, v in result.items():
                result[k] = deserialize_methods.get(k, lambda x: x)(v)
            return result
        else:
            return deserialize_methods.get(key, lambda x: x)(self.ds[idx][key])

    def __getitem__(self, idx):
        return self.fetch_ds(idx)

    def __len__(self):
        return len(self.ds)


def range_intersect(r1, r2):
    return r1[0] < r2[1] and r2[0] < r1[1]

class BinQueryCollator:
    def __init__(
        self,
        asm_tokenizer,
        src_tokenizer,
        desc_tokenizer,
        asm_max_length,
        src_max_length,
        desc_max_length,
    ):
        self.random = random.Random(42)
        self.asm_tokenizer = asm_tokenizer
        self.src_tokenizer = src_tokenizer
        self.desc_tokenizer = desc_tokenizer
        self.asm_max_length = asm_max_length
        self.src_max_length = src_max_length
        self.desc_max_length = desc_max_length

    def decide(self, prob: float = 0.5):
        return self.random.random() < prob

    def sample_desc(
        self, row, asm_token_range, src_token_range, src_range, sa_range_map
    ):
        exp = row["description"]
        func_desc = exp["function_description"]
        functionality = func_desc["functionality"]
        implementation = func_desc["implementation"]
        labels = func_desc["labels"]
        global_desc = functionality + "\n" + implementation + "\n" + " ".join(labels)
        frag_desc = exp["snippet_descriptions"]
        frags = list(frag_desc.keys())
        if len(frags) > 3:
            frags = self.random.sample(frags, 3)

        desc_map = {}
        for frag in frags:
            desc = self.random.choice(frag_desc[frag])["description"]
            related_sr_indices = []
            for sr_idx, sr in enumerate(src_range):
                if range_intersect(sr, frag):
                    related_sr_indices.append(sr_idx)
            related_ar_indices = []
            for sr_idx in related_sr_indices:
                related_ar_indices.extend(sa_range_map[sr_idx])

            related_asm_token_ranges = [asm_token_range[i] for i in related_ar_indices]
            related_src_token_ranges = [src_token_range[i] for i in related_sr_indices]
            desc_map[desc] = {
                "asm": related_asm_token_ranges,
                "src": related_src_token_ranges,
            }
        return global_desc, desc_map

    def sample_desc_batch(
        self,
        batch,
        asm_token_range_list: dict,
        src_token_range_list: dict,
        src_range_list: list,
        asm_src_range_map: list,
    ):
        desc_map_list = []
        desc_list = []
        for idx, row in enumerate(batch):
            single_map = {}
            global_desc, desc_map = self.sample_desc(
                row,
                asm_token_range_list[idx],
                src_token_range_list[idx],
                src_range_list[idx],
                asm_src_range_map[idx],
            )
            desc_list.append(global_desc)
            single_map["global_desc"] = len(desc_list) - 1
            single_map["frag_desc"] = {}
            for desc, content in desc_map.items():
                desc_list.append(desc)
                single_map["frag_desc"][len(desc_list) - 1] = content
            desc_map_list.append(single_map)
        return desc_list, desc_map_list

    def __call__(self, batch):
        asm = [b["asm"] for b in batch]
        src = [b["src"] for b in batch]
        asm_range_list = [b["asm_range_list"] for b in batch]
        src_range_list = [b["src_range_list"] for b in batch]
        src_asm_range_map = [b["src_asm_range_map"] for b in batch]

        asm_tokenized = self.asm_tokenizer(
            code=asm, code_range=asm_range_list, max_length=self.asm_max_length
        )
        src_tokenized = self.src_tokenizer(
            code=src, code_range=src_range_list, max_length=self.src_max_length
        )
        desc, desc_map = self.sample_desc_batch(
            batch,
            asm_tokenized["token_range"],
            src_tokenized["token_range"],
            src_range_list,
            src_asm_range_map,
        )
        desc_tokenized = self.desc_tokenizer(
            desc,
            max_length=self.desc_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            asm_tokenized,
            src_tokenized,
            desc_tokenized,
            desc_map,
        )


class BinQueryModel(torch.nn.Module):
    def __init__(self, asm_model, src_model, desc_model):
        super(BinQueryModel, self).__init__()
        self.asm_model = asm_model
        self.src_model = src_model
        self.desc_model = desc_model

    def forward(
        self,
        asm_input_ids,
        asm_attention_mask,
        asm_token_type_ids,
        src_input_ids,
        src_attention_mask,
        desc_input_ids,
        desc_attention_mask,
    ):
        asm_output = self.asm_model(
            input_ids=asm_input_ids,
            attention_mask=asm_attention_mask,
            token_type_ids=asm_token_type_ids,
        )
        src_output = self.src_model(
            input_ids=src_input_ids, attention_mask=src_attention_mask
        )
        desc_output = self.desc_model(
            input_ids=desc_input_ids, attention_mask=desc_attention_mask
        )
        return asm_output, src_output, desc_output


def save_model(model, checkpoint, step):
    os.makedirs(checkpoint, exist_ok=True)
    to_save = os.path.join(checkpoint, str(step))
    model.asm_model.save_pretrained(
        os.path.join(to_save, "asm"), safe_serialization=False
    )
    model.src_model.save_pretrained(
        os.path.join(to_save, "src"), safe_serialization=False
    )
    model.desc_model.save_pretrained(
        os.path.join(to_save, "desc"), safe_serialization=False
    )
    if os.path.exists(os.path.join(checkpoint, "latest")):
        os.unlink(os.path.join(checkpoint, "latest"))
    os.symlink(str(step), os.path.join(checkpoint, "latest"))


def get_global_embeddings(asm_hs, src_hs, desc_hs, asm_am, src_am, desc_am, desc_map):
    asm_emb = mean_pooling(asm_hs, asm_am)
    src_emb = mean_pooling(src_hs, src_am)

    # Get global description embeddings
    desc_indices = [single_map["global_desc"] for single_map in desc_map]
    desc_emb = mean_pooling(desc_hs, desc_am)
    desc_emb = desc_emb[desc_indices]
    return asm_emb, src_emb, desc_emb


def snippet_embedding(hs, am, range_list, c: float = 2.0):
    am = am.to(hs)
    for start, end in range_list:
        am[start:end] *= c
    hs = hs * am.unsqueeze(-1)
    embedding = hs.sum(0) / am.sum()
    return embedding


def get_snippet_embeddings(
    asm_hs,
    src_hs,
    desc_hs,
    asm_am,
    src_am,
    desc_am,
    desc_map,
    c: float = 100.0,
):
    desc_embs = mean_pooling(desc_hs, desc_am)
    asm_emb_list = []
    src_emb_list = []
    desc_emb_list = []
    per_group_length = []
    for idx, single_map in enumerate(desc_map):
        per_group_length.append(len(single_map["frag_desc"]))
        for desc_idx, as_range_info in single_map["frag_desc"].items():
            asm_token_ranges = as_range_info["asm"]
            src_token_ranges = as_range_info["src"]

            asm_emb = snippet_embedding(asm_hs[idx], asm_am[idx], asm_token_ranges, c)
            src_emb = snippet_embedding(src_hs[idx], src_am[idx], src_token_ranges, c)
            desc_emb = desc_embs[desc_idx]

            asm_emb_list.append(asm_emb)
            src_emb_list.append(src_emb)
            desc_emb_list.append(desc_emb)

    asm_frag_emb = torch.stack(asm_emb_list)
    src_frag_emb = torch.stack(src_emb_list)
    desc_frag_emb = torch.stack(desc_emb_list)

    extended_group_length = [0] + list(itertools.accumulate(per_group_length))
    group_ranges = list(zip(extended_group_length[:-1], extended_group_length[1:]))
    return asm_frag_emb, src_frag_emb, desc_frag_emb, group_ranges

def calc_sac_loss(
    query,
    key,
    x_range_list: List[int],
    y_range_list: List[int],
    T: float = 0.07,
    labels: torch.Tensor = None,
):
    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)

    logits = query @ key.t() / T
    mask = torch.zeros_like(logits, dtype=torch.bool)
    start_idx = 0
    
    """
    Snippets belonging to the same function are neither positive nor negative samples.
    """
    
    for idx in range(len(x_range_list)):
        x_start, x_end = x_range_list[idx]
        y_start, y_end = y_range_list[idx]
        size = x_end - x_start
        assert (y_end - y_start) == size
        mask[x_start:x_end, y_start:y_end] = ~torch.eye(size, dtype=torch.bool)
    mask = mask.to(logits.device)
    logits = logits.masked_fill(mask, float("-inf"))

    if labels is None:
        labels = torch.arange(query.shape[0], device=query.device)
    else:
        labels = labels.to(query.device)
    return F.cross_entropy(logits, labels)


def main(
    asm_model: Optional[str] = None,
    asm_max_length: int = 1024,
    src_model: Optional[str] = None,
    src_max_length: int = 1024,
    desc_model: Optional[str] = None,
    desc_max_length: int = 512,
    asm_tokenizer: Optional[str] = None,
    src_tokenizer: Optional[str] = None,
    desc_tokenizer: Optional[str] = None,
    dataset: Optional[str] = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    epochs: int = 1,
    num_data: Optional[int] = None,
    wandb: Optional[str] = None,
    warmup: Union[float, int] = 0.05,
    weight_decay: float = 0.01,
    learning_rate: float = 1e-4,
    checkpoint: Optional[str] = None,
    resume_from: Optional[Union[str, int]] = None,
    save_every: int = 1000,
    c: float = 10.0,
):
    if resume_from is not None and not os.path.exists(
        os.path.join(checkpoint, str(resume_from))
    ):
        resume_from = None

    # accelerator = Accelerator(
    #     log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps
    # )
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="bla",
        config=get_args(),
        init_kwargs={
            "wandb": {
                "name": wandb if wandb else "ERROR",
                "mode": "online" if wandb else "disabled",
            }
        },
    )

    # Dataset
    asm_tokenizer = CodeRangeTokenizer.from_pretrained(asm_tokenizer)
    src_tokenizer = CodeRangeTokenizer.from_pretrained(src_tokenizer)
    desc_tokenizer = AutoTokenizer.from_pretrained(desc_tokenizer)
    collator = BinQueryCollator(
        asm_tokenizer,
        src_tokenizer,
        desc_tokenizer,
        asm_max_length,
        src_max_length,
        desc_max_length,
    )
    ds = BLADataset(dataset)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=16,
        prefetch_factor=4,
    )

    # Model
    if checkpoint is not None and resume_from is not None:
        resume_from_path = os.path.join(checkpoint, str(resume_from))
        asm_model_path = os.path.join(resume_from_path, "asm")
        src_model_path = os.path.join(resume_from_path, "src")
        desc_model_path = os.path.join(resume_from_path, "desc")
        asm_model = AsmEncoder.from_pretrained(asm_model_path)
        src_model = NewModel.from_pretrained(src_model_path)
        desc_model = MPNetModel.from_pretrained(desc_model_path)
    else:
        asm_model = AsmEncoder.from_pretrained(asm_model)
        src_model = NewModel.from_pretrained(src_model)
        desc_model = MPNetModel.from_pretrained(desc_model)

    model = BinQueryModel(asm_model, src_model, desc_model)

    model.requires_grad_(True)

    # Optimizer
    grp = []
    grp.extend(param_group(model, learning_rate, weight_decay))
    optimizer = AdamW(grp)

    # Scheduler
    if num_data is None:
        num_data = len(dl) * epochs
    num_steps = num_data // (batch_size * accelerator.num_processes)
    if warmup < 1:
        warmup_steps = int(num_steps * warmup)
    else:
        warmup_steps = int(warmup)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )
    accelerator.print(f"num_warmpup_steps: {warmup_steps}")

    # Prepare
    model, optimizer, scheduler, dl = accelerator.prepare(
        model, optimizer, scheduler, dl 
    )

    if checkpoint is not None and resume_from is not None:
        try:
            step = int(resume_from)
        except ValueError:
            resume_from_path = os.path.join(checkpoint, str(resume_from))
            while os.path.islink(resume_from_path):
                resume_from_path = os.readlink(resume_from_path)
            step = int(os.path.basename(resume_from_path))
    else:
        step = 0

    prog_bar = tqdm(
        total=num_steps, disable=not accelerator.is_main_process, initial=step
    )
    while step <= num_steps:
        for (
            asm_tokenized,
            src_tokenized,
            desc_tokenized,
            desc_map,
        ) in dl:
            model.train()
            try:
                with accelerator.accumulate(model):
                    asm_input_ids = asm_tokenized["input_ids"].to(accelerator.device)
                    asm_attention_mask = asm_tokenized["attention_mask"].to(
                        accelerator.device
                    )
                    asm_token_type_ids = asm_tokenized["token_type_ids"].to(
                        accelerator.device
                    )
                    src_input_ids = src_tokenized["input_ids"].to(accelerator.device)
                    src_attention_mask = src_tokenized["attention_mask"].to(
                        accelerator.device
                    )
                    desc_input_ids = desc_tokenized["input_ids"].to(accelerator.device)
                    desc_attention_mask = desc_tokenized["attention_mask"].to(
                        accelerator.device
                    )
                    asm_output, src_output, desc_output = model(
                        asm_input_ids,
                        asm_attention_mask,
                        asm_token_type_ids,
                        src_input_ids,
                        src_attention_mask,
                        desc_input_ids,
                        desc_attention_mask,
                    )

                    asm_emb, src_emb, desc_emb = get_global_embeddings(
                        asm_output.last_hidden_state,
                        src_output.last_hidden_state,
                        desc_output.last_hidden_state,
                        asm_attention_mask,
                        src_attention_mask,
                        desc_attention_mask,
                        desc_map,
                    )

                    asm_frag_emb, src_frag_emb, desc_frag_emb, group_ranges = (
                        get_snippet_embeddings(
                            asm_output.last_hidden_state,
                            src_output.last_hidden_state,
                            desc_output.last_hidden_state,
                            asm_attention_mask,
                            src_attention_mask,
                            desc_attention_mask,
                            desc_map,
                            c,
                        )
                    )
                    fac_loss = 0
                    fac_loss += cross_entropy_loss(asm_emb, src_emb)
                    fac_loss += cross_entropy_loss(asm_emb, desc_emb)
                    fac_loss += cross_entropy_loss(src_emb, desc_emb)
                    fac_loss += cross_entropy_loss(src_emb, asm_emb)
                    fac_loss += cross_entropy_loss(desc_emb, asm_emb)
                    fac_loss += cross_entropy_loss(desc_emb, src_emb)
                    fac_loss /= 6

                    sac_loss = 0
                    sac_loss += calc_sac_loss(
                        asm_frag_emb,
                        src_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss += calc_sac_loss(
                        asm_frag_emb,
                        desc_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss += calc_sac_loss(
                        src_frag_emb,
                        desc_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss += calc_sac_loss(
                        src_frag_emb,
                        asm_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss += calc_sac_loss(
                        desc_frag_emb,
                        asm_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss += calc_sac_loss(
                        desc_frag_emb,
                        src_frag_emb,
                        x_range_list=group_ranges,
                        y_range_list=group_ranges,
                    )
                    sac_loss /= 6

                    binquery_loss = fac_loss + sac_loss

                    accelerator.backward(binquery_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            except Exception as e:
                # Handle CUDA OOM
                if "CUDA out of memory" in str(e):
                    accelerator.print(
                        "CUDA out of memory at step: ", step, " skipping batch"
                    )
                    continue

            accelerator.log(
                {
                    "loss": binquery_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=step,
            )
            prog_bar.set_description(
                f"loss: {binquery_loss.item():.8f} | lr: {scheduler.get_last_lr()[0]:.8f}"
            )

            prog_bar.update(1)
            step += 1

            if step % save_every == 0 and accelerator.is_main_process and checkpoint is not None:
                save_model(accelerator.unwrap_model(model), checkpoint, step)

            if step >= num_steps:
                break

    if accelerator.is_main_process and checkpoint is not None:
        save_model(accelerator.unwrap_model(model), checkpoint, step)


if __name__ == "__main__":
    main(
        asm_model="models/example_asm",
        asm_max_length=1024,
        src_model="models/example_src",
        src_max_length=1024,
        desc_model="models/example_desc",
        desc_max_length=512,
        asm_tokenizer="tokenizers/asm",
        src_tokenizer="tokenizers/src",
        desc_tokenizer="tokenizers/desc",
        dataset="dataset/data/function_snippets_with_descriptions",
        batch_size=4,
    )
