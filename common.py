import inspect
import torch
import itertools
import torch.distributed as dist
from torch.distributed.nn import all_gather, all_to_all
import torch.nn.functional as F
from typing import List, Optional


def get_args():
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    caller_args = {arg: values[arg] for arg in args}
    return caller_args


def param_group(model, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def mean_pooling(token_embeddings, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(token_embeddings)
    else:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings)
        )
    result = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return result


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    else:
        return torch.cat(all_gather(tensor), dim=0)


def cross_entropy_loss(
    query, key, additional_negative=None, labels: torch.Tensor = None, T: float = 0.07
):
    # Shape of result1 and result2: (batch_size, embedding_dimension)
    # Shape of negative: (negative_size, embedding_dimension)
    # normalization on the last dimension
    if additional_negative is not None:
        key = torch.cat([key, additional_negative.to(key.device)], dim=0)
    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)

    logits = query @ key.t() / T

    if labels is None:
        labels = torch.arange(query.shape[0], device=query.device)
    else:
        labels = labels.to(query.device)
    return F.cross_entropy(logits, labels)





def frag_weighted_pooling(
    token_embeddings, range_list, attention_mask=None, weights: float = 2.0
):
    if attention_mask is None:
        attention_mask = torch.ones_like(token_embeddings)


def recall_mrr(anchor, positive, poolsize, k_list=[1]):
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)

    negative_cnt = poolsize - 1
    negative = anchor[-negative_cnt:]
    anchor = anchor[:-negative_cnt]
    positive = positive[:-negative_cnt]

    # l_pos = torch.einsum("nc,nc->n", [anchor, positive]).unsqueeze(-1)
    # l_neg = torch.einsum("ic,jc->ij", [anchor, negative])

    l_pos = (anchor * positive).sum(dim=-1).unsqueeze(-1)
    l_neg = anchor @ negative.t()
    logits = torch.cat([l_pos, l_neg], dim=-1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

    _, indices = torch.sort(logits, dim=-1, descending=True)
    ranks = torch.nonzero(indices == labels.unsqueeze(-1), as_tuple=False)[:, -1]
    return {
        "mrr": float(torch.reciprocal(ranks.float() + 1).mean()),
        "recall": {k: float((ranks < k).float().mean()) for k in k_list},
    }


# def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
#     if not dist.is_available() or not dist.is_initialized():
#         return tensor
#     else:
#         return torch.cat(dist.nn.all_gather(tensor), dim=0)


def index_gather_with_grad(t: torch.Tensor):
    assert len(t.shape) == 2
    if not dist.is_available() or not dist.is_initialized():
        return t, torch.arange(t.size(0), device=t.device, dtype=torch.long)
    else:
        process_idx = dist.get_rank()
        world_size = dist.get_world_size()
        emb_dim = t.size(1)
        tensor_shape = torch.tensor(t.shape, device=t.device)
        shape_list = [torch.empty_like(tensor_shape) for _ in range(world_size)]
        dist.all_gather(shape_list, tensor_shape)
        length_list = [shape[0].item() for shape in shape_list]

        max_length = max(length_list)

        if t.size(0) < max_length:
            t = torch.cat(
                [
                    t,
                    torch.zeros(
                        max_length - t.size(0), emb_dim, device=t.device, dtype=t.dtype
                    ),
                ],
                dim=0,
            )

        tensor_list = all_gather(t)
        tensor_list = [
            tensor[:length] for tensor, length in zip(tensor_list, length_list)
        ]

        # Get range
        length_list = [0] + list(itertools.accumulate(length_list))
        range_list = [(length_list[i], length_list[i + 1]) for i in range(world_size)]
        current_range_list = range_list[dist.get_rank()]
        indices = torch.arange(*current_range_list, device=t.device, dtype=torch.long)
        # indices = list(range(*current_range_list))

        return torch.cat(tensor_list, dim=0), indices


if __name__ == "__main__":
    # query = torch.randn(6, 128)
    # key = torch.randn(6, 128)
    # negative = torch.randn(4, 128)
    # import ipdb

    # ipdb.set_trace()
    # loss = grouped_cross_entropy_loss(query, key, [2, 4], negative)
    # print(loss)

    from accelerate import Accelerator

    accelerator = Accelerator()
    process_idx = dist.get_rank()
    if process_idx == 0:
        t = torch.randn(4, 5, requires_grad=True, device=accelerator.device)
    elif process_idx == 1:
        t = torch.randn(6, 5, requires_grad=True, device=accelerator.device)
    else:
        t = torch.randn(8, 5, requires_grad=True, device=accelerator.device)
    # t = torch.randn(4, 5, requires_grad=True, device=accelerator.device)
    print(t.grad)
    n, indices = index_gather_with_grad(t)
    n[indices] = 0
    print(f"{process_idx}: {n}")

    loss = n.sum()
    print(f"{process_idx}: {loss}")
    loss.backward()

    print(f"{process_idx}: {t.grad}")
    dist.barrier()
