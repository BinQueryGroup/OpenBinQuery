from transformers import PreTrainedTokenizerFast
from typing import List, Tuple, Dict, Union, Optional
from itertools import accumulate
import torch

CODE_TYPE = List[str]
CODE_RANGE_TYPE = List[Tuple[int, int]]


def is_code_range_type(obj) -> bool:
    """Check if the given object is of type CODE_RANGE_TYPE."""
    return isinstance(obj, list) and all(
        isinstance(item, tuple) and len(item) == 2 for item in obj
    )


def is_code_range_list(obj) -> bool:
    """Check if the given object is a list of CODE_RANGE_TYPE."""
    return isinstance(obj, list) and all(is_code_range_type(item) for item in obj)


def is_code_type(obj) -> bool:
    """Check if the given object is of type SRC_CODE_TYPE."""
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)


def is_code_list(obj) -> bool:
    """Check if the given object is a list of SRC_CODE_TYPE."""
    return isinstance(obj, list) and all(is_code_type(item) for item in obj)


class CodeRangeTokenizer(PreTrainedTokenizerFast):
    def __call__(
        self,
        code: Union[List[CODE_TYPE], CODE_TYPE],
        code_range: Optional[Union[List[CODE_RANGE_TYPE], CODE_RANGE_TYPE]] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        batch_mode = is_code_list(code)
        has_code_range = code_range is not None

        code_list = code if batch_mode else [code]
        if has_code_range:
            code_range_list = code_range if batch_mode else [code_range]
        else:
            code_range_list = [[(0, len(code))] for code in code_list]

        if len(code_list) != len(code_range_list):
            raise ValueError("src_code and code_range must have the same length.")

        # union_src_code_list = []
        # length_list = []
        # for func_idx, code_range in enumerate(code_range_list):
        #     length_list.append(len(code_range))
        #     for start, end in code_range:
        #         union_src_code_list.append(
        #             "\n".join(code_list[func_idx][start:end])
        #         )

        per_func_line_num_list = [len(code) for code in code_list]
        _tmp = [0] + list(accumulate(per_func_line_num_list))
        per_func_line_range_list = [
            (start, end) for start, end in zip(_tmp[:-1], _tmp[1:])
        ]

        union_code_list = [row for code in code_list for row in code]
        tokenized = super(CodeRangeTokenizer, self).__call__(
            union_code_list,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

        per_func_input_ids = []
        per_func_attention_mask = []
        per_func_token_type_ids = []
        per_func_token_range = []

        union_input_ids = tokenized["input_ids"]
        union_attention_mask = tokenized["attention_mask"]
        for func_idx, (func_start, func_end) in enumerate(per_func_line_range_list):
            func_input_id_list = union_input_ids[func_start:func_end]
            func_attention_mask_list = union_attention_mask[func_start:func_end]
            func_token_type_ids_list = []
            for row_idx, row_input_ids in enumerate(func_input_id_list):
                row_token_types = [f"[INSTR{row_idx + 1}]"] * len(row_input_ids)
                row_token_type_ids = self.convert_tokens_to_ids(row_token_types)
                func_token_type_ids_list.append(row_token_type_ids)
            func_line_range_list = code_range_list[func_idx]
            row_length_list = [len(token_row) for token_row in func_input_id_list]
            _tmp = [0] + list(accumulate(row_length_list))
            row_token_ranges = [(start, end) for start, end in zip(_tmp[:-1], _tmp[1:])]
            token_range_list = []
            for line_start, line_end in func_line_range_list:
                start = row_token_ranges[line_start][0]
                end = row_token_ranges[line_end - 1][1]
                token_range_list.append((start, end))

            func_input_ids = [i for row in func_input_id_list for i in row]
            func_attention_mask = [i for row in func_attention_mask_list for i in row]
            func_token_type_ids = [i for row in func_token_type_ids_list for i in row]
            per_func_input_ids.append(func_input_ids)
            per_func_attention_mask.append(func_attention_mask)
            per_func_token_type_ids.append(func_token_type_ids)
            per_func_token_range.append(token_range_list)

        # token_type_list= []

        # union_range_list = [0] + list(accumulate(length_list))
        # tokenized = super(CodeRangeTokenizer, self).__call__(
        #     union_src_code_list, max_length=max_length, truncation=True
        # )

        # per_func_input_ids = []
        # per_func_attention_mask = []
        # per_func_token_range = []

        # for idx in range(len(code_list)):
        #     start = union_range_list[idx]
        #     end = union_range_list[idx + 1]
        #     input_ids_list = tokenized["input_ids"][start:end]
        #     attention_mask_list = tokenized["attention_mask"][start:end]

        #     input_ids = [i for row in input_ids_list for i in row]
        #     attention_mask = [i for row in attention_mask_list for i in row]

        #     length_list = [len(input_ids) for input_ids in input_ids_list]
        #     accumulate_length_list = [0] + list(accumulate(length_list))
        #     token_ranges = [
        #         (accumulate_length_list[i], accumulate_length_list[i + 1])
        #         for i in range(len(accumulate_length_list) - 1)
        #     ]

        #     per_func_input_ids.append(input_ids)
        #     per_func_attention_mask.append(attention_mask)
        #     per_func_token_range.append(token_ranges)

        max_length = max_length if max_length is not None else self.model_max_length

        new_per_func_input_ids = []
        new_per_func_attention_mask = []
        new_per_func_token_type_ids = []
        new_per_func_token_range = []

        for data_idx in range(len(per_func_input_ids)):
            input_ids = per_func_input_ids[data_idx]
            attention_mask = per_func_attention_mask[data_idx]
            token_type_ids = per_func_token_type_ids[data_idx]
            token_range = per_func_token_range[data_idx]

            new_token_range = []
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]
                for start, end in token_range:
                    if start >= max_length:
                        start = max_length
                    if end > max_length:
                        end = max_length
                    new_token_range.append((start, end))
                token_range = new_token_range

            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids += [self.pad_token_id] * pad_length
                token_type_ids += [0] * pad_length
                attention_mask += [0] * pad_length

            new_per_func_input_ids.append(input_ids)
            new_per_func_attention_mask.append(attention_mask)
            new_per_func_token_type_ids.append(token_type_ids)
            new_per_func_token_range.append(token_range)

        input_ids = torch.tensor(new_per_func_input_ids)
        attention_mask = torch.tensor(new_per_func_attention_mask)
        token_type_ids = torch.tensor(new_per_func_token_type_ids)
        token_range = new_per_func_token_range

        if not batch_mode:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
            token_type_ids = token_type_ids[0]
            token_range = token_range[0]

        if has_code_range:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "token_range": token_range,
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
