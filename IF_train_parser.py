#!/usr/bin/env python3
"""Preprocess IFBench/IFEvalG data for verl rule-based RL training."""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Any

import datasets

from verl.utils.reward_score.IF_Bench_Train import instructions_registry


DATA_SOURCE = "allenai/IFBench_train"


def parse_ground_truth(raw_ground_truth: Any) -> list[dict[str, Any]] | None:
    """Parse ground_truth into a non-empty list of dictionaries."""

    ground_truth = raw_ground_truth

    if isinstance(ground_truth, str):
        ground_truth = ground_truth.strip()

        if not ground_truth:
            return None

        # 优先按照 JSON 解析；失败后兼容 Python 字面量格式。
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            try:
                ground_truth = ast.literal_eval(ground_truth)
            except (ValueError, SyntaxError):
                return None

    if not isinstance(ground_truth, list) or not ground_truth:
        return None

    if not all(isinstance(item, dict) for item in ground_truth):
        return None

    return ground_truth


def normalize_instruction_metadata(
    raw_ground_truth: Any,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """
    Validate and normalize instruction metadata.

    Rules:
    1. instruction_id and kwargs must be aligned lists.
    2. None is allowed only for instructions that require no arguments.
    3. For no-argument instructions, None is converted to {}.
    4. Parameterized instructions with kwargs=None are rejected.
    5. Unknown instruction IDs and malformed kwargs are rejected.
    """

    ground_truth = parse_ground_truth(raw_ground_truth)
    if ground_truth is None:
        return None

    # 当前数据格式中，instruction metadata 位于 ground_truth[0]。
    instruction_info = ground_truth[0]

    instruction_ids = instruction_info.get("instruction_id")
    raw_kwargs_list = instruction_info.get("kwargs")

    if not isinstance(instruction_ids, list):
        return None

    if not isinstance(raw_kwargs_list, list):
        return None

    if not instruction_ids:
        return None

    if len(instruction_ids) != len(raw_kwargs_list):
        return None

    normalized_instruction_ids: list[str] = []
    normalized_kwargs_list: list[dict[str, Any]] = []

    for instruction_id, raw_kwargs in zip(
        instruction_ids,
        raw_kwargs_list,
        strict=True,
    ):
        if (
            not isinstance(instruction_id, str)
            or not instruction_id.strip()
        ):
            return None

        instruction_id = instruction_id.strip()

        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(
            instruction_id
        )
        if instruction_cls is None:
            # 当前评分器没有注册该 instruction。
            return None

        instruction = instruction_cls(instruction_id)

        try:
            expected_keys = instruction.get_instruction_args_keys()
        except Exception:
            # instruction 实现本身异常，避免产生无法评分的数据。
            return None

        if expected_keys is None:
            expected_keys = []

        if raw_kwargs is None:
            if expected_keys:
                # 该 instruction 需要参数，但数据没有提供参数。
                return None

            # 无参数 instruction：
            # None 与 {} 语义等价，但 evaluator 要求它必须是 dict。
            normalized_kwargs = {}

        elif isinstance(raw_kwargs, dict):
            # 与原 evaluator 一致，去除值为 None 的字段。
            normalized_kwargs = {
                key: value
                for key, value in raw_kwargs.items()
                if value is not None
            }

        else:
            # evaluator 后续会调用 kwargs.items()，
            # 因此 list、string、number 等类型均视为非法。
            return None

        normalized_instruction_ids.append(instruction_id)
        normalized_kwargs_list.append(normalized_kwargs)

    return normalized_instruction_ids, normalized_kwargs_list


def has_valid_messages(example: dict[str, Any]) -> bool:
    """Validate the chat-message structure."""

    messages = example.get("messages")

    if not isinstance(messages, list) or not messages:
        return False

    first_message = messages[0]

    if not isinstance(first_message, dict):
        return False

    content = first_message.get("content")

    return isinstance(content, str) and bool(content.strip())


def keep_valid_example(example: dict[str, Any]) -> bool:
    """Return True only when the example can be evaluated safely."""

    if not has_valid_messages(example):
        return False

    normalized = normalize_instruction_metadata(
        example.get("ground_truth")
    )

    return normalized is not None


def make_map_fn(split: str):
    """Create the final verl-format mapping function."""

    def process_fn(
        example: dict[str, Any],
        filtered_index: int,
    ) -> dict[str, Any]:
        normalized = normalize_instruction_metadata(
            example["ground_truth"]
        )

        # 经过 filter 后理论上不可能发生，仅作为防御性检查。
        if normalized is None:
            raise ValueError(
                "Invalid sample survived filtering: "
                f"original_index={example.get('original_index')}"
            )

        instruction_id_list, kwargs_list = normalized

        question = example["messages"][0]["content"].strip()

        original_index = example.get(
            "original_index",
            filtered_index,
        )

        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "Instruction_Following",
            "reward_model": {
                "style": "rule",
                "ground_truth": "just for placeholder",
            },
            "extra_info": {
                "split": split,
                "index": original_index,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
                "question": question,
            },
        }

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_dir",
        default="/share/nlp/baijun/shuhan/IF_Bench_train",
        help="Directory used to save the processed parquet file.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default="allenai/IF_multi_constraints_upto5_no_lang",
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--dataset_config",
        default="default",
        help="Hugging Face dataset configuration name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Source dataset split.",
    )
    parser.add_argument(
        "--output_name",
        default="train.parquet",
        help="Output parquet filename.",
    )

    args = parser.parse_args()

    dataset = datasets.load_dataset(
        args.local_dataset_path,
        args.dataset_config,
    )

    if args.split not in dataset:
        raise KeyError(
            f"Split {args.split!r} does not exist. "
            f"Available splits: {list(dataset.keys())}"
        )

    raw_dataset = dataset[args.split]
    original_size = len(raw_dataset)

    # 在过滤前保存原始数据索引。
    raw_dataset = raw_dataset.map(
        lambda _example, index: {
            "original_index": index,
        },
        with_indices=True,
        desc="Adding original indices",
        load_from_cache_file=False,
    )

    filtered_dataset = raw_dataset.filter(
        keep_valid_example,
        desc="Filtering invalid IFBench samples",
        load_from_cache_file=False,
    )

    filtered_size = len(filtered_dataset)
    removed_size = original_size - filtered_size

    print("=" * 70)
    print(f"Original samples : {original_size}")
    print(f"Valid samples    : {filtered_size}")
    print(f"Removed samples  : {removed_size}")

    if original_size:
        removed_ratio = removed_size / original_size
        print(f"Removed ratio    : {removed_ratio:.2%}")

    print("=" * 70)

    if filtered_size == 0:
        raise RuntimeError(
            "All samples were filtered out. "
            "Check the dataset structure and instruction registry."
        )

    processed_dataset = filtered_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=filtered_dataset.column_names,
        desc="Converting to verl training format",
        load_from_cache_file=False,
    )

    os.makedirs(args.local_dir, exist_ok=True)

    output_path = os.path.join(
        args.local_dir,
        args.output_name,
    )

    processed_dataset.to_parquet(output_path)

    print(f"Saved processed dataset to: {output_path}")
    print(f"Final dataset columns: {processed_dataset.column_names}")


if __name__ == "__main__":
    main()