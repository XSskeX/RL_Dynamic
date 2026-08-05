import argparse
import os
import re
import ast
from datasets import load_dataset, concatenate_datasets
import json
from verl.utils.hdfs_io import copy, makedirs
system_instruction='''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/KK")
    parser.add_argument("--local_dataset_path", default="K-and-K/knights-and-knaves")
    args = parser.parse_args()
    train_dataset = load_dataset(
        args.local_dataset_path,
        "train",
        split="2ppl",
    )
    test_dataset = load_dataset(
        args.local_dataset_path,
        "test",
        split="2ppl",
    )
    for i in range(3, 9):
        train_i = load_dataset(
            args.local_dataset_path,
            "train",
            split=f"{i}ppl",
        )
        test_i = load_dataset(
            args.local_dataset_path,
            "test",
            split=f"{i}ppl",
        )
        train_dataset = concatenate_datasets([
            train_dataset,
            train_i,
        ])
        test_dataset = concatenate_datasets([
            test_dataset,
            test_i,
        ])
    train_dataset = train_dataset.shuffle(seed=42)
    
    data_source = "K-and-K/knights-and-knaves"

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["quiz"]
            solution_text = example["solution_text"]
            question = (
                system_instruction
                + f"\n\n### Question: {question_raw}\n"
                + "### Answer: Let's think step by step"
            )

            return {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logical_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution_text,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "instruction_id_list": [],
                    "kwargs": [], 
                    "question": question_raw,
                    "metadata_json": "{}",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    #train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
