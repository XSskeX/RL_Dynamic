import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/DAPO17k_for_Diffing")
    parser.add_argument("--local_dataset_path", default="HuggingFaceH4/aime_2024")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['train']
    
    data_source = "HuggingFaceH4/aime_2024"
    instruction_following = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"

    def make_map_fn(split):
        def process_fn(example):
            idx = example.pop("id")
            question_raw = example.pop("problem")
            question = instruction_following + question_raw
            answer_raw = str(example.pop("answer")).strip()
            answer = str(answer_raw).strip()
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "instruction_id_list": [],
                    "kwargs": [], 
                    "question": question_raw,
                },
            }
            return data
        return process_fn
    #train_dataset = train_dataset.map(function=make_map_fn("train"))
    #test_dataset = test_dataset.map(function=make_map_fn("test"))
    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    #train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
