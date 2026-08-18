import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/OpenThoughts_for_Diffing")
    parser.add_argument("--local_dataset_path", default="open-r1/OpenThoughts-114k-math")
    parser.add_argument("--is_test", default=False)
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['train']
    train_raw_dataset, test_raw_dataset = raw_dataset.train_test_split(test_size=0.95, seed=42)
        
    data_source = "open-r1/OpenThoughts-114k-math"
    #instruction_following = ""

    def make_map_fn(split):
        def process_fn(example):
            idx = 1
            conversations = example.pop("conversations")
            instruction_following = conversations[0]['value']
            question_raw = conversations[1]['value']
            question = instruction_following + question_raw
            answer_raw = example.pop("reward_model")['ground_truth']
            answer = str(answer_raw).strip()
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "",
                    "index": "",
                    "instruction_id_list": [],
                    "kwargs": [], 
                    "question": "",
                },
            }
            return data
        return process_fn

    train_dataset = train_raw_dataset.map(function=make_map_fn("train"), remove_columns=train_raw_dataset.column_names)
    test_dataset = test_raw_dataset.map(function=make_map_fn("test"), remove_columns=test_raw_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))