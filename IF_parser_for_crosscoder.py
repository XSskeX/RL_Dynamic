import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/IF_bench_for_Diffing")
    parser.add_argument("--local_dataset_path", default="allenai/IF_multi_constraints_upto5_no_lang")
    parser.add_argument("--is_test", default=False)
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['train']
        
    data_source = "allenai/IF_multi_constraints_upto5_no_lang"
    instruction_following = ""

    def make_map_fn(split):
        def process_fn(example):
            idx = 1
            question_raw = example.pop("messages")[0]['content']
            question = question_raw + " " + instruction_following
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

    train_dataset = raw_dataset.map(function=make_map_fn("train"), remove_columns=raw_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
