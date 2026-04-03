import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/DAPO17k")
    parser.add_argument("--local_dataset_path", default="BytedTsinghua-SIA/DAPO-Math-17k")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['train']

    
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    #instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    instruction_following = ""

    def make_map_fn(split):
        def process_fn(example):
            idx = 1
            question_raw = example.pop("prompt")[0]['content']
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("reward_model")['ground_truth']
            answer = str(answer_raw).strip()
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
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
    train_dataset = raw_dataset.map(function=make_map_fn("train"))

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    #train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
