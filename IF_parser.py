import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/RL_Dynamic/IF_Bench")
    parser.add_argument("--local_dataset_path", default="allenai/IFBench_test")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['train']
    #split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    #train_dataset, test_dataset = split['train'], split['test']
    
    data_source = "allenai/IFBench_test"
    #instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example):
            idx = int(example.pop("key"))
            question_raw = example.pop("prompt")
            instruction_id_list = example.pop("instruction_id_list")
            kwargs = example.pop("kwargs")
            question = question_raw
            #answer_raw = str(example.pop("answer")).strip()
            answer = "just for placeholder"
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "Instruction_Following",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs, 
                    "question": question_raw,
                },
            }
            return data
        return process_fn
    #train_dataset = train_dataset.map(function=make_map_fn("train"))
    #test_dataset = test_dataset.map(function=make_map_fn("test"))
    test_dataset = raw_dataset.map(function=make_map_fn("test"))

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    #train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
