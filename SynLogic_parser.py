import argparse
import os
import re
import ast
import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/SynLogic")
    parser.add_argument("--local_dataset_path", default="MiniMaxAI/SynLogic")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "easy")
    raw_dataset = dataset['train']
    test_raw_dataset = dataset['validation']
    #split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    #train_dataset, test_dataset = split['train'], split['test']
    
    data_source = "MiniMaxAI/SynLogic"
    #instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["prompt"][0]["content"]
            game_data = example["extra_info"]["game_data_str"]
            data_class = example["data_source"]

            return {
                "data_source": "MiniMaxAI/SynLogic",
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "logical_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "just for placeholder",
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "game_data_str": game_data,
                    "question_class": data_class,
                    "instruction_id_list": [],
                    "kwargs": [],
                    "question": "just for placeholder",
                },
            }

        return process_fn
    #train_dataset = train_dataset.map(function=make_map_fn("train"))
    #test_dataset = test_dataset.map(function=make_map_fn("test"))
    train_dataset = raw_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_raw_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    #train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
