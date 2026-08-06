import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/GooseReason_stem")
    parser.add_argument("--local_dataset_path", default="nvidia/Nemotron-Research-GooseReason-0.7M")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path)
    raw_dataset = dataset['stem']
    split = raw_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset, test_dataset = split['train'], split['test']
    
    data_source = "nvidia/Nemotron-Research-GooseReason-0.7M"

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            options = example.pop("options")
            option_text = "\n".join(
                f"{chr(ord('A') + i)}. {option}"
                for i, option in enumerate(options)
            )
            question = (
                f"{question_raw}\n\n"
                f"**Options:**\n{option_text}\n\n"
                r"Output the letter of the correct option in \boxed{}."
            )
            answer_raw = str(example.pop("answer")).strip()
            answer = str(answer_raw).strip()
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "instruction_id_list": [],
                    "kwargs": [], 
                    "question": question_raw,
                    "metadata_json": "{}",
                },
            }
            return data
        return process_fn
    #train_dataset = train_dataset.map(function=make_map_fn("train"))
    #test_dataset = test_dataset.map(function=make_map_fn("test"))
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
