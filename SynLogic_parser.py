import argparse
import os
import re
import ast
import datasets

from verl.utils.hdfs_io import copy, makedirs

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
            question_raw = parse_ground_truth(example.pop("prompt")[0])["content"]
            if question_raw is None:
                raise ValueError(
                    "Invalid sample survived filtering: "
                    f"original_index={example.get('original_index')}"
                )
            question = question_raw
            game_data = parse_ground_truth(example.pop("extra_info"))["game_data_str"]
            if game_data is None:
                raise ValueError(
                    "Invalid sample survived filtering: "
                    f"original_index={example.get('original_index')}"
                )
            data_class = example.pop("data_source")
            #answer_raw = str(example.pop("answer")).strip()
            answer = "just for placeholder"
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "logical_reasoning",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "instruction_id_list": [],
                    "kwargs": [], 
                    "question": question_raw,
                    "game_data_str": game_data,
                    "question_class": data_class,
                },
            }
            return data
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
