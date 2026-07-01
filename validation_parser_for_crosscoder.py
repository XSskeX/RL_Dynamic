import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res



def parse_MMLU_Pro():
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", "default")
    raw_dataset = dataset['validation']
    dev_df = preprocess(raw_dataset)

    data_source = "TIGER-Lab/MMLU-Pro"

    def make_map_fn(split):
        def process_fn(example):
            idx = example.pop("question_id")

            question_raw = example.pop("question")
            options = example.pop("options")
            cot_content = example.pop("cot_content")
            question = format_example(question_raw, options)
            
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
    return test_dataset


def parse_AIME_2024():
    dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", "default")
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

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    return test_dataset


def parse_IF_bench():
    dataset = datasets.load_dataset("allenai/IFBench_test", "default")
    raw_dataset = dataset['train']

    data_source = "allenai/IFBench_test"

    def make_map_fn(split):
        def process_fn(example):
            idx = int(example.pop("key"))
            question_raw = example.pop("prompt")
            instruction_id_list = example.pop("instruction_id_list")
            kwargs = example.pop("kwargs")
            question = question_raw
            answer = "just for placeholder"
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "Instruction_Following",
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

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)

    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)

    return test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/DAPO17k_for_Diffing")
    args = parser.parse_args()
    mmlu_test_dataset = parse_MMLU_Pro()
    aime_test_dataset = parse_AIME_2024()
    if_test_dataset = parse_IF_bench()
    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    dataset_list = [mmlu_test_dataset, aime_test_dataset, if_test_dataset]
    test_dataset = datasets.concatenate_datasets(dataset_list)
    test_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))