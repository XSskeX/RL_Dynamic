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
            question_raw = example.pop("question")
            options = example.pop("options")
            cot_content = example.pop("cot_content")
            question = format_example(question_raw, options)
            
            data = {
                "prompt": question + cot_content,
            }
            return data
        return process_fn

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)

    return test_dataset


def parse_AIME_2024():
    dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", "default")
    raw_dataset = dataset['train']
    
    data_source = "HuggingFaceH4/aime_2024"
    instruction_following = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"

    def make_map_fn(split):
        def process_fn(example):
            question_raw = example.pop("problem")
            solution = example.pop("solution")
            question = instruction_following + question_raw + solution
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)
    return test_dataset

def parse_AIME_2025():
    dataset = datasets.load_dataset("MathArena/aime_2025", "default")
    raw_dataset = dataset['train']
    
    data_source = "MathArena/aime_2025"
    instruction_following = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"

    def make_map_fn(split):
        def process_fn(example):
            idx = example.pop("problem_idx")
            question_raw = example.pop("problem")
            question = instruction_following + question_raw
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)
    return test_dataset

def parse_AIME_2026():
    dataset = datasets.load_dataset("MathArena/aime_2026", "default")
    raw_dataset = dataset['train']
    
    data_source = "MathArena/aime_2026"
    instruction_following = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"

    def make_map_fn(split):
        def process_fn(example):
            question_raw = example.pop("problem")
            question = instruction_following + question_raw
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)
    return test_dataset


def parse_IF_bench():
    dataset = datasets.load_dataset("allenai/IFBench_test", "default")
    raw_dataset = dataset['train']

    data_source = "allenai/IFBench_test"

    def make_map_fn(split):
        def process_fn(example):
            question_raw = example.pop("prompt")
            question = question_raw
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    test_dataset = raw_dataset.map(function=make_map_fn("validation"), remove_columns=raw_dataset.column_names)

    return test_dataset

def parse_Dapo_17k():
    dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default")
    raw_dataset = dataset['train']
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    instruction_following = ""
    sampled_dataset = raw_dataset.shuffle(seed=42).select(range(1000))
    def make_map_fn(split):
        def process_fn(example):
            question_raw = example.pop("prompt")[0]['content']
            question = question_raw + " " + instruction_following
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    train_dataset = sampled_dataset.map(function=make_map_fn("train"))
    return train_dataset

def parse_IF_bench_train():
    dataset = datasets.load_dataset("allenai/IF_multi_constraints_upto5_no_lang", "default")
    raw_dataset = dataset['train']
    sampled_dataset = raw_dataset.shuffle(seed=42).select(range(1000))
    data_source = "allenai/IF_multi_constraints_upto5_no_lang"
    def make_map_fn(split):
        def process_fn(example):
            question = example["messages"][0]["content"].strip()
            data = {
                "prompt": question,
            }
            return data
        return process_fn

    train_dataset = sampled_dataset.map(function=make_map_fn("train"))
    return train_dataset

def parse_kk():
    system_instruction='''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

    You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

    CONCLUSION:
    (1) ...
    (2) ...
    (3) ...
    '''
    dataset = load_dataset(
        "K-and-K/knights-and-knaves",
        "train",
        split="3ppl",
    )
    sampled_dataset = dataset.shuffle(seed=42).select(range(1000))
    def make_map_fn(split):
        def process_fn(example):
            question_raw = example["quiz"]
            solution_text = example["solution_text"]
            question = (
                system_instruction
                + f"\n\n### Question: {question_raw}\n"
                + "### Answer: Let's think step by step"
            )
            data = {
                "prompt": question,
            }
            return data
        return process_fn
    train_dataset = sampled_dataset.map(function=make_map_fn("train"))
    return train_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/DAPO17k_for_Diffing")
    args = parser.parse_args()
    mmlu_test_dataset = parse_MMLU_Pro()
    aime2024_test_dataset = parse_AIME_2024()
    aime2025_test_dataset = parse_AIME_2025()
    aime2026_test_dataset = parse_AIME_2026()
    if_test_dataset = parse_IF_bench()
    dapo_train_dataset = parse_Dapo_17k()
    if_train_dataset = parse_IF_bench_train()
    kk_train_dataset = parse_kk()
    local_save_dir = args.local_dir
    os.makedirs(local_save_dir, exist_ok=True)
    dataset_list = [mmlu_test_dataset, aime2024_test_dataset, aime2025_test_dataset, aime2026_test_dataset, if_test_dataset, dapo_train_dataset, if_train_dataset, kk_train_dataset]
    test_dataset = datasets.concatenate_datasets(dataset_list)
    test_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))