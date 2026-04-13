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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/share/nlp/baijun/shuhan/MMLU_Pro_domains")
    parser.add_argument("--local_dataset_path", default="TIGER-Lab/MMLU-Pro")
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.local_dataset_path, "default")
    raw_dataset = dataset['test']
    dev_df = preprocess(raw_dataset)
    #split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    #train_dataset, test_dataset = split['train'], split['test']
    
    data_source = "TIGER-Lab/MMLU-Pro"

    def make_map_fn(split):
        def process_fn(example):    
            idx = example.pop("question_id")
            question_raw = example.pop("question")
            options = example.pop("options")
            cot_content = example.pop("cot_content")
            question = format_example(question_raw, options)
            domain = example.pop("category")
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
                    "kwargs": [domain], 
                    "question": question_raw,
                },
            }
            #print(data["prompt"][0]["content"])
            #print(data["reward_model"]["ground_truth"])
            return data
        return process_fn

    
    test_dataset = raw_dataset.map(function=make_map_fn("test"))

    # Get unique domains
    unique_domains = list(set(raw_dataset['category']))

    for domain in unique_domains:
        # Filter for this domain
        
        test_subset = test_dataset.filter(lambda x: x['extra_info']['kwargs'][0] == domain)

        local_save_dir = os.path.join(args.local_dir, domain.replace(" ", "_").replace("/", "_"))
        os.makedirs(local_save_dir, exist_ok=True)
        
        test_subset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
