import glob
import sys
import json
import re
import random


from pathlib import Path

def file_exists(path: str | Path) -> bool:
    return Path(path).is_file()

def extract_answer(text, level):
    if level == 'l1':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None
    elif level == 'l2':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)
    

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def extract_solution(solution_str):
    pred = extract_answer(solution_str, 'l2')

    return pred
        


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        if not file_exists("/share/nlp/baijun/shuhan/MMLU_Pro_groundtruth.txt"):
            with open("/share/nlp/baijun/shuhan/MMLU_Pro_groundtruth.txt", "w", encoding="utf-8") as f:
                f.write(answer + "\n" + ground_truth)
        if answer == ground_truth:
            return score
        else:
            return format_score