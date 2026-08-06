import numpy as np
def parse_cot_eval(pred_str, ans,
                   conclusion_patterns=['CONCLUSION:'],
                   verbose=False,
                   finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"],
                   reformat_gold_conditions=None):
    
    def judge_string(input_str, reformat_gold_conditions, wrong_reason, finish_patterns):
        correct_count = 0
        is_correct = False
        beyond_id = len(reformat_gold_conditions)+1
        beyond_id_pattern = f"({beyond_id})"

        for finish_pattern in finish_patterns:
            if finish_pattern in input_str:
                input_str = input_str.split(finish_pattern)[0]

        if beyond_id_pattern in input_str:
            is_correct = False
            wrong_reason = "beyond_list"
        elif "if" in input_str:
            is_correct = False
            wrong_reason = "contain_if"
        else:
            is_correct = True
            for gold_condition in reformat_gold_conditions:
                if gold_condition not in input_str:
                    is_correct = False
                    wrong_reason = "wrong_identity"
                else:
                    correct_count += 1
        correct_ratio = correct_count/len(reformat_gold_conditions)

        return is_correct, wrong_reason, correct_ratio

    def check_numbers_in_string(s, N):
        for i in range(1, N + 1):
            if f"({i})" not in s:
                return False
        return True
    
    original_str = pred_str
    pred_str = pred_str.split("### Question")[0]
    pred_answer = pred_str
    is_correct = False
    correct_ratio = 0
    if reformat_gold_conditions is None:
        gold = ans.replace(" and ", "").replace(".", "")
        gold_conditions = gold.split(",")
        reformat_gold_conditions = []
        for condition in gold_conditions:
            gold_condition = condition.strip()    # Remove leading and trailing spaces
            reformat_gold_conditions.append(gold_condition)

    wrong_reason = "no_conclusion_matched"
    for pattern in conclusion_patterns:
        pred = pred_str.split(pattern)
        if len(pred) > 1:
            if len(pred[1]) > 0:  # if the matched the answer is not empty
                pred_answer = pred[1]
                is_correct, wrong_reason, correct_ratio = judge_string(
                    pred_answer, reformat_gold_conditions, wrong_reason, finish_patterns)
                break
    if is_correct == False and wrong_reason == "no_conclusion_matched": 
        if check_numbers_in_string(pred_str, len(reformat_gold_conditions)): # the answer contains (1)..(2)..
            is_correct, wrong_reason, correct_ratio = judge_string(
                pred_str, reformat_gold_conditions, wrong_reason, finish_patterns)
    if is_correct == False and verbose == True:
        print("wrong_reason:",wrong_reason)
        print("********* \nprediction before parse:\n", original_str)
        print("********* \nprediction after parse:\n", pred_answer)

    return is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions


def kk_reward(answer: str, ground_truth: str, dense: bool = False) -> float:
    is_correct, _, _, correct_ratio, _ = parse_cot_eval(
        pred_str=answer,
        ans=ground_truth,
        conclusion_patterns=[
            "CONCLUSION:",
            "Conclusion:",
            "conclusion:",
        ],
    )

    if dense:
        return float(correct_ratio)

    return float(is_correct)

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):

    return kk_reward(answer=solution_str, ground_truth=ground_truth, dense=True)