import re
def extract_solution(solution_str):
    final_answer = None

    boxed_matches = re.findall(r"\\boxed\{\s*([A-Za-z]+)\s*\}", solution_str)
    if boxed_matches:
        final_answer = boxed_matches[-1].strip()
    
    if final_answer is None:
        letter_matches = re.findall(r"[A-Za-z]+", solution_str)
        if letter_matches:
            final_answer = letter_matches[-1]
                    
    return final_answer

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score