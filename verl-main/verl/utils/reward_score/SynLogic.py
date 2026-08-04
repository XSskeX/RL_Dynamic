# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300
from pathlib import Path
import time
import random
import json

from pydantic import BaseModel
import time
import random
from .SynLogic_fn.task2verifier import *
from .SynLogic_fn.base.data import Data
import re



THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def _extract_answer(text):
    # Define regex pattern to match content between <answer> and </answer>
    pattern = r'<answer>(.*?)</answer>'
    
    # Use re.search to find the first match
    match = re.search(pattern, text, re.DOTALL)
    
    # If match found, return the matched content
    if match:
        return match.group(1).strip()
    else:
        return None

def _extract_solution_with_thought(solution_str):
    model_output = solution_str
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    
    predict_answer = _extract_answer(model_output)
    if predict_answer is not None:
        return predict_answer
    else:
        return model_output
    

class Payload(BaseModel):
    response: str
    answer: str
    prompt: str
    solution: str
    data_source: str
    game_data: str


def synthetic_puzzle_process(payload_dict):
    format_res = 0
    # check format
    if (payload_dict["response"].startswith('<think>') and
        payload_dict["response"].endswith('</answer>') and
        payload_dict["response"].count('<think>') == 1 and
        payload_dict["response"].count('</think>') == 1 and
        payload_dict["response"].count('<answer>') == 1 and
        payload_dict["response"].count('</answer>') == 1 ):
        format_res = 1
    logic_res = 0
    if format_res > 0:
        verifier_class = verifier_classes[payload_dict['data_source']]
        verifier = verifier_class()
        game_data = Data.from_json_str(payload_dict['game_data'])
        test_solution = payload_dict['response']
        test_solution = _extract_solution_with_thought(test_solution)
        logic_res = verifier.verify(game_data, test_solution)

    ## reward merge
    final_score = logic_res * format_res

    result = {
            "rewards": {
                "format_reward": format_res,
                "accuracy_reward": logic_res,
                "final_reward": final_score
            },
        }

    payload_dict['result'] = result

    return result


def file_exists(path: str | Path) -> bool:
    return Path(path).is_file()



def extract_reward(solution_str, game_data, data_source):
    payload = Payload(
        response=solution_str,
        answer="just for placeholder",
        prompt="just for placeholder",
        solution=solution_str,
        data_source=data_source,
        game_data=game_data
    )
    return synthetic_puzzle_process(payload.dict())


def compute_score(solution_str, ground_truth, extra_info, format_score=0.0, score=1.0):
    """The scoring function for SynLogic.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    metadata = json.loads(extra_info["metadata_json"])

    data_source = metadata["data_source"]
    game_data = metadata["game_data_str"]

    result = extract_reward(solution_str=solution_str, game_data=game_data, data_source=data_source)
    return result['rewards']['final_reward']