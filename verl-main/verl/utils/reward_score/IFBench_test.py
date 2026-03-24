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

import os

from absl import app
from absl import logging

from .IF_Bench_Eval import IF_Bench_evaluation_lib

from pathlib import Path

def file_exists(path: str | Path) -> bool:
    return Path(path).is_file()


def if_bench_evaluation(response, key, kwargs, instruction_id_list, prompt, method):
    print("-" * 60 + "successfully enter evaluation" + "-" * 60)
    promptExample = IF_Bench_evaluation_lib.read_prompt_list(key, kwargs, instruction_id_list, prompt)
    print("-" * 60 + "successfully get promptExample" + "-" * 60)
    if method == "strict":
        output = IF_Bench_evaluation_lib.test_instruction_following_strict(promptExample, response) 
        print("-" * 60 + "successfully complete IF_Bench_evaluation_lib.test_instruction_following_strict" + "-" * 60)
        follow_all_instructions = output.follow_all_instructions
        return int(follow_all_instructions)
    else:
        output = IF_Bench_evaluation_lib.test_instruction_following_loose(promptExample, response) 
        follow_all_instructions = output.follow_all_instructions
        return int(follow_all_instructions)



def compute_score(solution_str, ground_truth, extra_info, method="soft", format_score=0.0, score=1.0):
    """
        compute_score for IF_Bench
        use accuracy as reward
    """
    response = solution_str
    print("--------------------successfully enter compute_score-----------------------------------")
    if response is None:
        return 0
    key = extra_info['index']
    kwargs = extra_info['kwargs']
    instruction_id_list = extra_info['instruction_id_list']
    prompt = extra_info['question']
    reward = if_bench_evaluation(response, key, kwargs, instruction_id_list, prompt, method)

    return reward