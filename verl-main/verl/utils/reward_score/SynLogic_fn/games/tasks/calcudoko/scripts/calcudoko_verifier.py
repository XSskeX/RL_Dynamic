from base.data import Data
from base.verifier import Verifier
import numpy as np
from typing import List, Dict
import re

class CalcudokoVerifier(Verifier):
    """
    Calcudoko 游戏的验证器
    """
    
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取数独解答
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案 (元组形式的字符串)
        """
        if not test_solution:
            return ""
            
        # 提取Python代码块中的内容
        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        
        if matches:
            # 取最后一个Python代码块
            python_code = matches[-1].strip()
            return python_code
            
        # 如果没有找到Python代码块，尝试找到任何类似于元组的结构
        tuple_pattern = r"\(\s*\(\s*\d+\s*,.*?\)\s*\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
            
        # 如果上面都没找到，直接返回原始答案
        return test_solution
    
    def parse_grid_from_answer(self, test_solution: str, data: Data) -> np.ndarray:
        """
        从模型的回答中解析出网格数据
        
        @param test_solution: 模型的完整回答
        @param data: 包含问题、元数据等信息的Data对象
        @return: numpy数组形式的网格
        """
        try:
            # 从data中获取grid_size
            grid_size = data.metadata.get("grid_size", 4)  # 默认为4
            
            # 先提取答案
            answer = self.extract_answer(test_solution)
            
            # 查找[[...]]格式的答案
            bracket_pattern = r'\[\[(.*?)\]\]'
            bracket_match = re.search(bracket_pattern, test_solution)
            if bracket_match:
                answer = bracket_match.group(1)
            
            # 清理字符串，只保留必要字符
            answer = ''.join(c for c in answer if c.isdigit() or c in '[],' or c.isspace())
            
            # 标准化分隔符
            answer = answer.replace('，', ',')  # 中文逗号转英文逗号
            
            # 去掉最外层的括号
            answer = answer.strip('[]')
            
            # 分割成行
            rows = [row.strip() for row in answer.split(',')]
            
            # 转换成数字网格
            grid = []
            for row in rows:
                # 分割并转换成整数
                numbers = [int(n) for n in row.split() if n.strip()]
                if len(numbers) != grid_size:
                    # 尝试修复长度不正确的行
                    # 如果长度小于grid_size，尝试查找更多数字
                    if len(numbers) < grid_size:
                        # 在原始答案中查找类似于 "第x行：1 2 3 4 5" 的行描述
                        row_patterns = [
                            r'第\s*\d+\s*行\s*[:：]\s*([\d\s]+)', 
                            r'行\s*\d+\s*[:：]\s*([\d\s]+)',
                            r'Row\s*\d+\s*[:：]\s*([\d\s]+)',
                            r'- 第\d+行[:：]?\s*([\d\s]+)',
                            r'- 行\d+[:：]?\s*([\d\s]+)',
                            r'- Row\d+[:：]?\s*([\d\s]+)'
                        ]
                        
                        found_rows = []
                        for pattern in row_patterns:
                            matches = re.findall(pattern, test_solution)
                            if matches:
                                found_rows.extend(matches)
                        
                        
                        if len(found_rows) == grid_size:
                            # 如果找到的行数正好等于grid_size，使用这些行
                            grid = []
                            for found_row in found_rows:
                                numbers = [int(n) for n in found_row.split() if n.strip()]
                                if len(numbers) != grid_size:
                                    # 如果长度不正确，填充或截断
                                    if len(numbers) < grid_size:
                                        numbers.extend([1] * (grid_size - len(numbers)))
                                    else:
                                        numbers = numbers[:grid_size]
                                grid.append(numbers)
                            break  # 成功处理，退出循环
                    
                    # 填充或截断使长度正确
                    if len(numbers) < grid_size:
                        numbers.extend([1] * (grid_size - len(numbers)))
                    else:
                        numbers = numbers[:grid_size]
                
                grid.append(numbers)
            
            # 检查行数
            if len(grid) != grid_size:
                # 如果行数小于grid_size，添加默认行
                if len(grid) < grid_size:
                    for _ in range(grid_size - len(grid)):
                        grid.append([1] * grid_size)
                else:
                    # 如果行数大于grid_size，截断
                    grid = grid[:grid_size]
            
            return np.array(grid)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # 尝试直接搜索[[...]]格式
            try:
                bracket_pattern = r'\[\[(.*?)\]\]'
                bracket_match = re.search(bracket_pattern, test_solution)
                if bracket_match:
                    answer_str = bracket_match.group(1)
                    rows = answer_str.split(',')
                    grid = []
                    for row in rows:
                        numbers = [int(n) for n in row.split() if n.strip()]
                        if len(numbers) != grid_size:
                            if len(numbers) < grid_size:
                                numbers.extend([1] * (grid_size - len(numbers)))
                            else:
                                numbers = numbers[:grid_size]
                        grid.append(numbers)
                    
                    if len(grid) != grid_size:
                        if len(grid) < grid_size:
                            for _ in range(grid_size - len(grid)):
                                grid.append([1] * grid_size)
                        else:
                            grid = grid[:grid_size]
                    
                    return np.array(grid)
            except:
                pass
            
            # 最终失败，返回默认网格
            return np.array([[i % grid_size + 1 for i in range(grid_size)] for _ in range(grid_size)])
    
    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案字符串
        @return: 回答是否正确的布尔值
        """
        try:
            # 从元数据中获取必要信息
            grid_size = data.metadata.get("grid_size", 4)  # 默认为4
            regions = data.metadata.get("regions", [])
            
            
            # 从答案中提取网格
            try:
                grid = self.parse_grid_from_answer(test_answer, data)
            except Exception as e:
                return False
            
            # 检查每一行
            for i, row in enumerate(grid):
                if len(set(row)) != grid_size:
                    return False
                if not all(1 <= n <= grid_size for n in row):
                    return False
                    
            # 检查每一列
            for i, col in enumerate(grid.T):
                if len(set(col)) != grid_size:
                    return False
                if not all(1 <= n <= grid_size for n in col):
                    return False
            
            # 验证区域规则
            for i, region in enumerate(regions):
                # 获取区域中的单元格
                cells = region["cells"]
                numbers = []
                
                # 提取数字
                for cell in cells:
                    try:
                        if isinstance(cell, str):
                            match = re.search(r'\((\d+)\s*,\s*(\d+)\)', cell)
                            if match:
                                r,c =  (int(match.group(1)), int(match.group(2)))
                                
                        # r, c = cell  # cell已经是元组了，直接解包
                        # 转换为0基索引
                        r, c = r-1, c-1
                        # 检查坐标是否有效
                        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
                            return False
                        numbers.append(grid[r][c])
                    except Exception as e:
                        return False
                
                
                # 检查数字是否有效（1到grid_size）
                if not all(1 <= n <= grid_size for n in numbers):
                    return False
                
                # 检查区域内数字是否唯一
                if len(set(numbers)) != len(numbers):
                    return False
                
                # 检查数字是否满足运算规则
                target = region["target"]
                operator = region["operator"]
                
                try:
                    if operator == '+':
                        result = sum(numbers)
                        valid = (result == target)
                        if not valid:
                            return False
                    elif operator == '-':
                        if len(numbers) != 2:
                            return False
                        result1 = numbers[0] - numbers[1]
                        result2 = numbers[1] - numbers[0]
                        valid = (result1 == target or result2 == target)
            
                        if not valid:
                            return False
                    elif operator == '*':
                        result = np.prod(numbers)
                        valid = (result == target)
                        if not valid:
                            return False
                    else:  # division
                        if len(numbers) != 2:
                            return False
                        if numbers[0] == 0 or numbers[1] == 0:
                            return False
                        result1 = numbers[0] / numbers[1]
                        result2 = numbers[1] / numbers[0]
                        valid = (abs(result1 - target) < 0.001 or abs(result2 - target) < 0.001)
                        if not valid:
                            return False
                except Exception as e:
                    return False
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

def main():
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Verify Calcudoko solutions')
    parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid (N)')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to save valid samples (optional)')
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = CalcudokoVerifier()
    
    total_puzzles = 0
    valid_answers = 0
    invalid_answers = 0
    valid_samples = []  # 用于存储有效样本
    
    def parse_regions_from_text(text):
        """从文本中解析regions信息"""
        regions = []
        # 查找Regions:后面的内容
        regions_text = re.search(r'Regions:\n(.*?)(?:\n\n|$)', text, re.DOTALL)
        if regions_text:
            for line in regions_text.group(1).strip().split('\n'):
                if line.strip():
                    # 解析每个region
                    cells_part, target = line.split(':')
                    # 使用正则表达式提取所有坐标对
                    cells = re.findall(r'\((\d+),(\d+)\)', cells_part)
                    # 将字符串转换为整数元组
                    cells = [tuple(map(int, pair)) for pair in cells]
                    # 处理target部分
                    target_value, operator = target[:-1], target[-1]
                    regions.append({
                        'cells': cells,
                        'target': int(target_value),
                        'operator': operator
                    })
        return regions
    
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_puzzles += 1
                    
                    # 尝试从不同位置获取regions信息
                    regions = None
                    if 'regions' in data:
                        regions = data['regions']
                    elif 'regions' in data['metadata']:
                        regions = data['metadata']['regions']
                    elif 'question_data' in data:
                        question_data = json.loads(data['question_data'])
                        if 'regions' in question_data:
                            regions = question_data['regions']
                    elif 'question' in data:
                        regions = parse_regions_from_text(data['question'])
                    
                    if not regions:
                        continue
                    
                    # 获取答案
                    answer = data.get('answer', '')
                    if not answer:
                        continue
                    
                    # 构造Data对象
                    data_obj = Data(
                        question=data['question'],
                        answer=answer,  # 添加 answer 参数
                        metadata={'grid_size': args.grid_size, 'regions': regions}
                    )
                    
                    # 验证答案
                    is_valid = verifier.verify(data_obj, answer)
                    if is_valid:
                        valid_answers += 1
                        # 将有效样本添加到列表中
                        valid_samples.append(data)
                    else:
                        invalid_answers += 1
                    
                except Exception as e:
                    invalid_answers += 1
                    
    except Exception as e:
        return
    
    # 保存有效样本
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
    

if __name__ == "__main__":
    main() 
    # response =  "[[2 3 4 6 1 5,1 2 3 4 5 6,3 1 2 5 6 4,4 5 6 1 2 3,5 6 1 3 4 2,6 4 5 2 3 1]]"
    # metadata = {"grid_size": 6, "regions": [{"cells": ["(2,1)", "(1,1)", "(1,2)"], "target": 6, "operator": "*"}, {"cells": ["(3,5)", "(3,6)"], "target": 2, "operator": "-"}, {"cells": ["(1,5)", "(2,5)"], "target": 5, "operator": "÷"}, {"cells": ["(4,5)", "(4,4)"], "target": 2, "operator": "÷"}, {"cells": ["(5,6)", "(5,5)", "(6,5)"], "target": 9, "operator": "+"}, {"cells": ["(3,3)", "(2,3)", "(2,4)", "(3,4)"], "target": 120, "operator": "*"}, {"cells": ["(1,3)", "(1,4)"], "target": 2, "operator": "-"}, {"cells": ["(2,6)", "(1,6)"], "target": 1, "operator": "-"}, {"cells": ["(3,2)", "(3,1)", "(4,1)", "(5,1)"], "target": 60, "operator": "*"}, {"cells": ["(5,3)", "(5,2)", "(6,2)"], "target": 11, "operator": "+"}, {"cells": ["(5,4)", "(6,4)", "(6,3)"], "target": 10, "operator": "+"}, {"cells": ["(4,2)", "(4,3)"], "target": 1, "operator": "-"}, {"cells": ["(6,1)", "(2,2)"], "target": 3, "operator": "÷"}, {"cells": ["(6,6)", "(4,6)"], "target": 2, "operator": "-"}]}
    # testClass = CalcudokoVerifier()
    # dataClass = Data(question="", answer="")
    # dataClass.metadata = metadata

    # test_answer = testClass.parse_grid_from_answer(test_solution=response, data=dataClass)
    # print(test_answer)
    # print(testClass.verify(data=dataClass, test_answer=response))