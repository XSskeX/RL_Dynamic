import json
from pathlib import Path

import pandas as pd


def get_feature_ids(matches_path="matches.json"):
    """读取 matches.json，返回其中按出现顺序排列的所有 feature_id。"""
    with Path(matches_path).open("r", encoding="utf-8") as file:
        matches = json.load(file)

    feature_ids = []
    for value in matches.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and "feature_id" in item:
                feature_ids.append(item["feature_id"])

    return feature_ids

def get_topk_range_indices(csv_path, top_k=5):
    df = pd.read_csv(csv_path, index_col=0)
    df['range'] = df.max(axis=1) - df.min(axis=1)
    top_k_indices = df['range'].nlargest(top_k).index.tolist()
    
    return top_k_indices

# ================= 使用示例 =================
if __name__ == "__main__":
    csv_file = "C:/Users/Administrator/Desktop/RL_Data/crosscoder_data/RL_Dynamics_data/crosscoder_output_for_8/llama32_3B_Instruct_latent_norm.csv"  
    k = 10                  
    
    result_indices = get_topk_range_indices(csv_file, top_k=k)
    print(result_indices)
