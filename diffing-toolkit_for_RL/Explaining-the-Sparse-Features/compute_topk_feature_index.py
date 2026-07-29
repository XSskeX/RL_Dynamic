import pandas as pd

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