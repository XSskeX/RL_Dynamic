import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, OlmoeForCausalLM
# 假设这些是您的自定义模块，保持导入不变
from sparsify.data import chunk_and_tokenize 
from delphi.latents import LatentCache
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.sparse_coders import load_hooks_sparse_coders
# from models.modeling_qwen3_moe import Qwen3MoeForCausalLM # 此行在原代码中未被使用，故注释掉

def main():
    # 🌟 参数解析
    parser = argparse.ArgumentParser(description="Generate latents for a sparse model.")
    
    # 模型路径
    parser.add_argument("--model_path", type=str, default="../../models/OLMoE-1B-7B-0125", help="Path to the base model.")
    parser.add_argument("--sparse_path", type=str, default="../../models/OLMoE-1B-7B-0125_tied_ef32_k64_bs1024_lr1e-4", help="Path to the sparse model/coders.")
    
    # 实验配置
    parser.add_argument("--sparse_type", type=str, choices=['sae', 'transcoder'], default='transcoder', help="Type of sparse coder ('sae' or 'transcoder').")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length for tokenization and caching.")
    parser.add_argument("--layers", nargs="+", type=int, help="list of layer indices")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading and cache generation.")
    parser.add_argument("--n_tokens", type=int, default=20000, help="Total number of tokens to process.")
    parser.add_argument("--n_splits", type=int, default=1, help="Number of splits for saving the latents.")
    parser.add_argument("--exp_topk", type=int, default=8, help="Top-K value for router weights (only used if save_router_weights_flag is True).")
    parser.add_argument("--save_router_weights_flag", action="store_true", help="Whether to save router weights.")

    args = parser.parse_args()

    print("Arguments:", args, flush=True)

    sparse_type = args.sparse_type
    max_seq_len = args.max_seq_len
    model_path = args.model_path
    sparse_path = args.sparse_path
    layers = args.layers
    batch_size = args.batch_size
    n_tokens = args.n_tokens
    n_splits = args.n_splits
    exp_topk = args.exp_topk
    save_router_weights_flag = args.save_router_weights_flag

    # 派生参数
    hookpoints=[f"layers.{i}.mlp.gate" for i in layers]
    model_name = os.path.basename(model_path)
    latents_path = Path(f"./outputs/{model_name}/{sparse_type}/raw_latents")
    
    # 打印运行信息
    print(f"Generating {n_tokens} tokens of latents for {model_name} at {sparse_type} hookpoints {hookpoints}", flush=True)

    # 📚 数据加载与分词
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("kotyKD/c4-pro-tiny", split="train")
    tokens = chunk_and_tokenize(dataset, tokenizer, max_seq_len=max_seq_len, text_key="text")["input_ids"]

    # 💾 模型加载 (使用 BitsAndBytesConfig 进行量化)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = OlmoeForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # ⚙️ 配置对象实例化
    cache_cfg = CacheConfig(
        batch_size=batch_size, 
        cache_ctx_len=max_seq_len, 
        n_tokens=n_tokens
    )

    run_cfg = RunConfig(
        constructor_cfg=ConstructorConfig(),
        sampler_cfg=SamplerConfig(),
        cache_cfg=cache_cfg,
        model=model_path,
        sparse_model=sparse_path,
        hookpoints=hookpoints,
    )

    # function of sae to encode
    hookpoint_to_sparse_encode, _ = load_hooks_sparse_coders(model, run_cfg)

    # 缓存对象实例化
    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=batch_size,
        transcode=True if sparse_type=='transcoder' else False,
        save_router_weights_flag=save_router_weights_flag,
        router_topk=exp_topk if save_router_weights_flag else None,
    )

    # 运行和保存
    print("Starting cache generation...", flush=True)
    cache.run(n_tokens=n_tokens, tokens=torch.stack(list(tokens), 0))
    print("Cache generation complete. Saving results...", flush=True)

    os.makedirs(latents_path, exist_ok=True)
    cache.save_splits(
        n_splits=n_splits,
        save_dir=latents_path
    )
    if save_router_weights_flag:
        cache.save_router_weights(
            save_dir=latents_path
        )
    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)
    print(f"Results saved to {latents_path}", flush=True)

if __name__ == "__main__":
    main()