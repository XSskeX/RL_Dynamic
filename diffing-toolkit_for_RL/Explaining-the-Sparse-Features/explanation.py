import os
import re
import json
import pandas as pd
import argparse
import orjson
import asyncio

from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from delphi.explainers import DefaultExplainer, explanation_loader
from delphi.clients import OpenRouter
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import DetectionScorer

import torch
from pathlib import Path
from transformers import AutoTokenizer
from functools import partial

# --- 配置参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run the explanation pipeline with customizable settings.")

    # 0. 实验配置
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--mode",
        type=str,
        default="explanation",
        choices=["explanation", "evaluation"],
        help="Mode to run: 'explanation' to generate explanations, 'evaluation' to evaluate them.",
    )

    # 1. 模型和稀疏化配置
    model_group = parser.add_argument_group("Model and Sparsity Configuration")
    model_group.add_argument(
        "--tokenizer_path",
        type=str,
        default="../../models/Qwen3-30B-A3B-Instruct-2507",
        help="Path to the pre-trained model directory.",
    )
    model_group.add_argument(
        "--hookpoint",
        type=str,
        default="mlp.gate",
        help="Hookpoint module name to extract latents from.",
    )
    model_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to save outputs such as explanations and evaluations.",
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=16,
        help="Number of layers in the model to process.",
    )
    model_group.add_argument(
        "--num_latents",
        type=int,
        default=65536,
        help="Total number of latents (neurons/features) per layer.",
    )
    model_group.add_argument(
        "--example_ctx_len",
        type=int,
        default=128,
        help="Context length for examples.",
    )

    # 2. 采样器配置 (SamplerConfig)
    sampler_group = parser.add_argument_group("Sampler Configuration")
    sampler_group.add_argument(
        "--n_examples_train",
        type=int,
        default=20,
        help="Number of training examples.",
    )
    sampler_group.add_argument(
        "--n_examples_test",
        type=int,
        default=20,
        help="Number of testing examples.",
    )
    sampler_group.add_argument(
        "--n_quantiles",
        type=int,
        default=10,
        help="Number of quantiles for sampling.",
    )
    sampler_group.add_argument(
        "--train_type",
        type=str,
        default="quantiles",
        help="Type of training sampling.",
    )
    sampler_group.add_argument(
        "--test_type",
        type=str,
        default="quantiles",
        help="Type of testing sampling.",
    )

    # 3. 构造器配置 (ConstructorConfig)
    constructor_group = parser.add_argument_group("Constructor Configuration")
    constructor_group.add_argument(
        "--min_examples",
        type=int,
        default=100,
        help="Minimum number of examples required.",
    )
    constructor_group.add_argument(
        "--non_activating_source",
        type=str,
        default='random',
        help="Source for non-activating examples.",
    )

    # 4. LLM 客户端和 Explainer 配置
    llm_group = parser.add_argument_group("LLM Client and Explainer Configuration")
    llm_group.add_argument(
        "--exp_model",
        type=str,
        default="qwen3",
        help="Model name for the explainer LLM client.",
    )
    llm_group.add_argument(
        "--base_url",
        type=str,
        default="http://0.0.0.0:8889/v1/chat/completions",
        help="Base URL for the LLM API.",
    )
    llm_group.add_argument(
        "--api_key",
        type=str,
        default="bigai",
        help="API key for the LLM service.",
    )
    llm_group.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens for LLM response.",
    )
    llm_group.add_argument(
        "--explainer_threshold",
        type=float,
        default=0.1,
        help="Activation threshold for the explainer.",
    )
    llm_group.add_argument(
        "--pipeline_limit",
        type=int,
        default=1,
        help="Limit the number of latents to process in the pipeline run.",
    )
    llm_group.add_argument(
        "--sentence_level",
        action='store_true',
        help="Whether to use sentence-level explanations.",
    )

    # 5. 评估配置
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--n_examples_shown",
        type=int,
        default=5,
        help="Number of examples to show in evaluation.",
    )

    return parser.parse_args()

# --- 主逻辑函数 ---
async def main():
    args = parse_args()

    # 1. 初始化模型和 Hookpoints
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    hookpoints = [args.hookpoint.format(layer_id=i) for i in range(args.num_layers)]

    latent_dict = {hp: torch.arange(0, args.num_latents) for hp in hookpoints}

    # 2. 初始化配置对象
    sampler_cfg = SamplerConfig(
        n_examples_train=args.n_examples_train,
        n_examples_test=args.n_examples_test,
        n_quantiles=args.n_quantiles,
        train_type=args.train_type,
        test_type=args.test_type,
    )
    constructor_cfg = ConstructorConfig(
        min_examples=args.min_examples,
        example_ctx_len=args.example_ctx_len,
        non_activating_source=args.non_activating_source,
    )

    # 3. 初始化 LatentDataset
    print("Initializing LatentDataset...")
    latent_dataset = LatentDataset(
        raw_dir=f"{args.output_dir}/raw_latents",
        modules=hookpoints,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        latents=latent_dict,
        tokenizer=tokenizer
    )

    # 4. 初始化 LLM Client 和 Explainer
    print("Initializing LLM Client and Explainer...")
    llm_client = OpenRouter(
        args.exp_model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens
    )

    explanation_dir = f'{args.output_dir}/explanations/{args.hookpoint}/'

    if args.mode == 'explanation':

        explainer = DefaultExplainer(
            llm_client,
            threshold=args.explainer_threshold,
            activations=True,
            cot=False,
            sentence_level=args.sentence_level,
        )

        # 5. 设置输出路径和后处理函数
        explanations_path = Path(explanation_dir)
        os.makedirs(explanations_path, exist_ok=True)

        def explainer_postprocess(result):
            # 写入 JSON 格式的解释到文件
            with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
                f.write(orjson.dumps(result.explanation))
            return result

        explainer_pipe = process_wrapper(explainer, postprocess=explainer_postprocess)

        # 6. 运行 Pipeline
        print("Starting pipeline run...")
        pipeline = Pipeline(latent_dataset, explainer_pipe)
        await pipeline.run(args.pipeline_limit)
        print("Pipeline finished.")

    elif args.mode == 'evaluation':

        def scorer_preprocess(result):
            record = result.record
            record.explanation = result.explanation
            record.extra_examples = record.not_active
            return record

        def scorer_postprocess(result, score_dir):
            safe_latent_name = str(result.record.latent).replace("/", "--")

            with open(Path(score_dir) / Path(f"{safe_latent_name}.txt"), "wb") as f:
                f.write(orjson.dumps(result.score))

        # If one wants to load the explanations they generated earlier
        explainer_pipe = partial(explanation_loader, explanation_dir=explanation_dir)

        score_dir = f'{args.output_dir}/evaluations'
        os.makedirs(score_dir, exist_ok=True)
        scorer_pipe = process_wrapper(
                DetectionScorer(llm_client, verbose=True, log_prob=True, tokenizer=latent_dataset.tokenizer, n_examples_shown=args.n_examples_shown),
                preprocess=scorer_preprocess,
                postprocess=partial(scorer_postprocess, score_dir=score_dir),
            )

        pipeline = Pipeline(
            latent_dataset,
            explainer_pipe,
            scorer_pipe,
        )
        await pipeline.run(args.pipeline_limit)
        print("Pipeline finished.")

        results = []

        # 遍历文件夹中的所有 json 文件
        for file in os.listdir(score_dir):
            
            match = re.search(r"layers.(\d+).", file)
            layer_id = int(match.group(1))

            match = re.search(r"latent(\d+)", file)
            latent_id = int(match.group(1))

            file_path = os.path.join(score_dir, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # data 是一个列表，每个元素是一个样本的预测结果
            total = len(data)

            total_positive = sum(1 for sample in data if sample['activating'])
            correct = sum(1 for sample in data if sample['correct'])
            accuracy = correct / total if total > 0 else 0.0

            results.append({
                "layer id": layer_id,
                "latent id": latent_id,
                "total": total,
                "total positives": total_positive,
                "correct": correct,
                "accuracy": accuracy
            })

        # 保存到 CSV
        df = pd.DataFrame(results)
        output_file = f'{score_dir}/detection_summary.csv'
        df.to_csv(output_file, index=False)

        print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")