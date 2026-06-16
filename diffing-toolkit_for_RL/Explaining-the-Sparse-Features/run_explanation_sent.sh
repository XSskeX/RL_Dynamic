#!/bin/bash

# --- 默认配置 (如果未通过命令行传入，将使用这些值) ---

# 模型配置
TOKENIZER_PATH=/data/baijun/models/Qwen3-4B
OUTPUT_DIR=outputs/e5-large/sae/
HOOKPOINT=embedding
NUM_LAYERS=1
NUM_LATENTS=8192
EXAMPLE_CTX_LEN=512

# 训练配置
MIN_EXAMPLES=50
NUM_TRAIN=10

# LLM 客户端配置
EXP_MODEL="qwen3"
BASE_URL="http://0.0.0.0:8889/v1/chat/completions"
API_KEY="bigai"
MAX_TOKENS=512
EXPL_THRESHOLD=0.1
PIPELINE_LIMIT=2

# --- 运行 Python 脚本 ---
python3 explanation.py \
    --mode explanation \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --hookpoint "$HOOKPOINT" \
    --num_layers "$NUM_LAYERS" \
    --num_latents "$NUM_LATENTS" \
    --example_ctx_len "$EXAMPLE_CTX_LEN" \
    --min_examples ${MIN_EXAMPLES} \
    --n_examples_train ${NUM_TRAIN} \
    --exp_model "$EXP_MODEL" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --max_tokens "$MAX_TOKENS" \
    --explainer_threshold "$EXPL_THRESHOLD" \
    --pipeline_limit "$PIPELINE_LIMIT" \
    --sentence_level
