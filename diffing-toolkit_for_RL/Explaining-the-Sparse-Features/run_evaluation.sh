#!/bin/bash

# --- 脚本说明 ---
# 这是一个用于运行 evaluation pipeline 的 Shell 脚本。

# --- 默认配置 (如果未通过命令行传入，将使用这些值) ---

# 模型配置
MODEL_PATH=/data/baijun/models/OLMoE-1B-7B-0125
SPARSE_TYPE=transcoder
NUM_LAYERS=2
NUM_LATENTS=512
EXAMPLE_CTX_LEN=128

# 评估配置
MIN_EXAMPLES=20
NUM_TEST=20
NUM_SHOWN=5

# LLM 客户端配置
EXP_MODEL="qwen3"
BASE_URL="http://0.0.0.0:8889/v1/chat/completions"
API_KEY="bigai"
MAX_TOKENS=512
EXPL_THRESHOLD=0.1
PIPELINE_LIMIT=1 # 默认只运行 1 个 latent 进行测试

# --- 打印运行时配置 ---
echo "--- Starting Explanation Pipeline with Configuration ---"
echo "Model Path:      $MODEL_PATH"
echo "Sparse Type:     $SPARSE_TYPE"
echo "Layers:          $NUM_LAYERS"
echo "LLM Model:       $EXP_MODEL"
echo "Pipeline Limit:  $PIPELINE_LIMIT"
echo "------------------------------------------"
echo ""

# --- 运行 Python 脚本 ---
python3 explanation.py \
    --mode evaluation \
    --model_path "$MODEL_PATH" \
    --sparse_type "$SPARSE_TYPE" \
    --num_layers "$NUM_LAYERS" \
    --num_latents "$NUM_LATENTS" \
    --example_ctx_len "$EXAMPLE_CTX_LEN" \
    --min_examples ${MIN_EXAMPLES} \
    --n_examples_test ${NUM_TEST} \
    --n_examples_shown ${NUM_SHOWN} \
    --exp_model "$EXP_MODEL" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --max_tokens "$MAX_TOKENS" \
    --explainer_threshold "$EXPL_THRESHOLD" \
    --pipeline_limit "$PIPELINE_LIMIT"