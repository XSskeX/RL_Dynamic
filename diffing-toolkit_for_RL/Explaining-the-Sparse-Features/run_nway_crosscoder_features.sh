#!/usr/bin/env bash
set -euo pipefail

LAYER="${LAYER:-0}"
CTX_LEN="${CTX_LEN:-128}"
N_SPLITS="${N_SPLITS:-4}"
NUM_LATENTS="${NUM_LATENTS:-65536}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_NUM_SAMPLES="${MAX_NUM_SAMPLES:-1000000}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
RAW_LATENTS_DIR="${RAW_LATENTS_DIR:-}"
FEATURES_DIR="${FEATURES_DIR:-}"

if [[ -z "$TOKENIZER_PATH" ]]; then
  echo "Set TOKENIZER_PATH to the tokenizer/base model path used for the activation cache." >&2
  exit 1
fi

if [[ -z "$RAW_LATENTS_DIR" ]]; then
  echo "Set RAW_LATENTS_DIR to the raw_latents output directory." >&2
  exit 1
fi

cache_args=(
  --layer "$LAYER"
  --ctx_len "$CTX_LEN"
  --n_splits "$N_SPLITS"
  --max_num_samples "$MAX_NUM_SAMPLES"
  --module_name "nway_crosscoder.layer_${LAYER}"
  --output_dir "$RAW_LATENTS_DIR"
  --overwrite
)
if [[ -n "$MODEL_PATH" ]]; then
  cache_args+=(--model_path "$MODEL_PATH")
fi
if [[ "$#" -gt 0 ]]; then
  cache_args+=(--hydra_overrides "$@")
fi

python nway_crosscoder_delphi_cache.py "${cache_args[@]}"

if [[ -z "$FEATURES_DIR" ]]; then
  FEATURES_DIR="${RAW_LATENTS_DIR%/raw_latents}/features"
fi

python latents_to_visfeatures.py \
  --model_path "$TOKENIZER_PATH" \
  --latent_dir "$RAW_LATENTS_DIR" \
  --save_dir "$FEATURES_DIR" \
  --layers "$LAYER" \
  --num_latents "$NUM_LATENTS" \
  --num_workers "$NUM_WORKERS" \
  --module_template "nway_crosscoder.layer_{layer_idx}" \
  --skip_feature_logits \
  --overwrite

echo "Feature files saved to: $FEATURES_DIR"
