export CUDA_VISIBLE_DEVICES=7

python latents_cache.py \
    --model_path /data/baijun/models/OLMoE-1B-7B-0125 \
    --layers {0..15} \
    --sparse_path /data/baijun/models/OLMoE-1B-7B-0125_tied_ef32_k64_bs1024_lr1e-4 \
    --sparse_type transcoder \
    --n_tokens 1000000 \
    --max_seq_len 128 \
    --batch_size 128 \
    --n_splits 4