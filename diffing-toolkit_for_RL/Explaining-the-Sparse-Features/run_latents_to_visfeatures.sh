export CUDA_VISIBLE_DEVICES=2

for layer in 2; do
    python latents_to_visfeatures.py \
        --model_path /data/baijun/models/OLMoE-1B-7B-0125 \
        --base_transcoder_path /data/baijun/models/OLMoE-1B-7B-0125_tied_ef32_k64_bs1024_lr1e-4/layers.{layer_idx}.mlp.gate \
        --latent_dir outputs/OLMoE-1B-7B-0125/transcoder/raw_latents/ \
        --save_dir /data/baijun/models/OLMoE-1B-7B-0125_tied_ef32_k64_bs1024_lr1e-4/features \
        --layers $layer \
        --num_latents 65536 \
        --num_workers 4 \
        --overwrite
done
