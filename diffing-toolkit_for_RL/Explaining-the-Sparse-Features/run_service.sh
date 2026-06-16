export CUDA_VISIBLE_DEVICES=2,3,4,5

vllm serve /data/baijun/models/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8889 \
    --served-model-name qwen3 \
    --tensor-parallel-size 4 \
    --api-key bigai \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
    #--enable-auto-tool-choice \
    #--tool-call-parser hermes

# vllm serve /data/baijun/models/Qwen3-14B \
# 	--host 0.0.0.0 \
# 	--port 8888 \
# 	--served-model-name qwen3-14b \
# 	--api-key bigai 
