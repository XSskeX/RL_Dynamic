set -xeuo pipefail

# 使用v1引擎
export VLLM_USE_V1=1
# 指定vllm 版本
export VLLM_VERSION=0.9.1

# 开启二级流水
export TASK_QUEUE_ENABLE=2
# 开启细绑核
export CPU_AFFINITY_CONF=1
# 使用jemalloc优化内存访问（依赖安装jemalloc）
#export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

trainer_n_gpus_per_node=4
trainer_nnodes=1
trainer_project_name='RL_Dynamics_lambda'
trainer_experiment_name="Llama3.2-3B-Instruct_format-0_entropy-0_grpo_4gpu"
export WANDB_API_KEY="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"
export WANDB_ENTITY="qinshuhanbuaa-beihang-university"
export WANDB_PROJECT="RL_Dynamic_lambda"
export WANDB_MODE="online"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/data/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${trainer_project_name}/${trainer_experiment_name}"}

export TENSORBOARD_DIR="${RAY_DATA_HOME}/tensorboard_dir/${trainer_project_name}/${trainer_experiment_name}"
mkdir -p "${RAY_DATA_HOME}/logs/${trainer_project_name}"
LOG_PATH="${RAY_DATA_HOME}/logs/${trainer_project_name}/${trainer_experiment_name}.log"

use_dynamic_bsz=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/share/nlp/baijun/shuhan/DAPO17k/train.parquet" \
    data.val_files="['/share/nlp/baijun/shuhan/IF_Bench/test.parquet', '/share/nlp/baijun/shuhan/AIME2024/test.parquet', '/share/nlp/baijun/shuhan/AIME2025/test.parquet', '/share/nlp/baijun/shuhan/AIME2026/test.parquet',  '/share/nlp/baijun/shuhan/MMLU_Pro/test.parquet']" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="meta-llama/Llama-3.2-3B-Instruct" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9300 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9300\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=9300 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.val_before_train=True 2>&1 | tee ${LOG_PATH}
