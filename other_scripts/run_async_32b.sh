#!/usr/bin/env bash
# GRPO | Qwen3-32B LoRA | Fully Async (4+4 GPU split) | veRL 0.8.0 | A100 80GB
# Rollout (4 GPU, TP=4) 和 Training (4 GPU) 同时运行，时间重叠
set -xeuo pipefail

PROJ=/data/yanziyi
SCRIPT_DIR=/data/yanziyi/gpu_test_0610
LOG_DIR=${SCRIPT_DIR}/outputs
mkdir -p "${LOG_DIR}"

RUN_ID=qwen3_32b_lora_async_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/${RUN_ID}.log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

MODEL_PATH=${PROJ}/models/Qwen3-32B
DATA_DIR=${PROJ}/data/gsm8k_verl

source ${SCRIPT_DIR}/venv/bin/activate

# 暴露全部 8 GPU 给 Ray，Ray 内部自动分配 trainer_pool vs rollout_pool
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1
export WANDB_MODE=offline

# ── GPU 分配 ─────────────────────────────────────────────────────
TRAINER_GPUS=4       # FSDP 训练（Actor + RefPolicy）
ROLLOUT_GPUS=4       # vLLM 推理（TP=4，1 个副本）

# ── 超参（与 sync baseline 对齐）────────────────────────────────
ppo_mini_batch_size=16
max_prompt_length=512
max_response_length=512
actor_lr=1e-6
kl_loss_coef=0.001
rollout_n=2
rollout_tp=4          # TP=4 → 4 GPU 组成 1 个 vLLM 副本

# LoRA 配置
lora_rank=64
lora_alpha=32

# 总 rollout 样本数 ≈ 50 个 sync step 的数据量
# 每 sync = trigger_sync_step * ppo_mini_batch * rollout_n = 4*16*2 = 128 samples
# 总 rollout prompts = 20 sync步 × trigger(4) × mini_batch(16) = 1280
total_rollout_steps=1280

# 异步控制参数
staleness_threshold=0.5   # 允许的最大 stale 样本比例
trigger_sync_step=4        # 每 4 个 mini-batch 更新同步一次参数
test_freq=10               # 每 N 次参数同步后做 validation

EXPERIMENT_NAME=${RUN_ID}

# ── 启动 GPU 监控 sidecar ─────────────────────────────────────────
MONITOR_CSV=${LOG_DIR}/${RUN_ID}_monitor.csv
python3 ${SCRIPT_DIR}/monitor_gpu.py \
    --output   "${MONITOR_CSV}" \
    --interval 1 &
MONITOR_PID=$!
echo "Monitor PID: ${MONITOR_PID}  CSV: ${MONITOR_CSV}"

cleanup() {
    echo "[cleanup] stopping monitor PID ${MONITOR_PID}"
    kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ── Fully Async 训练（32B LoRA）─────────────────────────────────
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    "data.truncation='error'" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_sync_step} \
    async_training.partial_rollout=True \
    async_training.require_batches=1 \
    async_training.use_trainer_do_validate=False \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
    rollout.n=${rollout_n} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    trainer.test_freq=${test_freq} \
    trainer.balance_batch=True \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=flagperf_qwen3_32b_lora_async \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${TRAINER_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    "$@"

# ── 训练完成后填写结果表格 ─────────────────────────────────────────────────────
cd /data/yanziyi/gpu_test_0610 && source venv/bin/activate
python3 fill_tables.py \
    --log     "${LOG_FILE}" \
    --tables  results_tables.md \
    --mode    async_32b \
    --thresh  ${staleness_threshold}
echo "[fill] results_tables.md updated (async_32b run)"
