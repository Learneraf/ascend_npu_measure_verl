#!/usr/bin/env bash
# GRPO | Qwen3-8B | veRL 0.8.0 main_ppo_sync | 8xA100 80GB
# 同步基准：与 run_async.sh 对比
set -xeuo pipefail

PROJ=/data/yanziyi
SCRIPT_DIR=/data/yanziyi/gpu_test_0610
LOG_DIR=${SCRIPT_DIR}/outputs
mkdir -p "${LOG_DIR}"


MODEL_PATH=${PROJ}/models/Qwen3-8B
DATA_DIR=${PROJ}/data/gsm8k_verl

source ${SCRIPT_DIR}/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_LOGGING_LEVEL=INFO

# ── 超参 ────────────────────────────────────────────────────────
NGPUS_PER_NODE=8
train_batch_size=${train_batch_size:-256}
ppo_mini_batch_size=${ppo_mini_batch_size:-64}
max_prompt_length=512
max_response_length=${max_response_length:-1024}
actor_lr=1e-6
kl_loss_coef=0.001
rollout_n=${rollout_n:-4}
rollout_tp=1
total_epochs=1
total_training_steps=20
save_freq=-1
test_freq=5

RUN_ID=qwen3_8b_grpo_bs${train_batch_size}_n${rollout_n}_resp${max_response_length}_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/${RUN_ID}.log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

# ── 启动系统监控 sidecar ─────────────────────────────────────────
MONITOR_CSV=${LOG_DIR}/${RUN_ID}_monitor.csv
python3 ${SCRIPT_DIR}/monitor_gpu.py \
    --output   "${MONITOR_CSV}" \
    --interval 1 &
MONITOR_PID=$!
echo "Monitor PID: ${MONITOR_PID}  CSV: ${MONITOR_CSV}"

cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ── 训练 ─────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo_sync \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    "data.truncation='error'" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    trainer.balance_batch=True \
    "trainer.logger=['console']" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    "$@"

# ── 自动填表 ──────────────────────────────────────────────────────
# FILL_MODE=skip 时跳过填表（供消融脚本调用时使用）
if [ "${FILL_MODE:-baseline}" != "skip" ]; then
    python3 ${SCRIPT_DIR}/fill_tables.py \
        --log     "${LOG_FILE}" \
        --tables  ${SCRIPT_DIR}/results_tables.md \
        --mode    "${FILL_MODE:-baseline}"
fi
