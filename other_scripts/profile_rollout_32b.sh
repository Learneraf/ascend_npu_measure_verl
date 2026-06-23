#!/usr/bin/env bash
# Rollout internal profiling run (32B LoRA): 5 steps, torch profiler on vLLM workers.
# Produces torch trace JSONs; usage: bash profile_rollout_32b.sh [extra hydra overrides]
set -xeuo pipefail

PROJ=/data/yanziyi
SCRIPT_DIR=/data/yanziyi/gpu_test_0610
LOG_DIR=${SCRIPT_DIR}/outputs
mkdir -p ${LOG_DIR}

RUN_ID=qwen3_32b_lora_grpo_bs64_n2_resp512_profile_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/${RUN_ID}.log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

MODEL_PATH=${PROJ}/models/Qwen3-32B
DATA_DIR=${PROJ}/data/gsm8k_verl
PROF_DIR=${LOG_DIR}/traces_${RUN_ID}
mkdir -p "${PROF_DIR}"

source ${SCRIPT_DIR}/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_LOGGING_LEVEL=INFO
export WANDB_MODE=offline

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

NGPUS_PER_NODE=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    "data.truncation='error'" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    trainer.balance_batch=True \
    "trainer.logger=['console']" \
    trainer.project_name=flagperf_rollout_profile_32b \
    trainer.experiment_name=${RUN_ID} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=5 \
    global_profiler.tool=torch \
    "global_profiler.steps=[3,4,5]" \
    global_profiler.save_path=${PROF_DIR} \
    actor_rollout_ref.actor.profiler.enable=True \
    "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,cpu]" \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
    "$@"

echo "[profile_rollout_32b] Done. Log: ${LOG_FILE}"
echo "[profile_rollout_32b] Traces at: ${PROF_DIR}"

# ── 解析 profiling trace，填写表16/17 ─────────────────────────────────────────
cd /data/yanziyi/gpu_test_0610 && source venv/bin/activate
python3 other_scripts/parse_rollout_profile.py \
    --trace-dir "${PROF_DIR}" \
    --log       "${LOG_FILE}" \
    --tables    results_tables.md \
    --out       "${LOG_DIR}/${RUN_ID}_rollout_breakdown.csv" \
    --mode      32b
echo "[fill] results_tables.md updated (profile_32b run)"
