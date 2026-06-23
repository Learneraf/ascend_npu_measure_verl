#!/usr/bin/env bash
# Ablation: rollout_n in {1, 2, 4, 8, 16}  fixed train_batch_size=64  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail
BASE=/data/yanziyi/gpu_test_0610
TABLES=${BASE}/results_tables.md

for n in 1 2 4 8 16; do
    echo "========== rollout_n=${n} =========="
    set +e
    rollout_n=${n} FILL_MODE=skip bash ${BASE}/run_qwen3_32b_lora_a100.sh \
        actor_rollout_ref.rollout.n=${n} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_rollout_n${n}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    cd ${BASE} && source venv/bin/activate
    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[rollout_n=${n}] FAILED (rc=${TRAIN_RC}), marking OOM"
        python3 fill_tables.py --tables "${TABLES}" --mode "ablation_n_32b_oom=${n}"
    else
        LATEST_LOG=$(ls -t ${BASE}/outputs/qwen3_32b_lora_grpo_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            python3 fill_tables.py --log "${LATEST_LOG}" --tables "${TABLES}" --mode "ablation_n_32b=${n}"
            echo "[fill] Table 8 row n=${n} updated from ${LATEST_LOG}"
        fi
    fi

    # ── Clean up stale VLLM/Ray processes before next run ────────────────────
    echo "[cleanup] Killing stale VLLM::* processes (prevent NCCL deadlock)..."
    pkill -9 -f 'VLLM::' 2>/dev/null || true
    sleep 3
    echo "[cleanup] Done."
    echo "========== rollout_n=${n} done =========="
done
echo "[ablation_rollout_n_32b] All done. results_tables.md updated."
