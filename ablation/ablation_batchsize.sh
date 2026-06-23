#!/usr/bin/env bash
# Ablation: train_batch_size in {64, 128, 256, 512}  fixed rollout_n=5
set -euo pipefail
BASE=/data/yanziyi/gpu_test_0610
TABLES=${BASE}/results_tables.md

for bsz in 64 128 256 512; do
    mini=$(( bsz / 4 ))
    echo "========== train_batch_size=${bsz} ppo_mini=${mini} =========="
    set +e
    train_batch_size=${bsz} ppo_mini_batch_size=${mini} FILL_MODE=skip bash ${BASE}/run_qwen3_8b_a100.sh \
        data.train_batch_size=${bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini} \
        trainer.total_epochs=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_bsz${bsz}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[warn] Training failed (rc=${TRAIN_RC}), skipping fill for bsz=${bsz}"
    else
        LATEST_LOG=$(ls -t ${BASE}/outputs/qwen3_8b_grpo_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            cd ${BASE} && source venv/bin/activate && \
            python3 fill_tables.py --log "${LATEST_LOG}" --tables "${TABLES}" --mode "ablation_bsz=${bsz}"
            echo "[fill] Table 9 row bsz=${bsz} updated"
        fi
    fi

    # 清理残留 VLLM 进程，避免下一轮冲突
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 3
    echo "========== bsz=${bsz} done =========="
done
echo "[ablation_batchsize] All done."
