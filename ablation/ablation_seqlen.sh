#!/usr/bin/env bash
# Ablation: max_response_length sweep
set -euo pipefail
BASE=/data/yanziyi/gpu_test_0610
TABLES=${BASE}/results_tables.md

for resp in 256 512 1024 2048; do
    echo "========== max_response_length=${resp} =========="
    set +e
    max_response_length=${resp} FILL_MODE=skip bash ${BASE}/run_qwen3_8b_a100.sh \
        data.max_response_length=${resp} \
        trainer.total_epochs=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_resp${resp}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[warn] Training failed (rc=${TRAIN_RC}), skipping fill for resp=${resp}"
    else
        LATEST_LOG=$(ls -t ${BASE}/outputs/qwen3_8b_grpo_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            cd ${BASE} && source venv/bin/activate && \
            python3 fill_tables.py --log "${LATEST_LOG}" --tables "${TABLES}" --mode "ablation_seq=${resp}"
            echo "[fill] Table 11 row seq=${resp} updated"
        fi
    fi

    # 清理残留 VLLM 进程，避免下一轮冲突
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 3
    echo "========== resp=${resp} done =========="
done
echo "[ablation_seqlen] All done."
