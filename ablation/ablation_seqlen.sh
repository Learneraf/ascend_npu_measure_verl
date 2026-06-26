#!/usr/bin/env bash
# Ablation: max_response_length sweep
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

for resp in 256 512 1024 2048; do
    echo "========== max_response_length=${resp} =========="
    set +e
    max_response_length=${resp} FILL_MODE=skip bash "${BASE}/run_qwen3_8b_a100.sh" \
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
        ablation_fill_latest "qwen3_8b_grpo_*.log" "ablation_seq=${resp}" "Table 11 row seq=${resp}" || true
    fi

    # 清理残留 VLLM 进程，避免下一轮冲突
    ablation_cleanup
    echo "========== resp=${resp} done =========="
done
echo "[ablation_seqlen] All done."
