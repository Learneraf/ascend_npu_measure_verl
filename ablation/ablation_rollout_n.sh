#!/usr/bin/env bash
# Ablation: rollout_n in {1, 2, 4, 8, 16}  fixed train_batch_size=256
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

for n in 1 2 4 8 16; do
    echo "========== rollout_n=${n} =========="
    set +e
    rollout_n=${n} FILL_MODE=skip bash "${BASE}/run_qwen3_8b_a100.sh" \
        actor_rollout_ref.rollout.n=${n} \
        trainer.total_epochs=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_rollout_n${n}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[warn] Training failed (rc=${TRAIN_RC}), skipping fill for n=${n}"
    else
        ablation_fill_latest "qwen3_8b_grpo_*.log" "ablation_n=${n}" "Table 7 row n=${n}" || true
    fi

    # 清理残留 VLLM 进程，避免下一轮冲突
    ablation_cleanup
    echo "========== rollout_n=${n} done =========="
done
echo "[ablation_rollout_n] All done. results_tables.md updated."
