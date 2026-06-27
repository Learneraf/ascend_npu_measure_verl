#!/usr/bin/env bash
# Ablation: rollout_n in {1, 2, 4, 8, 16}  fixed train_batch_size=64  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

for n in 1 2 4 8 16; do
    echo "========== rollout_n=${n} =========="
    set +e
    rollout_n=${n} FILL_MODE=skip ablation_run_32b \
        actor_rollout_ref.rollout.n=${n} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_rollout_n${n}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[rollout_n=${n}] FAILED (rc=${TRAIN_RC}), marking OOM"
        ablation_mark_oom "ablation_n_32b_oom=${n}" "Table 8 row n=${n}"
    else
        ablation_fill_latest "qwen3_32b_lora_grpo_*.log" "ablation_n_32b=${n}" "Table 8 row n=${n}" || true
    fi

    # ── Clean up stale VLLM/Ray processes before next run ────────────────────
    ablation_cleanup
    echo "========== rollout_n=${n} done =========="
done
echo "[ablation_rollout_n_32b] All done. results_tables.md updated."
