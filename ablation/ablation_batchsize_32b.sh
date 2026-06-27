#!/usr/bin/env bash
# Ablation: train_batch_size in {16, 32, 64, 128}  fixed rollout_n=2  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

for bsz in 16 32 64 128; do
    mini=$(( bsz / 4 ))
    echo "========== train_batch_size=${bsz} ppo_mini=${mini} =========="
    set +e
    train_batch_size=${bsz} ppo_mini_batch_size=${mini} FILL_MODE=skip ablation_run_32b \
        data.train_batch_size=${bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_bsz${bsz}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[bsz=${bsz}] FAILED (rc=${TRAIN_RC}), marking OOM"
        ablation_mark_oom "ablation_bsz_32b_oom=${bsz}" "Table 10 row bsz=${bsz}"
    else
        ablation_fill_latest "qwen3_32b_lora_grpo_*.log" "ablation_bsz_32b=${bsz}" "Table 10 row bsz=${bsz}" || true
    fi
    ablation_cleanup
    echo "========== bsz=${bsz} done =========="
done
echo "[ablation_batchsize_32b] All done."
