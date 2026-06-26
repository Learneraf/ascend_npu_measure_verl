#!/usr/bin/env bash
# Ablation: max_response_length in {256, 512, 1024, 2048}  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ablation_common.sh"

for resp in 256 512 1024 2048; do
    echo "========== max_response_length=${resp} =========="
    set +e
    max_response_length=${resp} FILL_MODE=skip bash "${BASE}/run_qwen3_32b_lora_a100.sh" \
        data.max_response_length=${resp} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_resp${resp}_$(date +%H%M%S)
    TRAIN_RC=$?
    set -e

    if [ ${TRAIN_RC} -ne 0 ]; then
        echo "[resp=${resp}] FAILED (rc=${TRAIN_RC}), marking OOM"
        ablation_mark_oom "ablation_seq_32b_oom=${resp}" "Table 12 row seq=${resp}"
    else
        ablation_fill_latest "qwen3_32b_lora_grpo_*.log" "ablation_seq_32b=${resp}" "Table 12 row seq=${resp}" || true
    fi
    ablation_cleanup
    echo "========== resp=${resp} done =========="
done
echo "[ablation_seqlen_32b] All done."
