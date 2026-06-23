#!/usr/bin/env bash
# Ablation: max_response_length in {256, 512, 1024, 2048}  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail
BASE=/data/yanziyi/gpu_test_0610
TABLES=${BASE}/results_tables.md

for resp in 256 512 1024 2048; do
    echo "========== max_response_length=${resp} =========="
    max_response_length=${resp} FILL_MODE=skip bash ${BASE}/run_qwen3_32b_lora_a100.sh \
        data.max_response_length=${resp} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_resp${resp}_$(date +%H%M%S)

    LATEST_LOG=$(ls -t ${BASE}/outputs/qwen3_32b_lora_grpo_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        cd ${BASE} && source venv/bin/activate && \
        python3 fill_tables.py --log "${LATEST_LOG}" --tables "${TABLES}" --mode "ablation_seq_32b=${resp}"
        echo "[fill] Table 12 row seq=${resp} updated"
    fi
    echo "========== resp=${resp} done =========="
done
echo "[ablation_seqlen_32b] All done."
