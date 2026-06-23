#!/usr/bin/env bash
# Ablation: train_batch_size in {16, 32, 64, 128}  fixed rollout_n=2  (32B LoRA)
# 每个配置跑20步（性能基准，不需要full epoch）
set -euo pipefail
BASE=/data/yanziyi/gpu_test_0610
TABLES=${BASE}/results_tables.md

for bsz in 16 32 64 128; do
    mini=$(( bsz / 4 ))
    echo "========== train_batch_size=${bsz} ppo_mini=${mini} =========="
    train_batch_size=${bsz} ppo_mini_batch_size=${mini} FILL_MODE=skip bash ${BASE}/run_qwen3_32b_lora_a100.sh \
        data.train_batch_size=${bsz} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=20 \
        trainer.save_freq=-1 \
        trainer.test_freq=999 \
        trainer.experiment_name=ablation_32b_bsz${bsz}_$(date +%H%M%S)

    LATEST_LOG=$(ls -t ${BASE}/outputs/qwen3_32b_lora_grpo_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        cd ${BASE} && source venv/bin/activate && \
        python3 fill_tables.py --log "${LATEST_LOG}" --tables "${TABLES}" --mode "ablation_bsz_32b=${bsz}"
        echo "[fill] Table 10 row bsz=${bsz} updated"
    fi
    echo "========== bsz=${bsz} done =========="
done
echo "[ablation_batchsize_32b] All done."
