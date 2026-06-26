# gpu_test_0610 Ascend baseline

本目录的两个 baseline 入口已经迁移为默认在 Ascend NPU 上运行，但保留原文件名：

```bash
cd /mnt/data/gpu_test_0610
bash run_qwen3_8b_a100.sh
bash run_qwen3_32b_lora_a100.sh
```

默认环境：

- `ACCELERATOR=ascend`
- `DEVICE_IDS=0,1,2,3,4,5,6,7`
- `PROJ=/mnt/data/FlagPerf`
- `MODEL_PATH=${PROJ}/models/Qwen3-8B` 或 `${PROJ}/models/Qwen3-32B`
- `DATA_DIR=${PROJ}/data/gsm8k_verl`

脚本会自动设置 `ASCEND_RT_VISIBLE_DEVICES`、`VLLM_USE_V1=1`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`HCCL_CONNECT_TIMEOUT=1500`，并传入 `trainer.device=npu`。如果当前 veRL 没有 `verl.trainer.main_ppo_sync`，脚本会使用当前环境存在的 `verl.trainer.main_ppo`。

## Smoke test

用于快速验证链路，不填表：

```bash
FILL_MODE=skip \
train_batch_size=8 ppo_mini_batch_size=8 \
max_response_length=32 rollout_n=1 rollout_tp=2 \
actor_micro_bsz=1 logprob_micro_bsz=1 \
total_training_steps=1 total_epochs=1 test_freq=-1 save_freq=-1 \
bash run_qwen3_8b_a100.sh
```

## 32B requirement

运行 32B LoRA 前必须准备：

```bash
/mnt/data/FlagPerf/models/Qwen3-32B
```

或者显式传入：

```bash
MODEL_PATH=/path/to/Qwen3-32B bash run_qwen3_32b_lora_a100.sh
```

如果模型目录不存在，脚本会在训练前预检失败，避免启动到中途才报错。

## Outputs

训练日志和 NPU/CPU/RAM 监控 CSV 写入：

```bash
/mnt/data/gpu_test_0610/outputs/
```

`monitor_gpu.py` 默认采集 NPU util、HBM used/total、power、temperature。HCCS 带宽采样会阻塞训练监控周期，默认关闭；确实需要时可设置 `MONITOR_NPU_HCCS_BW=1`。
