# gpu_test_0610 Ascend NPU 测试说明

本目录保留原有四部分实验操作流程。其中第一部分 baseline 和第二部分 ablation 已迁移为默认在 Ascend NPU 上运行，但入口文件名仍保留 `a100`，方便沿用原来的运行方式。

## 原实验操作流程

第一部分的实验：

```bash
cd /mnt/data/gpu_test_0610
bash run_qwen3_8b_a100.sh
bash run_qwen3_32b_lora_a100.sh
```

第二部分的实验：

```bash
cd /mnt/data/gpu_test_0610/ablation
bash ablation_rollout_n.sh
bash ablation_rollout_n_32b.sh
bash ablation_batchsize.sh
bash ablation_batchsize_32b.sh
bash ablation_seqlen.sh
bash ablation_seqlen_32b.sh
```

第三部分的实验：

```bash
cd /mnt/data/gpu_test_0610/other_scripts
bash profile_rollout.sh
bash profile_rollout_32b.sh
```

第四部分的实验（建议 verl 版本 0.8.x 及以上）：

```bash
cd /mnt/data/gpu_test_0610/other_scripts
bash run_async.sh
bash run_async_32b.sh
```

运行完毕后，结果会自动填入 `results_tables.md`，NPU/CPU/RAM 监控数据在 `./outputs/*_monitor.csv`。请检查 `results_tables.md` 和对应 monitor CSV 是否有空值，若无异常，再交付结果文件。

说明：当前 Ascend 迁移已覆盖第一部分和第二部分；第三、第四部分的 `other_scripts` 仍保留原操作流程，如需在 Ascend 上运行，需要单独适配对应脚本。

## Ascend 默认环境

baseline 和 ablation 脚本默认使用以下路径和环境：

- `ACCELERATOR=ascend`
- `DEVICE_IDS=0,1,2,3,4,5,6,7`
- `PROJ=/mnt/data/FlagPerf`
- `MODEL_PATH=${PROJ}/models/Qwen3-8B` 或自动选择 `${PROJ}/models/Qwen3-32B` / `${PROJ}/models/Qwen3-32B-local`
- `DATA_DIR=${PROJ}/data/gsm8k_verl`

脚本会自动设置 `ASCEND_RT_VISIBLE_DEVICES`、`VLLM_USE_V1=1`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`HCCL_CONNECT_TIMEOUT=1500`，并传入 `trainer.device=npu`。如果当前 veRL 没有 `verl.trainer.main_ppo_sync`，脚本会使用当前环境存在的 `verl.trainer.main_ppo`。

如需覆盖模型、数据或设备，可在运行时显式传入：

```bash
PROJ=/path/to/FlagPerf DEVICE_IDS=0,1,2,3 MODEL_PATH=/path/to/Qwen3-8B bash run_qwen3_8b_a100.sh
```

## 快速 smoke test

用于快速验证 8B 链路，不填表：

```bash
cd /mnt/data/gpu_test_0610
FILL_MODE=skip \
train_batch_size=8 ppo_mini_batch_size=8 \
max_response_length=32 rollout_n=1 rollout_tp=2 \
actor_micro_bsz=1 logprob_micro_bsz=1 \
total_training_steps=1 total_epochs=1 test_freq=-1 save_freq=-1 \
bash run_qwen3_8b_a100.sh
```

## 32B 模型要求

运行 32B LoRA 前必须准备：

```bash
/mnt/data/FlagPerf/models/Qwen3-32B
```

本机当前存在的 32B 模型目录为：

```bash
/mnt/data/FlagPerf/models/Qwen3-32B-local
```

脚本会优先使用 `Qwen3-32B`，不存在时自动回退到 `Qwen3-32B-local`。也可以显式传入：

```bash
MODEL_PATH=/path/to/Qwen3-32B bash run_qwen3_32b_lora_a100.sh
```

如果模型目录不存在，脚本会在训练前预检失败，避免启动到中途才报错。第二部分的 32B ablation 脚本会捕获这类失败，并在 `results_tables.md` 中写入对应失败/OOM 标记后继续下一组参数。

## 输出文件

训练日志和 NPU/CPU/RAM 监控 CSV 写入：

```bash
/mnt/data/gpu_test_0610/outputs/
```

`monitor_gpu.py` 默认采集 NPU util、HBM used/total、power、temperature。HCCS 带宽采样会阻塞训练监控周期，默认关闭；确实需要时可设置 `MONITOR_NPU_HCCS_BW=1`。

## Ablation 说明

`ablation/*.sh` 现在通过仓库相对路径定位 baseline 入口、`fill_tables.py` 和 `results_tables.md`，不再依赖 `/data/yanziyi/gpu_test_0610`。默认使用当前 Python 环境；如确实需要仓库内 `venv`，可设置：

```bash
USE_VENV=1 bash ablation_rollout_n.sh
```
