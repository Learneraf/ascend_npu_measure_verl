# Qwen3 GRPO 后训练系统性能实验结果（8B 全量 & 32B LoRA）

## 实验配置

### Qwen3-8B 全量配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-8B |
| 算法 | GRPO |
| GPU | 8 × A100-SXM4-80GB |
| Rollout 后端 | vLLM 0.20.2（TP=1） |
| train_batch_size | 256 |
| rollout_n | 4 |
| max_prompt_length | 512 |
| max_response_length | 1024 |
| 数据集 | GSM8K |

### Qwen3-32B LoRA 配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-32B |
| 微调方式 | LoRA（r=64 α=32 target=all-linear）|
| 算法 | GRPO |
| GPU | 8 × A100-SXM4-80GB |
| Rollout 后端 | vLLM 0.20.2（TP=4）|
| train_batch_size | 64 |
| rollout_n | 2 |
| max_prompt_length | 512 |
| max_response_length | 512 |
| param_offload | True |
| optimizer_offload | True |
| gpu_memory_utilization | 0.40 |
| 数据集 | GSM8K |

---

## 第一部分：基准性能

> 8B：veRL 0.8.0 main_ppo_sync，8×A100，vLLM TP=1，step 1 含冷启动开销。
> 32B：LoRA r=64 α=32，FSDP + param_offload + optimizer_offload，vLLM TP=4。

### 表 1：各阶段耗时

| step | gen(s) | old_log_prob(s) | ref(s) | update_actor(s) | update_weights(s) | step_total(s) | gen占比% | update占比% |
|------|--------|-----------------|--------|-----------------|-------------------|---------------|----------|-------------|
| 1 | - | - | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - | - |
| 5 | - | - | - | - | - | - | - | - |
| 6 | - | - | - | - | - | - | - | - |
| 7 | - | - | - | - | - | - | - | - |
| 8 | - | - | - | - | - | - | - | - |
| 9 | - | - | - | - | - | - | - | - |
| 10 | - | - | - | - | - | - | - | - |
| 11 | - | - | - | - | - | - | - | - |
| 12 | - | - | - | - | - | - | - | - |
| 13 | - | - | - | - | - | - | - | - |
| 14 | - | - | - | - | - | - | - | - |
| 15 | - | - | - | - | - | - | - | - |
| 16 | - | - | - | - | - | - | - | - |
| 17 | - | - | - | - | - | - | - | - |
| 18 | - | - | - | - | - | - | - | - |
| 19 | - | - | - | - | - | - | - | - |
| 20 | - | - | - | - | - | - | - | - |
| **均值** | - | - | - | - | - | - | - | - |
| **std** | - | - | - | - | - | - | - | - |

### 表 2：各阶段耗时（32B LoRA）

| step | gen(s) | old_log_prob(s) | ref(s) | update_actor(s) | update_weights(s) | step_total(s) | gen占比% | update占比% |
|------|--------|-----------------|--------|-----------------|-------------------|---------------|----------|-------------|
| 1 | - | - | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - | - |
| 5 | - | - | - | - | - | - | - | - |
| 6 | - | - | - | - | - | - | - | - |
| 7 | - | - | - | - | - | - | - | - |
| 8 | - | - | - | - | - | - | - | - |
| 9 | - | - | - | - | - | - | - | - |
| 10 | - | - | - | - | - | - | - | - |
| 11 | - | - | - | - | - | - | - | - |
| 12 | - | - | - | - | - | - | - | - |
| 13 | - | - | - | - | - | - | - | - |
| 14 | - | - | - | - | - | - | - | - |
| 15 | - | - | - | - | - | - | - | - |
| 16 | - | - | - | - | - | - | - | - |
| 17 | - | - | - | - | - | - | - | - |
| 18 | - | - | - | - | - | - | - | - |
| 19 | - | - | - | - | - | - | - | - |
| 20 | - | - | - | - | - | - | - | - |
| **均值** | - | - | - | - | - | - | - | - |
| **std** | - | - | - | - | - | - | - | - |

### 表 3：系统效率指标

| step | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 实际生成速率(seq/s) |
|------|------|---------------|-------------|---------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| 6 | - | - | - | - |
| 7 | - | - | - | - |
| 8 | - | - | - | - |
| 9 | - | - | - | - |
| 10 | - | - | - | - |
| 11 | - | - | - | - |
| 12 | - | - | - | - |
| 13 | - | - | - | - |
| 14 | - | - | - | - |
| 15 | - | - | - | - |
| 16 | - | - | - | - |
| 17 | - | - | - | - |
| 18 | - | - | - | - |
| 19 | - | - | - | - |
| 20 | - | - | - | - |
| **均值** | - | - | - | - |
| **std** | - | - | - | - |

### 表 4：系统效率指标（32B LoRA）

| step | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 实际生成速率(seq/s) |
|------|------|---------------|-------------|---------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| 6 | - | - | - | - |
| 7 | - | - | - | - |
| 8 | - | - | - | - |
| 9 | - | - | - | - |
| 10 | - | - | - | - |
| 11 | - | - | - | - |
| 12 | - | - | - | - |
| 13 | - | - | - | - |
| 14 | - | - | - | - |
| 15 | - | - | - | - |
| 16 | - | - | - | - |
| 17 | - | - | - | - |
| 18 | - | - | - | - |
| 19 | - | - | - | - |
| 20 | - | - | - | - |
| **均值** | - | - | - | - |
| **std** | - | - | - | - |

### 表 5：训练质量指标

| step | 平均回复长度(token) | 截断比例% | 平均奖励 |
|------|---------------------|-----------|----------|
| 1 | - | - | - |
| 2 | - | - | - |
| 3 | - | - | - |
| 4 | - | - | - |
| 5 | - | - | - |
| 6 | - | - | - |
| 7 | - | - | - |
| 8 | - | - | - |
| 9 | - | - | - |
| 10 | - | - | - |
| 11 | - | - | - |
| 12 | - | - | - |
| 13 | - | - | - |
| 14 | - | - | - |
| 15 | - | - | - |
| 16 | - | - | - |
| 17 | - | - | - |
| 18 | - | - | - |
| 19 | - | - | - |
| 20 | - | - | - |
| **均值** | - | - | - |
| **std** | - | - | - |

---

### 表 6：训练质量指标（32B LoRA）

| step | 响应长度均值 | 截断比例% | 平均奖励 |
|------|------------|-----------|----------|
| 1 | - | - | - |
| 2 | - | - | - |
| 3 | - | - | - |
| 4 | - | - | - |
| 5 | - | - | - |
| 6 | - | - | - |
| 7 | - | - | - |
| 8 | - | - | - |
| 9 | - | - | - |
| 10 | - | - | - |
| 11 | - | - | - |
| 12 | - | - | - |
| 13 | - | - | - |
| 14 | - | - | - |
| 15 | - | - | - |
| 16 | - | - | - |
| 17 | - | - | - |
| 18 | - | - | - |
| 19 | - | - | - |
| 20 | - | - | - |
| **均值** | - | - | - |
| **std** | - | - | - |

---

---

## 第二部分：超参消融实验

> 各组取稳定步（去掉 step 1 warmup）的均值±std；基准配置行以斜体标注。

### 表 7：Rollout 采样数消融（rollout_n）

> 固定：train_batch_size=256，max_response_length=1024，max_prompt_length=512

| rollout_n | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 实际生成速率(seq/s) |
|-----------|--------|-----------------|---------------|------|---------------|-------------|---------------------|
| 1 | - | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - |
| 8 | - | - | - | - | - | - | - |
| 16 | - | - | - | - | - | - | - |

### 表 8：Rollout 采样数消融（rollout_n，32B LoRA）

> 固定：train_batch_size=64，max_response_length=512，max_prompt_length=512

| rollout_n | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 实际生成速率(seq/s) |
|-----------|--------|-----------------|---------------|------|---------------|-------------|---------------------|
| 1 | - | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - |
| 8 | - | - | - | - | - | - | - |
| 16 | - | - | - | - | - | - | - |

### 表 9：训练批大小消融（train_batch_size）

> 固定：rollout_n=4，max_response_length=1024，max_prompt_length=512

| train_batch_size | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 理论并发序列数 |
|------------------|--------|-----------------|---------------|------|---------------|-------------|----------------|
| 64 | - | - | - | - | - | - | - |
| 128 | - | - | - | - | - | - | - |
| 256 | - | - | - | - | - | - | - |
| 512 | - | - | - | - | - | - | - |

### 表 10：训练批大小消融（train_batch_size，32B LoRA）

> 固定：rollout_n=2，max_response_length=512，max_prompt_length=512

| train_batch_size | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | GPU显存(GB) | 理论并发序列数 |
|------------------|--------|-----------------|---------------|------|---------------|-------------|----------------|
| 16 | - | - | - | - | - | - | - |
| 32 | - | - | - | - | - | - | - |
| 64 | - | - | - | - | - | - | - |
| 128 | - | - | - | - | - | - | - |

### 表 11：最大响应长度消融（max_response_length）

> 固定：train_batch_size=256，rollout_n=4，max_prompt_length=512

| max_response_length | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | 截断比例% | GPU显存(GB) |
|---------------------|--------|-----------------|---------------|------|---------------|-----------|-------------|
| 256 | - | - | - | - | - | - | - |
| 512 | - | - | - | - | - | - | - |
| 1024 | - | - | - | - | - | - | - |
| 2048 | - | - | - | - | - | - | - |

---

### 表 12：最大响应长度消融（max_response_length，32B LoRA）

> 固定：train_batch_size=64，rollout_n=2，max_prompt_length=512

| max_response_length | gen(s) | update_actor(s) | step_total(s) | MFU% | 吞吐量(tok/s) | 截断比例% | GPU显存(GB) |
|---------------------|--------|-----------------|---------------|------|---------------|-----------|-------------|
| 256 | - | - | - | - | - | - | - |
| 512 | - | - | - | - | - | - | - |
| 1024 | - | - | - | - | - | - | - |
| 2048 | - | - | - | - | - | - | - |

---

---

## 第三部分：GPU 算子级分析

> 数据来源：`python3 parse_rollout_profile.py --trace-dir outputs/rollout_profile_<RUN_ID> --log outputs/<RUN_ID>.log --tables results_tables.md`

| 参数 | 值 |
|------|-----|
| profiling_steps | [3, 4, 5] |
| profiler | torch（cuda + cpu） |
| actor.profiler contents | [cuda, cpu] |
| gpu_memory_utilization | 0.4 |

### 表 13：训练阶段算子分解（update_actor，step 3–5 均值）

> 独立 profiler 实验；FSDP 策略；8×A100；数值为 8 GPU 均值

| 算子类型 | GPU耗时(ms) | 占 update_actor% |
|----------|-------------|------------------|
| Attention（前向+反向） | - | - |
| Linear / GEMM（前向+反向） | - | - |
| AllReduce / NCCL 通信 | - | - |
| Optimizer（Adam step） | - | - |
| 其他（归一化/激活/调度） | - | - |
| **update_actor 合计** | - | - |

### 表 14：训练阶段算子分解（update_actor，32B LoRA step 3–5 均值）

> FSDP + param_offload；8×A100；数值为 8 GPU 均值

| 算子类型 | GPU耗时(ms) | 占 update_actor% |
|----------|-------------|------------------|
| Attention（前向+反向） | - | - |
| Linear / GEMM（前向+反向） | - | - |
| AllReduce / NCCL 通信 | - | - |
| Optimizer（Adam step） | - | - |
| 其他（归一化/激活/调度） | - | - |
| **update_actor 合计** | - | - |

> 数据来源：`python3 parse_rollout_profile.py --trace-dir outputs/rollout_profile_32b_<RUN_ID> --log outputs/<RUN_ID>.log --tables results_tables.md`

| 参数 | 值 |
|------|-----|
| profiling_steps | [3, 4, 5] |
| profiler | torch（cuda + cpu）|
| gpu_memory_utilization | 0.40 |
| vLLM TP | 4 |

### 表 15：Rollout 阶段算子分解（gen，step 3–5 均值）

> 独立 profiler 实验（profile_rollout.sh）；8 个 vLLM worker 均值；TP=1

| 阶段 / 算子类型 | GPU耗时(ms) | 占 gen 总时间% |
|----------------|-------------|----------------|
| **Prefill 阶段** | - | - |
| Prefill Attention（Flash-Attn varlen） | - | - |
| Prefill Linear / GEMM | - | - |
| Prefill 其他（RoPE/激活/归一化） | - | - |
| **Decode 阶段** | - | - |
| Decode Attention（Flash-Attn KV-cache） | - | - |
| Decode Linear / GEMM | - | - |
| Decode 其他（采样/softmax） | - | - |
| **GPU 合计** | - | - |

---

### 表 16：Rollout 阶段算子分解（gen，32B LoRA step 3–5 均值）

> vLLM TP=4；4 个 vLLM worker 均值

| 阶段 / 算子类型 | GPU耗时(ms) | 占 gen 总时间% |
|----------------|-------------|----------------|
| **Prefill 阶段** | - | - |
| Prefill Attention（Flash-Attn varlen） | - | - |
| Prefill Linear / GEMM | - | - |
| Prefill 其他（RoPE/激活/归一化） | - | - |
| **Decode 阶段** | - | - |
| Decode Attention（Flash-Attn KV-cache） | - | - |
| Decode Linear / GEMM | - | - |
| Decode 其他（采样/softmax） | - | - |
| **GPU 合计** | - | - |

---

---

## 第四部分：全异步训练

> veRL 0.8.0 fully_async_policy，8 GPU A100-SXM4-80GB（4 GPU FSDP Actor+RefPolicy + 4 GPU vLLM TP=1×4 独立副本）；
> Rollout pool 与 Training 完全重叠运行，每 trigger_sync_step=4 个 mini-batch 更新同步一次参数到 rollout pool。

| 参数 | 值 |
|------|-----|
| 训练 GPU | 4（FSDP Actor + RefPolicy）|
| 推理 GPU | 4（vLLM TP=1×4 副本）|
| ppo_mini_batch_size | 64 |
| rollout_n | 4 |
| staleness_threshold | 0.5 |
| trigger_sync_step | 4 |
| gpu_memory_utilization | 0.85 |

### 表 17：Async vs Sync 性能总览

> 对比 Sync 基准（表1均值，8 GPU 共享，Rollout+Training 串行）与 Async 4+4 GPU 分离（Rollout 与 Training 完全重叠）。
> **等效sync耗时** = async 连续 4 个 param-sync step 的总壁钟时间（处理 4×256=1024 samples，与 sync 单步数据量相同）。

| 配置 | GPU分配 | 吞吐量(tok/s) | 实际生成速率(seq/s) | MFU%(训练GPU) | 等效sync耗时(s) | 加速比 | staleness_ratio% |
|------|---------|--------------|---------------------|--------------|----------------|-------|------------------|
| Sync 基准 | - | - | - | - | - | - | - |
| Async 4+4 GPU | - | - | - | - | - | - | - |


---

> veRL 0.8.0 fully_async_policy；8 GPU A100-SXM4-80GB（4 GPU FSDP LoRA Actor+RefPolicy + 4 GPU vLLM TP=4）；
> Rollout pool 与 Training 完全重叠运行，每 trigger_sync_step=4 个 mini-batch 更新同步一次参数。

| 参数 | 值 |
|------|-----|
| 训练 GPU | 4（FSDP LoRA Actor + RefPolicy）|
| 推理 GPU | 4（vLLM TP=4 × 1 副本）|
| ppo_mini_batch_size | 16 |
| rollout_n | 2 |
| staleness_threshold | 0.5 |
| trigger_sync_step | 4 |
| gpu_memory_utilization | 0.85 |

### 表 18：Async vs Sync 性能总览（32B LoRA）

> 对比 Sync 基准（表2均值，8 GPU 共享，Rollout+Training 串行）与 Async 4+4 GPU 分离（Rollout 与 Training 完全重叠）。
> **等效sync耗时** = async 连续 4 个 param-sync step 的总壁钟时间。

| 配置 | GPU分配 | 吞吐量(tok/s) | 实际生成速率(seq/s) | MFU%(训练GPU) | 等效sync耗时(s) | 加速比 | staleness_ratio% |
|------|---------|--------------|---------------------|--------------|----------------|-------|------------------|
| Sync 基准 | - | - | - | - | - | - | - |
| Async 4+4 GPU | - | - | - | - | - | - | - |
