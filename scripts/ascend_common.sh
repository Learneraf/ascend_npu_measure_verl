#!/usr/bin/env bash
# Common runtime setup for running the A100-origin veRL scripts on Ascend NPU.

ascend_repo_root() {
    local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    local dir
    dir="$(cd "$(dirname "${src}")/.." && pwd)"
    printf '%s\n' "${dir}"
}

ascend_setup_runtime() {
    ACCELERATOR=${ACCELERATOR:-ascend}
    DEVICE_IDS=${DEVICE_IDS:-0,1,2,3,4,5,6,7}
    NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(awk -F, '{print NF}' <<<"${DEVICE_IDS}")}

    if [ "${ACCELERATOR}" = "ascend" ] || [ "${ACCELERATOR}" = "npu" ]; then
        export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${DEVICE_IDS}}"
        export VLLM_USE_V1="${VLLM_USE_V1:-1}"
        export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
        export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
        export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        TRAINER_DEVICE=npu
        MONITOR_GPU_IDS="${ASCEND_RT_VISIBLE_DEVICES}"
    elif [ "${ACCELERATOR}" = "cuda" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_IDS}}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        TRAINER_DEVICE=cuda
        MONITOR_GPU_IDS="${CUDA_VISIBLE_DEVICES}"
    else
        echo "[preflight] unsupported ACCELERATOR=${ACCELERATOR}; expected ascend or cuda" >&2
        return 2
    fi

    export ACCELERATOR DEVICE_IDS NGPUS_PER_NODE TRAINER_DEVICE MONITOR_GPU_IDS
}

ascend_find_trainer_module() {
    local preferred="${1:-}"
    python3 - "${preferred}" <<'PY'
import importlib.util
import sys

preferred = sys.argv[1]
candidates = []
if preferred:
    candidates.append(preferred)
candidates.extend(["verl.trainer.main_ppo_sync", "verl.trainer.main_ppo"])

seen = set()
for name in candidates:
    if name in seen:
        continue
    seen.add(name)
    if importlib.util.find_spec(name) is not None:
        print(name)
        raise SystemExit(0)

print("No veRL trainer module found. Tried: " + ", ".join(candidates), file=sys.stderr)
raise SystemExit(1)
PY
}

ascend_preflight() {
    local model_path="$1"
    local data_dir="$2"
    local ndevices="$3"

    if [ ! -d "${model_path}" ]; then
        echo "[preflight] model path not found: ${model_path}" >&2
        return 10
    fi
    if [ ! -f "${data_dir}/train.parquet" ] || [ ! -f "${data_dir}/test.parquet" ]; then
        echo "[preflight] expected train/test parquet under: ${data_dir}" >&2
        return 11
    fi

    if [ "${TRAINER_DEVICE}" = "npu" ]; then
        command -v npu-smi >/dev/null || {
            echo "[preflight] npu-smi not found in PATH" >&2
            return 12
        }
        python3 - "${ndevices}" <<'PY'
import sys

expected = int(sys.argv[1])
try:
    import torch
    import torch_npu  # noqa: F401
except Exception as exc:
    print(f"[preflight] torch_npu import failed: {exc}", file=sys.stderr)
    raise SystemExit(13)

if not torch.npu.is_available():
    print("[preflight] torch.npu.is_available() is False", file=sys.stderr)
    raise SystemExit(14)

count = torch.npu.device_count()
if count < expected:
    print(f"[preflight] expected at least {expected} NPU devices, found {count}", file=sys.stderr)
    raise SystemExit(15)

print(f"[preflight] torch_npu OK, visible NPU devices: {count}")
PY
    fi
}
