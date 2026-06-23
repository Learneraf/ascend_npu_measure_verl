#!/usr/bin/env bash
# =============================================================================
# setup.sh — Reproduce gpu_test_0610 environment on a new server
#
# Prerequisites:
#   - Python 3.12
#   - NVIDIA driver >= 570 (tested on 595.71 with A100-SXM4-80GB)
#   - CUDA 12.8 compatible hardware (adjust torch index URL if needed)
#   - 8× GPU with ≥40GB VRAM per card recommended (tested: 8×A100-80GB)
#
# Steps:
#   1. Edit the two variables in USER CONFIGURATION below
#   2. Download Qwen3-8B model manually (see note at bottom)
#   3. bash setup.sh
# =============================================================================
set -euo pipefail

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
PROJ=/data/yanziyi               # parent dir: models/ and data/ will go here
SCRIPT_DIR=/data/yanziyi/gpu_test_0610   # directory containing this script
# ─────────────────────────────────────────────────────────────────────────────

LOG=${SCRIPT_DIR}/setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup.sh started at $(date) ==="

VENV=${SCRIPT_DIR}/venv

# ── Step 1: Create venv ───────────────────────────────────────────────────────
echo "[1/6] Creating venv at ${VENV} ..."
if [ ! -f "${VENV}/bin/activate" ]; then
    python3.12 -m venv "${VENV}"
    echo "      venv created"
else
    echo "      venv already exists, skipping"
fi
source "${VENV}/bin/activate"

# ── Step 2: Install PyTorch + CUDA packages ───────────────────────────────────
# Tested: torch==2.11.0 with CUDA 13.0 (PyTorch nightly / future release)
# For CUDA 12.8 stable:  --index-url https://download.pytorch.org/whl/cu128
# For CUDA 12.4 stable:  --index-url https://download.pytorch.org/whl/cu124
# Adjust to match your driver + torch release available at time of setup.
echo "[2/6] Installing PyTorch (CUDA 12.8 wheels) ..."
TORCH_INDEX=https://download.pytorch.org/whl/cu128
pip install --quiet \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    "torchaudio==2.11.0" \
    --index-url "${TORCH_INDEX}"

# flash-attn must match torch+CUDA; build from source if wheel unavailable
echo "      Installing flash-attn ..."
pip install --quiet flash-attn==2.8.3 --no-build-isolation || \
    pip install --quiet flash-attn --no-build-isolation    # fallback: latest

# flashinfer
pip install --quiet flashinfer-python==0.6.11.post1 || true

# ── Step 3: Install vLLM, Ray, core ML packages ──────────────────────────────
echo "[3/6] Installing vLLM, Ray, core packages ..."
pip install --quiet \
    "vllm==0.20.2" \
    "ray[default]==2.55.1" \
    "transformers==5.6.0" \
    "accelerate==1.13.0" \
    "datasets==4.8.5" \
    "tensordict==0.10.0" \
    "omegaconf==2.3.0" \
    "hydra-core==1.3.2" \
    "wandb==0.26.1" \
    "tensorboard==2.20.0" \
    "pydantic==2.13.4" \
    "sentencepiece==0.2.1" \
    "tiktoken==0.12.0" \
    "codetiming==1.4.0" \
    "torchao==0.17.0"

# ── Step 4: Install veRL + project-specific packages ─────────────────────────
echo "[4/6] Installing veRL 0.8.0, TransferQueue, numpy pin ..."
pip install --quiet "verl==0.8.0"
pip install --quiet "TransferQueue==0.1.7"
# Pin numpy AFTER everything else (TransferQueue pulls in 1.x; we need 2.x)
pip install --quiet "numpy==2.3.5" --no-deps

# ── Step 5: Apply patches ─────────────────────────────────────────────────────
echo "[5/6] Applying patches ..."

# Patch 1 (REQUIRED): transformers flash_attention s_aux=None crash on Qwen3
python3 "${SCRIPT_DIR}/patches/patch_flash_attention.py"

# Patch 2 (REQUIRED): veRL main_ppo_sync gen_imbalance measurement
python3 "${SCRIPT_DIR}/patches/patch_main_ppo_sync.py"

# Patch 3 (OPTIONAL): HF rollout prefill/decode timer — only for standalone test
# Copies the pre-patched file; safe to skip if you don't run test_hf_rollout.py
HF_ROLLOUT_DST="${VENV}/lib/python3.12/site-packages/verl/workers/rollout/hf_rollout.py"
if [ -f "${SCRIPT_DIR}/patches/hf_rollout.patched.py" ] && [ -f "${HF_ROLLOUT_DST}" ]; then
    cp "${SCRIPT_DIR}/patches/hf_rollout.patched.py" "${HF_ROLLOUT_DST}"
    echo "[patch] hf_rollout.py: replaced with patched version"
fi

# ── Step 6: Prepare GSM8K data ────────────────────────────────────────────────
echo "[6/6] Preparing GSM8K data ..."
DATA_DIR=${PROJ}/data/gsm8k_verl
if [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/test.parquet" ]; then
    echo "      Data already exists at ${DATA_DIR}, skipping"
else
    python3 "${SCRIPT_DIR}/prepare_data.py" --output "${DATA_DIR}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "NEXT STEPS:"
echo "  1. Download Qwen3-8B model:"
echo "     huggingface-cli download Qwen/Qwen3-8B --local-dir ${PROJ}/models/Qwen3-8B"
echo "     (or: git clone https://huggingface.co/Qwen/Qwen3-8B ${PROJ}/models/Qwen3-8B)"
echo ""
echo "  2. Update paths in run scripts if PROJ/SCRIPT_DIR differ from defaults:"
echo "     PROJ=${PROJ}"
echo "     SCRIPT_DIR=${SCRIPT_DIR}"
echo ""
echo "  3. Run baseline:  bash ${SCRIPT_DIR}/run_qwen3_8b_a100.sh"
echo "     Run async:     bash ${SCRIPT_DIR}/run_async.sh"
echo "     Run ablations: bash ${SCRIPT_DIR}/ablation/ablation_rollout_n.sh"
echo ""
echo "  Log: ${LOG}"
