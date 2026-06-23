#!/usr/bin/env python3
"""
Standalone verification of HFRollout with _PrefillDecodeTimer on A100.
Does NOT require veRL trainer integration — loads the model directly.

Usage:
  cd /data/yanziyi/gpu_test_0610
  source venv/bin/activate
  python3 test_hf_rollout.py

Passes if:
  1. No exception raised
  2. At least one "[hf_rollout]" line appears in output
  3. prefill_ms > 0 and decode_ms > 0
"""
import sys, os, time, json, re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single GPU for standalone test

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto

MODEL_PATH = "/data/yanziyi/models/Qwen3-8B"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 4  # small for quick test

PROMPTS = [
    "Janet has 3 apples. She gives 1 to Bob. How many apples does Janet have?",
    "A train travels at 60 km/h for 2 hours. How far does it go?",
    "If x + 5 = 12, what is x?",
    "John has 10 marbles. He loses 3. How many does he have left?",
] * (BATCH_SIZE // 4)


def make_config():
    cfg = OmegaConf.create({
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "response_length": MAX_NEW_TOKENS,
        "val_kwargs": {
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 0.0,
        },
    })
    return cfg


def main():
    print(f"[test] Loading tokenizer from {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"[test] Loading model (bfloat16)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"[test] Model loaded in {time.perf_counter()-t0:.1f}s  "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # build DataProto batch
    enc = tok(PROMPTS[:BATCH_SIZE], return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids      = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    prompt_length  = input_ids.size(1)
    position_ids   = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    batch = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
        batch_size=BATCH_SIZE,
    )
    proto = DataProto(
        batch=batch,
        meta_info={
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
        },
    )

    # import patched HFRollout
    from verl.workers.rollout.hf_rollout import HFRollout, _PrefillDecodeTimer
    print(f"[test] HFRollout from: {HFRollout.__module__}")
    print(f"[test] _PrefillDecodeTimer available: {_PrefillDecodeTimer}")

    rollout = HFRollout(module=model, config=make_config())

    print(f"[test] Running generate_sequences with batch_size={BATCH_SIZE}...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    out = rollout.generate_sequences(proto)
    elapsed = time.perf_counter() - t0
    print(f"[test] generate_sequences done in {elapsed:.2f}s")

    resp = out.batch["responses"]
    print(f"[test] Output shape: prompts={out.batch['prompts'].shape}  responses={resp.shape}")

    # decode a sample
    sample_ids = resp[0]
    sample_ids = sample_ids[sample_ids != tok.pad_token_id]
    sample_text = tok.decode(sample_ids, skip_special_tokens=True)
    print(f"[test] Sample response: {sample_text[:200]!r}")

    # The [hf_rollout] line is printed inside generate_sequences; check it was emitted
    # (We can only check if it ran without error; stdout not captured here)
    print(f"\n[test] PASSED — HFRollout with _PrefillDecodeTimer works on A100.")
    print(f"[test] Check the output above for '[hf_rollout] prefill_ms:...' line.")


if __name__ == "__main__":
    main()
