#!/usr/bin/env python3
"""
prepare_data.py — Download GSM8K from HuggingFace and convert to veRL parquet format.

Usage:
  python3 prepare_data.py --output /data/yanziyi/data/gsm8k_verl

Output files:
  <output>/train.parquet   (7473 rows)
  <output>/test.parquet    (1319 rows)

Format (matches veRL's expected schema):
  data_source : str  — "openai/gsm8k"
  prompt      : list[dict]  — [{"role": "system", ...}, {"role": "user", ...}]
  ability     : str  — "math"
  reward_model: dict — {"style": "rule", "ground_truth": "<answer>"}
  extra_info  : dict — {"solution": "<full solution text>"}
"""
import argparse, re, numpy as np
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, "
    "then give the final numerical answer on the last line in the format:\n#### <number>"
)

def extract_answer(solution: str) -> str:
    """Extract the number after '####' in GSM8K solution text."""
    m = re.search(r"####\s*([\d,\.\-]+)", solution)
    if m:
        return m.group(1).replace(",", "").strip()
    return solution.strip().split("\n")[-1].strip()

def convert(split: str, output_dir: Path):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)

    rows = []
    for ex in ds:
        answer = extract_answer(ex["answer"])
        rows.append({
            "data_source": "openai/gsm8k",
            "prompt": np.array([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["question"]},
            ], dtype=object),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info":   {"solution": ex["answer"]},
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out = output_dir / f"{split}.parquet"
    df.to_parquet(out, index=False)
    print(f"[data] {split}: {len(df)} rows -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    convert("train", out)
    convert("test",  out)
    print("[data] Done.")
