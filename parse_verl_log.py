#!/usr/bin/env python3
"""
parse veRL 0.7.1 log -> per-step structured CSV.
Usage: python3 parse_verl_log.py --log train.log --output out.csv [--summary]
"""
import argparse, csv, json, re, sys
from pathlib import Path

_RE_STEP = re.compile(r"\bstep:(\d+)\b")
_RE_KV   = re.compile(
    r"([\w/@-]+):"
    r"(?:np\.(?:float64|int32|int64)\((-?[\d.e+\-]+)\)|(-?[\d.e+\-]+))"
)
_RE_VLLM_PROMPT   = re.compile(r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s")
_RE_VLLM_GEN      = re.compile(r"Avg generation throughput:\s*([\d.]+)\s*tokens/s")
_RE_VLLM_RUNNING  = re.compile(r"Running:\s*(\d+)\s*reqs")
_RE_VLLM_WAITING  = re.compile(r"Waiting:\s*(\d+)\s*reqs")
_RE_VLLM_KVCACHE  = re.compile(r"GPU KV cache usage:\s*([\d.]+)%")
_RE_HF_ROLLOUT    = re.compile(
    r"\[hf_rollout\]\s+prefill_ms:([\d.]+)\s+decode_ms:([\d.]+)\s+"
    r"decode_steps:(\d+)\s+prompt_toks:(\d+)\s+resp_toks:(\d+)"
)

_RE_CFG = {
    "cfg/rollout_n":           re.compile(r"actor_rollout_ref\.rollout\.n=(\d+)"),
    "cfg/train_batch_size":    re.compile(r"data\.train_batch_size=(\d+)"),
    "cfg/max_response_length": re.compile(r"data\.max_response_length=(\d+)"),
    "cfg/max_prompt_length":   re.compile(r"data\.max_prompt_length=(\d+)"),
    "cfg/n_gpus":              re.compile(r"trainer\.n_gpus_per_node=(\d+)"),
    "cfg/ppo_mini_batch_size": re.compile(r"actor_rollout_ref\.actor\.ppo_mini_batch_size=(\d+)"),
}

OUTPUT_FIELDS = [
    "step",
    "cfg/rollout_n", "cfg/train_batch_size", "cfg/max_response_length",
    "cfg/max_prompt_length", "cfg/n_gpus",
    "timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/adv",
    "timing_s/update_actor", "timing_s/update_weights", "timing_s/step", "timing_s/testing",
    "timing_per_token_ms/gen", "timing_per_token_ms/ref", "timing_per_token_ms/update_actor",
    "timing_s/agent_loop/generate_sequences/min",
    "timing_s/agent_loop/generate_sequences/mean",
    "timing_s/agent_loop/generate_sequences/max",
    "timing_s/agent_loop/slowest/generate_sequences",
    "timing_s/agent_loop/slowest/prompt_length",
    "timing_s/agent_loop/slowest/response_length",
    "global_seqlen/min", "global_seqlen/max", "global_seqlen/mean",
    "global_seqlen/minmax_diff", "global_seqlen/balanced_min", "global_seqlen/balanced_max",
    "response_length/mean", "response_length/min", "response_length/max",
    "response_length/clip_ratio", "prompt_length/mean", "prompt_length/max",
    "critic/score/mean", "critic/score/max", "critic/score/min",
    "critic/rewards/mean", "val-core/openai/gsm8k/acc/mean@1",
    "perf/mfu/actor", "perf/mfu/actor_infer",
    "perf/throughput", "perf/total_num_tokens", "perf/time_per_step",
    "perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb", "perf/cpu_memory_used_gb",
    "actor/grad_norm", "actor/kl_loss", "actor/ppo_kl",
    "num_turns/mean", "num_turns/min", "num_turns/max",
    "vllm/prompt_throughput_toks", "vllm/gen_throughput_toks",
    "vllm/running_reqs/mean", "vllm/running_reqs/max", "vllm/running_reqs/min",
    "vllm/waiting_reqs/mean", "vllm/kv_cache_usage_pct/mean",
    "derived/actual_concurrency_pct",
    "derived/gen_frac", "derived/update_frac", "derived/update_w_frac",
    "derived/gen_imbalance", "derived/seqlen_imbalance", "derived/rollout_vs_train_ratio",
    "derived/theoretical_concurrency",
    "derived/actual_seqs_per_sec",
    "derived/prefill_frac",
    "hf/prefill_ms", "hf/decode_ms", "hf/decode_steps",
    "hf/prompt_toks", "hf/resp_toks",
    "hf/prefill_throughput_toks", "hf/decode_throughput_toks",
]


def _f(s):
    try: return float(s)
    except: return None


def parse_config(path):
    cfg = {}
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 200:
                break
            for key, pat in _RE_CFG.items():
                if key not in cfg:
                    m = pat.search(line)
                    if m:
                        cfg[key] = float(m.group(1))
    return cfg


def parse_line(line):
    rec = {}
    m = _RE_STEP.search(line)
    if not m:
        return rec
    rec["step"] = int(m.group(1))
    for m in _RE_KV.finditer(line):
        v = _f(m.group(2) if m.group(2) is not None else m.group(3))
        if v is not None:
            rec[m.group(1)] = v
    for pat, k in [(_RE_VLLM_PROMPT, "vllm/prompt_throughput_toks"),
                   (_RE_VLLM_GEN,    "vllm/gen_throughput_toks")]:
        mv = pat.search(line)
        if mv:
            rec[k] = float(mv.group(1))
    return rec


def compute_derived(r, cfg):
    r.update(cfg)
    tg = r.get("timing_s/gen", 0)
    ts = r.get("timing_s/step", 0)
    tu = r.get("timing_s/update_actor", 0)
    tw = r.get("timing_s/update_weights", 0)
    if ts > 0:
        r["derived/gen_frac"]      = round(tg / ts, 4)
        r["derived/update_frac"]   = round(tu / ts, 4)
        r["derived/update_w_frac"] = round(tw / ts, 4)
        tt = ts - tg
        if tt > 0:
            r["derived/rollout_vs_train_ratio"] = round(tg / tt, 3)
    gmn = r.get("timing_s/agent_loop/generate_sequences/min")
    gmx = r.get("timing_s/agent_loop/generate_sequences/max")
    if gmn and gmx and gmn > 0:
        r["derived/gen_imbalance"] = round(gmx / gmn, 3)
    # veRL 0.8.0: rollout/gen_imbalance logged directly by patched main_ppo_sync.py
    gi_direct = r.get("rollout/gen_imbalance")
    if gi_direct and "derived/gen_imbalance" not in r:
        r["derived/gen_imbalance"] = round(gi_direct, 3)
    smn = r.get("global_seqlen/min")
    smx = r.get("global_seqlen/max")
    if smn and smx and smn > 0:
        r["derived/seqlen_imbalance"] = round(smx / smn, 3)
    rn  = cfg.get("cfg/rollout_n")
    bsz = cfg.get("cfg/train_batch_size")
    if rn and bsz:
        r["derived/theoretical_concurrency"] = int(rn * bsz)
    total_toks = r.get("perf/total_num_tokens")
    resp_len   = r.get("response_length/mean")
    if total_toks and resp_len and resp_len > 0 and tg > 0:
        total_seqs = total_toks / resp_len
        r["derived/actual_seqs_per_sec"] = round(total_seqs / tg, 2)
    pf = r.get("vllm/prompt_throughput_toks")
    dc = r.get("vllm/gen_throughput_toks")
    if pf and dc and (pf + dc) > 0:
        r["derived/prefill_frac"] = round(pf / (pf + dc), 4)
    # HF rollout: compute throughput and prefill_frac from timer data
    hf_prefill_ms = r.get("hf/prefill_ms")
    hf_decode_ms  = r.get("hf/decode_ms")
    hf_prompt_toks = r.get("hf/prompt_toks")
    hf_resp_toks   = r.get("hf/resp_toks")
    if hf_prefill_ms and hf_prefill_ms > 0 and hf_prompt_toks:
        r["hf/prefill_throughput_toks"] = round(hf_prompt_toks / (hf_prefill_ms / 1000.0), 1)
    if hf_decode_ms and hf_decode_ms > 0 and hf_resp_toks:
        r["hf/decode_throughput_toks"] = round(hf_resp_toks / (hf_decode_ms / 1000.0), 1)
    if hf_prefill_ms and hf_decode_ms and (hf_prefill_ms + hf_decode_ms) > 0:
        r["derived/prefill_frac"] = round(hf_prefill_ms / (hf_prefill_ms + hf_decode_ms), 4)
    # actual concurrency utilization: mean running reqs vs theoretical per GPU
    run_mean = r.get("vllm/running_reqs/mean")
    n_gpus = cfg.get("cfg/n_gpus")
    theo = r.get("derived/theoretical_concurrency")
    if run_mean is not None and n_gpus and theo:
        theo_per_gpu = theo / n_gpus
        if theo_per_gpu > 0:
            r["derived/actual_concurrency_pct"] = round(run_mean / theo_per_gpu * 100, 1)
    return r


def parse_log(path):
    cfg  = parse_config(path)
    recs = {}
    last = None
    # accumulate vLLM stat samples per step
    vllm_samples = {}  # step -> {"running": [], "waiting": [], "kv": [], "prompt": [], "gen": []}

    def _samples(s):
        if s not in vllm_samples:
            vllm_samples[s] = {"running": [], "waiting": [], "kv": [], "prompt": [], "gen": []}
        return vllm_samples[s]

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if _RE_STEP.search(line):
                rec = parse_line(line)
                if "step" in rec:
                    s = rec["step"]
                    if s not in recs:
                        recs[s] = rec
                    else:
                        recs[s].update(rec)
                    last = s
            elif last is not None and (
                "Avg prompt throughput" in line or "Running:" in line or "KV cache" in line
            ):
                sp = _samples(last)
                for pat, key in [(_RE_VLLM_PROMPT, "prompt"), (_RE_VLLM_GEN, "gen")]:
                    mv = pat.search(line)
                    if mv:
                        sp[key].append(float(mv.group(1)))
                for pat, key in [(_RE_VLLM_RUNNING, "running"), (_RE_VLLM_WAITING, "waiting")]:
                    mv = pat.search(line)
                    if mv:
                        sp[key].append(int(mv.group(1)))
                mv = _RE_VLLM_KVCACHE.search(line)
                if mv:
                    sp["kv"].append(float(mv.group(1)))
            elif "[hf_rollout]" in line:
                mh = _RE_HF_ROLLOUT.search(line)
                if mh and last is not None:
                    # one line per generate_sequences call; overwrite (last wins)
                    if last not in recs:
                        recs[last] = {"step": last}
                    recs[last]["hf/prefill_ms"]   = float(mh.group(1))
                    recs[last]["hf/decode_ms"]     = float(mh.group(2))
                    recs[last]["hf/decode_steps"]  = int(mh.group(3))
                    recs[last]["hf/prompt_toks"]   = int(mh.group(4))
                    recs[last]["hf/resp_toks"]     = int(mh.group(5))

    # Merge vLLM samples into step records
    for s, sp in vllm_samples.items():
        if s not in recs:
            continue
        if sp["prompt"]:
            recs[s]["vllm/prompt_throughput_toks"] = round(sum(sp["prompt"]) / len(sp["prompt"]), 1)
        if sp["gen"]:
            recs[s]["vllm/gen_throughput_toks"] = round(sum(sp["gen"]) / len(sp["gen"]), 1)
        if sp["running"]:
            recs[s]["vllm/running_reqs/mean"] = round(sum(sp["running"]) / len(sp["running"]), 1)
            recs[s]["vllm/running_reqs/max"]  = max(sp["running"])
            recs[s]["vllm/running_reqs/min"]  = min(sp["running"])
        if sp["waiting"]:
            recs[s]["vllm/waiting_reqs/mean"] = round(sum(sp["waiting"]) / len(sp["waiting"]), 1)
        if sp["kv"]:
            recs[s]["vllm/kv_cache_usage_pct/mean"] = round(sum(sp["kv"]) / len(sp["kv"]), 1)

    return [compute_derived(recs[s], cfg) for s in sorted(recs)]


def print_summary(records):
    from statistics import mean, stdev
    train_recs = [r for r in records if r.get("step", 0) > 0 and "timing_s/gen" in r]
    if not train_recs:
        print("[warn] no training steps found")
        return
    print("\n=== per-step timing summary ===")
    keys = [
        "timing_s/gen", "timing_s/old_log_prob", "timing_s/ref",
        "timing_s/update_actor", "timing_s/update_weights", "timing_s/step",
        "derived/gen_frac", "derived/update_frac",
        "response_length/mean", "response_length/clip_ratio",
        "perf/throughput", "perf/mfu/actor",
        "global_seqlen/minmax_diff", "derived/seqlen_imbalance", "derived/gen_imbalance",
        "derived/theoretical_concurrency", "derived/actual_seqs_per_sec",
        "derived/prefill_frac",
        "vllm/prompt_throughput_toks", "vllm/gen_throughput_toks",
        "vllm/running_reqs/mean", "vllm/running_reqs/max",
        "vllm/kv_cache_usage_pct/mean", "derived/actual_concurrency_pct",
        "num_turns/mean",
        "hf/prefill_ms", "hf/decode_ms", "hf/decode_steps",
        "hf/prefill_throughput_toks", "hf/decode_throughput_toks",
    ]
    print(f"{'metric':<50} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print("-" * 82)
    for k in keys:
        vs = [r[k] for r in train_recs if k in r]
        if not vs:
            continue
        print(f"{k:<50} {mean(vs):>8.3f} {(stdev(vs) if len(vs)>1 else 0):>8.3f} "
              f"{min(vs):>8.3f} {max(vs):>8.3f}")
    vr = [(r["step"], r["val-core/openai/gsm8k/acc/mean@1"])
          for r in records if "val-core/openai/gsm8k/acc/mean@1" in r]
    if vr:
        print("\n=== Val Accuracy ===")
        for s, a in vr:
            print(f"  step {s:4d}  acc={a:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",     required=True)
    ap.add_argument("--output",  default="step_metrics.csv")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    records = parse_log(args.log)
    if not records:
        print("[warn] no steps found", file=sys.stderr)
        sys.exit(1)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    jo = str(Path(args.output).with_suffix(".json"))
    with open(jo, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[parse] {len(records)} steps -> {args.output}  {jo}")
    if args.summary:
        print_summary(records)


if __name__ == "__main__":
    main()
