#!/usr/bin/env python3
"""
parse_rollout_profile.py
Parse vLLM (rollout) torch profiler traces -> Prefill/Decode/Attention/Linear/Comm/Idle breakdown.

Usage:
  python3 parse_rollout_profile.py --trace-dir /data/yanziyi/outputs/rollout_profile_XXX \
                                   [--log /path/to/log] [--out breakdown.csv]

The script looks for trace files at:
  <trace-dir>/agent_loop_rollout_replica_*/  -> vLLM GPU worker traces (.json.gz or .pt.trace.json)

Each CUDA kernel is classified into:
  - attention_prefill   : flash_attn varlen (prefill)
  - attention_decode    : flash_attn with kvcache / paged attention
  - attention_other     : any other attention kernel
  - linear              : GEMM / cublasLt / matmul
  - communication       : NCCL (AllReduce, AllGather, etc.)
  - memory              : Memcpy / Memset
  - sampling            : sampling / topk / softmax
  - other_gpu           : remaining GPU kernels

Additionally parses --log for vLLM "Avg prompt/generation throughput" lines
to get coarse Prefill/Decode throughput ratio.
"""
import argparse, csv, gzip, json, os, re, sys
from pathlib import Path
from statistics import mean

# Kernel classification regexes (against lowercase kernel name)
_ATTN_PREFILL = re.compile(
    r"flash_fwd_varlen|flash_attn_varlen|fa_fwd_varlen|"
    r"flash_fwd(?!.*splitkv)(?!.*kvcache)"
)
_ATTN_DECODE = re.compile(
    r"flash_fwd_splitkv|flash_attn_with_kvcache|paged_attention_v\d|"
    r"flash_decoding|fmha.*decode|flashinfer.*decode|flashinfer.*paged"
)
_ATTN_GENERIC = re.compile(r"flash|attention|attn")
_LINEAR = re.compile(
    r"gemm|sgemm|hgemm|cutlass|cublas|ampere_.*gemm|volta_.*gemm|"
    r"sm80.*gemm|sm86.*gemm|sm90.*gemm|aten::mm|aten::addmm|aten::linear|"
    r"triton.*matmul|Cijk_Ailk_Bljk"
)
_COMM = re.compile(r"nccl|ncclall|allreduce|allgather|reducescatter")
_MEMORY = re.compile(r"memcpy|memset|dmamemcpy")
_SAMPLING = re.compile(r"sampling|topk|top_k|softmax|argmax|multinomial|gumbel")


def classify_kernel(name):
    n = name.lower()
    # Use only the function name before template args "<...>" to avoid
    # false matches on parameter type names (e.g. flash_fwd_params).
    func = n.split('<')[0].rstrip()
    if _ATTN_PREFILL.search(func):
        return "attention_prefill"
    if _ATTN_DECODE.search(func):
        return "attention_decode"
    if _ATTN_GENERIC.search(func):
        return "attention_other"
    if _LINEAR.search(func):
        return "linear"
    if _COMM.search(func):
        return "communication"
    if _MEMORY.search(func):
        return "memory"
    if _SAMPLING.search(func):
        return "sampling"
    return "other_gpu"


def load_trace(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, "r") as f:
            raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("traceEvents", [])
    if isinstance(raw, list):
        return raw
    return []


def analyze_trace(events):
    categories = {
        "attention_prefill": 0.0, "attention_decode": 0.0, "attention_other": 0.0,
        "linear": 0.0, "communication": 0.0, "memory": 0.0,
        "sampling": 0.0, "other_gpu": 0.0,
    }
    total_gpu_us = 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat", "") not in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel"):
            continue
        dur = ev.get("dur", 0) or 0
        if dur <= 0:
            continue
        kind = classify_kernel(ev.get("name", ""))
        categories[kind] += dur
        total_gpu_us += dur
    total_ms = total_gpu_us / 1000.0
    result = {"total_gpu_ms": total_ms}
    for k, v in categories.items():
        result[k + "_ms"] = v / 1000.0
        result[k + "_pct"] = (v / total_gpu_us * 100) if total_gpu_us > 0 else 0.0
    return result


def parse_vllm_stats(log_path):
    re_prompt = re.compile(r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s")
    re_gen = re.compile(r"Avg generation throughput:\s*([\d.]+)\s*tokens/s")
    prompt_vals, gen_vals = [], []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = re_prompt.search(line)
            if m:
                prompt_vals.append(float(m.group(1)))
            m = re_gen.search(line)
            if m:
                gen_vals.append(float(m.group(1)))
    result = {}
    if prompt_vals:
        result["vllm_prompt_tput_mean"] = mean(prompt_vals)
        result["n_prompt_samples"] = len(prompt_vals)
    if gen_vals:
        result["vllm_gen_tput_mean"] = mean(gen_vals)
        result["n_gen_samples"] = len(gen_vals)
    if prompt_vals and gen_vals:
        pt, gt = mean(prompt_vals), mean(gen_vals)
        result["prefill_frac"] = pt / (pt + gt) if (pt + gt) > 0 else None
    return result


def find_trace_files(trace_dir):
    traces = []
    for subdir in sorted(trace_dir.glob("agent_loop_rollout_replica_*")):
        if subdir.is_dir():
            traces += sorted(subdir.rglob("*.json.gz"))
            traces += sorted(subdir.rglob("*.pt.trace.json"))
    if not traces:
        print(f"[warn] No agent_loop_rollout_replica_* dirs found in {trace_dir}. "
              "Rollout worker traces missing -- skipping rollout analysis.", file=sys.stderr)
    return traces


def aggregate_traces(trace_files):
    results = []
    for tf in trace_files:
        print(f"  Analyzing: {tf.name} ({tf.stat().st_size // 1024} KB)", file=sys.stderr)
        try:
            events = load_trace(tf)
            r = analyze_trace(events)
            if r["total_gpu_ms"] > 0:
                results.append(r)
        except Exception as e:
            print(f"  [warn] Failed: {tf}: {e}", file=sys.stderr)
    if not results:
        return {}
    keys = list(results[0].keys())
    agg = {k: mean([r[k] for r in results if k in r]) for k in keys}
    agg["n_traces"] = len(results)
    return agg


def print_breakdown(agg, vllm_stats):
    total = agg.get("total_gpu_ms", 0)
    n = agg.get("n_traces", "?")
    print(f"\n=== Rollout GPU Kernel Breakdown (avg over {n} worker traces) ===")
    print(f"{'Category':<30} {'Time(ms)':>10} {'GPU%':>8}")
    print("-" * 52)
    cats = ["attention_prefill", "attention_decode", "attention_other",
            "linear", "communication", "memory", "sampling", "other_gpu"]
    for c in cats:
        ms = agg.get(f"{c}_ms", 0)
        pct = agg.get(f"{c}_pct", 0)
        print(f"  {c:<28} {ms:>10.1f} {pct:>7.1f}%")
    print(f"  {'[GPU Total]':<28} {total:>10.1f} {'100.0':>7}%")

    attn_ms = sum(agg.get(f"{c}_ms", 0) for c in ["attention_prefill", "attention_decode", "attention_other"])
    lin_ms = agg.get("linear_ms", 0)
    comm_ms = agg.get("communication_ms", 0)
    other_ms = total - attn_ms - lin_ms - comm_ms
    print(f"\n=== Summary ===")
    print(f"  {'Attention (total)':<28} {attn_ms:>10.1f} {attn_ms/total*100 if total>0 else 0:>7.1f}%")
    if agg.get("attention_prefill_ms", 0) > 0 or agg.get("attention_decode_ms", 0) > 0:
        print(f"    {'Prefill FA':<26} {agg.get('attention_prefill_ms',0):>10.1f} "
              f"{agg.get('attention_prefill_pct',0):>7.1f}%")
        print(f"    {'Decode FA':<26} {agg.get('attention_decode_ms',0):>10.1f} "
              f"{agg.get('attention_decode_pct',0):>7.1f}%")
    print(f"  {'Linear (GEMM)':<28} {lin_ms:>10.1f} {lin_ms/total*100 if total>0 else 0:>7.1f}%")
    print(f"  {'Communication (NCCL)':<28} {comm_ms:>10.1f} {comm_ms/total*100 if total>0 else 0:>7.1f}%")
    print(f"  {'Other/Sampling/Mem':<28} {other_ms:>10.1f} {other_ms/total*100 if total>0 else 0:>7.1f}%")

    if vllm_stats:
        print(f"\n=== vLLM Throughput Stats ===")
        if "vllm_prompt_tput_mean" in vllm_stats:
            print(f"  Avg Prefill throughput : {vllm_stats['vllm_prompt_tput_mean']:.1f} tok/s")
        if "vllm_gen_tput_mean" in vllm_stats:
            print(f"  Avg Decode throughput  : {vllm_stats['vllm_gen_tput_mean']:.1f} tok/s")
        if vllm_stats.get("prefill_frac"):
            print(f"  Prefill frac (by tput) : {vllm_stats['prefill_frac']*100:.1f}%")


def save_csv(agg, vllm_stats, out_path):
    row = {**agg}
    row.update({k: v for k, v in vllm_stats.items() if not isinstance(v, list)})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n[parse_rollout_profile] Saved to {out_path}")



# ─── Actor trace 解析（表7: update_actor 算子分解）─────────────────────────────

_ACTOR_ATTN = re.compile(
    r"flash_fwd|flash_bwd|flash_attention|sdpa_flash|"
    r"mem_efficient_attention|xformers.*attn", re.I)
_ACTOR_GEMM = re.compile(
    r"gemm|sgemm|hgemm|s16816|ampere_fp16|volta_fp16|sm80|sm90|"
    r"cublas|cutlass|wgrad|Cijk_Ailk_Bljk|aten::addmm|aten::mm", re.I)
_ACTOR_NCCL = re.compile(r"nccl|allreduce|all_reduce|allgather|all_gather|reducescatter|reduce_scatter", re.I)
_ACTOR_ADAM = re.compile(r"adam|multi_tensor_adam|fused_adam|adamw|multi_tensor_apply", re.I)


def classify_actor_kernel(name):
    n = name.lower()
    if _ACTOR_ATTN.search(n):  return "attention"
    if _ACTOR_GEMM.search(n):  return "linear"
    if _ACTOR_NCCL.search(n):  return "communication"
    if _ACTOR_ADAM.search(n):  return "optimizer"
    return "other"


def find_actor_trace_files(trace_dir: "Path") -> list:
    """查找 actor update 侧 trace 文件。
    支持以下布局：
      actor_update/prof_rank-*.json.gz  (discrete profiler, global_profiler.tool=torch + discrete=True)
      actor_train-step*/rank*.json.gz   (旧版 actor.profiler 独立输出)
      e2e/prof_rank-*.json.gz           (global_profiler e2e 非 discrete 输出)
    """
    traces = []
    # 1. actor_update/ (discrete=True 时 veRL 保存到此目录)
    for subdir in sorted(trace_dir.glob("actor_update*")):
        if subdir.is_dir():
            traces += sorted(subdir.rglob("*.json.gz"))
            traces += sorted(subdir.rglob("*.pt.trace.json"))
    # 2. actor_train* subdirs (旧布局)
    if not traces:
        for subdir in sorted(trace_dir.glob("actor_train*")):
            if subdir.is_dir():
                traces += sorted(subdir.rglob("*.json.gz"))
                traces += sorted(subdir.rglob("*.pt.trace.json"))
    # 3. e2e/ subdir (non-discrete global_profiler)
    if not traces:
        e2e_dir = trace_dir / "e2e"
        if e2e_dir.is_dir():
            traces += sorted(e2e_dir.glob("prof_rank-*.json.gz"))
            traces += sorted(e2e_dir.glob("prof_rank-*.pt.trace.json"))
    # 4. direct rank*.json fallback
    if not traces:
        traces += sorted(trace_dir.rglob("rank*.json"))
    return [t for t in traces if t.stat().st_size > 1000]


def analyze_actor_trace(events: list) -> dict:
    """分析 actor 侧 trace，返回各算子类型耗时（ms）。"""
    cats = {"attention": 0.0, "linear": 0.0, "communication": 0.0,
            "optimizer": 0.0, "other": 0.0}
    total_us = 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat", "") not in ("kernel", "gpu_memcpy", "Kernel", "cuda_kernel"):
            continue
        dur = ev.get("dur", 0) or 0
        if dur <= 0:
            continue
        kind = classify_actor_kernel(ev.get("name", ""))
        cats[kind] += dur
        total_us += dur
    total_ms = total_us / 1000.0
    result = {"total_gpu_ms": total_ms}
    for k, v in cats.items():
        result[k + "_ms"] = v / 1000.0
        result[k + "_pct"] = (v / total_us * 100) if total_us > 0 else 0.0
    return result


def aggregate_actor_traces(trace_files: list) -> dict:
    results = []
    for tf in trace_files:
        print(f"  [actor] {tf.name} ({tf.stat().st_size // 1024} KB)", file=sys.stderr)
        try:
            events = load_trace(tf)
            r = analyze_actor_trace(events)
            if r["total_gpu_ms"] > 0:
                results.append(r)
        except Exception as e:
            print(f"  [warn] Failed: {tf}: {e}", file=sys.stderr)
    if not results:
        return {}
    keys = list(results[0].keys())
    agg = {k: mean([r[k] for r in results if k in r]) for k in keys}
    agg["n_traces"] = len(results)
    return agg


# ─── Markdown 表格填写 ─────────────────────────────────────────────────────────

def _replace_row(text: str, section_header: str, col1_key: str, new_row: str) -> str:
    """在指定 section 内替换首列包含 col1_key 的行。"""
    start = text.find(section_header)
    if start == -1:
        return text
    import re as _re
    next_h = _re.search(r"^###", text[start + len(section_header):], _re.MULTILINE)
    end = start + len(section_header) + next_h.start() if next_h else len(text)
    section = text[start:end]
    # 匹配含 col1_key 的行（忽略 ├└│ 等前缀和 ** 粗体）
    escaped = _re.escape(col1_key.strip("*├└│ "))
    pat = _re.compile(r"^\|[^|\n]*" + escaped + r"[^|\n]*\|.*$", _re.MULTILINE)
    new_section = pat.sub(new_row, section, count=1)
    return text[:start] + new_section + text[end:]


def fill_table7(text: str, actor_agg: dict) -> str:
    """填写 表13：训练阶段算子分解（update_actor，8B）。"""
    total = actor_agg.get("total_gpu_ms", 0)

    def r(ms): return f"{ms:.0f}"
    def p(ms): return f"{ms / total * 100:.1f}" if total > 0 else "-"

    rows = [
        ("Attention（前向+反向）",     actor_agg.get("attention_ms", 0)),
        ("Linear / GEMM（前向+反向）", actor_agg.get("linear_ms", 0)),
        ("AllReduce / NCCL 通信",       actor_agg.get("communication_ms", 0)),
        ("Optimizer（Adam step）",      actor_agg.get("optimizer_ms", 0)),
        ("其他（归一化/激活/调度）",    actor_agg.get("other_ms", 0)),
    ]
    for col1, ms in rows:
        new_row = f"| {col1} | {r(ms)} | {p(ms)} |"
        text = _replace_row(text, "### 表 13：", col1, new_row)

    total_row = f"| **update_actor 合计** | {r(total)} | **100%** |"
    text = _replace_row(text, "### 表 13：", "update_actor 合计", total_row)
    return text



def fill_table14(text: str, actor_agg: dict) -> str:
    """填写 表14：训练阶段算子分解（update_actor，32B LoRA step 3-5 均值）。"""
    total = actor_agg.get("total_gpu_ms", 0)
    def r(ms): return f"{ms:.0f}"
    def p(ms): return f"{ms / total * 100:.1f}" if total > 0 else "-"
    rows = [
        ("Attention（前向+反向）",     actor_agg.get("attention_ms", 0)),
        ("Linear / GEMM（前向+反向）", actor_agg.get("linear_ms", 0)),
        ("AllReduce / NCCL 通信",       actor_agg.get("communication_ms", 0)),
        ("Optimizer（Adam step）",      actor_agg.get("optimizer_ms", 0)),
        ("其他（归一化/激活/调度）",    actor_agg.get("other_ms", 0)),
    ]
    for col1, ms in rows:
        new_row = f"| {col1} | {r(ms)} | {p(ms)} |"
        text = _replace_row(text, "### 表 14：", col1, new_row)
    total_row = f"| **update_actor 合计** | {r(total)} | **100%** |"
    text = _replace_row(text, "### 表 14：", "update_actor 合计", total_row)
    return text


def fill_table16(text: str, rollout_agg: dict, vllm_stats: dict) -> str:
    """填写 表16：Rollout 阶段算子分解（gen，32B LoRA step 3-5 均值）。"""
    total = rollout_agg.get("total_gpu_ms", 0)
    if total <= 0:
        return text
    pf_attn  = rollout_agg.get("attention_prefill_ms", 0)
    dc_attn  = rollout_agg.get("attention_decode_ms", 0)
    attn_oth = rollout_agg.get("attention_other_ms", 0)
    linear   = rollout_agg.get("linear_ms", 0)
    sampling = rollout_agg.get("sampling_ms", 0)
    other    = rollout_agg.get("other_gpu_ms", 0)
    comm     = rollout_agg.get("communication_ms", 0)
    mem      = rollout_agg.get("memory_ms", 0)
    pf_frac  = vllm_stats.get("prefill_frac")
    if pf_frac is None:
        attn_total = pf_attn + dc_attn
        pf_frac = pf_attn / attn_total if attn_total > 0 else 0.5
    dc_frac = 1.0 - pf_frac
    if pf_attn == 0 and dc_attn > 0:
        attn_ms = dc_attn + attn_oth
        pf_attn = attn_ms * pf_frac
        dc_attn = attn_ms * dc_frac
        attn_oth = 0.0
    pf_gemm  = linear * pf_frac
    dc_gemm  = linear * dc_frac
    pf_other = (attn_oth + other) * pf_frac
    dc_other = sampling + (attn_oth + other) * dc_frac
    pf_total  = pf_attn + pf_gemm + pf_other
    dc_total  = dc_attn + dc_gemm + dc_other
    # Use actual total_gpu_ms as denominator so all % are out of true GPU total
    def r(ms): return f"{ms:.0f}"
    def p(ms): return f"{ms / total * 100:.1f}" if total > 0 else "-"
    updates = [
        ("Prefill 阶段",                                  pf_total),
        ("Prefill Attention（Flash-Attn varlen）",        pf_attn),
        ("Prefill Linear / GEMM",                         pf_gemm),
        ("Prefill 其他（RoPE/激活/归一化）",              pf_other),
        ("Decode 阶段",                                   dc_total),
        ("Decode Attention（Flash-Attn KV-cache）",       dc_attn),
        ("Decode Linear / GEMM",                          dc_gemm),
        ("Decode 其他（采样/softmax）",                   dc_other),
        ("NCCL 通信（TP=4 AllReduce）",                   comm),
        ("Memcpy（权重/KV 操作）",                        mem),
        ("GPU 合计",                                      total),
    ]
    for col1, ms in updates:
        pct_str = "**100%**" if "合计" in col1 else p(ms)
        if col1.endswith("阶段") or "合计" in col1:
            row = f"| **{col1}** | {r(ms)} | {pct_str} |"
        else:
            row = f"| {col1} | {r(ms)} | {pct_str} |"
        text = _replace_row(text, "### 表 16：", col1, row)
    return text


def fill_table8(text: str, rollout_agg: dict, vllm_stats: dict) -> str:
    """填写 表15：Rollout 阶段算子分解（gen，8B）。
    用 vLLM 吞吐量比例将 linear/other 拆分到 Prefill 和 Decode。
    """
    total = rollout_agg.get("total_gpu_ms", 0)
    if total <= 0:
        return text

    pf_attn = rollout_agg.get("attention_prefill_ms", 0)
    dc_attn = rollout_agg.get("attention_decode_ms", 0)
    attn_oth = rollout_agg.get("attention_other_ms", 0)
    linear  = rollout_agg.get("linear_ms", 0)
    sampling = rollout_agg.get("sampling_ms", 0)
    other   = rollout_agg.get("other_gpu_ms", 0)
    mem     = rollout_agg.get("memory_ms", 0)
    comm    = rollout_agg.get("communication_ms", 0)

    # 用 vLLM prompt/gen 吞吐量比例估算 Prefill vs Decode 比例
    pf_frac = vllm_stats.get("prefill_frac")
    if pf_frac is None:
        # fallback: prefill attn / (prefill + decode attn)
        attn_total = pf_attn + dc_attn
        pf_frac = pf_attn / attn_total if attn_total > 0 else 0.5
    dc_frac = 1.0 - pf_frac

    # 当 vLLM 用 flash_fwd_splitkv 同时处理 prefill+decode，
    # kernel 名无法区分时（pf_attn=0, dc_attn>0），用 throughput ratio 拆分 attention
    if pf_attn == 0 and dc_attn > 0 and pf_frac is not None:
        attn_ms = dc_attn + attn_oth
        pf_attn = attn_ms * pf_frac
        dc_attn = attn_ms * dc_frac
        attn_oth = 0.0

    # Prefill/Decode GEMM split by throughput ratio heuristic
    pf_gemm = linear * pf_frac
    dc_gemm = linear * dc_frac
    # Prefill other: attn_other * pf_frac + other * pf_frac
    pf_other = (attn_oth + other) * pf_frac
    # Decode other: sampling + rest
    dc_other = sampling + (attn_oth + other) * dc_frac

    pf_total = pf_attn + pf_gemm + pf_other
    dc_total = dc_attn + dc_gemm + dc_other

    def r(ms): return f"{ms:.0f}"
    def p(ms): return f"{ms / total * 100:.1f}" if total > 0 else "-"

    updates = [
        ("Prefill 阶段",                      pf_total),
        ("Prefill Attention（Flash-Attn varlen）",   pf_attn),
        ("Prefill Linear / GEMM",                   pf_gemm),
        ("Prefill 其他（RoPE/激活/归一化）",         pf_other),
        ("Decode 阶段",                        dc_total),
        ("Decode Attention（Flash-Attn KV-cache）", dc_attn),
        ("Decode Linear / GEMM",                   dc_gemm),
        ("Decode 其他（采样/softmax）",             dc_other),
        ("Memcpy（显存操作）",                  mem),
        ("GPU 合计",                           total),
    ]
    for col1, ms in updates:
        pct_str = "**100%**" if "合计" in col1 else p(ms)
        if col1.endswith("阶段") or "合计" in col1:
            row = f"| **{col1}** | {r(ms)} | {pct_str} |"
        else:
            row = f"| {col1} | {r(ms)} | {pct_str} |"
        text = _replace_row(text, "### 表 15：", col1, row)
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True,
                    help="Profiler output dir (contains agent_loop_rollout_replica_*/ and actor_train*/)")
    ap.add_argument("--log",     default=None, help="Training log for vLLM stats")
    ap.add_argument("--out",     default="rollout_breakdown.csv")
    ap.add_argument("--mode",    default="8b", choices=["8b", "32b"],
                    help="8b=fill tables 7&8, 32b=fill tables 14&16")
    ap.add_argument("--tables",  default=None,
                    help="results_tables.md path; if set, fill Tables 7 and 8")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"[error] trace-dir not found: {trace_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Rollout 侧 traces ─────────────────────────────────────────────────────
    rollout_files = find_trace_files(trace_dir)
    if not rollout_files:
        print(f"[warn] No rollout trace files found in {trace_dir}", file=sys.stderr)
        print("  Expected: agent_loop_rollout_replica_*/ subdirs with .json.gz files",
              file=sys.stderr)
    else:
        print(f"Found {len(rollout_files)} rollout trace files", file=sys.stderr)

    rollout_agg = aggregate_traces(rollout_files) if rollout_files else {}
    vllm_stats  = parse_vllm_stats(args.log) if args.log else {}

    if not vllm_stats and args.log:
        print("[warn] No vLLM throughput stats in log "
              "(disable_log_stats may still be True)", file=sys.stderr)

    if rollout_agg:
        print_breakdown(rollout_agg, vllm_stats)
        save_csv(rollout_agg, vllm_stats, args.out)
    elif vllm_stats:
        print("\nvLLM stats only (no rollout trace files found):")
        print(vllm_stats)

    # ── Actor 侧 traces（表7: update_actor 算子分解）──────────────────────────
    actor_files = find_actor_trace_files(trace_dir)
    actor_agg = {}
    if actor_files:
        print(f"\nFound {len(actor_files)} actor trace files", file=sys.stderr)
        actor_agg = aggregate_actor_traces(actor_files)
        if actor_agg:
            total = actor_agg.get("total_gpu_ms", 0)
            print(f"\n=== Actor (update_actor) GPU Kernel Breakdown ===")
            print(f"  Total GPU time: {total:.0f} ms")
            for k in ["attention", "linear", "communication", "optimizer", "other"]:
                ms  = actor_agg.get(f"{k}_ms", 0)
                pct = actor_agg.get(f"{k}_pct", 0)
                print(f"  {k:<20} {ms:>8.1f} ms  {pct:>5.1f}%")
    else:
        print("\n[info] No actor trace files found (actor_train*/ subdirs).", file=sys.stderr)

    # ── 填写 results_tables.md ─────────────────────────────────────────────────
    if args.tables:
        import os
        if not os.path.exists(args.tables):
            print(f"[error] --tables file not found: {args.tables}", file=sys.stderr)
        else:
            with open(args.tables, "r", encoding="utf-8") as f:
                md = f.read()

            if args.mode == "32b":
                if actor_agg:
                    md = fill_table14(md, actor_agg)
                    print(f"[fill] Table 14 updated")
                else:
                    print("[skip] Table 14: no actor traces")
                if rollout_agg:
                    md = fill_table16(md, rollout_agg, vllm_stats)
                    print(f"[fill] Table 16 updated")
                else:
                    print("[skip] Table 16: no rollout traces")
            else:
                if actor_agg:
                    md = fill_table7(md, actor_agg)
                    print(f"[fill] Table 7 updated")
                else:
                    print("[skip] Table 7: no actor traces")
                if rollout_agg:
                    md = fill_table8(md, rollout_agg, vllm_stats)
                    print(f"[fill] Table 8 updated")
                else:
                    print("[skip] Table 8: no rollout traces")

            with open(args.tables, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"[fill] Written: {args.tables}")

    if not rollout_agg and not actor_agg:
        print("[error] Nothing to report", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
