#!/usr/bin/env python3
"""
fill_tables.py — 解析 veRL 训练日志，自动填写 results_tables.md 中所有行。

Usage:
  python3 fill_tables.py --log <log> --tables <md> --mode <mode> [--out <out>]

Modes:
  baseline       → 表1/2/3 逐步行 + 均值/std 行；表4/5/6 基准行
  ablation_n=N   → 表4 rollout_n=N 行
  ablation_bsz=N → 表5 train_batch_size=N 行
  ablation_seq=N → 表6 max_response_length=N 行
"""
import argparse, re, sys
from pathlib import Path
from statistics import mean, stdev

# ─── 格式化工具 ────────────────────────────────────────────────────────────────

def _vs(vals):
    """过滤 None，返回有效值列表"""
    return [v for v in vals if v is not None]

def stats(vals):
    vs = _vs(vals)
    if not vs:
        return None, None
    m = mean(vs)
    s = stdev(vs) if len(vs) > 1 else 0.0
    return m, s

def fmt(m, s=None, dec=1):
    """mean±std，若 m=None 则返回空串"""
    if m is None:
        return ""
    if s is not None and s > 0:
        return f"{m:.{dec}f}±{s:.{dec}f}"
    return f"{m:.{dec}f}"

def fmtm(m, dec=1):
    """仅均值"""
    if m is None:
        return ""
    return f"{m:.{dec}f}"

def fmts(s, dec=1):
    """仅 std"""
    if s is None:
        return ""
    return f"{s:.{dec}f}"

def pct_str(v, dec=1):
    if v is None:
        return ""
    return f"{v * 100:.{dec}f}"


# ─── 日志解析 ─────────────────────────────────────────────────────────────────

def load_records(log_path: str) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    import parse_verl_log as pv
    return pv.parse_log(log_path)

def stable_recs(records, skip=1):
    train = [r for r in records if r.get("step", 0) > 0 and "timing_s/gen" in r]
    return train[skip:]

def val_accuracy(records):
    vs = [r.get("val-core/openai/gsm8k/acc/mean@1")
          for r in records if "val-core/openai/gsm8k/acc/mean@1" in r]
    return vs[-1] if vs else None

def mem_gb(records):
    vals = _vs([r.get("actor/perf/max_memory_allocated_gb")
                or r.get("perf/max_memory_allocated_gb") for r in records])
    return mean(vals) if vals else None

def reward_key(r):
    return r.get("critic/rewards/mean") or r.get("critic/score/mean")



# ─── Markdown 行替换 ──────────────────────────────────────────────────────────

def replace_row_in_section(text: str, section_prefix: str, col1: str, new_row: str) -> str:
    """在 section_prefix 开头的段落内，替换第一列为 col1 的行（支持 * 斜体和 ├└ 前缀）。"""
    start = text.find(section_prefix)
    if start == -1:
        return text
    next_h = re.search(r"^###", text[start + len(section_prefix):], re.MULTILINE)
    end = start + len(section_prefix) + next_h.start() if next_h else len(text)
    section = text[start:end]
    # 匹配 "| [*├└ ]? col1 [*]? |" 格式
    escaped = re.escape(col1)
    pattern = re.compile(
        r"^\|\s*[*├└\s]*\*?" + escaped + r"\*?\s*\|.*$",
        re.MULTILINE
    )
    new_section = pattern.sub(new_row, section, count=1)
    return text[:start] + new_section + text[end:]

def replace_row_global(text: str, col1: str, new_row: str) -> str:
    """全文替换首列为 col1 的行（适用于全局唯一行）。"""
    escaped = re.escape(col1)
    pattern = re.compile(
        r"^\|\s*[*├└\s]*\*?" + escaped + r"\*?\s*\|.*$",
        re.MULTILINE
    )
    return pattern.sub(new_row, text, count=1)




def _parse_sync_step_time(text: str, section_prefix: str) -> float | None:
    """从 results_tables.md 的 Sync 基准行读取 step 时间（第6列）。
    行格式: | Sync 基准 | ... | ... | ... | ... | <step_s> | 1.0× | 0 |
    """
    start = text.find(section_prefix)
    if start == -1:
        return None
    import re as _re
    next_h = _re.search(r"^###", text[start + len(section_prefix):], _re.MULTILINE)
    end = start + len(section_prefix) + next_h.start() if next_h else len(text)
    section = text[start:end]
    for line in section.splitlines():
        if "Sync 基准" in line:
            cols = [c.strip().strip("*") for c in line.split("|")]
            # cols[0]='' cols[1]='Sync 基准' ... cols[6]=step_s cols[7]='1.0×'
            try:
                return float(cols[6])
            except (ValueError, IndexError):
                return None
    return None


def fill_oom_row(text: str, section_prefix: str, key: str, n_data_cols: int = 7) -> str:
    """Replace a table row with OOM markers."""
    oom_row = f"| {key} | " + " | ".join(["OOM"] * n_data_cols) + " |"
    return replace_row_in_section(text, section_prefix, key, oom_row)

# ─── fill_baseline：表1/2/3 逐步 + 均值/std；表4/5/6 基准行 ──────────────────

def fill_baseline(text: str, records: list) -> str:
    train = [r for r in records if r.get("step", 0) > 0 and "timing_s/gen" in r]
    if not train:
        print("[warn] no training records found")
        return text

    # ── 表1 逐步行 ──────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])   # step 存为 float，转 int 再 str 避免 "12.0" 匹配失败
        tg = r.get("timing_s/gen") or 0
        ts = r.get("timing_s/step") or 0
        tu = r.get("timing_s/update_actor") or 0
        row = (f"| {s} | {fmtm(tg)} | {fmtm(r.get('timing_s/old_log_prob'))} |"
               f" {fmtm(r.get('timing_s/ref'))} | {fmtm(tu)} |"
               f" {fmtm(r.get('timing_s/update_weights'))} | {fmtm(ts)} |"
               f" {fmtm(tg/ts*100 if ts else None)} | {fmtm(tu/ts*100 if ts else None)} |")
        text = replace_row_in_section(text, "### 表 1：", str(s), row)

    # ── 表2 逐步行 ──────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])
        mfu = r.get("perf/mfu/actor")
        thr = r.get("perf/throughput")
        mem = r.get("actor/perf/max_memory_allocated_gb") or r.get("perf/max_memory_allocated_gb")
        sq  = r.get("derived/actual_seqs_per_sec")
        row = (f"| {s} | {fmtm(mfu*100 if mfu else None)} | {fmtm(thr, 0)} |"
               f" {fmtm(mem)} | {fmtm(sq)} |")
        text = replace_row_in_section(text, "### 表 3：", str(s), row)

    # ── 表3 逐步行 ──────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])
        rl = r.get("response_length/mean")
        cl = r.get("response_length/clip_ratio")
        rw = reward_key(r)
        row = (f"| {s} | {fmtm(rl, 0)} | {fmtm(cl*100 if cl else None)} |"
               f" {fmtm(rw, 3)} |")
        text = replace_row_in_section(text, "### 表 5：", str(s), row)

    # ── 表1/2/3 均值行和 std 行（跳过 step 1 warmup）───────────────────────
    stable = stable_recs(records)

    # 表1 均值/std
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    olm,ols = stats([r.get("timing_s/old_log_prob") for r in stable])
    rm,rs   = stats([r.get("timing_s/ref") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    wm,ws   = stats([r.get("timing_s/update_weights") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    gfm,gfs = stats([r.get("derived/gen_frac") for r in stable])
    ufm,ufs = stats([r.get("derived/update_frac") for r in stable])

    mean1 = (f"| **均值** | **{fmtm(gm)}** | **{fmtm(olm)}** |"
             f" **{fmtm(rm)}** | **{fmtm(um)}** | **{fmtm(wm)}** | **{fmtm(sm)}** |"
             f" **{fmtm(gfm*100 if gfm else None)}** | **{fmtm(ufm*100 if ufm else None)}** |")
    std1  = (f"| **std** | **{fmts(gs)}** | **{fmts(ols)}** |"
             f" **{fmts(rs)}** | **{fmts(us)}** | **{fmts(ws)}** | **{fmts(ss)}** |"
             f" **{fmts(gfs*100 if gfs else None, 1)}** | **{fmts(ufs*100 if ufs else None, 1)}** |")
    text = replace_row_in_section(text, "### 表 1：", "**均值**", mean1)
    text = replace_row_in_section(text, "### 表 1：", "**std**",  std1)

    # 表2 均值/std
    mfum,mfus = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,thrs = stats([r.get("perf/throughput") for r in stable])
    memm,mems = stats([r.get("actor/perf/max_memory_allocated_gb")
                       or r.get("perf/max_memory_allocated_gb") for r in stable])
    sqm,sqs   = stats([r.get("derived/actual_seqs_per_sec") for r in stable])

    mean2 = (f"| **均值** | **{fmtm(mfum*100 if mfum else None)}** | **{fmtm(thrm, 0)}** |"
             f" **{fmtm(memm)}** | **{fmtm(sqm)}** |")
    std2  = (f"| **std** | **{fmts(mfus*100 if mfus else None)}** | **{fmts(thrs, 0)}** |"
             f" **{fmts(mems)}** | **{fmts(sqs)}** |")
    text = replace_row_in_section(text, "### 表 3：", "**均值**", mean2)
    text = replace_row_in_section(text, "### 表 3：", "**std**",  std2)

    # 表3 均值/std
    rlm,rls = stats([r.get("response_length/mean") for r in stable])
    clm,cls = stats([r.get("response_length/clip_ratio") for r in stable])
    rwm,rws = stats([reward_key(r) for r in stable])
    mean3 = (f"| **均值** | **{fmtm(rlm, 0)}** | **{fmtm(clm*100 if clm else None)}** |"
             f" **{fmtm(rwm, 3)}** |")
    std3  = (f"| **std** | **{fmts(rls, 0)}** | **{fmts(cls*100 if cls else None)}** |"
             f" **{fmts(rws, 3)}** |")
    text = replace_row_in_section(text, "### 表 5：", "**均值**", mean3)
    text = replace_row_in_section(text, "### 表 5：", "**std**",  std3)

    # 表9: Sync 基准行（与异步实验对比用）
    sm9, _  = stats([r.get("timing_s/step") for r in stable])
    thr9, _ = stats([r.get("perf/throughput") for r in stable])
    mfu9, _ = stats([r.get("perf/mfu/actor") for r in stable])
    sq9, _  = stats([r.get("derived/actual_seqs_per_sec") for r in stable])
    ngpu    = int(stable[0].get("cfg/n_gpus", 8)) if stable else 8
    row9_sync = (f"| Sync 基准 | {ngpu} GPU 共享 | {fmtm(thr9, 0)} | {fmtm(sq9)} |"
                 f" {fmtm(mfu9 * 100 if mfu9 else None)} | {fmtm(sm9)} | 1.0× | 0 |")
    text = replace_row_in_section(text, "### 表 17：", "Sync 基准", row9_sync)

    return text


# ─── fill_ablation_n/bsz/seq ──────────────────────────────────────────────────

def fill_ablation_n(text: str, records: list, n: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    memm    = mem_gb(stable)
    sqm,sqs = stats([r.get("derived/actual_seqs_per_sec") for r in stable])

    row = (f"| {n} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} |"
           f" {fmt(thrm,ts2,0)} | {fmtm(memm)} | {fmt(sqm,sqs)} |")
    # 使用 section-aware，避免 rollout_n=1/2 与 step=1/2 冲突
    return replace_row_in_section(text, "### 表 7：", str(n), row)


def fill_ablation_bsz(text: str, records: list, bsz: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    memm    = mem_gb(stable)
    rn      = records[0].get("cfg/rollout_n", 5) if records else 5
    theo    = int(bsz * rn)

    row = (f"| {bsz} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} | {fmt(thrm,ts2,0)} |"
           f" {fmtm(memm)} | {theo} |")
    return replace_row_in_section(text, "### 表 9：", str(bsz), row)


def fill_ablation_seq(text: str, records: list, seqlen: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    clm,_   = stats([r.get("response_length/clip_ratio") for r in stable])
    memm    = mem_gb(stable)

    row = (f"| {seqlen} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} | {fmt(thrm,ts2,0)} |"
           f" {fmtm(clm*100 if clm else None)} | {fmtm(memm)} |")
    return replace_row_in_section(text, "### 表 11：", str(seqlen), row)




# ─── fill_async：表9/10 全异步训练 ───────────────────────────────────────────

def fill_async(text: str, records: list, thresh: float) -> str:
    """Fill Table 9 async summary row and Table 10 per-sync-step rows."""
    train = [r for r in records if r.get("step", 0) > 0]
    stable = stable_recs(records)

    def _staleness(r):
        for k in ["async_training/staleness_ratio", "async/staleness_ratio",
                  "rollout/staleness_ratio", "staleness_ratio"]:
            v = r.get(k)
            if v is not None:
                return v
        # compute from count keys (fully_async trainer log format)
        # use max_required_samples (queue capacity) as denominator for stable ratio
        ss = r.get("fully_async/count/staleness_samples")
        max_req = r.get("fully_async/static/max_required_samples")
        if ss is not None and max_req and max_req > 0:
            return ss / max_req
        # fallback: use total_generated_samples (cumulative, less stable)
        ts = r.get("fully_async/count/total_generated_samples")
        if ss is not None and ts and ts > 0:
            return ss / ts
        return None

    def _rollout_thr(r):
        for k in ["async_training/rollout_throughput", "rollout/throughput",
                  "vllm/gen_throughput_toks", "perf/throughput"]:
            v = r.get(k)
            if v is not None:
                return v
        return None

    # === 表9: Async 汇总行 ===
    if stable:
        sm, ss     = stats([r.get("timing_s/step") for r in stable])
        thrm, thrs = stats([r.get("perf/throughput") for r in stable])
        mfum, _    = stats([r.get("perf/mfu/actor") for r in stable])
        sqm, sqs   = stats([r.get("derived/actual_seqs_per_sec") for r in stable])
        stm, _     = stats([_staleness(r) for r in stable])

        # 等效sync耗时 = 4个param-sync step总时间（4×256=1024 samples ≈ sync 1步）
        equiv_m = sm * 4 if sm else None
        _sync_s = _parse_sync_step_time(text, "### 表 17：") or 146.1
        speedup = (f"{_sync_s / equiv_m:.2f}×" if equiv_m else "-")
        row9 = (f"| Async 4+4 GPU | 4+4 GPU 分离 | {fmt(thrm, thrs, 0)} | {fmt(sqm, sqs)} |"
                f" {fmtm(mfum * 100 if mfum else None)} | {fmtm(equiv_m)} | {speedup} |"
                f" {fmtm(stm * 100 if stm else None)} |")
        text = replace_row_in_section(text, "### 表 17：", "Async 4+4 GPU", row9)


    return text



# ─── fill_baseline_32b：表10/11/12 逐步 + 均值/std；表18 Sync 基准行 ─────────

def fill_baseline_32b(text: str, records: list) -> str:
    train = [r for r in records if r.get("step", 0) > 0 and "timing_s/gen" in r]
    if not train:
        print("[warn] no training records found (32b)")
        return text

    # ── 表10 逐步行 ─────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])
        tg = r.get("timing_s/gen") or 0
        ts = r.get("timing_s/step") or 0
        tu = r.get("timing_s/update_actor") or 0
        row = (f"| {s} | {fmtm(tg)} | {fmtm(r.get('timing_s/old_log_prob'))} |"
               f" {fmtm(r.get('timing_s/ref'))} | {fmtm(tu)} |"
               f" {fmtm(r.get('timing_s/update_weights'))} | {fmtm(ts)} |"
               f" {fmtm(tg/ts*100 if ts else None)} | {fmtm(tu/ts*100 if ts else None)} |")
        text = replace_row_in_section(text, "### 表 2：", str(s), row)

    # ── 表11 逐步行 ─────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])
        mfu = r.get("perf/mfu/actor")
        thr = r.get("perf/throughput")
        mem = r.get("actor/perf/max_memory_allocated_gb") or r.get("perf/max_memory_allocated_gb")
        sq  = r.get("derived/actual_seqs_per_sec")
        row = (f"| {s} | {fmtm(mfu*100 if mfu else None)} | {fmtm(thr, 0)} |"
               f" {fmtm(mem)} | {fmtm(sq)} |")
        text = replace_row_in_section(text, "### 表 4：", str(s), row)

    # ── 表12 逐步行 ─────────────────────────────────────────────────────────
    for r in train:
        s = int(r["step"])
        rl = r.get("response_length/mean")
        cl = r.get("response_length/clip_ratio")
        rw = reward_key(r)
        row = (f"| {s} | {fmtm(rl, 0)} | {fmtm(cl*100 if cl else None)} |"
               f" {fmtm(rw, 3)} |")
        text = replace_row_in_section(text, "### 表 6：", str(s), row)

    # ── 表10/11/12 均值/std（跳过 step 1 warmup）───────────────────────────
    stable = stable_recs(records)

    # 表10 均值/std
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    olm,ols = stats([r.get("timing_s/old_log_prob") for r in stable])
    rm,rs   = stats([r.get("timing_s/ref") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    wm,ws   = stats([r.get("timing_s/update_weights") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    gfm,gfs = stats([r.get("derived/gen_frac") for r in stable])
    ufm,ufs = stats([r.get("derived/update_frac") for r in stable])

    mean10 = (f"| **均值** | **{fmtm(gm)}** | **{fmtm(olm)}** |"
              f" **{fmtm(rm)}** | **{fmtm(um)}** | **{fmtm(wm)}** | **{fmtm(sm)}** |"
              f" **{fmtm(gfm*100 if gfm else None)}** | **{fmtm(ufm*100 if ufm else None)}** |")
    std10  = (f"| **std** | **{fmts(gs)}** | **{fmts(ols)}** |"
              f" **{fmts(rs)}** | **{fmts(us)}** | **{fmts(ws)}** | **{fmts(ss)}** |"
              f" **{fmts(gfs*100 if gfs else None, 1)}** | **{fmts(ufs*100 if ufs else None, 1)}** |")
    text = replace_row_in_section(text, "### 表 2：", "**均值**", mean10)
    text = replace_row_in_section(text, "### 表 2：", "**std**",  std10)

    # 表11 均值/std
    mfum,mfus = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,thrs = stats([r.get("perf/throughput") for r in stable])
    memm,mems = stats([r.get("actor/perf/max_memory_allocated_gb")
                       or r.get("perf/max_memory_allocated_gb") for r in stable])
    sqm,sqs   = stats([r.get("derived/actual_seqs_per_sec") for r in stable])

    mean11 = (f"| **均值** | **{fmtm(mfum*100 if mfum else None)}** | **{fmtm(thrm, 0)}** |"
              f" **{fmtm(memm)}** | **{fmtm(sqm)}** |")
    std11  = (f"| **std** | **{fmts(mfus*100 if mfus else None)}** | **{fmts(thrs, 0)}** |"
              f" **{fmts(mems)}** | **{fmts(sqs)}** |")
    text = replace_row_in_section(text, "### 表 4：", "**均值**", mean11)
    text = replace_row_in_section(text, "### 表 4：", "**std**",  std11)

    # 表12 均值/std
    rlm,rls = stats([r.get("response_length/mean") for r in stable])
    clm,cls = stats([r.get("response_length/clip_ratio") for r in stable])
    rwm,rws = stats([reward_key(r) for r in stable])
    mean12 = (f"| **均值** | **{fmtm(rlm, 0)}** | **{fmtm(clm*100 if clm else None)}** |"
              f" **{fmtm(rwm, 3)}** |")
    std12  = (f"| **std** | **{fmts(rls, 0)}** | **{fmts(cls*100 if cls else None)}** |"
              f" **{fmts(rws, 3)}** |")
    text = replace_row_in_section(text, "### 表 6：", "**均值**", mean12)
    text = replace_row_in_section(text, "### 表 6：", "**std**",  std12)

    # 表18: Sync 基准行（与32B异步实验对比用）
    sm18, _  = stats([r.get("timing_s/step") for r in stable])
    thr18, _ = stats([r.get("perf/throughput") for r in stable])
    mfu18, _ = stats([r.get("perf/mfu/actor") for r in stable])
    sq18, _  = stats([r.get("derived/actual_seqs_per_sec") for r in stable])
    ngpu     = int(stable[0].get("cfg/n_gpus", 8)) if stable else 8
    row18_sync = (f"| Sync 基准 | {ngpu} GPU 共享 | {fmtm(thr18, 0)} | {fmtm(sq18)} |"
                  f" {fmtm(mfu18 * 100 if mfu18 else None)} | {fmtm(sm18)} | 1.0× | 0 |")
    text = replace_row_in_section(text, "### 表 18：", "Sync 基准", row18_sync)

    return text


# ─── fill_ablation_n/bsz/seq 32B ─────────────────────────────────────────────

def fill_ablation_n_32b(text: str, records: list, n: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    memm    = mem_gb(stable)
    sqm,sqs = stats([r.get("derived/actual_seqs_per_sec") for r in stable])

    row = (f"| {n} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} |"
           f" {fmt(thrm,ts2,0)} | {fmtm(memm)} | {fmt(sqm,sqs)} |")
    return replace_row_in_section(text, "### 表 8：", str(n), row)


def fill_ablation_bsz_32b(text: str, records: list, bsz: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    memm    = mem_gb(stable)
    rn      = records[0].get("cfg/rollout_n", 2) if records else 2
    theo    = int(bsz * rn)

    row = (f"| {bsz} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} | {fmt(thrm,ts2,0)} |"
           f" {fmtm(memm)} | {theo} |")
    return replace_row_in_section(text, "### 表 10：", str(bsz), row)


def fill_ablation_seq_32b(text: str, records: list, seqlen: int) -> str:
    stable = stable_recs(records)
    gm,gs   = stats([r.get("timing_s/gen") for r in stable])
    um,us   = stats([r.get("timing_s/update_actor") for r in stable])
    sm,ss   = stats([r.get("timing_s/step") for r in stable])
    mfum,_  = stats([r.get("perf/mfu/actor") for r in stable])
    thrm,ts2= stats([r.get("perf/throughput") for r in stable])
    clm,_   = stats([r.get("response_length/clip_ratio") for r in stable])
    memm    = mem_gb(stable)

    row = (f"| {seqlen} | {fmt(gm,gs)} | {fmt(um,us)} | {fmt(sm,ss)} |"
           f" {fmtm(mfum*100 if mfum else None)} | {fmt(thrm,ts2,0)} |"
           f" {fmtm(clm*100 if clm else None)} | {fmtm(memm)} |")
    return replace_row_in_section(text, "### 表 12：", str(seqlen), row)


# ─── fill_async_32b：表18 全异步训练（32B LoRA）──────────────────────────────

def fill_async_32b(text: str, records: list, thresh: float) -> str:
    """Fill Table 18 async summary row for 32B LoRA experiment."""
    stable = stable_recs(records)

    def _staleness(r):
        for k in ["async_training/staleness_ratio", "async/staleness_ratio",
                  "rollout/staleness_ratio", "staleness_ratio"]:
            v = r.get(k)
            if v is not None:
                return v
        ss = r.get("fully_async/count/staleness_samples")
        max_req = r.get("fully_async/static/max_required_samples")
        if ss is not None and max_req and max_req > 0:
            return ss / max_req
        ts = r.get("fully_async/count/total_generated_samples")
        if ss is not None and ts and ts > 0:
            return ss / ts
        return None

    if stable:
        sm, ss     = stats([r.get("timing_s/step") for r in stable])
        thrm, thrs = stats([r.get("perf/throughput") for r in stable])
        mfum, _    = stats([r.get("perf/mfu/actor") for r in stable])
        sqm, sqs   = stats([r.get("derived/actual_seqs_per_sec") for r in stable])
        stm, _     = stats([_staleness(r) for r in stable])

        equiv_m = sm * 4 if sm else None
        _sync_s = _parse_sync_step_time(text, "### 表 18：") or 139.7
        speedup = (f"{_sync_s / equiv_m:.2f}×" if equiv_m else "-")
        row18 = (f"| Async 4+4 GPU | 4+4 GPU 分离 | {fmt(thrm, thrs, 0)} | {fmt(sqm, sqs)} |"
                 f" {fmtm(mfum * 100 if mfum else None)} | {fmtm(equiv_m)} | {speedup} |"
                 f" {fmtm(stm * 100 if stm else None)} |")
        text = replace_row_in_section(text, "### 表 18：", "Async 4+4 GPU", row18)

    return text


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",    required=False, default=None, help="veRL training log file")
    ap.add_argument("--tables", required=True,  help="results_tables.md path")
    ap.add_argument("--mode",   required=True,
                    help="baseline | ablation_n=N | ablation_bsz=N | ablation_seq=N | async")
    ap.add_argument("--thresh", type=float, default=0.5, help="staleness threshold (async mode)")
    ap.add_argument("--out",      default=None,   help="output md path (default: overwrite --tables)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap records to first N steps (default: use all)")
    args = ap.parse_args()
    mode = args.mode

    # OOM modes don't need log data
    if "_oom=" in mode:
        with open(args.tables, encoding="utf-8") as f:
            text = f.read()
        out = args.out or args.tables
        if mode.startswith("ablation_n_32b_oom="):
            text = fill_oom_row(text, "### 表 8：", mode.split("=")[1])
        elif mode.startswith("ablation_bsz_32b_oom="):
            text = fill_oom_row(text, "### 表 10：", mode.split("=")[1])
        elif mode.startswith("ablation_seq_32b_oom="):
            text = fill_oom_row(text, "### 表 12：", mode.split("=")[1])
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[fill] {out} updated (mode={mode}, OOM)", flush=True)
        return

    records = load_records(args.log)
    if args.max_steps is not None:
        records = [r for r in records if r.get("step", 0) <= args.max_steps]
    if not records:
        print(f"[fill] WARNING: no step records found in {args.log}", flush=True)

    with open(args.tables, encoding="utf-8") as f:
        text = f.read()

    mode = args.mode
    if mode == "baseline":
        text = fill_baseline(text, records)
    elif mode == "async":
        text = fill_async(text, records, args.thresh)
    elif mode.startswith("ablation_n="):
        text = fill_ablation_n(text, records, int(mode.split("=")[1]))
    elif mode.startswith("ablation_bsz="):
        text = fill_ablation_bsz(text, records, int(mode.split("=")[1]))
    elif mode.startswith("ablation_seq="):
        text = fill_ablation_seq(text, records, int(mode.split("=")[1]))
    elif mode == "baseline_32b":
        text = fill_baseline_32b(text, records)
    elif mode == "async_32b":
        text = fill_async_32b(text, records, args.thresh)
    elif mode.startswith("ablation_n_32b="):
        text = fill_ablation_n_32b(text, records, int(mode.split("=")[1]))
    elif mode.startswith("ablation_bsz_32b="):
        text = fill_ablation_bsz_32b(text, records, int(mode.split("=")[1]))
    elif mode.startswith("ablation_seq_32b="):
        text = fill_ablation_seq_32b(text, records, int(mode.split("=")[1]))
    elif mode.startswith("ablation_n_32b_oom="):
        text = fill_oom_row(text, "### 表 8：", mode.split("=")[1])
    elif mode.startswith("ablation_bsz_32b_oom="):
        text = fill_oom_row(text, "### 表 10：", mode.split("=")[1])
    elif mode.startswith("ablation_seq_32b_oom="):
        text = fill_oom_row(text, "### 表 12：", mode.split("=")[1])
    else:
        print(f"[fill] Unknown mode: {mode}", flush=True)
        return

    out = args.out or args.tables
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[fill] {out} updated (mode={mode}, {len(records)} step records)", flush=True)


if __name__ == "__main__":
    main()
