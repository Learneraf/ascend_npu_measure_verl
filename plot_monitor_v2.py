"""
折线图生成脚本 v2
- 硬件指标来自 monitor CSV（GPU/CPU/RAM）
- step 级指标来自训练 log（timing/MFU/reward 等）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re, os

CSV = "/data/yanziyi/gpu_test_0610/outputs/qwen3_8b_grpo_20260612_144803_monitor.csv"
LOG = "/data/yanziyi/gpu_test_0610/outputs/qwen3_8b_grpo_20260612_144803.log"
OUT = "/data/yanziyi/gpu_test_0610/outputs/plots_v2"
os.makedirs(OUT, exist_ok=True)

# ── monitor CSV ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
t = df['elapsed_s']

# ── 解析训练 log → step 级 DataFrame ─────────────────────────────────────────
RE_STEP = re.compile(r'\bstep:(\d+)\b')
RE_KV   = re.compile(r'([\w/@-]+):(np\.(?:float64|int32|int64)\(([^)]+)\)|(-?[\d.e+\-]+))')
RE_VP   = re.compile(r'Avg prompt throughput:\s*([\d.]+)\s*tokens/s')
RE_VG   = re.compile(r'Avg generation throughput:\s*([\d.]+)\s*tokens/s')

WANT = {
    'timing_s/gen', 'timing_s/old_log_prob', 'timing_s/ref', 'timing_s/adv',
    'timing_s/update_actor', 'timing_s/update_weights', 'timing_s/step',
    'timing_s/testing', 'response_length/mean', 'response_length/clip_ratio',
    'perf/mfu/actor', 'perf/throughput', 'critic/rewards/mean',
    'val-core/openai/gsm8k/acc/mean@1',
}

step_rows = {}
with open(LOG) as f:
    for line in f:
        if '(TaskRunner' not in line:
            continue
        m = RE_STEP.search(line)
        if not m:
            continue
        s = int(m.group(1))
        row = step_rows.setdefault(s, {'step': s})
        for m2 in RE_KV.finditer(line):
            key = m2.group(1)
            if key not in WANT:
                continue
            raw = m2.group(3) if m2.group(3) else m2.group(4)
            try:
                row[key] = float(raw)
            except Exception:
                pass
        mp = RE_VP.search(line)
        if mp:
            row['vllm/prompt_throughput_toks'] = float(mp.group(1))
        mg = RE_VG.search(line)
        if mg:
            row['vllm/gen_throughput_toks'] = float(mg.group(1))

df_s = pd.DataFrame(list(step_rows.values())).sort_values('step').reset_index(drop=True)
print(f"Monitor rows: {len(df)},  Step rows: {len(df_s)},  Steps: {df_s['step'].tolist()}")

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}', dpi=150)
    plt.close(fig)
    print(f'  saved {name}')

def step_plot(ax, col, label, color='steelblue'):
    if col not in df_s:
        ax.set_visible(False)
        return
    s = df_s[col].dropna()
    if s.empty:
        ax.set_visible(False)
        return
    ax.plot(df_s.loc[s.index, 'step'], s, 'o-', ms=4, lw=1.2, color=color)
    ax.set(title=label, xlabel='Step')
    ax.grid(alpha=0.3)

# ── 01 GPU 利用率 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for i in range(8):
    ax.plot(t, df[f'gpu{i}_util_pct'], lw=0.8, alpha=0.8, label=f'GPU{i}')
ax.set(xlabel='Elapsed (s)', ylabel='Utilization (%)', title='GPU Utilization')
ax.set_ylim(0, 105); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
savefig(fig, '01_gpu_util.png')

# ── 02 GPU 显存 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for i in range(8):
    ax.plot(t, df[f'gpu{i}_mem_used_gb'], lw=0.8, alpha=0.8, label=f'GPU{i}')
ax.axhline(80.0, color='red', lw=0.8, ls='--', label='80GB limit')
ax.set(xlabel='Elapsed (s)', ylabel='Memory Used (GB)', title='GPU Memory Usage')
ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
savefig(fig, '02_gpu_mem.png')

# ── 03 GPU 功耗 & 温度 ────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i in range(8):
    a1.plot(t, df[f'gpu{i}_power_w'], lw=0.8, alpha=0.8, label=f'GPU{i}')
    a2.plot(t, df[f'gpu{i}_temp_c'],  lw=0.8, alpha=0.8, label=f'GPU{i}')
a1.set(ylabel='Power (W)', title='GPU Power'); a1.legend(ncol=4, fontsize=8); a1.grid(alpha=0.3)
a2.set(xlabel='Elapsed (s)', ylabel='Temp (°C)', title='GPU Temperature')
a2.legend(ncol=4, fontsize=8); a2.grid(alpha=0.3)
savefig(fig, '03_gpu_power_temp.png')

# ── 04 CPU & RAM ──────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
a1.plot(t, df['cpu_util_pct'], lw=0.8, color='steelblue')
a1.set(ylabel='CPU Util (%)', title='CPU Utilization'); a1.set_ylim(0, 105); a1.grid(alpha=0.3)
a2.plot(t, df['ram_used_gb'], lw=0.8, color='darkorange', label='Used')
a2.axhline(df['ram_total_gb'].iloc[0], color='red', lw=0.8, ls='--',
           label=f"Total {df['ram_total_gb'].iloc[0]:.0f}GB")
a2.set(xlabel='Elapsed (s)', ylabel='RAM (GB)', title='RAM Usage')
a2.legend(fontsize=8); a2.grid(alpha=0.3)
savefig(fig, '04_cpu_ram.png')

# ── 05 各阶段耗时（log） ──────────────────────────────────────────────────────
if 'timing_s/step' in df_s.columns:
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = ['gen', 'old_log_prob', 'ref', 'adv', 'update_actor', 'update_weights']
    cols   = [f'timing_s/{l}' for l in labels]
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(labels)))
    bottom = np.zeros(len(df_s))
    xs = np.arange(len(df_s))
    for col, label, color in zip(cols, labels, colors):
        if col in df_s.columns:
            vals = df_s[col].fillna(0).values
            ax.bar(xs, vals, bottom=bottom, label=label, color=color, width=0.8)
            bottom += vals
    ax.plot(xs, df_s['timing_s/step'].fillna(0).values, 'k-o', ms=4, lw=1.2, label='step total')
    # 标注 validation steps
    if 'timing_s/testing' in df_s.columns:
        for xi, row in df_s.iterrows():
            if pd.notna(row.get('timing_s/testing')) and row['timing_s/testing'] > 0:
                ax.axvline(xi, color='gray', lw=0.8, ls=':')
                ax.text(xi, ax.get_ylim()[1] * 0.95, 'val', ha='center', fontsize=7, color='gray')
    ax.set_xticks(xs)
    ax.set_xticklabels(df_s['step'].tolist(), fontsize=8)
    ax.set(xlabel='Step', ylabel='Time (s)', title='Per-Step Timing Breakdown')
    ax.legend(ncol=4, fontsize=8); ax.grid(axis='y', alpha=0.3)
    savefig(fig, '05_step_timing.png')

# ── 06 MFU & 吞吐量（log） ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
step_plot(axes[0, 0], 'perf/mfu/actor',           'Actor MFU (%)',               'steelblue')
step_plot(axes[0, 1], 'perf/throughput',           'Training Throughput (tok/s)', 'darkorange')
step_plot(axes[1, 0], 'vllm/gen_throughput_toks',  'vLLM Gen Throughput (tok/s)', 'forestgreen')
step_plot(axes[1, 1], 'response_length/mean',      'Avg Response Length (tok)',   'purple')
savefig(fig, '06_perf_metrics.png')

# ── 07 训练质量（log） ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
step_plot(axes[0], 'response_length/mean',       'Avg Response Length (tok)', 'purple')
step_plot(axes[1], 'response_length/clip_ratio', 'Clip Ratio',                'tomato')
step_plot(axes[2], 'critic/rewards/mean',         'Mean Reward',              'steelblue')
savefig(fig, '07_training_quality.png')

# ── 08 GSM8K 验证精度（log） ──────────────────────────────────────────────────
val_col = 'val-core/openai/gsm8k/acc/mean@1'
if val_col in df_s.columns:
    dv = df_s.dropna(subset=[val_col])
    if not dv.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dv['step'], dv[val_col], 'o-', ms=5, lw=1.5, color='darkgreen')
        ax.set(xlabel='Step', ylabel='Accuracy', title='GSM8K Validation Accuracy')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
        ax.grid(alpha=0.3)
        savefig(fig, '08_gsm8k_acc.png')

print("All done ->", OUT)
