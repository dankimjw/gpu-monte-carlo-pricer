#!/usr/bin/env python3
"""
Generate performance charts for the GPU Monte Carlo Option Pricing Engine.
Reads CSVs from output/ and writes PNGs back to output/.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(__file__), '..', 'output')

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
GRID_CLR  = '#21262d'
TEXT_CLR  = '#e6edf3'
ACCENT    = '#58a6ff'   # blue highlight
GREEN     = '#3fb950'
RED       = '#f85149'
ORANGE    = '#d29922'
PURPLE    = '#bc8cff'
CYAN      = '#39d353'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_CLR,
    'axes.labelcolor':   TEXT_CLR,
    'axes.titlecolor':   TEXT_CLR,
    'xtick.color':       TEXT_CLR,
    'ytick.color':       TEXT_CLR,
    'text.color':        TEXT_CLR,
    'grid.color':        GRID_CLR,
    'grid.linewidth':    0.8,
    'font.family':       'monospace',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
})

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  Saved {path}')

# ── Load CSVs ────────────────────────────────────────────────────────────────
def load_csv(name):
    path = os.path.join(OUT, name)
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})
    return rows

cpu_gpu  = load_csv('benchmark_cpu_gpu.csv')
blocksize = load_csv('benchmark_blocksize.csv')
paths     = load_csv('benchmark_paths.csv')

# ── 1. CPU vs GPU Timing ─────────────────────────────────────────────────────
print('Generating: cpu_vs_gpu.png')
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)

labels    = [f"{int(r['num_paths'])//1000}K" if int(r['num_paths']) < 1_000_000
             else f"{int(r['num_paths'])//1_000_000}M"
             for r in cpu_gpu]
cpu_ms    = [float(r['cpu_ms'])      for r in cpu_gpu]
gpu_ms    = [float(r['gpu_total_ms']) for r in cpu_gpu]
speedups  = [float(r['speedup'])     for r in cpu_gpu]

x = np.arange(len(labels))
w = 0.35

bars_cpu = ax.bar(x - w/2, cpu_ms, w, color=RED,    label='CPU (single-threaded)', zorder=3)
bars_gpu = ax.bar(x + w/2, gpu_ms, w, color=GREEN,  label='GPU (RTX 3060 Ti)',     zorder=3)

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f'{v:.0f}ms' if v >= 1 else f'{v:.2f}ms'))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Time (ms, log scale)')
ax.set_title('CPU vs GPU Execution Time')
ax.grid(axis='y', zorder=0)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR)

# Annotate bars with times
for bar, ms in zip(bars_cpu, cpu_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f'{ms:.0f}ms', ha='center', va='bottom', fontsize=9, color=RED)
for bar, ms in zip(bars_gpu, gpu_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f'{ms:.2f}ms', ha='center', va='bottom', fontsize=9, color=GREEN)

# Speedup labels above each pair
for i, (sp, xi) in enumerate(zip(speedups, x)):
    ymax = max(cpu_ms[i], gpu_ms[i])
    ax.text(xi, ymax * 4.5, f'{sp:.0f}×', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=ACCENT)

ax.text(0.5, 0.97, 'Speedup (×) shown above each pair',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9, color=ACCENT, style='italic')

fig.tight_layout()
savefig(fig, 'cpu_vs_gpu.png')

# ── 2. Block Size Sweep ──────────────────────────────────────────────────────
print('Generating: block_size_sweep.png')
fig, ax1 = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)

bs_vals    = [int(r['block_size'])         for r in blocksize]
thru_vals  = [float(r['throughput_mpps'])  for r in blocksize]
kernel_ms  = [float(r['kernel_ms'])        for r in blocksize]

colors = [ACCENT if b != 256 else ORANGE for b in bs_vals]
bars = ax1.bar(range(len(bs_vals)), thru_vals, color=colors, zorder=3, width=0.6)

ax1.set_xticks(range(len(bs_vals)))
ax1.set_xticklabels([str(b) for b in bs_vals])
ax1.set_xlabel('Block Size (threads/block)')
ax1.set_ylabel('Throughput (Mpaths/sec)', color=TEXT_CLR)
ax1.set_title('Block Size Sweep — 1M Paths, 252 Steps')
ax1.grid(axis='y', zorder=0)

# Throughput value labels
for bar, t, k in zip(bars, thru_vals, kernel_ms):
    col = ORANGE if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(ORANGE) else ACCENT
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{t:.0f}', ha='center', va='bottom', fontsize=10, color=col)

# Kernel ms as secondary line
ax2 = ax1.twinx()
ax2.set_facecolor(PANEL_BG)
ax2.plot(range(len(bs_vals)), kernel_ms, color=PURPLE, marker='o',
         linewidth=2, markersize=7, label='Kernel time (ms)', zorder=4)
ax2.set_ylabel('Kernel Time (ms)', color=PURPLE)
ax2.tick_params(axis='y', colors=PURPLE)
ax2.spines['right'].set_color(PURPLE)

legend_patches = [
    Patch(color=ACCENT,  label='Block size'),
    Patch(color=ORANGE,  label='Optimal (256) — 605 Mpaths/s'),
    Patch(color=PURPLE,  label='Kernel time (ms)'),
]
ax1.legend(handles=legend_patches, facecolor=PANEL_BG, edgecolor=GRID_CLR, fontsize=9)

fig.tight_layout()
savefig(fig, 'block_size_sweep.png')

# ── 3. MC Convergence (Error + Std Error vs Paths) ──────────────────────────
print('Generating: mc_convergence.png')
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)

n_vals    = [int(r['num_paths'])    for r in paths]
err_vals  = [float(r['error_pct'])  for r in paths]
stderr_v  = [float(r['std_err'])    for r in paths]

ax.loglog(n_vals, err_vals,  color=ORANGE, marker='o', linewidth=2,
          markersize=8, label='Error vs Black-Scholes (%)', zorder=3)
ax.loglog(n_vals, stderr_v, color=CYAN,   marker='s', linewidth=2,
          markersize=7, linestyle='--', label='Std Error ($)', zorder=3)

# Theoretical O(1/sqrt(N)) reference line
n_ref = np.array([1e3, 1e7])
ref_y = err_vals[0] * np.sqrt(n_vals[0] / n_ref)
ax.loglog(n_ref, ref_y, color=GRID_CLR, linewidth=1.5, linestyle=':',
          label='O(1/√N) reference', zorder=2)

# Annotate each point with error value
for n, e in zip(n_vals, err_vals):
    ax.annotate(f'{e:.2f}%', (n, e), textcoords='offset points',
                xytext=(6, 4), fontsize=8, color=ORANGE)

ax.set_xlabel('Number of Paths')
ax.set_ylabel('Error / Std Error')
ax.set_title('Monte Carlo Convergence vs Black-Scholes')
ax.grid(True, which='both', zorder=0)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f'{int(v):,}'))
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR)

fig.tight_layout()
savefig(fig, 'mc_convergence.png')

# ── 4. Kernel vs Total Time (overhead breakdown) ─────────────────────────────
print('Generating: timing_breakdown.png')
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)

n_labels   = []
k_times    = []
overhead   = []
for r in paths:
    n = int(r['num_paths'])
    k = float(r['kernel_ms'])
    t = float(r['total_ms'])
    n_labels.append(f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M")
    k_times.append(k)
    overhead.append(max(0, t - k))

x = np.arange(len(n_labels))
w = 0.5
ax.bar(x, k_times,  w, color=GREEN,  label='MC Kernel',          zorder=3)
ax.bar(x, overhead, w, bottom=k_times, color=PURPLE, alpha=0.8,
       label='RNG Init + Reduction overhead', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(n_labels)
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Time (ms)')
ax.set_title('GPU Timing Breakdown: Kernel vs Overhead')
ax.grid(axis='y', zorder=0)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR)

# Total time label on top
for i, (k, ov) in enumerate(zip(k_times, overhead)):
    total = k + ov
    ax.text(i, total + 1, f'{total:.1f}ms', ha='center', va='bottom',
            fontsize=9, color=TEXT_CLR)

fig.tight_layout()
savefig(fig, 'timing_breakdown.png')

# ── 5. Combined 2×2 figure ────────────────────────────────────────────────────
print('Generating: benchmarks.png  (combined 2x2)')
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('GPU Monte Carlo Option Pricing Engine — Performance Benchmarks',
             fontsize=14, color=TEXT_CLR, y=1.01)

# ── Panel A: CPU vs GPU ──
ax = axes[0, 0]
ax.set_facecolor(PANEL_BG)
xa = np.arange(len(labels))
wa = 0.35
bars_c = ax.bar(xa - wa/2, cpu_ms, wa, color=RED,   label='CPU', zorder=3)
bars_g = ax.bar(xa + wa/2, gpu_ms, wa, color=GREEN, label='GPU', zorder=3)
ax.set_yscale('log')
ax.set_xticks(xa)
ax.set_xticklabels(labels)
ax.set_title('CPU vs GPU Execution Time')
ax.set_ylabel('Time (ms, log)')
ax.set_xlabel('Paths')
ax.grid(axis='y', zorder=0)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, fontsize=9)
for i, sp in enumerate(speedups):
    ymax = max(cpu_ms[i], gpu_ms[i])
    ax.text(i, ymax * 5, f'{sp:.0f}×', ha='center', fontsize=11,
            fontweight='bold', color=ACCENT)
for bar, ms in zip(bars_c, cpu_ms):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2,
            f'{ms:.0f}', ha='center', fontsize=8, color=RED)
for bar, ms in zip(bars_g, gpu_ms):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2,
            f'{ms:.2f}', ha='center', fontsize=8, color=GREEN)

# ── Panel B: Block Size ──
ax = axes[0, 1]
ax.set_facecolor(PANEL_BG)
cols = [ACCENT if b != 256 else ORANGE for b in bs_vals]
ax.bar(range(len(bs_vals)), thru_vals, color=cols, zorder=3, width=0.6)
ax.set_xticks(range(len(bs_vals)))
ax.set_xticklabels([str(b) for b in bs_vals])
ax.set_title('Block Size vs Throughput (1M paths)')
ax.set_ylabel('Mpaths/sec')
ax.set_xlabel('Block Size')
ax.grid(axis='y', zorder=0)
for i, t in enumerate(thru_vals):
    ax.text(i, t + 5, f'{t:.0f}', ha='center', fontsize=9,
            color=ORANGE if bs_vals[i] == 256 else ACCENT)
ax2b = ax.twinx()
ax2b.set_facecolor(PANEL_BG)
ax2b.plot(range(len(bs_vals)), kernel_ms, color=PURPLE, marker='o',
          linewidth=2, markersize=5)
ax2b.set_ylabel('Kernel ms', color=PURPLE)
ax2b.tick_params(axis='y', colors=PURPLE)
ax2b.spines['right'].set_color(PURPLE)

# ── Panel C: Convergence ──
ax = axes[1, 0]
ax.set_facecolor(PANEL_BG)
ax.loglog(n_vals, err_vals,  color=ORANGE, marker='o', linewidth=2, markersize=7, label='Error vs BS (%)')
ax.loglog(n_vals, stderr_v,  color=CYAN,   marker='s', linewidth=2, markersize=6, linestyle='--', label='Std Error ($)')
ax.loglog(n_ref, ref_y, color=GRID_CLR, linewidth=1.5, linestyle=':', label='O(1/√N)')
ax.set_title('MC Convergence')
ax.set_ylabel('Error / Std Error')
ax.set_xlabel('Num Paths')
ax.grid(True, which='both', zorder=0)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, fontsize=9)

# ── Panel D: Timing Breakdown ──
ax = axes[1, 1]
ax.set_facecolor(PANEL_BG)
xd = np.arange(len(n_labels))
ax.bar(xd, k_times,  0.5, color=GREEN,  label='Kernel', zorder=3)
ax.bar(xd, overhead, 0.5, bottom=k_times, color=PURPLE, alpha=0.8,
       label='Overhead', zorder=3)
ax.set_xticks(xd)
ax.set_xticklabels(n_labels)
ax.set_title('Timing: Kernel vs Overhead')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Paths')
ax.grid(axis='y', zorder=0)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, fontsize=9)
for i, (k, ov) in enumerate(zip(k_times, overhead)):
    ax.text(i, k + ov + 1, f'{k+ov:.1f}', ha='center', fontsize=8, color=TEXT_CLR)

fig.tight_layout()
savefig(fig, 'benchmarks.png')

print('\nAll charts saved to output/')
