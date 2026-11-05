# -*- coding: utf-8 -*-
"""Produce Fig 3 to Fig 9 of the paper from figures/data/paper_data.json.

  python figures/build_data.py && python figures/make_figures.py

Covers two models (Mixtral-8x7B, 8x22B) and four systems (Fiddlestick,
Fiddler, DAOP, vLLM). Phi-3.5-MoE and KTransformers are not part of the paper.
"""
import json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "out")
os.makedirs(OUT, exist_ok=True)
P = json.load(open(os.path.join(ROOT, "figures", "data", "paper_data.json")))
D = {"exp1": P["throughput"], "exp2": P["beam"], "exp4": P["budget"],
     "heldout": P["heldout"]}
PF = P["latency"]
HT = P["hit"]
RESID = {m: P["residency"][m] for m in P["residency"]}
CELLS = [f"{i}x{o}" for i in (16, 32, 64) for o in (2, 4, 8, 16)]
MS = ("x7b", "c22")
MN = {"x7b": "Mixtral-8x7B", "c22": "Mixtral-8x22B"}
SYS = [("Fiddlestick", "#4c72b0"), ("Fiddler", "#8c8c8c"),
       ("DAOP", "#c44e52"), ("vLLM", "#8172b3")]

def g(a, b): return 100 * (st.mean([a[c] / b[c] for c in CELLS if c in a and c in b]) - 1)
def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print(p)

# ===== Fig3 — Experiment 1: throughput =====
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
for ax, m in zip(axes, MS):
    e = D["exp1"][m]
    vals = [(n, e[n]["mean"], c) for n, c in SYS if e.get(n)]
    hi = max(v for _, v, _ in vals)
    for i, (n, v, c) in enumerate(vals):
        ax.bar(i, v, .62, color=c)
        ax.text(i, v + hi * .03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([n for n, _, _ in vals], fontsize=9)
    ax.set_title(MN[m], fontsize=11); ax.set_ylim(0, hi * 1.18)
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
axes[0].set_ylabel("Throughput (tokens/s)")
save(fig, "Fig3.png")

# ===== Fig4 — Experiment 2: absolute prefill / decode latency =====
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
x = np.arange(2); w = .34
for ax, key, ylab, ttl in (
        (axes[0], "prefill", "Prefill latency (s)", "(a) Prefill"),
        (axes[1], "decode_tok", "Decode latency per token (ms)", "(b) Decode")):
    sc = 1000 if key == "decode_tok" else 1
    fs = [PF[m]["Fiddlestick"][key] * sc for m in MS]
    fd = [PF[m]["Fiddler"][key] * sc for m in MS]
    ax.bar(x - w/2, fs, w, color="#4c72b0", label="Fiddlestick")
    ax.bar(x + w/2, fd, w, color="#8c8c8c", label="Fiddler")
    hi = max(fd + fs)
    for i, (b, a) in enumerate(zip(fs, fd)):
        ax.text(i - w/2, b + hi*.025, f"{b:.2f}" if sc == 1 else f"{b:.0f}",
                ha="center", fontsize=9)
        ax.text(i + w/2, a + hi*.025, f"{a:.2f}" if sc == 1 else f"{a:.0f}",
                ha="center", fontsize=9)
        ax.text(i, max(a, b) + hi*.115, f"{100*(b/a-1):+.1f}%", ha="center",
                fontsize=9.5, fontweight="bold",
                color="#2a6a3a" if b < a else "#8a2a2e")
    ax.set_xticks(x); ax.set_xticklabels([MN[m] for m in MS], fontsize=9)
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11)
    ax.set_ylim(0, hi * 1.34); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", framealpha=.95)
save(fig, "Fig4.png")

# ===== Fig5 — Experiment 3: measured minimum GPU residency =====
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(2); w = .19
RSYS = [(n, c) for n, c in SYS if n in RESID["x7b"]]
w = .22
for j, (n, c) in enumerate(RSYS):
    v = [RESID[m][n] / 1024 for m in MS]
    ax.bar(x + (j - (len(RSYS) - 1) / 2) * w, v, w, color=c, label=n)
    for i, val in enumerate(v):
        ax.text(i + (j - (len(RSYS) - 1) / 2) * w, val + .25, f"{val:.2f}",
                ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([MN[m] for m in MS], fontsize=10)
ax.set_ylabel("Minimum GPU residency (GiB)"); ax.set_ylim(0, 15.5)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
save(fig, "Fig5.png")

# ===== Fig6 — Experiment 4: throughput vs VRAM budget =====
GB = [8, 16, 24, 32, 40]
MK = {"Fiddler": "o", "Fiddlestick": "o", "DAOP": "s", "vLLM": "^"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
for ax, m in zip(axes, MS):
    for n, c in SYS:
        xs, ys = [], []
        for gb in GB:
            e = D["exp4"].get(f"{m}|{gb}", {}).get(n)
            if e: xs.append(gb); ys.append(e["mean"])
        if xs: ax.plot(xs, ys, MK[n] + "-", ms=5.5, lw=1.9, color=c, label=n)
    # A line starting at 16 GB means the system could not run below that budget;
# say so in one note rather than shading the region.
    notes = []
    for gb in GB:
        e = D["exp4"].get(f"{m}|{gb}", {})
        if not e.get("Fiddlestick"):
            continue
        missing = [n for n, _ in SYS if n != "Fiddlestick" and not e.get(n)]
        if missing:
            notes.append(f"{', '.join(missing)}: out of memory at {gb} GB")
    if notes:
        ax.text(.03, .97, "\n".join(notes), transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="#c44e52")
    ax.set_xticks(GB); ax.set_xlabel("VRAM budget (GB)")
    ax.set_title(MN[m], fontsize=11); ax.grid(alpha=.3); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="lower right" if m == "c22" else "upper left")
axes[0].set_ylabel("Throughput (tokens/s)")
save(fig, "Fig6.png")

# ===== Fig7 — beam width =====
fig, ax = plt.subplots(figsize=(6.4, 4.2))
BE = [1, 2, 4, 8]
for m, c, mk, dy in (("x7b", "#4c72b0", "o", -15), ("c22", "#dd8452", "s", 11)):
    ys = []
    for b in BE:
        e = D["exp2"].get(f"{m}|{b}", {})
        ys.append(g(e["Fiddlestick"]["cells"], e["Fiddler"]["cells"]))
    ax.plot(BE, ys, mk + "-", ms=8, lw=2.2, color=c, label=MN[m])
    for b, y in zip(BE, ys):
        ha = "right" if b == BE[-1] else "center"
        dx = -8 if b == BE[-1] else 0
        ax.annotate(f"{y:+.1f}%", (b, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=c)
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log", base=2); ax.set_xticks(BE); ax.set_xticklabels(BE)
ax.set_xlabel("Beam width"); ax.set_ylabel("Throughput gain over Fiddler (%)")
ax.set_ylim(0, 30); ax.set_xlim(0.88, 9.6); ax.grid(alpha=.3); ax.set_axisbelow(True); ax.legend(fontsize=10, loc="lower right")
save(fig, "Fig7.png")

# ===== Fig8 — cache hit rate (immediately before held-out) =====
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(2); w = .25
TOT = {"x7b": 256, "c22": 448}
for j, (n, c) in enumerate((("Fiddlestick", "#4c72b0"), ("Fiddler", "#8c8c8c"),
                            ("DAOP", "#c44e52"))):
    k = n
    h = [HT[m]["ShareGPT"][k]["hit"] for m in MS]
    cv = [HT[m]["ShareGPT"][k]["coverage"] for m in MS]
    ne = [HT[m]["ShareGPT"][k]["nexp"] for m in MS]
    ax.bar(x + (j - 1) * w, h, w, color=c, label=n)
    for i, (v, cc, nn) in enumerate(zip(h, cv, ne)):
        ax.text(i + (j - 1) * w, v + .006, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(i + (j - 1) * w, .006, f"{nn}", ha="center", fontsize=7, color="white")
        ax.hlines(cc, i + (j - 1) * w - w/2, i + (j - 1) * w + w/2,
                  color="#111", ls=":", lw=1.3)
ax.plot([], [], ls=":", color="#111", lw=1.3, label="Coverage (random-placement baseline)")
ax.set_xticks(x); ax.set_xticklabels([MN[m] for m in MS], fontsize=10)
ax.set_ylabel("Expert cache hit rate"); ax.set_ylim(0, .25)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
save(fig, "Fig8.png")

# ===== Fig9 — held-out validation: hit rate, per model =====
SYS9 = (("Fiddlestick", "#4c72b0"), ("Fiddler", "#8c8c8c"), ("DAOP", "#c44e52"))
KEY = {n: n for n, _ in SYS9}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, m, tag in zip(axes, MS, "ab"):
    x = np.arange(len(SYS9)); w = .34
    sg = [HT[m]["ShareGPT"][KEY[n]]["hit"] for n, _ in SYS9]
    c4 = [HT[m]["C4"][KEY[n]]["hit"] for n, _ in SYS9]
    cols = [c for _, c in SYS9]
    ax.bar(x - w/2, sg, w, color=cols)
    ax.bar(x + w/2, c4, w, color=cols, hatch="///", edgecolor="white", linewidth=0)
    hi = max(sg)
    for i2, (a, b) in enumerate(zip(sg, c4)):
        ax.text(i2 - w/2, a + hi*.018, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.text(i2 + w/2, b + hi*.018, f"{b:.3f}", ha="center", fontsize=8.5)
        ax.text(i2, max(a, b) + hi*.095, f"{100*(b/a-1):+.0f}%", ha="center",
                fontsize=9.5, fontweight="bold", color="#8a2a2e")
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in SYS9], fontsize=9.5)
    ax.set_ylim(0, hi * 1.25); ax.set_title(f"({tag}) {MN[m]}", fontsize=11)
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
axes[0].set_ylabel("Expert cache hit rate")
import matplotlib.patches as mpatches
axes[0].legend(handles=[
    mpatches.Patch(facecolor="#777", label="ShareGPT (calibration)"),
    mpatches.Patch(facecolor="#777", hatch="///", edgecolor="white",
                   label="C4 (held-out)"),
], fontsize=9, loc="upper right")
fig.tight_layout()
save(fig, "Fig9.png")
