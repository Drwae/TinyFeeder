#!/usr/bin/env python
"""Figures for a finished run: Pareto front, confusion matrix, training
curves, robustness / modality weights and sample inputs.

  python scripts/make_figures.py --run runs/demo_synthetic --data data/synthetic/avffia_synth.npz
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEVELS = ["None", "Weak", "Medium", "Strong"]
BLUE, RED, GRAY, GREEN = "#1f5fa8", "#c0392b", "#9aa5b1", "#2e8b57"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data")
ap.add_argument("--out", default=None)
a = ap.parse_args()
run = Path(a.run)
out = Path(a.out or run / "figures")
out.mkdir(parents=True, exist_ok=True)
J = lambda n: json.load(open(run / n))

# 1. NAS Pareto front ---------------------------------------------------------
s, sel = J("nas/search.json"), J("nas/selected_arch.json")
ev, pa = s["evaluated"], s["pareto"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.scatter([r["pred_latency_ms"] for r in ev], [100 * r["acc"] for r in ev], s=12, c=GRAY, alpha=0.6, label="evaluated")
pa_s = sorted(pa, key=lambda r: r["pred_latency_ms"])
ax.plot([r["pred_latency_ms"] for r in pa_s], [100 * r["acc"] for r in pa_s], "o-", c=BLUE, ms=4, lw=1.2, label="Pareto front (3 objectives, projected)")
ax.scatter([sel["pred_latency_ms"]], [100 * sel["acc"]], marker="*", s=220, c=RED, zorder=5, label="selected")
ax.set_xlabel("Predicted latency (ms)"), ax.set_ylabel("Proxy validation accuracy (%)")
ax.set_title(f"HW-NAS (NSGA-II): {len(ev)} architectures, {len(pa)} on front", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(), fig.savefig(out / "nas_pareto.png"), plt.close(fig)

# 2. Confusion matrix -----------------------------------------------------------
R = J("results.json")
cm = np.array(R["test"]["stage3_qat_kd"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.2, 3.6))
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}\n{100 * cmn[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xticks(range(4), LEVELS), ax.set_yticks(range(4), LEVELS)
ax.set_xlabel("Predicted"), ax.set_ylabel("Actual")
acc = R["test"]["stage3_qat_kd"]["accuracy"]
ax.set_title(f"TinyFeeder (QAT+KD), test accuracy {100 * acc:.1f}%", fontsize=10)
fig.tight_layout(), fig.savefig(out / "confusion_matrix.png"), plt.close(fig)

# 3. Training curves --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.4))
off = 0
for name, col in [("stage1", BLUE), ("stage2", GREEN), ("stage3", RED)]:
    h = J(f"{name}.json")["history"]
    x = np.arange(off + 1, off + len(h) + 1)
    ax.plot(x, [100 * e["val_acc"] for e in h], c=col, lw=1.5, label={"stage1": "Stage 1: FP32",
            "stage2": "Stage 2: mixed-precision QAT", "stage3": "Stage 3: multi-teacher KD"}[name])
    off += len(h)
ax.set_xlabel("Epoch (cumulative)"), ax.set_ylabel("Validation accuracy (%)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(), fig.savefig(out / "training_curves.png"), plt.close(fig)

# 4. Robustness and modality weight ---------------------------------------------
rob = R["robustness"]
keys = ["clean", "turbidity", "dark", "visual_off", "audio_off"]
lab = ["Clean", "Heavy\nturbidity", "Low light", "Camera\nfailure", "Hydrophone\nfailure"]
fig, ax1 = plt.subplots(figsize=(6, 3.4))
ax1.bar(range(5), [100 * rob[k]["accuracy"] for k in keys], color=[BLUE, GRAY, GRAY, GRAY, GRAY])
ax1.set_xticks(range(5), lab, fontsize=8), ax1.set_ylabel("Test accuracy (%)")
ax2 = ax1.twinx()
ax2.plot(range(5), [rob[k]["mean_alpha_v"] for k in keys], "o--", c=RED)
ax2.set_ylabel("Mean visual weight $\\alpha_v$", color=RED), ax2.set_ylim(0, 1)
ax2.spines["top"].set_visible(False)
fig.tight_layout(), fig.savefig(out / "robustness.png"), plt.close(fig)

# 5. Sample inputs ------------------------------------------------------------------
if a.data:
    d = np.load(a.data)
    lbl = d["labels"]
    fig, axs = plt.subplots(2, 4, figsize=(8, 4.2))
    rng = np.random.default_rng(3)
    for c in range(4):
        i = rng.choice(np.where(lbl == c)[0])
        axs[0, c].imshow(d["frames"][i]), axs[0, c].set_title(LEVELS[c], fontsize=10)
        axs[1, c].imshow(d["mels"][i].astype(float), origin="lower", aspect="auto", cmap="magma")
        for r in range(2):
            axs[r, c].set_xticks([]), axs[r, c].set_yticks([])
    axs[0, 0].set_ylabel("RGB frame"), axs[1, 0].set_ylabel("log-Mel")
    fig.suptitle("Synthetic AV-FFIA-like samples", fontsize=10)
    fig.tight_layout(), fig.savefig(out / "samples.png"), plt.close(fig)
print("figures written to", out)
