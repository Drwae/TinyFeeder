#!/usr/bin/env python
"""Build every results table of the article from ONE set of saved test
predictions (<run>/predictions/*.npz) plus the run's JSON logs.

  python scripts/paper_tables.py --run runs/avffia [--jetson runs/avffia/jetson_benchmark.json]

Writes <run>/tables.md (human readable) and <run>/tables.json (all numbers).
Statistics:
  * accuracy 95% CI      - cluster bootstrap over recording sessions (split units)
  * paired comparison    - exact McNemar test on discordant clips, Holm-adjusted
                           over all comparisons, plus a cluster-bootstrap 95% CI
                           of the accuracy difference (clips of one session are
                           resampled together, so within-session correlation is
                           respected)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyfeeder.engine import bootstrap_ci, classification_report  # noqa: E402

LEVELS = ["None", "Weak", "Medium", "Strong"]
FINAL = "tinyfeeder_final"


def load_preds(run):
    return {f.stem: dict(np.load(f)) for f in sorted((run / "predictions").glob("*.npz"))}


def cluster_diff_ci(y, p_a, p_b, units, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = (p_a == y).astype(float) - (p_b == y).astype(float)
    ug = np.unique(units)
    s = np.array([d[units == u].sum() for u in ug])
    c = np.array([(units == u).sum() for u in ug])
    boot = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(n))]
    return float(d.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


def fmt(x, nd=1):
    return f"{100 * x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jetson", help="JSON from deploy/parse_jetson.py (optional)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run = Path(a.run)
    P = load_preds(run)
    info = json.load(open(run / "model_info.json"))["models"]
    jet = json.load(open(a.jetson)) if a.jetson else {}
    T, md = {}, []

    # ---------------------------------------------------------------- data
    ds = json.load(open(run / "data_stats.json"))
    T["data"] = ds
    md += ["## Dataset split", "",
           f"Split unit: `{ds['split_unit']}`; overlap train/test = {ds['unit_overlap_train_test']}, "
           f"val/test = {ds['unit_overlap_val_test']}.", "",
           "| Split | Units | Clips | " + " | ".join(LEVELS) + " |", "|---|---:|---:|" + "---:|" * 4]
    for k in ("train", "val", "test"):
        md.append(f"| {k} | {ds[k]['units']} | {ds[k]['samples']} | " + " | ".join(map(str, ds[k]["per_level"])) + " |")

    # ---------------------------------------------------------------- main comparison
    ref = P[FINAL]
    y, units = ref["y"], ref["unit"]
    models = [k for k in P if not k.startswith("robust_")]
    rows = {}
    for k in models:
        r = classification_report(P[k]["y"], P[k]["p"])
        r["ci95"] = bootstrap_ci(P[k]["y"], P[k]["p"], n=a.boot, groups=P[k]["unit"])
        r.update(info.get(k, {}))
        r.update({f"jetson_{kk}": vv for kk, vv in jet.get(k, {}).items()})
        rows[k] = r
    T["models"] = rows
    md += ["", "## Test-set comparison (one set of predictions)", "",
           "| Model | Params | Size (MB) | MFLOPs | Acc. % [95% CI] | Macro-F1 | Weighted-F1 | CPU ms | Jetson ms | Jetson FPS |",
           "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for k, r in sorted(rows.items(), key=lambda kv: kv[1]["params"]):
        jl = r.get("jetson_latency_ms_median")
        md.append(f"| {k} | {r['params'] / 1e3:,.0f}K | {r['size_mb']:.2f} | {r['flops'] / 1e6:,.1f} | "
                  f"{fmt(r['accuracy'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | {r['macro_f1']:.3f} | "
                  f"{r['weighted_f1']:.3f} | {r['host_cpu_latency_ms']:.1f} | "
                  f"{'' if jl is None else f'{jl:.1f}'} | {'' if jl is None else f'{1000 / jl:.1f}'} |")
    md += ["", "Jetson FPS = 1000 / median single-clip latency (batch 1, no test-time augmentation)."]

    # ---------------------------------------------------------------- confusion matrix
    r = rows[FINAL]
    cm = np.array(r["confusion_matrix"])
    md += ["", "## Confusion matrix of the final TinyFeeder model", "",
           "| Actual \\ Predicted | " + " | ".join(LEVELS) + " | Total | Recall % |", "|---|" + "---:|" * 6]
    for i in range(4):
        md.append(f"| {LEVELS[i]} | " + " | ".join(f"{v:,}" for v in cm[i]) +
                  f" | {cm[i].sum():,} | {fmt(r['recall'][i])} |")
    md.append("| Precision % | " + " | ".join(fmt(v) for v in r["precision"]) + f" | {cm.sum():,} | |")
    md += ["", f"Correct = {int(np.trace(cm)):,} / {int(cm.sum()):,} = {fmt(r['accuracy'], 2)}%; "
               f"errors = {int(cm.sum() - np.trace(cm)):,}; macro-F1 = {r['macro_f1']:.3f}; "
               f"weighted-F1 = {r['weighted_f1']:.3f}; None<->Strong errors = {int(cm[0, 3] + cm[3, 0])}."]

    # ---------------------------------------------------------------- compression stages
    md += ["", "## Compression stages (same architecture after NAS; distillation does not change size)", "",
           "| Stage | Params | Size (MB) | Acc. % | Macro-F1 |", "|---|---:|---:|---:|---:|"]
    for k, nm in [("tinyfeeder_stage1_fp32", "1 NAS architecture, FP32"),
                  ("tinyfeeder_stage2_qat", "2 + mixed-precision QAT"), (FINAL, "3 + multi-teacher KD")]:
        if k in rows:
            md.append(f"| {nm} | {rows[k]['params'] / 1e3:,.0f}K | {rows[k]['size_mb']:.3f} | "
                      f"{fmt(rows[k]['accuracy'])} | {rows[k]['macro_f1']:.3f} |")

    # ---------------------------------------------------------------- statistics
    others = [k for k in models if k != FINAL]
    stats, pv = {}, []
    for k in others:
        assert np.array_equal(P[k]["y"], y), f"{k}: different test set"
        a_ok, b_ok = ref["p"] == y, P[k]["p"] == y
        b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        diff, ci = cluster_diff_ci(y, ref["p"], P[k]["p"], units, a.boot)
        stats[k] = {"tinyfeeder_only_correct": b, "other_only_correct": c, "p_mcnemar_exact": float(p),
                    "acc_diff": diff, "acc_diff_ci95_cluster": ci}
        pv.append(p)
    for k, adj in zip(others, holm(np.array(pv)) if pv else []):
        stats[k]["p_holm"] = float(adj)
        stats[k]["verdict"] = ("no significant difference" if adj >= 0.05 else
                               ("TinyFeeder better" if stats[k]["acc_diff"] > 0 else "TinyFeeder worse"))
    T["statistics"] = stats
    md += ["", "## Paired comparison with TinyFeeder (exact McNemar, Holm-adjusted; cluster-bootstrap CI)", "",
           "| Model | TF-only correct | Other-only correct | Δacc (TF − other) % [95% CI] | p (exact) | p (Holm) | Verdict |",
           "|---|---:|---:|---|---:|---:|---|"]
    for k, s in stats.items():
        md.append(f"| {k} | {s['tinyfeeder_only_correct']} | {s['other_only_correct']} | "
                  f"{fmt(s['acc_diff'], 2)} [{fmt(s['acc_diff_ci95_cluster'][0], 2)}, {fmt(s['acc_diff_ci95_cluster'][1], 2)}] | "
                  f"{s['p_mcnemar_exact']:.2g} | {s['p_holm']:.2g} | {s['verdict']} |")

    # ---------------------------------------------------------------- robustness
    rob = {"clean": {"accuracy": rows[FINAL]["accuracy"], "mean_alpha_v": float(ref["alpha_v"].mean())}}
    for k in P:
        if k.startswith("robust_"):
            rob[k[7:]] = {"accuracy": float((P[k]["y"] == P[k]["p"]).mean()),
                          "mean_alpha_v": float(P[k]["alpha_v"].mean())}
    T["robustness"] = rob
    md += ["", "## Robustness of the final model (synthetic input corruptions on the test split)", "",
           "| Condition | Acc. % | Mean visual gate weight α_v |", "|---|---:|---:|"]
    md += [f"| {k} | {fmt(v['accuracy'])} | {v['mean_alpha_v']:.2f} |" for k, v in rob.items()]
    md += ["", "α_v = mean(σ(g_v)) / (mean(σ(g_v)) + mean(σ(g_a))): a summary of the two independent sigmoid "
               "channel gates of the fusion module; the gates themselves are not constrained to sum to one."]

    # ---------------------------------------------------------------- NAS / QAT / KD settings
    nas = json.load(open(run / "nas/search.json"))
    s2 = json.load(open(run / "stage2.json"))
    s3 = json.load(open(run / "stage3.json"))
    T["nas"] = {"evaluated": len(nas["evaluated"]), "pareto": len(nas["pareto"]), "predictor": nas["predictor"],
                "proxy": nas.get("proxy"), "selected": json.load(open(run / "nas/selected_arch.json"))}
    T["qat"] = {"fp16_layers": s2["fp16_layers"], "n_layers": len(s2["precision_map"]),
                "size_mb": s2["size_mb"], "fp32_size_mb": s2["fp32_size_mb"], "ablation": s2["ablation"]}
    T["kd"] = {k: s3.get(k) for k in ("tau", "beta_max", "lambdas")}
    md += ["", "## Search, quantization and distillation settings actually used", "",
           f"- NAS: {T['nas']['evaluated']} architectures trained on the proxy task, Pareto front of "
           f"{T['nas']['pareto']}; proxy = {nas.get('proxy')}; latency predictor = {nas['predictor']}.",
           f"- QAT: {len(s2['fp16_layers'])} of {len(s2['precision_map'])} layers kept in FP16; size "
           f"{s2['size_mb']:.3f} MB vs {s2['fp32_size_mb']:.3f} MB FP32 ({s2['fp32_size_mb'] / s2['size_mb']:.1f}× smaller); "
           f"same-budget ablations: {s2['ablation']}.",
           f"- KD: τ = {T['kd']['tau']}, β_max = {T['kd']['beta_max']}, λ = {T['kd']['lambdas']}."]

    (run / "tables.md").write_text("\n".join(md) + "\n")
    json.dump(T, open(run / "tables.json", "w"), indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    main()
