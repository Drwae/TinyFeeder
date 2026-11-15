#!/usr/bin/env python
"""Render README.md from docs/README_template.md and a finished run."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo_synthetic")
zen = sys.argv[2] if len(sys.argv) > 2 else "ZENODO_ID"
R = json.load(open(run / "results.json"))
S2 = json.load(open(run / "stage2.json"))
EX = json.load(open(run / "export.json"))
NAS = json.load(open(run / "nas/search.json"))
SEL = json.load(open(run / "nas/selected_arch.json"))
DS = json.load(open(run / "data_stats.json"))
t = R["test"]

pct = lambda x: f"{100 * x:.1f}"
rows = [
    "| Model | Params | Size (MB) | Test acc. (%) | Macro-F1 | Weighted-F1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for k, v in t.items():
    if k.startswith("teacher_"):
        rows.append(f"| Teacher: {k[8:]} | {v['params'] / 1e6:.2f}M | {4 * v['params'] / 1e6:.1f} | {pct(v['accuracy'])} | "
                    f"{v['macro_f1']:.3f} | {v['weighted_f1']:.3f} |")
names = {"stage1_fp32": "TinyFeeder, Stage 1 (FP32, NAS arch.)", "stage2_qat": "TinyFeeder, Stage 2 (+ mixed-precision QAT)",
         "stage3_qat_kd": "**TinyFeeder, Stage 3 (+ multi-teacher KD)**"}
P = R["params"]["total"]
for k, nm in names.items():
    v = t[k]
    rows.append(f"| {nm} | {P / 1e3:.0f}K | {v['size_mb']:.2f} | {pct(v['accuracy'])} | {v['macro_f1']:.3f} | {v['weighted_f1']:.3f} |")
ci = R["final_ci95_accuracy"]
rob = R["robustness"]
lines = rows + [
    "",
    f"- Data: {DS['train']['videos']}/{DS['val']['videos']}/{DS['test']['videos']} videos (train/val/test), "
    f"{DS['test']['samples']} test samples; video overlap between train and test: {DS['video_overlap_train_test']}.",
    f"- HW-NAS: {len(NAS['evaluated'])} architectures evaluated, {len(NAS['pareto'])} on the Pareto front; latency predictor "
    f"({NAS['predictor']['regressor']}, {NAS['predictor']['n_profiled']} profiled architectures) hold-out MAPE "
    f"{NAS['predictor']['holdout_mape_pct']:.1f}%.",
    f"- Selected architecture: {P / 1e3:.0f}K parameters ({R['params']['visual'] / 1e3:.0f}K visual, "
    f"{R['params']['acoustic'] / 1e3:.0f}K acoustic, {R['params']['fusion'] / 1e3:.0f}K fusion), {R['flops'] / 1e6:.1f} MFLOPs at 96 px.",
    f"- Mixed precision: {len(S2['fp16_layers'])} of {len(S2['precision_map'])} layers in FP16; deployment size "
    f"{S2['size_mb']:.2f} MB vs {S2['fp32_size_mb']:.2f} MB FP32. Same-budget ablations (val. acc.): "
    + ", ".join(f"{k.replace('_', ' ')} {pct(v['val_acc'])}%" for k, v in S2["ablation"].items()) + ".",
    f"- Final model: test accuracy {pct(t['stage3_qat_kd']['accuracy'])}% (bootstrap 95% CI {pct(ci[0])}–{pct(ci[1])}%); "
    f"ONNX Runtime CPU latency {EX['onnxruntime_cpu_latency_ms']['median']:.2f} ms (median, host CPU, batch 1).",
    f"- Modality reliance: mean visual weight α_v {rob['clean']['mean_alpha_v']:.2f} on clean inputs, "
    f"{rob['turbidity']['mean_alpha_v']:.2f} under added turbidity; accuracy with camera failure "
    f"{pct(rob['visual_off']['accuracy'])}%, with hydrophone failure {pct(rob['audio_off']['accuracy'])}%.",
]
tpl = open("docs/README_template.md").read()
open("README.md", "w").write(tpl.replace("RESULTS_TABLE", "\n".join(lines)).replace("ZENODO_ID", zen))
print("README.md written")
