"""K-fold cross-validation over recording sessions (default 5 folds) for the paper's final numbers.

Every fold holds out a different 1/K of the sessions as its test set, so each session is tested exactly once. The
teachers and the pretrained MobileNetV3 baseline are retrained in every fold (a teacher trained on a session that is
another fold's test session would leak it into the student through distillation). The student is then trained in
several variants on the same fold, sharing the fold's teachers:

  full    the final design (pretrained tiny backbone, modality dropout, mixed-precision QAT, KD)
  no_kd   the same without distillation (KD weight 0 in stage 1 and stage 3)
  no_md   the same without modality dropout

Runs are resumable: finished stages are skipped, so rerun the same command after an interruption.

  python scripts/cv_sessions.py --config configs/avffia_md.yaml --device cuda --workers 8
  python scripts/cv_sessions.py --config configs/avffia_md.yaml --summary-only     # tables from finished folds

Writes runs/cv/fold<k>/{shared,full,no_kd,no_md}/ and runs/cv/summary.{json,md}."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyfeeder.data.dataset import load_npz, unit_key  # noqa: E402

VARIANTS = ("full", "no_kd", "no_md")
STUDENT = ("stage1", "stage2", "stage3")


def folds(data, k, seed):
    """Sessions dealt to k folds, stratified by each session's dominant feeding level."""
    key = unit_key(data)
    units, labels = np.asarray(data[key]), np.asarray(data["labels"])
    rng = np.random.default_rng(seed)
    dom = {int(u): int(np.bincount(labels[units == u], minlength=4).argmax()) for u in np.unique(units)}
    order = []
    for lvl in sorted(set(dom.values())):
        us = [u for u, d in dom.items() if d == lvl]
        rng.shuffle(us)
        order += us
    out = [[] for _ in range(k)]
    for i, u in enumerate(order):
        out[i % k].append(u)
    return out


def split_for(fs, f, n_val):
    """Fold f is the test set; n_val sessions of the next fold (stratified order) validate; the rest train."""
    test = fs[f]
    nxt = fs[(f + 1) % len(fs)]
    val = nxt[:n_val]
    train = [u for i, x in enumerate(fs) if i != f for u in x if u not in val]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def variant_cfg(base, variant, out):
    c = copy.deepcopy(base)
    c["out_dir"] = str(out)
    c["baselines"]["backbones"] = base.get("cv_baselines", ["mobilenet_v3_large"])
    c.pop("baselines_scratch", None)
    if variant == "no_kd":
        c["stage1"]["kd"] = False
        c["stage3"]["beta_max"] = 0.0
    if variant == "no_md":
        for s in STUDENT:
            c[s]["modality_dropout"] = 0.0
    return c


def run(cfg, stages, args, tmp):
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = [sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--config", str(tmp), "--device", args.device,
           "--workers", str(args.workers), "--stages", *stages]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def link(src: Path, dst: Path):
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def metrics(npz):
    d = np.load(npz, allow_pickle=True)
    y, p = d["y"], d["p"]
    f1 = []
    for k in range(4):
        tp = ((p == k) & (y == k)).sum()
        pr, rc = tp / max((p == k).sum(), 1), tp / max((y == k).sum(), 1)
        f1.append(2 * pr * rc / max(pr + rc, 1e-9))
    return {"acc": float((y == p).mean()), "macro_f1": float(np.mean(f1)), "n": int(len(y)), "y": y, "p": p}


def summary(out: Path, k: int):
    names = ["tinyfeeder_final", "tinyfeeder_stage2_qat", "tinyfeeder_stage1_fp32", "robust_audio_off",
             "robust_visual_off", "robust_dark", "robust_turbidity"]
    rows = {}
    for v in VARIANTS:
        for n in names:
            rows[f"{v}/{n}"] = [out / f"fold{f}" / v / "predictions" / f"{n}.npz" for f in range(k)]
    for n in ("baseline_mobilenet_v3_large", "teacher_resnest50d", "teacher_swin_tiny_patch4_window7_224",
              "teacher_convnext_base"):
        rows[n] = [out / f"fold{f}" / "full" / "predictions" / f"{n}.npz" for f in range(k)]
    res, lines = {}, ["| model | folds | accuracy mean ± sd | macro-F1 mean ± sd | pooled accuracy |",
                      "|---|---|---|---|---|"]
    for name, paths in rows.items():
        ms = [metrics(p) for p in paths if p.exists()]
        if not ms:
            continue
        acc, f1 = np.array([m["acc"] for m in ms]), np.array([m["macro_f1"] for m in ms])
        pooled = float(np.mean(np.concatenate([m["y"] == m["p"] for m in ms])))
        res[name] = {"folds": len(ms), "acc_per_fold": acc.round(4).tolist(), "acc_mean": float(acc.mean()),
                     "acc_sd": float(acc.std(ddof=1)) if len(ms) > 1 else 0.0, "macro_f1_mean": float(f1.mean()),
                     "macro_f1_sd": float(f1.std(ddof=1)) if len(ms) > 1 else 0.0, "pooled_acc": pooled}
        r = res[name]
        lines.append(f"| {name} | {len(ms)} | {100 * r['acc_mean']:.1f} ± {100 * r['acc_sd']:.1f} | "
                     f"{100 * r['macro_f1_mean']:.1f} ± {100 * r['macro_f1_sd']:.1f} | {100 * pooled:.1f} |")
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", type=int, nargs="*", help="run only these fold indices")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--val-sessions", type=int, default=6)
    ap.add_argument("--out", default="runs/cv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        return summary(out, args.folds)

    data = load_npz(ROOT / base["data"]["path"])
    fs = folds(data, args.folds, base["seed"])
    (out / "folds.json").write_text(json.dumps(fs, indent=2))
    for f in (args.only if args.only is not None else range(args.folds)):
        split = split_for(fs, f, args.val_sessions)
        shared = out / f"fold{f}" / "shared"
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "splits.json").write_text(json.dumps(split, indent=2))
        print(f"== fold {f}: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} sessions",
              flush=True)
        run(variant_cfg(base, "full", shared), ["teachers", "baselines"], args, out / f"fold{f}" / "shared.yaml")
        for v in args.variants:
            d = out / f"fold{f}" / v
            d.mkdir(parents=True, exist_ok=True)
            (d / "splits.json").write_text(json.dumps(split, indent=2))
            for name in ("teachers", "teachers.json", "baselines", "baselines.json"):
                link(shared / name, d / name)
            run(variant_cfg(base, v, d), ["stage1", "stage2", "stage3", "evaluate"], args, out / f"fold{f}" / f"{v}.yaml")
        summary(out, args.folds)


if __name__ == "__main__":
    main()
