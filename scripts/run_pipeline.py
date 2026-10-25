#!/usr/bin/env python
"""End-to-end TinyFeeder pipeline.

  data -> teachers -> nas -> stage1 (FP32) -> stage2 (mixed-precision QAT)
       -> stage3 (multi-teacher KD) -> evaluate -> export

Usage:
  python scripts/run_pipeline.py --config configs/demo_synthetic.yaml
  python scripts/run_pipeline.py --config configs/demo_synthetic.yaml --stages evaluate export
Every stage writes its artefacts to <out_dir>/ and is skipped if they exist
(use --force to recompute).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyfeeder.data import AVFeedingDataset, SynthConfig, generate, load_npz, video_level_split  # noqa: E402
from tinyfeeder.data.dataset import save_split  # noqa: E402
from tinyfeeder.distill import MultiTeacherKD, beta_schedule  # noqa: E402
from tinyfeeder.engine import (bootstrap_ci, classification_report, count_flops, evaluate, fit,  # noqa: E402
                               measure_latency, predict)
from tinyfeeder.models import Teacher, TinyFeeder, param_breakdown  # noqa: E402
from tinyfeeder.nas import nsga2, profile_and_fit, select  # noqa: E402
from tinyfeeder.quant import model_size_mb, precision_map, prepare_qat, quantizable_layers, strip_qat  # noqa: E402

STAGES = ["data", "teachers", "nas", "stage1", "stage2", "stage3", "evaluate", "export"]


def seed_all(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s)


class Ctx:
    def __init__(self, cfg, force):
        self.cfg, self.force = cfg, force
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "pipeline.log", "a")
        self._data = None

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.logf.write(msg + "\n"), self.logf.flush()

    def done(self, *names):
        return not self.force and all((self.out / n).exists() for n in names)

    def save_json(self, obj, name):
        with open(self.out / name, "w") as f:
            json.dump(obj, f, indent=2)

    def load_json(self, name):
        return json.load(open(self.out / name))

    # data ---------------------------------------------------------------
    @property
    def data(self):
        if self._data is None:
            self._data = load_npz(self.cfg["data"]["path"])
        return self._data

    @property
    def split(self):
        return self.load_json("splits.json")

    def ds(self, part, train=False, corrupt=None, videos=None):
        return AVFeedingDataset(self.data, videos if videos is not None else self.split[part], train, corrupt)

    @property
    def sizes(self):
        return self.cfg["data"]["img_size"], self.cfg["data"]["n_mels"]


# ---------------------------------------------------------------- stages
def st_data(c: Ctx):
    d = c.cfg["data"]
    p = Path(d["path"])
    if not p.exists() or c.force:
        if not d.get("synthetic"):
            raise SystemExit(f"{p} not found. Prepare the real dataset or use the synthetic config.")
        c.log(f"[data] generating synthetic AV-FFIA-like dataset -> {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = generate(SynthConfig(n_videos=d["n_videos"], samples_per_video=d["samples_per_video"],
                                   img_size=d["img_size"], n_mels=d["n_mels"], seed=c.cfg["seed"]))
        np.savez_compressed(p, **arr)
    data = c.data
    split = video_level_split(data["video_id"], data["labels"], tuple(d["split_fracs"]), c.cfg["seed"])
    save_split(split, c.out / "splits.json")
    stats = {}
    for k, v in split.items():
        m = np.isin(data["video_id"], v)
        stats[k] = {"videos": len(v), "samples": int(m.sum()),
                    "per_level": np.bincount(data["labels"][m], minlength=4).tolist()}
    stats["video_overlap_train_test"] = len(set(split["train"]) & set(split["test"]))
    c.save_json(stats, "data_stats.json")
    c.log("[data] video-level split:", json.dumps(stats))


def st_teachers(c: Ctx):
    t = c.cfg["teachers"]
    tdir = c.out / "teachers"
    tdir.mkdir(exist_ok=True)
    res = {}
    for name in t["backbones"]:
        f = tdir / f"{name}.pt"
        if f.exists() and not c.force:
            continue
        seed_all(c.cfg["seed"])
        c.log(f"[teachers] training {name} (training split only)")
        m = Teacher(name, use_timm=t.get("use_timm", False))
        m, hist = fit(m, c.ds("train", True), c.ds("val"), t["epochs"], t["lr"], t["batch_size"], log=c.log)
        torch.save(m.state_dict(), f)
        res[name] = {"val": evaluate(m, DataLoader(c.ds("val"), batch_size=256)), "history": hist}
    if res:
        old = c.load_json("teachers.json") if (c.out / "teachers.json").exists() else {}
        old.update(res)
        c.save_json(old, "teachers.json")


def load_teachers(c: Ctx):
    t = c.cfg["teachers"]
    ts = []
    for name in t["backbones"]:
        m = Teacher(name, use_timm=t.get("use_timm", False))
        m.load_state_dict(torch.load(c.out / "teachers" / f"{name}.pt"))
        ts.append(m.eval())
    return ts


def st_nas(c: Ctx):
    if c.done("nas/selected_arch.json"):
        return
    n = c.cfg["nas"]
    (c.out / "nas").mkdir(exist_ok=True)
    img, mel = c.sizes
    c.log(f"[nas] profiling {n['profile_archs']} architectures for the latency predictor")
    pred, pstats = profile_and_fit(n["profile_archs"], img, mel, c.cfg["seed"], c.log)
    c.log(f"[nas] latency predictor: {pstats}")
    tr_videos = c.split["train"]
    if n.get("proxy_train_videos"):
        tr_videos = random.Random(0).sample(tr_videos, n["proxy_train_videos"])
    proxy_tr, proxy_va = c.ds(None, True, videos=tr_videos), c.ds("val")

    def proxy_acc(arch):
        seed_all(c.cfg["seed"])
        m = TinyFeeder(visual_genes=arch)
        m, _ = fit(m, proxy_tr, proxy_va, n["proxy_epochs"], 3e-3, 64, log=lambda *a: None)
        return evaluate(m, DataLoader(proxy_va, batch_size=256))["accuracy"]

    c.log(f"[nas] NSGA-II: population {n['population']}, generations {n['generations']}")
    allr, pareto, hist = nsga2(proxy_acc, pred, n["population"], n["generations"], img, mel,
                               c.cfg["seed"], c.log, n.get("patience", 10))
    best = select(pareto, n["latency_budget_ms"], n["params_budget"])
    c.save_json({"predictor": pstats, "history": hist, "evaluated": allr, "pareto": pareto}, "nas/search.json")
    c.save_json(best, "nas/selected_arch.json")
    c.log(f"[nas] {len(allr)} evaluated, Pareto front {len(pareto)}; selected acc {best['acc']:.4f}, "
          f"pred latency {best['pred_latency_ms']:.2f} ms, params {best['params']}")


def selected_arch(c):
    return c.load_json("nas/selected_arch.json")["arch"]


def st_stage1(c: Ctx):
    if c.done("stage1.pt"):
        return
    s = c.cfg["stage1"]
    seed_all(c.cfg["seed"])
    m = TinyFeeder(visual_genes=selected_arch(c))
    c.log(f"[stage1] FP32 training of the NAS architecture: {param_breakdown(m)}")
    m, hist = fit(m, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], s["batch_size"], log=c.log)
    torch.save(m.state_dict(), c.out / "stage1.pt")
    c.save_json({"history": hist}, "stage1.json")


def st_stage2(c: Ctx):
    if c.done("stage2.pt", "stage2.json"):
        return
    s = c.cfg["stage2"]
    arch = selected_arch(c)
    base = TinyFeeder(visual_genes=arch)
    base.load_state_dict(torch.load(c.out / "stage1.pt"))
    vids = c.split["val"]
    if s.get("sensitivity_val_videos"):
        vids = random.Random(1).sample(vids, s["sensitivity_val_videos"])
    vl = DataLoader(c.ds(None, videos=vids), batch_size=256)
    # calibrate observers on training data, then measure per-layer INT8 drop
    tl = DataLoader(c.ds("train"), batch_size=128, shuffle=True)
    acc_fp32 = evaluate(base, vl)["accuracy"]
    sens = {}
    c.log(f"[stage2] layer-wise sensitivity analysis over {len(quantizable_layers(base))} layers (FP32 acc {acc_fp32:.4f})")
    for name in quantizable_layers(base):
        torch.manual_seed(c.cfg["seed"])  # identical calibration batches for every layer
        m = prepare_qat(copy.deepcopy(base), only=name)
        m.train()
        with torch.no_grad():
            for b, (x, a, _) in enumerate(tl):
                m(x, a)
                if b >= 3:
                    break
        sens[name] = acc_fp32 - evaluate(m, vl)["accuracy"]
    ranked = sorted(sens, key=lambda k: -sens[k])
    k = max(1, int(round(s["fp16_fraction"] * len(ranked))))
    fp16 = ranked[:k]
    c.log(f"[stage2] FP16 layers (top {k}): {fp16}")
    q = prepare_qat(copy.deepcopy(base), fp16_layers=set(fp16))
    seed_all(c.cfg["seed"])
    q, hist = fit(q, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], 64, log=c.log)
    torch.save(q.state_dict(), c.out / "stage2.pt")
    # ablation: per-tensor INT8 and uniform INT8 (no FP16 layers) after the same QAT budget
    abl = {}
    for tag, kw in {"int8_per_channel": dict(fp16_layers=set()),
                    "int8_per_tensor": dict(fp16_layers=set(), per_channel=False)}.items():
        seed_all(c.cfg["seed"])
        mq = prepare_qat(copy.deepcopy(base), **kw)
        mq, _ = fit(mq, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], 64, log=lambda *a: None)
        abl[tag] = {"val_acc": evaluate(mq, DataLoader(c.ds("val"), batch_size=256))["accuracy"],
                    "size_mb": model_size_mb(mq)}
        c.log(f"[stage2] ablation {tag}: {abl[tag]}")
    c.save_json({"fp32_val_acc": acc_fp32, "sensitivity": sens, "fp16_layers": fp16,
                 "precision_map": precision_map(q), "size_mb": model_size_mb(q),
                 "fp32_size_mb": sum(p.numel() for p in base.parameters()) * 4 / 1e6,
                 "ablation": abl, "history": hist}, "stage2.json")


def quantized_student(c):
    st2 = c.load_json("stage2.json")
    m = prepare_qat(TinyFeeder(visual_genes=selected_arch(c)), fp16_layers=set(st2["fp16_layers"]))
    return m


def st_stage3(c: Ctx):
    if c.done("tinyfeeder_final.pt"):
        return
    s = c.cfg["stage3"]
    student = quantized_student(c)
    student.load_state_dict(torch.load(c.out / "stage2.pt"))
    teachers = load_teachers(c)
    kd = MultiTeacherKD(student.visual.out_dim + student.acoustic.out_dim, [t.feat_dim for t in teachers], s["tau"], tuple(s["lambdas"]))
    seed_all(c.cfg["seed"])
    c.log(f"[stage3] multi-teacher KD from {[t.name for t in teachers]}")
    student, hist = fit(student, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], 64, log=c.log,
                        teachers=teachers, kd=kd, extra_params=kd.parameters(),
                        beta_fn=lambda e: beta_schedule(e, s["warmup"], s["epochs"] - s["warmup"], s["beta_max"]))
    torch.save(student.state_dict(), c.out / "tinyfeeder_final.pt")
    c.save_json({"history": hist}, "stage3.json")


def st_evaluate(c: Ctx):
    img, mel = c.sizes
    te = DataLoader(c.ds("test"), batch_size=256)
    res = {"note": "Synthetic data - not comparable with the paper's farm-data results."
           if c.cfg["data"].get("synthetic") else "AV-FFIA test split"}
    arch = selected_arch(c)
    fp32 = TinyFeeder(visual_genes=arch)
    fp32.load_state_dict(torch.load(c.out / "stage1.pt"))
    qat = quantized_student(c)
    qat.load_state_dict(torch.load(c.out / "stage2.pt"))
    final = quantized_student(c)
    final.load_state_dict(torch.load(c.out / "tinyfeeder_final.pt"))
    rows = {}
    for tag, m in {"stage1_fp32": fp32, "stage2_qat": qat, "stage3_qat_kd": final}.items():
        rows[tag] = evaluate(m, te)
        rows[tag]["size_mb"] = model_size_mb(m) if tag != "stage1_fp32" else \
            sum(p.numel() for p in m.parameters()) * 4 / 1e6
    for t in load_teachers(c):
        rows[f"teacher_{t.name}"] = evaluate(t, te)
        rows[f"teacher_{t.name}"]["params"] = sum(p.numel() for p in t.parameters())
    res["test"] = rows
    y, p, pr, alpha = predict(final, te, return_alpha=True)
    res["final_ci95_accuracy"] = bootstrap_ci(y, p)
    res["params"] = param_breakdown(fp32)
    res["flops"] = count_flops(fp32, img, mel)
    res["host_cpu_latency_ms"] = measure_latency(strip_qat(copy.deepcopy(final)), img, mel)
    # robustness / modality reliance
    rob = {}
    for cor in ["turbidity", "dark", "visual_off", "audio_off"]:
        yl, pl, _, al = predict(final, DataLoader(c.ds("test", corrupt=cor), batch_size=256), return_alpha=True)
        rob[cor] = {"accuracy": float((yl == pl).mean()), "mean_alpha_v": float(al.mean())}
    rob["clean"] = {"accuracy": float((y == p).mean()), "mean_alpha_v": float(alpha.mean())}
    res["robustness"] = rob
    # accuracy by natural turbidity tercile of the session
    d = c.data
    idx = AVFeedingDataset(d, c.split["test"]).idx
    turb = d["turbidity"][idx]
    q = np.quantile(turb, [1 / 3, 2 / 3])
    res["by_turbidity"] = {nm: {"accuracy": float((y[m] == p[m]).mean()), "mean_alpha_v": float(alpha[m].mean()),
                                "n": int(m.sum())}
                           for nm, m in {"low": turb <= q[0], "mid": (turb > q[0]) & (turb <= q[1]),
                                         "high": turb > q[1]}.items()}
    c.save_json(res, "results.json")
    np.savez(c.out / "test_predictions.npz", y=y, p=p, prob=pr, alpha_v=alpha)
    c.log("[evaluate]", json.dumps({k: {"acc": round(v["accuracy"], 4), "macroF1": round(v["macro_f1"], 4)}
                                    for k, v in rows.items()}))
    c.log("[evaluate] robustness", json.dumps(rob))


def st_export(c: Ctx):
    img, mel = c.sizes
    final = quantized_student(c)
    final.load_state_dict(torch.load(c.out / "tinyfeeder_final.pt"))
    m = strip_qat(final).eval()
    x, a = torch.randn(1, 3, img, img), torch.randn(1, 1, mel, mel)
    path = c.out / "tinyfeeder.onnx"
    torch.onnx.export(m, (x, a), path, input_names=["frame", "mel"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"frame": {0: "batch"}, "mel": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    out = {"onnx": str(path), "onnx_mb": os.path.getsize(path) / 1e6}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {"frame": x.numpy(), "mel": a.numpy()}
        ref = m(x, a).detach().numpy()
        o = sess.run(None, feed)[0]
        out["max_abs_diff_vs_torch"] = float(np.abs(o - ref).max())
        for _ in range(20):
            sess.run(None, feed)
        ts = []
        for _ in range(c.cfg["export"]["benchmark_runs"]):
            t = time.perf_counter()
            sess.run(None, feed)
            ts.append((time.perf_counter() - t) * 1e3)
        out["onnxruntime_cpu_latency_ms"] = {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)),
                                             "p99": float(np.percentile(ts, 99))}
    except ImportError:
        pass
    c.save_json(out, "export.json")
    c.log("[export]", json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(open(args.config))
    c = Ctx(cfg, args.force)
    seed_all(cfg["seed"])
    for st in STAGES:
        if st in args.stages:
            t = time.time()
            globals()[f"st_{st}"](c)
            c.log(f"== stage {st} finished in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
