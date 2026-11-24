#!/usr/bin/env python
"""End-to-end TinyFeeder pipeline.

  data -> teachers -> baselines -> nas -> stage1 (FP32) -> stage2 (mixed-precision QAT)
       -> stage3 (multi-teacher KD) -> evaluate -> export

Usage:
  python scripts/run_pipeline.py --config configs/avffia.yaml --device cuda --workers 8
  python scripts/run_pipeline.py --config configs/avffia.yaml --stages evaluate export
Every stage writes its artefacts to <out_dir>/ and is skipped if they exist
(use --force to recompute). `evaluate` stores the test predictions of EVERY
model in <out_dir>/predictions/, and scripts/paper_tables.py builds every
table of the article from those files, so all reported numbers come from one
set of predictions.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tinyfeeder.engine as E  # noqa: E402
from tinyfeeder.data import (AVFeedingDataset, SynthConfig, generate, load_npz, unit_key,  # noqa: E402
                             video_level_split)
from tinyfeeder.data.dataset import save_split  # noqa: E402
from tinyfeeder.distill import MultiTeacherKD, beta_schedule  # noqa: E402
from tinyfeeder.engine import count_flops, evaluate_ds, fit, measure_latency, predict_ds  # noqa: E402
from tinyfeeder.models import Teacher, TinyFeeder, param_breakdown  # noqa: E402
from tinyfeeder.nas import nsga2, profile_and_fit, select  # noqa: E402
from tinyfeeder.nas.search import fit_from_csv  # noqa: E402
from tinyfeeder.quant.fakequant import ActObserver  # noqa: E402
from tinyfeeder.quant import model_size_mb, precision_map, prepare_qat, quantizable_layers, strip_qat  # noqa: E402

STAGES = ["data", "teachers", "baselines", "nas", "stage1", "stage2", "stage3", "evaluate", "export"]


def seed_all(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


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
        (self.out / name).parent.mkdir(parents=True, exist_ok=True)
        with open(self.out / name, "w") as f:
            json.dump(obj, f, indent=2)

    def load_json(self, name):
        return json.load(open(self.out / name))

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
            raise SystemExit(f"{p} not found. Run scripts/prepare_avffia.py first (see docs/REPRODUCE.md).")
        c.log(f"[data] generating synthetic AV-FFIA-like dataset -> {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = generate(SynthConfig(n_videos=d["n_videos"], samples_per_video=d["samples_per_video"],
                                   img_size=d["img_size"], n_mels=d["n_mels"], seed=c.cfg["seed"]))
        np.savez_compressed(p, **arr)
    if c.done("splits.json"):
        return
    data = c.data
    key = unit_key(data)
    split = video_level_split(data[key], data["labels"], tuple(d["split_fracs"]), c.cfg["seed"])
    save_split(split, c.out / "splits.json")
    stats = {"split_unit": key}
    for k, v in split.items():
        m = np.isin(data[key], v)
        stats[k] = {"units": len(v), "samples": int(m.sum()),
                    "per_level": np.bincount(data["labels"][m], minlength=4).tolist()}
    stats["unit_overlap_train_test"] = len(set(split["train"]) & set(split["test"]))
    stats["unit_overlap_val_test"] = len(set(split["val"]) & set(split["test"]))
    c.save_json(stats, "data_stats.json")
    c.log("[data] split by", key, json.dumps(stats))


def _tr(c: Ctx, s: dict | None = None) -> dict:
    """Training protocol shared by every model (config `train`, overridable per section): weight averaging and how
    the final weights are chosen."""
    g = dict(c.cfg.get("train") or {})
    g.update({k: v for k, v in (s or {}).items() if k in ("ema_decay", "select", "weight_decay")})
    return {"ema_decay": g.get("ema_decay"), "select": g.get("select", "best"), "weight_decay": g.get("weight_decay", 1e-4)}


def _train_two_stream(c: Ctx, section: str, sub: str):
    t = c.cfg[section]
    tdir = c.out / sub
    tdir.mkdir(exist_ok=True)
    res = c.load_json(f"{sub}.json") if (c.out / f"{sub}.json").exists() else {}
    for name in t["backbones"]:
        f = tdir / f"{name}.pt"
        if f.exists() and not c.force:
            continue
        seed_all(c.cfg["seed"])
        c.log(f"[{sub}] training {name} (training split only)")
        m = Teacher(name, use_timm=t.get("use_timm", False), pretrained=t.get("pretrained", True))
        m, hist = fit(m, c.ds("train", True), c.ds("val"), t["epochs"], t["lr"], t["batch_size"], log=c.log,
                      **_tr(c, t))
        torch.save(m.state_dict(), f)
        res[name] = {"val": evaluate_ds(m, c.ds("val")), "params": n_params(m), "history": hist}
        c.save_json(res, f"{sub}.json")


def st_teachers(c: Ctx):
    _train_two_stream(c, "teachers", "teachers")


def st_baselines(c: Ctx):
    if c.cfg.get("baselines", {}).get("backbones"):
        _train_two_stream(c, "baselines", "baselines")
    # the same backbones trained from scratch with TinyFeeder's protocol: the like-for-like test of the searched
    # architecture (the ImageNet-pretrained ones stay as the upper reference)
    if c.cfg.get("baselines_scratch", {}).get("backbones"):
        _train_two_stream(c, "baselines_scratch", "baselines_scratch")


def load_two_stream(c: Ctx, section: str, sub: str):
    t = c.cfg.get(section, {})
    ms = []
    for name in t.get("backbones", []):
        m = Teacher(name, use_timm=t.get("use_timm", False), pretrained=False)
        m.load_state_dict(torch.load(c.out / sub / f"{name}.pt", map_location="cpu"))
        ms.append(m.eval())
    return ms


def st_nas(c: Ctx):
    if c.done("nas/selected_arch.json"):
        return
    n = c.cfg["nas"]
    (c.out / "nas").mkdir(exist_ok=True)
    img, mel = c.sizes
    if n.get("latency_csv") and Path(n["latency_csv"]).exists():
        pred, pstats = fit_from_csv(n["latency_csv"], img, mel, c.log)
        pstats["source"] = f"device profile {n['latency_csv']}"
    else:
        if n.get("latency_csv"):
            c.log(f"[nas] WARNING: {n['latency_csv']} not found - falling back to host-CPU latency "
                  "(must be reported as such; re-run --stages nas ... --force once the device profile exists)")
        c.log(f"[nas] no device profile given - profiling {n['profile_archs']} architectures on THIS host")
        pred, pstats = profile_and_fit(n["profile_archs"], img, mel, c.cfg["seed"], c.log)
        pstats["source"] = "host CPU (not the target device)"
    tr_units = c.split["train"]
    if n.get("proxy_train_units"):
        tr_units = random.Random(0).sample(tr_units, min(n["proxy_train_units"], len(tr_units)))
    proxy_tr, proxy_va = c.ds(None, True, videos=tr_units), c.ds("val")
    proxy = {"train_units": len(tr_units), "train_samples": len(proxy_tr), "val_samples": len(proxy_va),
             "epochs": n["proxy_epochs"], "lr": n.get("proxy_lr", 3e-3), "weight_sharing": False}
    c.log(f"[nas] proxy task: {proxy}")

    def proxy_acc(arch):
        seed_all(c.cfg["seed"])
        m = TinyFeeder(visual_genes=arch)
        m, _ = fit(m, proxy_tr, proxy_va, n["proxy_epochs"], proxy["lr"], n.get("proxy_batch_size", 128),
                   log=lambda *a: None)
        return evaluate_ds(m, proxy_va)["accuracy"]

    t0 = time.time()
    c.log(f"[nas] NSGA-II: population {n['population']}, generations {n['generations']}")
    allr, pareto, hist = nsga2(proxy_acc, pred, n["population"], n["generations"], img, mel,
                               c.cfg["seed"], c.log, n.get("patience", 10))
    best = select(pareto, n["latency_budget_ms"], n["params_budget"])
    proxy["search_hours"] = (time.time() - t0) / 3600
    proxy["device"] = str(E.DEVICE)
    c.save_json({"predictor": pstats, "proxy": proxy, "history": hist, "evaluated": allr, "pareto": pareto},
                "nas/search.json")
    c.save_json(best, "nas/selected_arch.json")
    c.log(f"[nas] {len(allr)} evaluated, Pareto front {len(pareto)}; selected proxy acc {best['acc']:.4f}, "
          f"pred latency {best['pred_latency_ms']:.2f} ms, params {best['params']}")


def selected_arch(c):
    return c.load_json("nas/selected_arch.json")["arch"]


def student(c, pretrained=True):
    """The TinyFeeder student: the NAS-selected visual encoder, or (config `student.visual_backbone`) an
    ImageNet-pretrained tiny backbone with the same acoustic encoder and fusion."""
    st = c.cfg.get("student") or {}
    bb = st.get("visual_backbone")
    return TinyFeeder(visual_genes=None if bb else selected_arch(c), visual_backbone=bb,
                      pretrained=pretrained and st.get("pretrained", True))


def st_stage1(c: Ctx):
    if c.done("stage1.pt"):
        return
    s = c.cfg["stage1"]
    seed_all(c.cfg["seed"])
    m = student(c)
    c.log(f"[stage1] FP32 training of the NAS architecture: {param_breakdown(m)}")
    kw = {}
    if s.get("kd"):   # distil from the teachers from the first epoch: the small network trained from scratch on
        #               26 sessions otherwise memorises them (stage-3 distillation alone comes too late)
        teachers = load_two_stream(c, "teachers", "teachers")
        kd = MultiTeacherKD(m.visual.out_dim + m.acoustic.out_dim, [t.feat_dim for t in teachers],
                            s["tau"], tuple(s["lambdas"]))
        c.log(f"[stage1] with multi-teacher KD from {[t.name for t in teachers]}")
        kw = dict(teachers=teachers, kd=kd, extra_params=kd.parameters(),
                  beta_fn=lambda e: beta_schedule(e, s.get("warmup", 0), s.get("ramp", 5), s["beta_max"]))
    m, hist = fit(m, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], s["batch_size"], log=c.log,
                  **_tr(c, s), **kw)
    torch.save(m.state_dict(), c.out / "stage1.pt")
    c.save_json({"params": param_breakdown(m), "kd": bool(s.get("kd")), "protocol": _tr(c, s), "history": hist},
                "stage1.json")


def st_stage2(c: Ctx):
    if c.done("stage2.pt", "stage2.json"):
        return
    s = c.cfg["stage2"]
    base = student(c, pretrained=False)
    base.load_state_dict(torch.load(c.out / "stage1.pt", map_location="cpu"))
    units = c.split["val"]
    if s.get("sensitivity_val_units"):
        units = random.Random(1).sample(units, min(s["sensitivity_val_units"], len(units)))
    # the sensitivity set and the calibration batches are read once and kept in memory: a fresh DataLoader per layer
    # (64 layers with a pretrained backbone) exhausted the pod's shared memory («Bus error» in the workers)
    torch.manual_seed(c.cfg["seed"])
    sens_ds = list(E._loader(c.ds(None, videos=units), 128))
    calib = []
    for b, batch in enumerate(E._loader(c.ds("train"), 128, shuffle=True)):
        calib.append(batch)
        if b >= 15:
            break
    acc_fp32 = E.evaluate(base, sens_ds)["accuracy"]
    sens = {}
    c.log(f"[stage2] layer-wise sensitivity over {len(quantizable_layers(base))} layers (FP32 acc {acc_fp32:.4f})")
    for name in quantizable_layers(base):
        torch.manual_seed(c.cfg["seed"])  # identical calibration batches for every layer
        m = prepare_qat(copy.deepcopy(base), only=name).to(E.DEVICE)
        m.eval()   # calibrate only the activation observers: train mode would also shift BatchNorm statistics,
        for mod in m.modules():   # which cost every layer the same ~2.5 points and hid the real sensitivities
            if isinstance(mod, ActObserver):
                mod.train()
        with torch.no_grad():
            for b, (x, a, _) in enumerate(calib):
                m(*E._to(x, a))
                if b >= 3:
                    break
        sens[name] = acc_fp32 - E.evaluate(m, sens_ds)["accuracy"]
    # check that quantization itself is sound before any QAT training: every layer INT8, observers calibrated on a
    # few training batches (BatchNorm untouched), compared with FP32 on the validation and test sessions
    torch.manual_seed(c.cfg["seed"])
    ptq = prepare_qat(copy.deepcopy(base)).to(E.DEVICE)
    ptq.eval()
    for mod in ptq.modules():
        if isinstance(mod, ActObserver):
            mod.train()
    with torch.no_grad():
        for b, (x, a, _) in enumerate(calib):
            ptq(*E._to(x, a))
            if b >= 15:
                break
    ptq.eval()
    ptq_check = {part: {"fp32": evaluate_ds(base, c.ds(part))["accuracy"], "int8_ptq": evaluate_ds(ptq, c.ds(part))["accuracy"]}
                 for part in ("val", "test")}
    c.log(f"[stage2] PTQ check (no training): {ptq_check}")
    ranked = sorted(sens, key=lambda k: -sens[k])
    k = max(1, int(round(s["fp16_fraction"] * len(ranked))))
    fp16 = ranked[:k]
    c.log(f"[stage2] FP16 layers (top {k}): {fp16}")
    q = prepare_qat(copy.deepcopy(base), fp16_layers=set(fp16))
    seed_all(c.cfg["seed"])
    q, hist = fit(q, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], s.get("batch_size", 64), log=c.log,
                  **_tr(c, s))
    torch.save(q.state_dict(), c.out / "stage2.pt")
    abl = {}
    for tag, kw in {"int8_per_channel": dict(fp16_layers=set()),
                    "int8_per_tensor": dict(fp16_layers=set(), per_channel=False)}.items():
        seed_all(c.cfg["seed"])
        mq = prepare_qat(copy.deepcopy(base), **kw)
        mq, _ = fit(mq, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], s.get("batch_size", 64),
                    log=lambda *a: None, **_tr(c, s))
        abl[tag] = {"val_acc": evaluate_ds(mq, c.ds("val"))["accuracy"], "size_mb": model_size_mb(mq)}
        c.log(f"[stage2] ablation {tag}: {abl[tag]}")
    c.save_json({"fp32_val_acc": acc_fp32, "ptq_check": ptq_check, "sensitivity": sens, "fp16_layers": fp16,
                 "precision_map": precision_map(q), "size_mb": model_size_mb(q),
                 "fp32_size_mb": n_params(base) * 4 / 1e6, "ablation": abl, "history": hist}, "stage2.json")


def quantized_student(c):
    st2 = c.load_json("stage2.json")
    return prepare_qat(student(c, pretrained=False), fp16_layers=set(st2["fp16_layers"]))


def st_stage3(c: Ctx):
    if c.done("tinyfeeder_final.pt"):
        return
    s = c.cfg["stage3"]
    student = quantized_student(c)
    student.load_state_dict(torch.load(c.out / "stage2.pt", map_location="cpu"))
    teachers = load_two_stream(c, "teachers", "teachers")
    kd = MultiTeacherKD(student.visual.out_dim + student.acoustic.out_dim, [t.feat_dim for t in teachers],
                        s["tau"], tuple(s["lambdas"]))
    seed_all(c.cfg["seed"])
    c.log(f"[stage3] multi-teacher KD from {[t.name for t in teachers]} - same student architecture as stage 2")
    student, hist = fit(student, c.ds("train", True), c.ds("val"), s["epochs"], s["lr"], s.get("batch_size", 64),
                        log=c.log, teachers=teachers, kd=kd, extra_params=kd.parameters(),
                        beta_fn=lambda e: beta_schedule(e, s["warmup"], s["epochs"] - s["warmup"], s["beta_max"]),
                        **_tr(c, s))
    torch.save(student.state_dict(), c.out / "tinyfeeder_final.pt")
    c.save_json({"tau": s["tau"], "beta_max": s["beta_max"], "lambdas": s["lambdas"], "history": hist}, "stage3.json")


def _save_pred(c, name, y, p, prob, alpha=None):
    d = c.out / "predictions"
    d.mkdir(exist_ok=True)
    ds = c.ds("test")
    key = unit_key(c.data)
    extra = {"alpha_v": alpha} if alpha is not None else {}
    np.savez(d / f"{name}.npz", y=y, p=p, prob=prob, unit=c.data[key][ds.idx], sample_index=ds.idx, **extra)


def st_evaluate(c: Ctx):
    img, mel = c.sizes
    te = c.ds("test")
    models = {}
    fp32 = student(c, pretrained=False)
    fp32.load_state_dict(torch.load(c.out / "stage1.pt", map_location="cpu"))
    models["tinyfeeder_stage1_fp32"] = fp32
    qat = quantized_student(c)
    qat.load_state_dict(torch.load(c.out / "stage2.pt", map_location="cpu"))
    models["tinyfeeder_stage2_qat"] = qat
    final = quantized_student(c)
    final.load_state_dict(torch.load(c.out / "tinyfeeder_final.pt", map_location="cpu"))
    models["tinyfeeder_final"] = final
    for t in load_two_stream(c, "teachers", "teachers"):
        models[f"teacher_{t.name}"] = t
    for b in load_two_stream(c, "baselines", "baselines"):
        models[f"baseline_{b.name}"] = b
    if c.cfg.get("baselines_scratch", {}).get("backbones"):
        for b in load_two_stream(c, "baselines_scratch", "baselines_scratch"):
            models[f"scratch_{b.name}"] = b
    info = {}
    for name, m in models.items():
        y, p, pr, al = predict_ds(m, te, return_alpha=True)
        _save_pred(c, name, y, p, pr, al if al.size else None)
        is_q = name in ("tinyfeeder_stage2_qat", "tinyfeeder_final")
        info[name] = {"params": n_params(m),
                      "size_mb": model_size_mb(m) if is_q else n_params(m) * 4 / 1e6,
                      "flops": count_flops(strip_qat(copy.deepcopy(m)) if is_q else m, img, mel),
                      "host_cpu_latency_ms": measure_latency(strip_qat(copy.deepcopy(m)) if is_q else m, img, mel)}
        c.log(f"[evaluate] {name}: acc {(y == p).mean():.4f}  {info[name]}")
    # robustness of the final model to input corruption and modality loss
    for cor in ["turbidity", "dark", "visual_off", "audio_off"]:
        y, p, pr, al = predict_ds(final, c.ds("test", corrupt=cor), return_alpha=True)
        _save_pred(c, f"robust_{cor}", y, p, pr, al)
    c.save_json({"models": info, "params_breakdown": param_breakdown(fp32),
                 "data": "synthetic" if c.cfg["data"].get("synthetic") else c.cfg["data"]["path"]},
                "model_info.json")


def st_export(c: Ctx):
    img, mel = c.sizes
    final = quantized_student(c)
    final.load_state_dict(torch.load(c.out / "tinyfeeder_final.pt", map_location="cpu"))
    m = strip_qat(final).eval().cpu()
    x, a = torch.randn(1, 3, img, img), torch.randn(1, 1, mel, mel)
    path = c.out / "tinyfeeder.onnx"
    torch.onnx.export(m, (x, a), path, input_names=["frame", "mel"], output_names=["logits"],
                      opset_version=17,
                      dynamic_axes={"frame": {0: "batch"}, "mel": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    out = {"onnx": str(path), "onnx_mb": os.path.getsize(path) / 1e6,
           "note": "FP32 graph; INT8/FP16 precision is applied when the TensorRT engine is built on the device"}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {"frame": x.numpy(), "mel": a.numpy()}
        out["max_abs_diff_vs_torch"] = float(np.abs(sess.run(None, feed)[0] - m(x, a).detach().numpy()).max())
    except ImportError:
        pass
    # ONNX graphs of the comparison models, for benchmarking them on the same device
    odir = c.out / "onnx"
    odir.mkdir(exist_ok=True)
    for sec, sub in (("teachers", "teachers"), ("baselines", "baselines")):
        for t in load_two_stream(c, sec, sub):
            torch.onnx.export(t.cpu(), (x, a), odir / f"{sub[:-1]}_{t.name}.onnx", input_names=["frame", "mel"],
                              output_names=["logits"], opset_version=17, dynamo=False)
    out["comparison_onnx"] = sorted(str(p) for p in odir.glob("*.onnx"))
    c.save_json(out, "export.json")
    c.log("[export]", json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = E.configure(args.device, args.workers)
    cfg = yaml.safe_load(open(args.config))
    c = Ctx(cfg, args.force)
    c.log(f"== device {dev}, torch {torch.__version__}, config {args.config}")
    seed_all(cfg["seed"])
    for st in STAGES:
        if st in args.stages:
            t = time.time()
            globals()[f"st_{st}"](c)
            c.log(f"== stage {st} finished in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
