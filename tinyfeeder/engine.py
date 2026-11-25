"""Training / evaluation loops shared by all stages."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Runtime settings, configured once by scripts/run_pipeline.py
DEVICE = torch.device("cpu")
NUM_WORKERS = 0


def configure(device="auto", num_workers=0):
    global DEVICE, NUM_WORKERS
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE, NUM_WORKERS = torch.device(device), num_workers
    if num_workers > 0:
        # NAS builds a fresh DataLoader for every candidate: share tensors through the file system and lift the
        # open-file limit, or a long search dies with «OSError: Too many open files»
        torch.multiprocessing.set_sharing_strategy("file_system")
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
        except (ImportError, ValueError, OSError):
            pass
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return DEVICE


def _loader(ds, batch_size, shuffle=False, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                      persistent_workers=False)   # persistent workers of discarded loaders pile up


def _to(*xs):
    return [x.to(DEVICE, non_blocking=True) for x in xs]


def classification_report(y, p, n_classes=4):
    cm = np.zeros((n_classes, n_classes), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    sup = cm.sum(1)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * sup).sum() / sup.sum()),
        "precision": prec.round(4).tolist(),
        "recall": rec.round(4).tolist(),
        "f1": f1.round(4).tolist(),
        "support": sup.tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict(model, loader, return_alpha=False):
    model.to(DEVICE).eval()
    ys, ps, probs, alphas = [], [], [], []
    for x, a, y in loader:
        x, a = _to(x, a)
        out = model(x, a, return_dict=True)
        pr = out["logits"].float().softmax(1).cpu()
        ys.append(y), ps.append(pr.argmax(1)), probs.append(pr)
        if return_alpha and "alpha_v" in out:
            alphas.append(out["alpha_v"].float().cpu())
    y, p, pr = torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()
    if return_alpha:
        return y, p, pr, (torch.cat(alphas).numpy() if alphas else np.zeros(0, np.float32))
    return y, p, pr


def evaluate(model, loader):
    y, p, _ = predict(model, loader)
    return classification_report(y, p)


def bootstrap_ci(y, p, n=1000, seed=0, groups=None):
    """95% bootstrap CI of accuracy. With `groups`, whole groups (recording
    sessions) are resampled - a cluster bootstrap that respects the fact
    that samples from the same session are not independent."""
    rng = np.random.default_rng(seed)
    hit = (np.asarray(y) == np.asarray(p)).astype(float)
    if groups is None:
        acc = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n)]
    else:
        g = np.asarray(groups)
        ug = np.unique(g)
        s = np.array([hit[g == u].sum() for u in ug])
        c = np.array([(g == u).sum() for u in ug])
        acc = []
        for _ in range(n):
            i = rng.integers(0, len(ug), len(ug))
            acc.append(s[i].sum() / c[i].sum())
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def fit(model, train_ds, val_ds, epochs, lr, batch_size=64, weight_decay=1e-4,
        loss_fn=None, extra_params=(), step_hook=None, log=print, label_smoothing=0.1,
        teachers=None, kd=None, beta_fn=None, max_batches=None, ema_decay=None, select="best",
        modality_dropout=0.0):
    """Generic AdamW + cosine trainer. If `teachers`/`kd` are given the
    multi-teacher distillation objective is used.

    ema_decay : keep an exponential moving average of the weights (and BN/observer buffers); validation and the
                returned model use the averaged weights.
    modality_dropout : probability that a training sample loses one modality (half the time the video - set to the
                same constant frame as the `visual_off` corruption - half the time the audio, zeroed as `audio_off`),
                so the fusion learns to use either stream alone; teachers still see both.
    select    : "best" - the epoch with the highest validation accuracy (prone to picking a lucky epoch when the
                validation split has few recording sessions); "last" - the weights at the end of training, so the
                validation split is only monitored and never used to choose the model."""
    tl = _loader(train_ds, batch_size, shuffle=True, drop_last=True)
    vl = _loader(val_ds, 256)
    model.to(DEVICE)
    if kd is not None:
        kd.to(DEVICE)
    for t in teachers or []:
        t.to(DEVICE).eval()
    params = list(model.parameters()) + list(extra_params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = epochs * (len(tl) if max_batches is None else min(len(tl), max_batches))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.1,
                                                anneal_strategy="cos")
    ce = loss_fn or nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best, best_state, hist = -1, None, []
    ema = copy.deepcopy(model).eval() if ema_decay else None
    step = 0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        beta = beta_fn(ep) if beta_fn else 0.0
        for b, (x, a, y) in enumerate(tl):
            if max_batches and b >= max_batches:
                break
            x, a, y = _to(x, a, y)
            xs, as_ = (_drop_modality(x, a, modality_dropout) if modality_dropout else (x, a))
            if teachers is not None:
                with torch.no_grad():
                    t_outs = [t(x, a, return_dict=True) for t in teachers]
                s_out = model(xs, as_, return_dict=True)
                loss, _ = kd(s_out, t_outs, y, beta)
            else:
                loss = ce(model(xs, as_), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step()
            step += 1
            if ema is not None:
                _ema_update(ema, model, min(ema_decay, (1 + step) / (10 + step)))
            tot, n = tot + loss.item() * len(y), n + len(y)
        net = ema if ema is not None else model
        val = evaluate(net, vl)
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_acc": val["accuracy"],
                     "val_macro_f1": val["macro_f1"], "beta": beta, "sec": time.time() - t0})
        log(f"    ep {ep + 1:3d}/{epochs}  loss {tot / max(n, 1):.4f}  val_acc {val['accuracy']:.4f}"
            f"  macroF1 {val['macro_f1']:.4f}  beta {beta:.2f}  ({time.time() - t0:.0f}s)")
        if select == "best" and val["accuracy"] > best:
            best = val["accuracy"]
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if select == "last":
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


_OFF = None


@torch.no_grad()
def _drop_modality(x, a, p):
    global _OFF
    if _OFF is None or _OFF.device != x.device:
        from tinyfeeder.data.dataset import MEAN, STD
        _OFF = torch.tensor(-MEAN / STD, device=x.device).view(1, 3, 1, 1)   # an all-zero frame after normalisation
    r = torch.rand(x.shape[0], device=x.device)
    v_off, a_off = r < p / 2, (r >= p / 2) & (r < p)
    x = torch.where(v_off.view(-1, 1, 1, 1), _OFF.to(x.dtype).expand_as(x), x)
    a = torch.where(a_off.view(-1, 1, 1, 1), torch.zeros_like(a), a)
    return x, a


@torch.no_grad()
def _ema_update(ema, model, d):
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
        else:
            v.copy_(msd[k])


def evaluate_ds(model, ds, batch_size=256):
    return evaluate(model, _loader(ds, batch_size))


def predict_ds(model, ds, return_alpha=False, batch_size=256):
    return predict(model, _loader(ds, batch_size), return_alpha)


@torch.no_grad()
def measure_latency(model, img_size, mel_size, runs=50, warmup=10, threads=1):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x, a = torch.randn(1, 3, img_size, img_size), torch.randn(1, 1, mel_size, mel_size)
    for _ in range(warmup):
        model(x, a)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x, a)
        ts.append((time.perf_counter() - t) * 1e3)
    torch.set_num_threads(prev)
    return float(np.median(ts))


def count_flops(model, img_size, mel_size):
    """Multiply-accumulate count of conv/linear layers (FLOPs = 2 x MACs)."""
    macs = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs[0] += o.numel() * k
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features

    model = model.cpu()   # it may still sit on the GPU after predict_ds
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()
        model(torch.zeros(1, 3, img_size, img_size), torch.zeros(1, 1, mel_size, mel_size))
    for h in hs:
        h.remove()
    return 2 * macs[0]
