"""Mixed-precision quantization-aware training (Section 3.4).

Weights : symmetric, per-channel (or per-tensor) INT8, scale s_w = max|W_c| / 127
Acts    : asymmetric UINT8 with EMA min/max observers, scale (max-min)/255
FP16    : layers flagged as quantization-sensitive are simulated in half precision
Gradients pass through rounding via the straight-through estimator (STE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x, q):
    return x + (q - x).detach()


def fq_weight_sym(w, per_channel=True):
    if per_channel:
        dims = tuple(range(1, w.ndim))
        s = w.detach().abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / 127.0
    else:
        s = w.detach().abs().max().clamp_min(1e-8) / 127.0
    return _ste(w, torch.clamp(torch.round(w / s), -128, 127) * s)


class ActObserver(nn.Module):
    def __init__(self, momentum=0.01):
        super().__init__()
        self.m = momentum
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("init", torch.tensor(False))

    def forward(self, x):
        if self.training:
            lo, hi = x.detach().min(), x.detach().max()
            if not bool(self.init):
                self.lo.copy_(lo), self.hi.copy_(hi), self.init.fill_(True)
            else:
                self.lo.mul_(1 - self.m).add_(self.m * lo)
                self.hi.mul_(1 - self.m).add_(self.m * hi)
        s = ((self.hi - self.lo) / 255.0).clamp_min(1e-8)
        z = self.lo
        q = torch.clamp(torch.round((x - z) / s), 0, 255) * s + z
        return _ste(x, q)


class QWrap(nn.Module):
    """Wraps a Conv2d / Linear and simulates its deployment precision."""

    def __init__(self, layer: nn.Module, mode="int8", per_channel=True):
        super().__init__()
        self.layer, self.mode, self.per_channel = layer, mode, per_channel
        self.act = ActObserver()

    def forward(self, x):
        w, b = self.layer.weight, self.layer.bias
        if self.mode == "int8":
            x = self.act(x)
            w = fq_weight_sym(w, self.per_channel)
        elif self.mode == "fp16":
            x = _ste(x, x.half().float())
            w = _ste(w, w.half().float())
        if isinstance(self.layer, nn.Conv2d):
            L = self.layer
            return F.conv2d(x, w, b, L.stride, L.padding, L.dilation, L.groups)
        return F.linear(x, w, b)


def quantizable_layers(model):
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _set(model, name, new):
    parent = model
    *path, last = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, new)


def prepare_qat(model, fp16_layers=(), per_channel=True, only=None):
    """Replace every Conv2d/Linear with a QWrap.

    fp16_layers : names simulated in FP16 (sensitive layers)
    only        : if given, only this single layer is quantized to INT8 and
                  all others stay FP32 (used for sensitivity analysis)
    """
    for n in quantizable_layers(model):
        mod = model.get_submodule(n)
        if only is not None:
            mode = "int8" if n == only else "fp32"
        else:
            mode = "fp16" if n in fp16_layers else "int8"
        _set(model, n, QWrap(mod, mode, per_channel))
    return model


def strip_qat(model):
    """Return wrapped layers to plain modules (weights unchanged)."""
    for n, m in list(model.named_modules()):
        if isinstance(m, QWrap):
            _set(model, n, m.layer)
    return model


def precision_map(model):
    return {n: m.mode for n, m in model.named_modules() if isinstance(m, QWrap)}


def model_size_mb(model):
    """Deployment size: INT8 layers 1 B/param, FP16 2 B, everything else FP32.
    BatchNorm is assumed folded into the preceding convolution."""
    total, seen = 0, set()
    for n, m in model.named_modules():
        if isinstance(m, QWrap):
            nb = {"int8": 1, "fp16": 2, "fp32": 4}[m.mode]
            for p in m.layer.parameters():
                total += p.numel() * (nb if p.ndim > 1 else 4)
                seen.add(id(p))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                seen.add(id(p))
    for p in model.parameters():
        if id(p) not in seen:
            total += 4 * p.numel()
    return total / 1e6
