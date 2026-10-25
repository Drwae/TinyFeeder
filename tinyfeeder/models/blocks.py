"""Building blocks of the TinyFeeder search space (Section 3.3.1).

Three block families are searched:
  * IR     - MobileNetV3 inverted residual (optional SE, hard-swish)
  * MBConv - EfficientNet-Lite MBConv (ReLU6, no SE by default)
  * DWSep  - custom depthwise-separable convolution
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    return {"hswish": nn.Hardswish(), "relu": nn.ReLU(inplace=True), "relu6": nn.ReLU6(inplace=True)}[name]


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act="hswish"):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            _act(act) if act else nn.Identity(),
        )


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(8, c // r)
        self.fc1, self.fc2 = nn.Conv2d(c, h, 1), nn.Conv2d(h, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * F.hardsigmoid(self.fc2(F.relu(self.fc1(s))))


class InvertedResidual(nn.Module):
    """MobileNetV3 IR(k, e, s) block."""

    def __init__(self, cin, cout, k=3, e=4, s=1, se=False, act="hswish"):
        super().__init__()
        hid = cin * e
        layers = []
        if e != 1:
            layers.append(ConvBNAct(cin, hid, 1, act=act))
        layers.append(ConvBNAct(hid, hid, k, s, groups=hid, act=act))
        if se:
            layers.append(SqueezeExcite(hid))
        layers.append(ConvBNAct(hid, cout, 1, act=None))
        self.body = nn.Sequential(*layers)
        self.use_res = s == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_res else y


class MBConv(InvertedResidual):
    """EfficientNet-Lite MBConv(k, e, s, SE) block (ReLU6 activation)."""

    def __init__(self, cin, cout, k=3, e=4, s=1, se=False):
        super().__init__(cin, cout, k, e, s, se=se, act="relu6")


class DWSep(nn.Module):
    """Depthwise-separable convolution: DWConv(k) -> PWConv."""

    def __init__(self, cin, cout, k=3, s=1, act="relu"):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, k, s, groups=cin, act=act)
        self.pw = ConvBNAct(cin, cout, 1, act=act)
        self.use_res = s == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.use_res else y


def make_block(spec: dict, cin: int) -> nn.Module:
    """Instantiate a block from a search-space gene, e.g.
    {"type": "IR", "k": 5, "e": 3, "s": 2, "c": 32, "se": False}."""
    t, cout = spec["type"], spec["c"]
    if t == "IR":
        return InvertedResidual(cin, cout, spec["k"], spec["e"], spec["s"], spec.get("se", False))
    if t == "MBConv":
        return MBConv(cin, cout, spec["k"], spec["e"], spec["s"], spec.get("se", False))
    if t == "DWSep":
        return DWSep(cin, cout, spec["k"], spec["s"])
    raise ValueError(f"unknown block type {t}")
