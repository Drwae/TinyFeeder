"""TinyFeeder multimodal network (Section 3.6, Tables 1-3).

Visual encoder (RGB frame) + acoustic encoder (log-Mel spectrogram) +
gated attention fusion -> 4 feeding-intensity classes
{0: none, 1: weak, 2: medium, 3: strong}.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn

from .blocks import ConvBNAct, DWSep, make_block

# Architecture discovered by HW-NAS (Table 1 / Table 16 composition):
# layers 2-3 MobileNetV3 IR, layers 4-6 EfficientNet-Lite MBConv, layer 7 DWSep.
TINYFEEDER_VISUAL = [
    {"type": "IR", "k": 3, "e": 1, "s": 1, "c": 16},
    {"type": "IR", "k": 3, "e": 4, "s": 2, "c": 24},
    {"type": "MBConv", "k": 3, "e": 4, "s": 1, "c": 24},
    {"type": "MBConv", "k": 5, "e": 3, "s": 2, "c": 32},
    {"type": "MBConv", "k": 5, "e": 3, "s": 1, "c": 32},
    {"type": "DWSep", "k": 3, "e": 1, "s": 2, "c": 48},
]


class VisualEncoder(nn.Module):
    def __init__(self, genes=None, stem_c=16, out_c=64):
        super().__init__()
        genes = copy.deepcopy(genes or TINYFEEDER_VISUAL)
        self.stem = ConvBNAct(3, stem_c, 3, 2, act="hswish")
        blocks, c = [], stem_c
        for g in genes:
            blocks.append(make_block(g, c))
            c = g["c"]
        self.blocks = nn.Sequential(*blocks)
        self.proj = ConvBNAct(c, out_c, 1, act="hswish")
        self.out_dim = out_c

    def forward(self, x):
        fmap = self.proj(self.blocks(self.stem(x)))
        return fmap.mean((2, 3)), fmap


class AcousticEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(1, 32, 3, 2, act="relu"),
            DWSep(32, 32, 3, 1),
            ConvBNAct(32, 64, 3, 2, act="relu"),
            DWSep(64, 64, 3, 1),
            ConvBNAct(64, 128, 3, 2, act="relu"),
        )
        self.fc = nn.Linear(128, out_dim)
        self.out_dim = out_dim

    def forward(self, a):
        return self.fc(self.body(a).mean((2, 3)))


class GatedFusion(nn.Module):
    """Channel-wise gated attention fusion (Table 3)."""

    def __init__(self, dv=64, da=128, n_classes=4):
        super().__init__()
        self.gv, self.ga = nn.Linear(dv, dv), nn.Linear(da, da)
        self.head = nn.Linear(dv + da, n_classes)

    def forward(self, v, a):
        av, aa = torch.sigmoid(self.gv(v)), torch.sigmoid(self.ga(a))
        z = torch.cat([v * av, a * aa], 1)
        # scalar modality weight used for Figure 4-style visualisation
        mv, ma = av.mean(1), aa.mean(1)
        alpha_v = mv / (mv + ma + 1e-8)
        return self.head(z), z, alpha_v


class TinyFeeder(nn.Module):
    def __init__(self, visual_genes=None, n_classes=4, modality="both"):
        super().__init__()
        self.modality = modality
        self.visual = VisualEncoder(visual_genes)
        self.acoustic = AcousticEncoder()
        self.fusion = GatedFusion(self.visual.out_dim, self.acoustic.out_dim, n_classes)

    def forward(self, frame, mel, return_dict=False):
        v, fmap = self.visual(frame)
        a = self.acoustic(mel)
        if self.modality == "visual":
            a = torch.zeros_like(a)
        elif self.modality == "audio":
            v = torch.zeros_like(v)
        logits, z, alpha_v = self.fusion(v, a)
        if return_dict:
            return {"logits": logits, "feat": z, "vmap": fmap, "alpha_v": alpha_v}
        return logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def param_breakdown(m: TinyFeeder) -> dict:
    return {
        "visual": count_params(m.visual),
        "acoustic": count_params(m.acoustic),
        "fusion": count_params(m.fusion),
        "total": count_params(m),
    }
