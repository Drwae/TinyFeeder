"""Teacher networks for multi-teacher distillation (Section 3.5.1).

The paper's teachers are ResNeSt50, Swin-T and ConvNeXt-Base (ImageNet
pre-trained, via `timm`). Any `timm` model name can be used for the visual
stream; when `timm` is unavailable, or for the CPU demo, torchvision
backbones are used instead. Each teacher is two-stream: visual backbone +
acoustic CNN, fused by concatenation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .blocks import ConvBNAct

TORCHVISION = {
    "resnet18": (lambda: tvm.resnet18(weights=None), 512),
    "resnet50": (lambda: tvm.resnet50(weights=None), 2048),
    "mobilenet_v3_large": (lambda: tvm.mobilenet_v3_large(weights=None), 960),
    "shufflenet_v2_x1_0": (lambda: tvm.shufflenet_v2_x1_0(weights=None), 1024),
    "convnext_tiny": (lambda: tvm.convnext_tiny(weights=None), 768),
}


class _TVTrunk(nn.Module):
    """Wrap a torchvision classifier so it returns (pooled, feature-map)."""

    def __init__(self, name):
        super().__init__()
        ctor, self.dim = TORCHVISION[name]
        m = ctor()
        if name.startswith("resnet"):
            self.features = nn.Sequential(*list(m.children())[:-2])
        elif name.startswith("shufflenet"):
            self.features = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5)
        else:
            self.features = m.features

    def forward(self, x):
        f = self.features(x)
        return f.mean((2, 3)), f


class _TimmTrunk(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        import timm  # optional dependency

        self.m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        self.dim = self.m.num_features

    def forward(self, x):
        f = self.m.forward_features(x)
        if f.ndim == 3:  # transformer tokens -> map
            b, n, c = f.shape
            s = int(n ** 0.5)
            f = f[:, -s * s:].transpose(1, 2).reshape(b, c, s, s)
        elif f.ndim == 4 and f.shape[-1] == self.dim and f.shape[1] != self.dim:  # NHWC (Swin)
            f = f.permute(0, 3, 1, 2)
        return f.mean((2, 3)), f


class Teacher(nn.Module):
    def __init__(self, backbone="resnet18", n_classes=4, use_timm=False, pretrained=True):
        super().__init__()
        self.name = backbone
        self.visual = _TimmTrunk(backbone, pretrained) if use_timm else _TVTrunk(backbone)
        self.acoustic = nn.Sequential(
            ConvBNAct(1, 32, 3, 2, act="relu"), ConvBNAct(32, 64, 3, 2, act="relu"),
            ConvBNAct(64, 128, 3, 2, act="relu"), ConvBNAct(128, 256, 3, 2, act="relu"),
        )
        self.feat_dim = self.visual.dim + 256
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.feat_dim, n_classes))

    def forward(self, frame, mel, return_dict=False):
        v, fmap = self.visual(frame)
        a = self.acoustic(mel).mean((2, 3))
        z = torch.cat([v, a], 1)
        logits = self.head(z)
        if return_dict:
            return {"logits": logits, "feat": z, "vmap": fmap}
        return logits
