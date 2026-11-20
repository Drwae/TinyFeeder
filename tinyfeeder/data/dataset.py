"""Dataset, split and I/O for the `.npz` layout of docs/DATA.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet statistics (the teachers are ImageNet-pretrained)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CORRUPTIONS = ("turbidity", "dark", "visual_off", "audio_off")


def load_npz(path) -> dict:
    """All arrays of the file, loaded into memory (frames stay uint8, mels float16)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def unit_key(data: dict) -> str:
    """The split unit: the recording session when the data has one, else the video."""
    return "group_id" if "group_id" in data else "video_id"


def video_level_split(units, labels, fracs=(0.64, 0.16, 0.20), seed=0) -> dict:
    """Assign every unit (session / video) to exactly one of train/val/test, stratified by the unit's dominant
    feeding level, so no clip of a test unit is ever seen in training."""
    units, labels = np.asarray(units), np.asarray(labels)
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    dom = {int(u): int(np.bincount(labels[units == u], minlength=4).argmax()) for u in uniq}
    out = {"train": [], "val": [], "test": []}
    f_tr, f_va = fracs[0] / sum(fracs), fracs[1] / sum(fracs)
    for lvl in sorted(set(dom.values())):
        us = np.array([u for u, d in dom.items() if d == lvl])
        rng.shuffle(us)
        n = len(us)
        n_tr = int(round(n * f_tr))
        n_va = int(round(n * f_va))
        if n >= 3:                                   # every part gets at least one unit of the level when possible
            n_tr, n_va = max(1, min(n_tr, n - 2)), max(1, min(n_va, n - 1 - max(1, min(n_tr, n - 2))))
        out["train"] += us[:n_tr].tolist()
        out["val"] += us[n_tr:n_tr + n_va].tolist()
        out["test"] += us[n_tr + n_va:].tolist()
    return {k: sorted(int(u) for u in v) for k, v in out.items()}


def save_split(split: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


class AVFeedingDataset(Dataset):
    """(frame, mel, label) for the clips of the given units.

    frame: float tensor (3, H, W), ImageNet-normalised; mel: float tensor (1, F, T) in [-1, 1].
    `train` adds light augmentation (flip, brightness/contrast, Mel time/frequency masking);
    `corrupt` applies one test-time corruption of CORRUPTIONS to every clip."""

    def __init__(self, data: dict, units, train: bool = False, corrupt: str | None = None):
        if corrupt is not None and corrupt not in CORRUPTIONS:
            raise ValueError(f"corrupt must be one of {CORRUPTIONS}")
        self.d, self.train, self.corrupt = data, train, corrupt
        key = unit_key(data)
        self.idx = np.flatnonzero(np.isin(data[key], np.asarray(list(units))))
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.idx)

    def _frame(self, f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float32) / 255.0
        if self.train:
            if self.rng.random() < 0.5:
                x = x[:, ::-1]
            x = np.clip((x - 0.5) * self.rng.uniform(0.8, 1.2) + 0.5 + self.rng.uniform(-0.1, 0.1), 0, 1)
        if self.corrupt == "turbidity":
            k = 0.55
            haze = np.array([0.45, 0.55, 0.40], np.float32)
            x = (1 - k) * x + k * haze
            x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5
        elif self.corrupt == "dark":
            x = x * 0.25
        elif self.corrupt == "visual_off":
            x = np.zeros_like(x)
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def _mel(self, m: np.ndarray) -> np.ndarray:
        a = m.astype(np.float32).copy()
        if self.train:
            F, T = a.shape
            for _ in range(2):
                w = int(self.rng.integers(0, max(1, F // 8)))
                s = int(self.rng.integers(0, F - w + 1))
                a[s:s + w] = a.mean()
                w = int(self.rng.integers(0, max(1, T // 8)))
                s = int(self.rng.integers(0, T - w + 1))
                a[:, s:s + w] = a.mean()
        if self.corrupt == "audio_off":
            a = np.zeros_like(a)
        return a[None]

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.ascontiguousarray(self._frame(self.d["frames"][j])))
        a = torch.from_numpy(np.ascontiguousarray(self._mel(self.d["mels"][j])))
        return x, a, torch.tensor(int(self.d["labels"][j]), dtype=torch.long)
