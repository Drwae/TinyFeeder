"""Synthetic AV-FFIA-like data (docs/DATA.md): sessions with their own water colour (three farms), turbidity,
light, glare, pump hum and noise; a latent feeding intensity that ramps up, plateaus and decays (a third of the
sessions never reach strong feeding). For pipeline testing only — never for reported results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FARMS = [np.array([0.20, 0.35, 0.30]), np.array([0.30, 0.38, 0.22]), np.array([0.18, 0.28, 0.40])]


@dataclass
class SynthConfig:
    n_videos: int = 240
    samples_per_video: int = 32
    img_size: int = 96
    n_mels: int = 64
    seed: int = 0


def _intensity(t: np.ndarray, low: bool, rng) -> np.ndarray:
    peak = rng.uniform(1.3, 2.2) if low else rng.uniform(2.6, 3.4)
    up, down = rng.uniform(0.15, 0.3), rng.uniform(0.6, 0.85)
    v = np.where(t < up, t / up, np.where(t < down, 1.0, np.maximum(0, 1 - (t - down) / (1 - down))))
    return np.clip(v * peak + rng.normal(0, 0.25, t.shape), 0, 3)


def _frame(level: float, farm: np.ndarray, turb: float, light: float, glare: float, S: int, rng) -> np.ndarray:
    yy, xx = np.mgrid[0:S, 0:S] / S
    img = np.ones((S, S, 3)) * farm * light
    n = int(4 + 10 * level)                       # aggregation: more fish as feeding grows
    surf = 0.15 + 0.6 * (1 - level / 3)           # rise to the surface (top of the frame)
    for _ in range(n):
        cy = np.clip(rng.normal(surf, 0.15), 0, 1)
        cx = np.clip(rng.normal(0.5, 0.25 - 0.04 * level), 0, 1)
        r = rng.uniform(0.03, 0.06)
        m = ((yy - cy) ** 2 / (1 + 0.6 * level) + (xx - cx) ** 2) < r ** 2
        img[m] = img[m] * 0.4 + 0.55
    for _ in range(int(level ** 2 * 2)):          # splashes
        cy, cx = rng.uniform(0, 0.3), rng.uniform(0, 1)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < rng.uniform(0.0005, 0.003)
        img[m] = 0.95
    if level > 1.5:                               # motion blur
        img = (img + np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 3
    img = (1 - turb) * img + turb * (farm + 0.25)
    img += glare * np.exp(-((yy - 0.1) ** 2 + (xx - 0.8) ** 2) / 0.02)[..., None]
    img += rng.normal(0, 0.03, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _mel(level: float, hum: float, noise: float, M: int, rng) -> np.ndarray:
    f = np.arange(M)[:, None] / M
    t = np.arange(M)[None, :] / M
    s = rng.normal(0, noise, (M, M))
    for h in (0.04, 0.08, 0.12):                   # pump hum harmonics
        s += hum * np.exp(-((f - h) ** 2) / 0.0004)
    if rng.random() < 0.2:                         # aerator burst
        s += 0.6 * np.exp(-((t - rng.uniform()) ** 2) / 0.01) * (f > 0.3)
    for _ in range(int(level * 4)):                # splash transients: broadband, short
        s += rng.uniform(0.5, 1.0) * np.exp(-((t - rng.uniform()) ** 2) / 0.0006) * np.exp(-f * 1.5)
    for _ in range(int(level * 6)):                # chewing / snapping clicks: high band
        s += 0.7 * np.exp(-((t - rng.uniform()) ** 2) / 0.0002) * (f > 0.6)
    s = np.log1p(np.maximum(s, 0) * 5)
    s = 2 * (s - s.min()) / max(s.max() - s.min(), 1e-6) - 1
    return s.astype(np.float16)


def generate(cfg: SynthConfig, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n_videos * cfg.samples_per_video
    S, M = cfg.img_size, cfg.n_mels
    frames = np.zeros((N, S, S, 3), np.uint8)
    mels = np.zeros((N, M, M), np.float16)
    labels = np.zeros(N, np.int64)
    vid = np.repeat(np.arange(cfg.n_videos, dtype=np.int32), cfg.samples_per_video)
    farm_id = np.zeros(N, np.int32)
    k = 0
    for v in range(cfg.n_videos):
        fi = v % 3
        farm, turb = FARMS[fi], rng.uniform(0.05, 0.45)
        light, glare = rng.uniform(0.6, 1.2), rng.uniform(0, 0.3)
        hum, noise = rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.25)
        lat = _intensity(np.linspace(0, 1, cfg.samples_per_video), rng.random() < 1 / 3, rng)
        for lv in lat:
            frames[k] = _frame(lv, farm, turb, light, glare, S, rng)
            mels[k] = _mel(lv, hum, noise, M, rng)
            labels[k], farm_id[k] = int(np.clip(np.rint(lv), 0, 3)), fi
            k += 1
        if verbose and (v + 1) % 40 == 0:
            print(f"  synthetic: {v + 1}/{cfg.n_videos} sessions", flush=True)
    return {"frames": frames, "mels": mels, "labels": labels, "video_id": vid, "farm_id": farm_id}
