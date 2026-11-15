#!/usr/bin/env python
"""Write an INT8 calibration batch set for TensorRT / TFLite.

Samples are drawn from the TRAINING split only (never validation/test), so
calibration cannot leak test information (Section 4.1.1).

  python deploy/calibrate.py --config configs/paper.yaml --n 500
writes <out_dir>/calibration.npz with arrays `frame` and `mel`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyfeeder.data import AVFeedingDataset, load_npz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
out = Path(cfg["out_dir"])
import json

split = json.load(open(out / "splits.json"))
ds = AVFeedingDataset(load_npz(cfg["data"]["path"]), split["train"])
idx = np.random.default_rng(0).choice(len(ds), min(args.n, len(ds)), replace=False)
fr, me = zip(*[(ds[i][0], ds[i][1]) for i in idx])
np.savez(out / "calibration.npz", frame=torch.stack(fr).numpy(), mel=torch.stack(me).numpy())
print(f"wrote {len(idx)} calibration samples to {out / 'calibration.npz'}")
