#!/usr/bin/env python
"""Profile random search-space architectures ON THE TARGET DEVICE (e.g. Jetson
Nano) to build the training set of the hardware latency predictor
(Section 3.3.2). Copy the resulting CSV back and pass it to the NAS stage.

  python deploy/profile_device.py --n 5000 --img 224 --mel 128 --out jetson_latency.csv
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyfeeder.engine import measure_latency  # noqa: E402
from tinyfeeder.nas.search import build, random_arch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--mel", type=int, default=128)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="device_latency.csv")
a = ap.parse_args()

rng = random.Random(0)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arch_json", "latency_ms"])
    for i in range(a.n):
        arch = random_arch(rng)
        m = build(arch).to(a.device).half() if a.device == "cuda" else build(arch)
        if a.device == "cuda":
            x = torch.randn(1, 3, a.img, a.img, device="cuda").half()
            y = torch.randn(1, 1, a.mel, a.mel, device="cuda").half()
            with torch.no_grad():
                for _ in range(10):
                    m(x, y)
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                ts = []
                for _ in range(50):
                    st.record(); m(x, y); en.record(); torch.cuda.synchronize()
                    ts.append(st.elapsed_time(en))
            lat = sorted(ts)[len(ts) // 2]
        else:
            lat = measure_latency(m, a.img, a.mel)
        w.writerow([json.dumps(arch), f"{lat:.4f}"])
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}", flush=True)
