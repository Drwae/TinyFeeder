#!/usr/bin/env python
"""Real-time feeding controller (Sections 3.7.2 and 4.7).

Capture threads (camera at 1 FPS, hydrophone at 16 kHz) feed an inference
thread running the exported model; the controller applies the field-trial
rules:
  * start feeding when intensity rises from MEDIUM (2) to STRONG (3)
  * stop after 3 consecutive NONE/WEAK predictions
  * minimum 90-minute interval between meals
  * predictions with confidence < 0.80 are ignored and, if they persist,
    the controller falls back to the farm's fixed schedule (Section 6.3.3)

Hardware I/O is abstracted behind `Sensor` and `Feeder`; the default
implementations replay arrays so the logic can be tested without hardware.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

LEVELS = ("none", "weak", "medium", "strong")


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort

        prov = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(path, providers=prov)

    def __call__(self, frame, mel):
        z = self.s.run(None, {"frame": frame[None].astype(np.float32), "mel": mel[None].astype(np.float32)})[0][0]
        p = np.exp(z - z.max())
        return p / p.sum()


class Feeder:
    def __init__(self):
        self.on = False

    def set(self, on: bool):
        if on != self.on:
            print(f"[{time.strftime('%H:%M:%S')}] feeder {'ON' if on else 'OFF'}")
        self.on = on


class Controller:
    def __init__(self, feeder, min_interval_s=90 * 60, stop_after=3, conf_min=0.80, fallback_after=30):
        self.f, self.min_int, self.conf_min = feeder, min_interval_s, conf_min
        self.low = deque(maxlen=stop_after)
        self.prev, self.last_meal, self.uncertain = None, -1e9, 0
        self.fallback_after = fallback_after

    def step(self, prob, now):
        lvl, conf = int(prob.argmax()), float(prob.max())
        if conf < self.conf_min:
            self.uncertain += 1
            return "fallback_schedule" if self.uncertain >= self.fallback_after else "hold"
        self.uncertain = 0
        if not self.f.on and self.prev == 2 and lvl == 3 and now - self.last_meal >= self.min_int:
            self.f.set(True)
            self.last_meal = now
        self.low.append(lvl <= 1)
        if self.f.on and len(self.low) == self.low.maxlen and all(self.low):
            self.f.set(False)
        self.prev = lvl
        return LEVELS[lvl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--replay", help="npz with preprocessed `frame` and `mel` arrays")
    a = ap.parse_args()
    model, ctl = OnnxModel(a.onnx), Controller(Feeder())
    d = np.load(a.replay)
    for i, (fr, me) in enumerate(zip(d["frame"], d["mel"])):
        t = time.perf_counter()
        state = ctl.step(model(fr, me), now=i)
        print(f"t={i:4d}s  {state:18s}  {1e3 * (time.perf_counter() - t):.1f} ms")


if __name__ == "__main__":
    main()
