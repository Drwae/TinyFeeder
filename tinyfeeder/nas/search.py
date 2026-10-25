"""Hardware-aware NAS (Section 3.3): search space, latency predictor, NSGA-II.

Search space
  6 visual blocks; for each: type in {IR, MBConv, DWSep}, kernel in {3, 5},
  expansion in {3, 4, 6} (IR/MBConv), output channels from a per-position
  menu, optional squeeze-excitation. Strides are fixed per position so all
  candidates share the same resolution schedule. The acoustic encoder and the
  fusion module are fixed (Section 3.3.1).

Latency predictor
  Gradient-boosted regression trees on an architecture encoding (block
  one-hots, kernel/expansion/channels, params, FLOPs). Train it on latencies
  profiled on the *target* device (scripts/profile_device.py on the Jetson);
  the demo profiles the host CPU instead.
"""
from __future__ import annotations

import copy
import json
import random
import time

import numpy as np
import torch

from ..engine import count_flops, measure_latency
from ..models.tinyfeeder import TinyFeeder, count_params

TYPES = ("IR", "MBConv", "DWSep")
STRIDES = (1, 2, 1, 2, 1, 2)
CHANNELS = ((16,), (16, 24, 32), (16, 24, 32), (24, 32, 48), (24, 32, 48), (32, 48, 64))


def random_gene(pos, rng):
    t = rng.choice(TYPES)
    return {"type": t, "k": rng.choice((3, 5)), "e": 1 if t == "DWSep" else rng.choice((3, 4, 6)),
            "s": STRIDES[pos], "c": rng.choice(CHANNELS[pos]), "se": bool(t != "DWSep" and rng.random() < 0.3)}


def random_arch(rng):
    return [random_gene(i, rng) for i in range(len(STRIDES))]


def arch_key(arch):
    return json.dumps(arch, sort_keys=True)


def build(arch):
    return TinyFeeder(visual_genes=arch)


def encode(arch, img_size, mel_size):
    f = []
    for g in arch:
        f += [float(g["type"] == t) for t in TYPES] + [g["k"], g["e"], g["c"], g["s"], float(g["se"])]
    m = build(arch)
    f += [count_params(m) / 1e5, count_flops(m, img_size, mel_size) / 1e7]
    return np.array(f, np.float32)


# ------------------------------------------------------------------ predictor
class LatencyPredictor:
    def __init__(self):
        try:
            from xgboost import XGBRegressor

            self.reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9)
            self.kind = "XGBoost"
        except Exception:  # pragma: no cover
            from sklearn.ensemble import GradientBoostingRegressor

            self.reg = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
            self.kind = "sklearn-GBRT"

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(np.atleast_2d(X))


def profile_and_fit(n_archs, img_size, mel_size, seed=0, log=print):
    rng = random.Random(seed)
    archs = [random_arch(rng) for _ in range(n_archs)]
    X, y = [], []
    for i, a in enumerate(archs):
        X.append(encode(a, img_size, mel_size))
        y.append(measure_latency(build(a), img_size, mel_size, runs=20, warmup=5))
        if (i + 1) % 50 == 0:
            log(f"    profiled {i + 1}/{n_archs} architectures")
    X, y = np.stack(X), np.array(y)
    n_tr = int(0.8 * len(y))
    pred = LatencyPredictor().fit(X[:n_tr], y[:n_tr])
    mape = float(np.mean(np.abs(pred.predict(X[n_tr:]) - y[n_tr:]) / y[n_tr:]) * 100)
    pred.fit(X, y)
    return pred, {"n_profiled": n_archs, "holdout_mape_pct": mape, "regressor": pred.kind,
                  "latency_ms_range": [float(y.min()), float(y.max())]}


# ------------------------------------------------------------------ NSGA-II
def dominates(a, b):
    """objectives are all to be MINIMISED: (-acc, latency, params)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(F):
    n = len(F)
    S, cnt, fronts = [[] for _ in range(n)], [0] * n, [[]]
    for p in range(n):
        for q in range(n):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F, front):
    d = {i: 0.0 for i in front}
    for m in range(len(F[0])):
        s = sorted(front, key=lambda i: F[i][m])
        d[s[0]] = d[s[-1]] = float("inf")
        rng = F[s[-1]][m] - F[s[0]][m] or 1.0
        for j in range(1, len(s) - 1):
            d[s[j]] += (F[s[j + 1]][m] - F[s[j - 1]][m]) / rng
    return d


def crossover(a, b, rng, p=0.8):
    if rng.random() > p:
        return copy.deepcopy(a)
    i, j = sorted(rng.sample(range(len(a)), 2))
    return copy.deepcopy(a[:i] + b[i:j] + a[j:])


def mutate(a, rng, p=0.2):
    a = copy.deepcopy(a)
    for i in range(len(a)):
        if rng.random() < p:
            a[i] = random_gene(i, rng)
    return a


def nsga2(evaluate_fn, predictor, pop_size, generations, img_size, mel_size, seed=0, log=print,
          patience=10, tol=0.005):
    """evaluate_fn(arch) -> proxy validation accuracy (short training run)."""
    rng = random.Random(seed)
    cache = {}

    def objectives(arch):
        k = arch_key(arch)
        if k not in cache:
            enc = encode(arch, img_size, mel_size)
            lat = float(predictor.predict(enc)[0])
            acc = evaluate_fn(arch)
            cache[k] = {"arch": arch, "acc": acc, "pred_latency_ms": lat,
                        "params": count_params(build(arch))}
        r = cache[k]
        return (-r["acc"], r["pred_latency_ms"], r["params"])

    pop = [random_arch(rng) for _ in range(pop_size)]
    F = [objectives(a) for a in pop]
    history, best_hv, stall = [], None, 0
    for g in range(1, generations + 1):
        t0 = time.time()
        fronts = non_dominated_sort(F)
        rank = {i: r for r, fr in enumerate(fronts) for i in fr}
        cd = {}
        for fr in fronts:
            cd.update(crowding(F, fr))

        def tour():
            i, j = rng.sample(range(len(pop)), 2)
            return i if (rank[i], -cd[i]) < (rank[j], -cd[j]) else j

        kids = [mutate(crossover(pop[tour()], pop[tour()], rng), rng) for _ in range(pop_size)]
        R = pop + kids
        FR = F + [objectives(a) for a in kids]
        new = []
        for fr in non_dominated_sort(FR):
            if len(new) + len(fr) <= pop_size:
                new += fr
            else:
                d = crowding(FR, fr)
                new += sorted(fr, key=lambda i: -d[i])[: pop_size - len(new)]
                break
        pop, F = [R[i] for i in new], [FR[i] for i in new]
        front0 = [F[i] for i in non_dominated_sort(F)[0]]
        best_acc = max(-f[0] for f in front0)
        history.append({"generation": g, "evaluated": len(cache), "front_size": len(front0),
                        "best_acc": best_acc, "sec": time.time() - t0})
        log(f"    gen {g:3d}/{generations}  evaluated {len(cache):4d}  front {len(front0):2d}  best proxy acc {best_acc:.4f}")
        stall = stall + 1 if best_hv is not None and best_acc - best_hv < tol else 0
        best_hv = max(best_acc, best_hv or 0)
        if stall >= patience:
            log("    converged (Pareto improvement < tol for `patience` generations)")
            break
    allr = list(cache.values())
    Fa = [(-r["acc"], r["pred_latency_ms"], r["params"]) for r in allr]
    pareto = [allr[i] for i in non_dominated_sort(Fa)[0]]
    return allr, pareto, history


def select(pareto, lat_max, params_max):
    """score(a) = acc(a) * 1[L(a) <= L_max] * 1[params(a) <= P_max] (Section 3.3.4)."""
    ok = [r for r in pareto if r["pred_latency_ms"] <= lat_max and r["params"] <= params_max]
    return max(ok or pareto, key=lambda r: r["acc"])
