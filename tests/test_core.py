import copy
import random

import numpy as np
import torch

from tinyfeeder.data import AVFeedingDataset, SynthConfig, generate, video_level_split
from tinyfeeder.distill import MultiTeacherKD, beta_schedule
from tinyfeeder.models import TINYFEEDER_VISUAL, Teacher, TinyFeeder, param_breakdown
from tinyfeeder.nas.search import crowding, non_dominated_sort, random_arch
from tinyfeeder.quant import model_size_mb, precision_map, prepare_qat, strip_qat


def tiny_data():
    return generate(SynthConfig(n_videos=8, samples_per_video=6, img_size=64, n_mels=32), verbose=False)


def test_forward_shapes():
    m = TinyFeeder()
    out = m(torch.randn(2, 3, 96, 96), torch.randn(2, 1, 64, 64), return_dict=True)
    assert out["logits"].shape == (2, 4) and out["feat"].shape == (2, 192)
    assert torch.all((out["alpha_v"] > 0) & (out["alpha_v"] < 1))
    assert param_breakdown(m)["total"] < 500_000


def test_video_level_split_has_no_leakage():
    d = tiny_data()
    s = video_level_split(d["video_id"], d["labels"], seed=0)
    assert not set(s["train"]) & set(s["test"]) and not set(s["val"]) & set(s["test"])
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(np.unique(d["video_id"]).tolist())
    ds = AVFeedingDataset(d, s["test"])
    assert set(d["video_id"][ds.idx]) <= set(s["test"])


def test_qat_roundtrip_and_size():
    m = TinyFeeder()
    q = prepare_qat(copy.deepcopy(m), fp16_layers={"visual.stem.0"})
    pm = precision_map(q)
    assert pm["visual.stem.0"] == "fp16" and list(pm.values()).count("int8") == len(pm) - 1
    q.train()
    q(torch.randn(4, 3, 64, 64), torch.randn(4, 1, 32, 32)).sum().backward()
    assert model_size_mb(q) < sum(p.numel() for p in m.parameters()) * 4 / 1e6
    strip_qat(q)
    assert precision_map(q) == {}


def test_kd_loss_backward():
    s, t = TinyFeeder(), Teacher("resnet18").eval()
    x, a, y = torch.randn(4, 3, 64, 64), torch.randn(4, 1, 32, 32), torch.randint(0, 4, (4,))
    kd = MultiTeacherKD(192, [t.feat_dim])
    with torch.no_grad():
        to = [t(x, a, return_dict=True)]
    loss, parts = kd(s(x, a, return_dict=True), to, y, beta=0.7)
    loss.backward()
    assert all(np.isfinite(v) for v in parts.values())
    assert beta_schedule(0, 3, 10) == 0 and abs(beta_schedule(100, 3, 10) - 0.7) < 1e-9


def test_nsga2_primitives():
    F = [(0, 1), (1, 0), (1, 1), (2, 2)]
    fronts = non_dominated_sort(F)
    assert sorted(fronts[0]) == [0, 1] and fronts[-1] == [3]
    d = crowding(F, fronts[0])
    assert all(v == float("inf") for v in d.values())
    arch = random_arch(random.Random(0))
    TinyFeeder(visual_genes=arch)(torch.randn(1, 3, 64, 64), torch.randn(1, 1, 32, 32))
    assert len(TINYFEEDER_VISUAL) == len(arch)
