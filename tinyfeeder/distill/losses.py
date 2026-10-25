"""Multi-teacher knowledge distillation losses (Section 3.5.2)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(s_logits, t_logits_list, tau=4.0):
    """tau^2 * KL( softmax(z_s/tau) || mean_m softmax(z_t^m/tau) )."""
    p_t = torch.stack([F.softmax(t / tau, 1) for t in t_logits_list]).mean(0)
    return F.kl_div(F.log_softmax(s_logits / tau, 1), p_t, reduction="batchmean") * tau ** 2


def attention_map(fmap):
    a = fmap.pow(2).sum(1)  # sum_c F_c^2
    return F.normalize(a.flatten(1), dim=1), a.shape[-2:]


def attention_transfer_loss(s_map, t_map):
    if s_map.shape[-2:] != t_map.shape[-2:]:
        t_map = F.interpolate(t_map, size=s_map.shape[-2:], mode="bilinear", align_corners=False)
    a_s, _ = attention_map(s_map)
    a_t, _ = attention_map(t_map)
    return (a_s - a_t).pow(2).sum(1).mean()


class MultiTeacherKD(nn.Module):
    """L = (1-beta) CE + beta (l1 L_soft + l2 L_feat + l3 L_attn)."""

    def __init__(self, s_feat_dim, t_feat_dims, tau=4.0, lambdas=(1.0, 0.5, 0.3), label_smoothing=0.1):
        super().__init__()
        self.tau, self.l1, self.l2, self.l3 = tau, *lambdas
        # learned projections W_l aligning student features to each teacher
        self.proj = nn.ModuleList([nn.Linear(s_feat_dim, d) for d in t_feat_dims])
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, s_out, t_outs, y, beta):
        ce = self.ce(s_out["logits"], y)
        if beta <= 0:
            return ce, {"ce": ce.item(), "soft": 0.0, "feat": 0.0, "attn": 0.0}
        soft = soft_target_loss(s_out["logits"], [t["logits"] for t in t_outs], self.tau)
        feat = sum(F.mse_loss(F.normalize(p(s_out["feat"]), dim=1), F.normalize(t["feat"], dim=1))
                   for p, t in zip(self.proj, t_outs)) / len(t_outs)
        attn = sum(attention_transfer_loss(s_out["vmap"], t["vmap"]) for t in t_outs) / len(t_outs)
        kd = self.l1 * soft + self.l2 * feat + self.l3 * attn
        loss = (1 - beta) * ce + beta * kd
        return loss, {"ce": ce.item(), "soft": soft.item(), "feat": feat.item(), "attn": attn.item()}


def beta_schedule(epoch, warmup, ramp, beta_max=0.7):
    """Phase 1: hard labels only; Phase 2: beta grows linearly to beta_max."""
    if epoch < warmup:
        return 0.0
    return beta_max * min(1.0, (epoch - warmup + 1) / max(1, ramp))
