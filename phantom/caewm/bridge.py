"""GeoBridge: geometry-anchored bidirectional vision<->touch exchange between DiT blocks.

tac -> img: per-quadrant summaries of the tactile-obs frame (slot x common/diff)
            are fused and added to every video token, weighted by the TCP Gaussian.
img -> tac: the Gaussian-weighted mean of the conditioning-frame tokens around the
            fingertips is added to every tactile token (OBS_MECH + CONTACT) — the
            channel through which the no-sensor mode imagines touch from the image.
Both output projections are zero-initialised, so at step 0 the network equals the
bridge-free one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.caewm.packing import slot_quadrant_tokens
from phantom.model.sequence import FrameGroup


class GeoBridge(nn.Module):
    def __init__(self, dim: int, rank: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.t_in = nn.Linear(dim, rank)
        self.t_out = nn.Linear(4 * rank, dim)
        self.v_in = nn.Linear(dim, rank)
        self.v_out = nn.Linear(rank, dim)
        for m in (self.t_out, self.v_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_B_T_H_W_D: torch.Tensor, G_B_H_W: torch.Tensor, layout) -> torch.Tensor:
        B, T, Hp, Wp, D = x_B_T_H_W_D.shape
        x = x_B_T_H_W_D
        G = G_B_H_W.to(x.dtype)
        obs = x[:, layout.frame_slice(FrameGroup.OBS_MECH)].flatten(2, 3)      # (B,1,S,D)
        q = slot_quadrant_tokens(self.norm(obs), Hp, Wp).flatten(1, 2)         # (B,4,D)
        s = self.t_out(F.silu(self.t_in(q)).flatten(1))                       # (B,D)
        vc = self.norm(x[:, layout.frame_slice(FrameGroup.VIDEO_COND)][:, 0])  # (B,Hp,Wp,D)
        w = G / G.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        v = self.v_out(F.silu(self.v_in((vc * w[..., None]).sum(dim=(1, 2)))))

        vid = torch.zeros(T, device=x.device, dtype=x.dtype)
        tac = torch.zeros(T, device=x.device, dtype=x.dtype)
        for g in (FrameGroup.VIDEO_COND, FrameGroup.VIDEO_GEN):
            if layout.has(g):
                vid[layout.frame_slice(g)] = 1
        for g in (FrameGroup.OBS_MECH, FrameGroup.CONTACT):
            tac[layout.frame_slice(g)] = 1
        delta = (vid.view(1, T, 1, 1, 1) * G[:, None, :, :, None] * s[:, None, None, None, :]
                 + tac.view(1, T, 1, 1, 1) * v[:, None, None, None, :])
        return x + delta
