"""Layer 2: future-aware high-resolution fingertip bridge.

CropEncoder   frozen DINOv3 ViT-L/16 over the two scene crops (TCP at t0 and the TCP
              extrapolated ahead); patch tokens only. Attached to the flow model
              outside the net, so it is neither trained nor saved.
CropBridge    every tactile token (OBS_MECH and CONTACT frames) queries the crop tokens
              by cross-attention. Queries carry a quadrant embedding, because each
              tactile frame holds two time slots x (common, differential) quadrants, and
              keys carry a view embedding (current / ahead). A learned null token keeps
              the attention defined when both crops fall outside the frame. The output
              projection starts at zero.
GeoBridge2    the V1 GeoBridge restricted to its touch -> image direction; with the crop
              bridge in place, the image -> touch direction comes from the crops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.caewm.packing import slot_quadrant_tokens
from phantom.model.sequence import FrameGroup

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropEncoder(nn.Module):
    def __init__(self, name: str, device, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, local_files_only=True).to(device, dtype).eval()
        self.model.requires_grad_(False)
        self.n_skip = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.dim = int(self.model.config.hidden_size)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, crops_B_V_3_R_R: torch.Tensor) -> torch.Tensor:
        """uint8 or [0, 1] float crops (B, V, 3, R, R) -> patch tokens (B, V, N, dim)."""
        B, V = crops_B_V_3_R_R.shape[:2]
        x = crops_B_V_3_R_R.flatten(0, 1).float()
        if crops_B_V_3_R_R.dtype == torch.uint8:
            x = x / 255.0
        x = ((x - self.mean.to(x.device)) / self.std.to(x.device)).to(next(self.model.parameters()).dtype)
        tok = self.model(pixel_values=x).last_hidden_state[:, self.n_skip:]
        return tok.reshape(B, V, tok.shape[1], tok.shape[2])


def quadrant_ids(hp: int, wp: int, device) -> torch.Tensor:
    """(hp*wp,) quadrant index of every token of a tactile frame:
    0 slot-a common, 1 slot-a diff, 2 slot-b common, 3 slot-b diff (packing.py layout)."""
    r = torch.arange(hp, device=device).view(hp, 1) >= hp // 2
    c = torch.arange(wp, device=device).view(1, wp) >= wp // 2
    return (r.long() * 2 + c.long()).flatten()


class CropBridge(nn.Module):
    def __init__(self, dim: int, kv_dim: int, rank: int = 256, heads: int = 4, n_views: int = 2):
        super().__init__()
        self.heads, self.rank = heads, rank
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.kv_norm = nn.LayerNorm(kv_dim, elementwise_affine=False, eps=1e-6)
        self.q = nn.Linear(dim, rank)
        self.k = nn.Linear(kv_dim, rank)
        self.v = nn.Linear(kv_dim, rank)
        self.out = nn.Linear(rank, dim)
        self.quad = nn.Parameter(torch.zeros(4, rank))
        self.view = nn.Parameter(torch.zeros(n_views, rank))
        self.null_k = nn.Parameter(torch.zeros(1, rank))
        self.null_v = nn.Parameter(torch.zeros(1, rank))
        nn.init.normal_(self.quad, std=0.02)
        nn.init.normal_(self.view, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_B_T_H_W_D, kv_B_V_N_C, valid_B_V, layout):
        B, T, Hp, Wp, D = x_B_T_H_W_D.shape
        V, N = kv_B_V_N_C.shape[1:3]
        frames = torch.cat([torch.arange(T, device=x_B_T_H_W_D.device)[layout.frame_slice(g)]
                            for g in (FrameGroup.OBS_MECH, FrameGroup.CONTACT)])
        tac = x_B_T_H_W_D[:, frames].reshape(B, len(frames), Hp * Wp, D)
        q = self.q(self.norm(tac)) + self.quad.to(tac.dtype)[quadrant_ids(Hp, Wp, tac.device)]
        kv = self.kv_norm(kv_B_V_N_C.to(tac.dtype))
        k = self.k(kv) + self.view.to(tac.dtype)[:, None]
        v = self.v(kv)
        k = torch.cat([self.null_k.to(k.dtype).expand(B, 1, -1), k.flatten(1, 2)], 1)   # (B,1+V*N,r)
        v = torch.cat([self.null_v.to(v.dtype).expand(B, 1, -1), v.flatten(1, 2)], 1)
        keep = torch.cat([torch.ones(B, 1, device=k.device, dtype=torch.bool),
                          (valid_B_V > 0.5).repeat_interleave(N, 1)], 1)                  # (B,1+V*N)
        h, dh = self.heads, self.rank // self.heads
        qh = q.reshape(B, -1, h, dh).transpose(1, 2)                                     # (B,h,L,dh)
        kh = k.reshape(B, -1, h, dh).transpose(1, 2)
        vh = v.reshape(B, -1, h, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=keep[:, None, None, :])
        o = self.out(o.transpose(1, 2).reshape(B, -1, self.rank)).reshape(B, len(frames), Hp, Wp, D)
        x = x_B_T_H_W_D.clone()
        x[:, frames] = x[:, frames] + o
        return x


class GeoBridge2(nn.Module):
    """Touch -> image half of the V1 GeoBridge (same computation and zero init)."""

    def __init__(self, dim: int, rank: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.t_in = nn.Linear(dim, rank)
        self.t_out = nn.Linear(4 * rank, dim)
        nn.init.zeros_(self.t_out.weight)
        nn.init.zeros_(self.t_out.bias)

    def forward(self, x_B_T_H_W_D, G_B_H_W, layout):
        B, T, Hp, Wp, D = x_B_T_H_W_D.shape
        G = G_B_H_W.to(x_B_T_H_W_D.dtype)
        obs = x_B_T_H_W_D[:, layout.frame_slice(FrameGroup.OBS_MECH)].flatten(2, 3)
        q = slot_quadrant_tokens(self.norm(obs), Hp, Wp).flatten(1, 2)
        s = self.t_out(F.silu(self.t_in(q)).flatten(1))
        vid = torch.zeros(T, device=x_B_T_H_W_D.device, dtype=x_B_T_H_W_D.dtype)
        for g in (FrameGroup.VIDEO_COND, FrameGroup.VIDEO_GEN):
            if layout.has(g):
                vid[layout.frame_slice(g)] = 1
        return x_B_T_H_W_D + vid.view(1, T, 1, 1, 1) * G[:, None, :, :, None] * s[:, None, None, None, :]
