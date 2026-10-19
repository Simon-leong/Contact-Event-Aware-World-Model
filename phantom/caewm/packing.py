"""Lossless packing of tactile-tokenizer latents into backbone frame containers.

A frame is (16, 32, 40) = 16x20 tokens after the 2x2 patch embedder. One tactile
latent (16, 8, 10) is nearest-upsampled 2x into one 16x20 quadrant, so each token
holds exactly one tactile cell. A frame carries two time slots (rows) x
(common, diff) (columns):

    [ slot a common | slot a diff ]
    [ slot b common | slot b diff ]

Unpacking is a 2x2 average pool. Noise for these frames is drawn per cell and
packed the same way, so the unpacked noise has unit variance (the analogue of
the per-strip ACTION noise, rf.py P6).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pack_slots(z):
    """(B, S, 2, C, h, w), S even -> (B, C, S//2, 4h, 4w)."""
    B, S, two, C, h, w = z.shape
    z = z.reshape(B, S // 2, 2, 2, C, h, w).permute(0, 4, 1, 2, 5, 3, 6)
    z = z.reshape(B, C, S // 2, 2 * h, 2 * w)
    return z.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def unpack_slots(x):
    """(B, C, Tf, 4h, 4w) -> (B, 2*Tf, 2, C, h, w)."""
    B, C, Tf, H, W = x.shape
    x = F.avg_pool2d(x.flatten(1, 2), 2).reshape(B, C, Tf, H // 2, W // 2)
    h, w = H // 4, W // 4
    x = x.reshape(B, C, Tf, 2, h, 2, w).permute(0, 2, 3, 5, 1, 4, 6)
    return x.reshape(B, 2 * Tf, 2, C, h, w)


def to_common_diff(z_fingers):
    """(..., 2 fingers, C, h, w) [right already mirrored] -> (..., 2 [common, diff], ...)."""
    zl, zr = z_fingers.unbind(-4)
    return torch.stack([0.5 * (zl + zr), 0.5 * (zl - zr)], dim=-4)


def to_fingers(z_cd):
    zc, zd = z_cd.unbind(-4)
    return torch.stack([zc + zd, zc - zd], dim=-4)


def slot_quadrant_tokens(hidden_B_Tf_S_D, hp: int, wp: int):
    """CONTACT/OBS_MECH hidden (B, Tf, hp*wp, D) -> per-slot (common, diff) token
    means (B, 2*Tf, 2, D)."""
    B, Tf, S, D = hidden_B_Tf_S_D.shape
    h = hidden_B_Tf_S_D.reshape(B, Tf, 2, hp // 2, 2, wp // 2, D)
    return h.mean(dim=(3, 5)).reshape(B, 2 * Tf, 2, D)
