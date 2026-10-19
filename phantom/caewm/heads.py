"""Slip / contact readout over the six future tactile slots (+0.5 ... +3 s)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.caewm.packing import slot_quadrant_tokens


class TactileEventHead(nn.Module):
    """CONTACT hidden (B, 3, hp*wp, D) -> logits (B, 6, 2 fingers, 2 [slip, contact]).
    Causal over slots: slot k sees slots <= k."""

    def __init__(self, dim: int, hp: int, wp: int, width: int = 256):
        super().__init__()
        self.hp, self.wp = hp, wp
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(2 * dim, width)
        self.conv = nn.Conv1d(width, width, kernel_size=3)
        self.out = nn.Linear(width, 4)

    def forward(self, contact_hidden):
        h = slot_quadrant_tokens(contact_hidden.float(), self.hp, self.wp)    # (B,6,2,D)
        h = F.silu(self.proj(self.norm(h).flatten(2)))                         # (B,6,W)
        c = self.conv(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2)
        h = h + F.silu(c)
        B, S, _ = h.shape
        return self.out(h).reshape(B, S, 2, 2)


def focal_bce(logits, target, mask, gamma: float):
    """Masked binary focal loss (mean over masked entries; 0 when the mask is empty)."""
    logits, target, mask = logits.float(), target.float(), mask.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - pt) ** gamma
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce(logits, target, mask):
    ce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
