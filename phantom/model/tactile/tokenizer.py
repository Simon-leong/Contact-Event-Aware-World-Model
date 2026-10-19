"""Causal spatio-temporal tactile tokenizer for CAEWM.

Per finger, the encoder sees the newest HIST field frames (5 channels: deformation
x/y, depth, shear x/y; normalised by per-channel std) plus their frame-to-frame
differences, and emits one latent z_t of shape (C_TOK, 8, 10). The decoder sees z_t
only — the world model predicts future latents with no future history — and
reconstructs the current field x_t and its change dx_t = x_t - x_{t-1}.

Both fingers share one network. The right pad is mirrored along H before encoding
(measured: depth corr L vs flipH(R) = 0.40, vs unflipped 0.05), so the common/diff
split (zL +- zR') / 2 used by the world model compares the same physical cells.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HIST = 4
N_CH = 5
C_TOK = 16
LAT_HW = (8, 10)
DEPTH_CH = 2
TAU_DEPTH = 0.05


def _gn(c):
    return nn.GroupNorm(min(32, c // 4), c)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.n1, self.c1 = _gn(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.n2, self.c2 = _gn(c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CausalTactileAE(nn.Module):
    def __init__(self, width=(64, 128, 256), c_tok: int = C_TOK):
        super().__init__()
        c_in = N_CH * HIST + N_CH * (HIST - 1)
        w0, w1, w2 = width
        self.register_buffer("ch_std", torch.ones(N_CH))
        self.register_buffer("z_mean", torch.zeros(c_tok))
        self.register_buffer("z_std", torch.ones(c_tok))
        self.register_buffer("dz_std", torch.ones(c_tok))
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, w0, 3, padding=1), ResBlock(w0, w0),
            nn.Conv2d(w0, w1, 3, stride=2, padding=1), ResBlock(w1, w1),
            nn.Conv2d(w1, w2, 3, stride=2, padding=1), ResBlock(w2, w2),
            nn.Conv2d(w2, w2, 3, stride=2, padding=1), ResBlock(w2, w2), ResBlock(w2, w2),
            _gn(w2), nn.SiLU(), nn.Conv2d(w2, 2 * c_tok, 1),
        )
        up = lambda a, b: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv2d(a, b, 3, padding=1))
        self.dec = nn.Sequential(
            nn.Conv2d(c_tok, w2, 3, padding=1), ResBlock(w2, w2), ResBlock(w2, w2),
            up(w2, w2), ResBlock(w2, w2),
            up(w2, w1), ResBlock(w1, w1),
            up(w1, w0), ResBlock(w0, w0),
            _gn(w0), nn.SiLU(), nn.Conv2d(w0, 2 * N_CH, 3, padding=1),
        )
        self.c_tok = c_tok

    # --- normalisation -------------------------------------------------
    def norm(self, x):                       # raw (..., 5, H, W) -> normalised
        return x / self.ch_std.view(-1, 1, 1)

    def denorm(self, x):
        return x * self.ch_std.view(-1, 1, 1)

    # --- encode / decode (normalised field space) ----------------------
    def stack_input(self, hist_B_K_C_H_W):
        """(B, HIST, 5, H, W) normalised, oldest first -> (B, 35, H, W)."""
        d = hist_B_K_C_H_W[:, 1:] - hist_B_K_C_H_W[:, :-1]
        return torch.cat([hist_B_K_C_H_W.flatten(1, 2), d.flatten(1, 2)], 1)

    def encode_dist(self, hist_B_K_C_H_W):
        mu, logvar = self.enc(self.stack_input(hist_B_K_C_H_W)).chunk(2, 1)
        return mu, logvar.clamp(-12.0, 4.0)

    def decode(self, z):
        """z (B, C_TOK, 8, 10) -> (x_t, dx_t), each (B, 5, H, W) normalised."""
        return self.dec(z).chunk(2, 1)

    # --- raw-unit helpers used by the world model ----------------------
    @torch.no_grad()
    def encode_raw(self, hist_raw, mirror: bool = False):
        """Raw cached fields (B, HIST, 5, H, W) -> standardised latent (B, C_TOK, 8, 10).
        mirror=True for the right finger."""
        x = self.norm(hist_raw.float())
        if mirror:
            x = x.flip(-2)
        mu, _ = self.encode_dist(x)
        return (mu - self.z_mean.view(-1, 1, 1)) / self.z_std.view(-1, 1, 1)

    def decode_raw(self, z_std, mirror: bool = False):
        """Standardised latent -> raw-unit (x_t, dx_t)."""
        z = z_std * self.z_std.view(-1, 1, 1) + self.z_mean.view(-1, 1, 1)
        x, dx = self.decode(z)
        x, dx = self.denorm(x), self.denorm(dx)
        if mirror:
            x, dx = x.flip(-2), dx.flip(-2)
        return x, dx


def common_diff(zl, zr):
    """Both standardised, zr already mirrored: (common, diff)."""
    return 0.5 * (zl + zr), 0.5 * (zl - zr)


def split_common_diff(zc, zd):
    return zc + zd, zc - zd


def contact_mask(x_raw):
    return x_raw[..., DEPTH_CH, :, :].abs() > TAU_DEPTH
