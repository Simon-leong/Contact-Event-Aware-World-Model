"""Layer 2, from the current fingertip to the planned one: GeoBridge3.

V1's GeoBridge anchors the whole exchange at the current fingertip: one TCP Gaussian G on
the token grid weights where the tactile summary enters the video, and pools the
conditioning frame into one vector added to every tactile frame. GeoBridge3 keeps that
exchange for the observed frames (OBS_MECH, VIDEO_COND) and indexes the future ones by
time: CONTACT frame k reads the conditioning frame around G_con[k], the fingertip
projected at frame k's time along the model's own estimated trajectory, and the
generated video frame k receives CONTACT frame k's summary around G_vid[k]. The tactile
side never reads a generated video frame, so the video stays droppable for the actions
and the touch. Without future anchors (geo_future None) it is exactly V1's GeoBridge, the
first sampling step and part of training run that way. Output projections start at zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from phantom.caewm.bridge import GeoBridge
from phantom.caewm.packing import slot_quadrant_tokens
from phantom.model.sequence import FrameGroup


class GeoBridge3(GeoBridge):
    def _pool(self, vc, G):
        """vc (B, Hp, Wp, D) normalised conditioning tokens, G (B, K, Hp, Wp) -> (B, K, D)."""
        w = G / G.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return self.v_out(F.silu(self.v_in(torch.einsum("bkhw,bhwd->bkd", w, vc))))

    def _summ(self, frames, Hp, Wp):
        """(B, K, S, D) tactile frames -> (B, K, D) quadrant summaries."""
        B, K = frames.shape[:2]
        q = slot_quadrant_tokens(self.norm(frames.flatten(0, 1)[:, None]), Hp, Wp)       # (BK,1,4,D)
        return self.t_out(F.silu(self.t_in(q.flatten(1, 2))).flatten(1)).view(B, K, -1)

    def forward(self, x_B_T_H_W_D, G_B_H_W, layout, geo_future=None):
        if geo_future is None:
            return super().forward(x_B_T_H_W_D, G_B_H_W, layout)
        B, T, Hp, Wp, D = x_B_T_H_W_D.shape
        x = x_B_T_H_W_D
        G = G_B_H_W.to(x.dtype)
        G_con, G_vid = geo_future["con"].to(x.dtype), geo_future["vid"].to(x.dtype)
        sl_obs, sl_con = layout.frame_slice(FrameGroup.OBS_MECH), layout.frame_slice(FrameGroup.CONTACT)
        sl_vc, sl_vg = layout.frame_slice(FrameGroup.VIDEO_COND), layout.frame_slice(FrameGroup.VIDEO_GEN)
        s_obs = self._summ(x[:, sl_obs].flatten(2, 3), Hp, Wp)                           # (B,1,D)
        s_con = self._summ(x[:, sl_con].flatten(2, 3), Hp, Wp)                           # (B,Kc,D)
        vc = self.norm(x[:, sl_vc][:, 0])
        v_obs = self._pool(vc, G[:, None])                                               # (B,1,D)
        v_con = self._pool(vc, G_con)                                                    # (B,Kc,D)

        delta = torch.zeros_like(x)
        delta[:, sl_vc] = G[:, None, :, :, None] * s_obs[:, :, None, None, :]
        if layout.has(FrameGroup.VIDEO_GEN):
            delta[:, sl_vg] = G_vid[..., None] * s_con[:, :, None, None, :]
        delta[:, sl_obs] = v_obs[:, :, None, None, :].expand(-1, -1, Hp, Wp, -1)
        delta[:, sl_con] = v_con[:, :, None, None, :].expand(-1, -1, Hp, Wp, -1)
        return x + delta
