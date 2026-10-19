"""CAEWM sequence layout (13 frames):

    [VIDEO_COND | VIDEO_GEN x3 | OBS_MECH (tactile obs) | OBS_PROPRIO |
     CONTACT x3 (future tactile) | ACTION x4]

OBS_MECH and CONTACT carry tactile-tokenizer latents (packing.py), not the
HapticWAM mechanical map / contact package. OBS_MECH stays in COND_GROUPS, but the
rectified flow pins it per sample (Touch Forcing), so `cond_mask_T` is only the
default with-touch mask.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields

import numpy as np

from phantom.model.sequence import FrameGroup, GroupSlot, SequenceLayout

TAC_GROUPS = (FrameGroup.OBS_MECH, FrameGroup.CONTACT)


class CaewmLayout(SequenceLayout):
    @classmethod
    def build(cls, bb, mc, hw, *, student=None, drop_video: bool = False) -> "CaewmLayout":
        base = SequenceLayout.build(bb, mc, hw, student=True, drop_video=drop_video)
        order = []
        for s in base.slots:
            if s.group == FrameGroup.OBS_PROPRIO:
                order.append((FrameGroup.OBS_MECH, 1))
            order.append((s.group, s.t_len))
        slots, t = [], 0
        for g, n in order:
            slots.append(GroupSlot(g, t, n))
            t += n
        vals = {f.name: getattr(base, f.name) for f in dc_fields(base)}
        vals.update(slots=tuple(slots), student=False)
        return cls(**vals)

    def tactile_frames_T(self) -> np.ndarray:
        m = np.zeros(self.t_total, dtype=bool)
        for g in TAC_GROUPS:
            if self.has(g):
                m[self.frame_slice(g)] = True
        return m

    def haptic_group_token_mask(self) -> np.ndarray:
        m = np.zeros(self.n_tokens, dtype=bool)
        for g in TAC_GROUPS:
            if self.has(g):
                m[self.token_slice(g)] = True
        return m
