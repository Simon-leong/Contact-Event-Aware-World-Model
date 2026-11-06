"""V3 window sampler: the V2 sampler plus the TCP position at t0 in the UR base frame,
the origin of the trajectory that layer 2 anchors the future frames on."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phantom.caewm2.windows import Caewm2WindowSampler
from phantom.data.schema import STREAM_ARM_TCP_POSE


class Caewm3WindowSampler(Caewm2WindowSampler):
    def sample(self, ep: Path, t0: float | None = None) -> dict:
        if t0 is None:
            lo, hi = self.valid_range(ep)
            t0 = float(self.rng.uniform(lo, hi))
        w = super().sample(ep, t0)
        p = np.asarray(self._ep(ep).at(STREAM_ARM_TCP_POSE, t0), np.float32)[:3]
        w["tcp_xyz"] = torch.from_numpy(p.copy())
        return w
