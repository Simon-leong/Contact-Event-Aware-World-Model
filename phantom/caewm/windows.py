"""CaewmWindowSampler: HapticWAM windows (student keys: no gel / keyframe / reactive
reads) plus the CAEWM tactile keys, all read from the per-frame latent cache:

  tac_obs     (2 slots [t0-0.5, t0], 2 fingers, C, h, w)  standardised z, R mirrored
  tac_fut     (6 slots [t0+0.5 .. t0+3], 2, C, h, w)       (z(t0+tau) - z(t0)) / dz_std
  tac_slip    (6, 2)  debounced slip in (t0+tau-0.5, t0+tau]: two consecutive native
                      frames in contact with slip > tau_slip
  tac_contact (6, 2)  mask_frac > tau_contact_area at the newest frame <= t0+tau
  tcp_uv (2,), tcp_valid ()  TCP projected into the scene camera at t0
Every lookup is causal (newest frame at or before the query time).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phantom.caewm.geometry import Camera
from phantom.data.schema import STREAM_ARM_TCP_POSE
from phantom.data.tactile_cache import FINGERS
from phantom.data.windows import WindowSampler

OBS_LAGS = (-0.5, 0.0)
FUT_TAUS = tuple(0.5 * (k + 1) for k in range(6))


class _LatEpisode:
    def __init__(self, lat_root: Path, tac_root: Path, name: str):
        d, c = lat_root / name, tac_root / name
        self.z = {f: np.load(d / f"lat_{f}.npy", mmap_mode="r") for f in FINGERS}
        self.ts = {f: np.load(c / f"ts_{f}.npy") for f in FINGERS}
        self.mask_frac = {f: np.load(c / f"mask_frac_{f}.npy") for f in FINGERS}
        self.slip = {f: np.load(c / f"slip_{f}.npy") for f in FINGERS}

    def idx(self, f, t):
        ts = self.ts[f]
        return np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)


class CaewmWindowSampler(WindowSampler):
    def __init__(self, hw, bb, norm, *, lat_cache: str, tac_cache: str, dz_std, **kw):
        kw["student"] = True
        super().__init__(hw, bb, norm, **kw)
        self.lat_root, self.tac_root = Path(lat_cache), Path(tac_cache)
        self.dz_std = np.asarray(dz_std, np.float32).reshape(1, -1, 1, 1)
        self.cam = Camera()
        self._lat: dict[str, _LatEpisode] = {}

    def _lat_ep(self, name):
        e = self._lat.get(name)
        if e is None:
            e = self._lat[name] = _LatEpisode(self.lat_root, self.tac_root, name)
        return e

    def tactile_keys(self, ep: Path, t0: float) -> dict:
        d = self.hw.derived
        e = self._lat_ep(Path(ep).name)
        obs, fut, slip, con = [], [], np.zeros((6, 2), np.float32), np.zeros((6, 2), np.float32)
        for fi, f in enumerate(FINGERS):
            zo = np.asarray(e.z[f][e.idx(f, t0 + np.asarray(OBS_LAGS))], np.float32)
            zf = np.asarray(e.z[f][e.idx(f, t0 + np.asarray(FUT_TAUS))], np.float32)
            obs.append(zo)
            fut.append((zf - zo[-1:]) / self.dz_std)
            ts, mf, sl = e.ts[f], e.mask_frac[f], e.slip[f]
            hot = (sl > d.tau_slip) & (mf > d.tau_contact_area)
            two = hot & np.concatenate([[False], hot[:-1]])
            for k, tau in enumerate(FUT_TAUS):
                hi = t0 + tau
                sel = (ts > hi - 0.5) & (ts <= hi)
                slip[k, fi] = float(two[sel].any())
                con[k, fi] = float(mf[e.idx(f, hi)] > d.tau_contact_area)
        uv, ok = self.cam.uv_norm(np.asarray(self._ep(ep).at(STREAM_ARM_TCP_POSE, t0))[:3])
        return {
            "tac_obs": torch.from_numpy(np.stack(obs, 1)),
            "tac_fut": torch.from_numpy(np.stack(fut, 1)),
            "tac_slip": torch.from_numpy(slip),
            "tac_contact": torch.from_numpy(con),
            "tcp_uv": torch.from_numpy(uv),
            "tcp_valid": torch.tensor(float(ok)),
        }

    def sample(self, ep: Path, t0: float | None = None) -> dict:
        if t0 is None:
            lo, hi = self.valid_range(ep)
            t0 = float(self.rng.uniform(lo, hi))
        w = super().sample(ep, t0)
        w.update(self.tactile_keys(ep, t0))
        return w
