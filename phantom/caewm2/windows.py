"""Caewm2WindowSampler: the V1 windows plus the V2 targets and inputs.

Layer 1 targets (all from times after t0, never inputs):
  finger_bin (2 fingers, 2 [onset, slip]) long   first event bin in (t0, t0 + 3 s], 12 = none
  phase_bin  (3 [close, lift, release])   long   same, independent phase events
  pad_fut    (6, 2, 3)  pad force increments [|fz|, |fxy|, area] at +0.5..3 s, standardised
                        (pad_transform "std") or slog-compressed then standardised ("slog")
  wrist_fut  (6, 6)     standardised wrist wrench increments at +0.5..3 s
Layer 2 input:
  crops      (2, 3, R, R) uint8   scene-frame crops of crop_px pixels, centred on the
             projected TCP at t0 and on the TCP extrapolated crop_lead_s ahead with the
             measured TCP velocity (causal); R = crop_res
  crop_valid (2,)          float  projection inside the frame
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from phantom.caewm.windows import CaewmWindowSampler
from phantom.caewm2.events import FINGERS, force_increments, slog
from phantom.data.schema import STREAM_ARM_TCP_POSE, STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE
from phantom.data.windows import bilinear_resize

N_BINS = 12


def first_bin(times: np.ndarray, t0: float, bin_dt: float, n_bins: int = N_BINS) -> int:
    later = times[(times > t0) & (times <= t0 + bin_dt * n_bins)]
    if not len(later):
        return n_bins
    return int(np.clip(np.ceil((later.min() - t0) / bin_dt) - 1, 0, n_bins - 1))     # bin k = (k dt, (k+1) dt]


def crop_at(img: np.ndarray, u: float, v: float, px: int, res: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, res, res) uint8 crop of px pixels centred on (u, v),
    zero-padded where it leaves the frame."""
    H, W = img.shape[:2]
    h = px // 2
    out = np.zeros((px, px, 3), np.uint8)
    x0, y0 = int(round(u)) - h, int(round(v)) - h
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + px, W), min(y0 + px, H)
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    out = bilinear_resize(out.astype(np.float32), (res, res))
    return np.clip(out, 0, 255).astype(np.uint8).transpose(2, 0, 1)


class Caewm2WindowSampler(CaewmWindowSampler):
    def __init__(self, hw, bb, norm, *, event_cache: str, use_crop: bool = True,
                 crop_px: int = 160, crop_res: int = 224, crop_lead_s: float = 0.5,
                 bin_dt: float = 0.25, pad_transform: str = "std", **kw):
        super().__init__(hw, bb, norm, **kw)
        self.event_root = Path(event_cache)
        st = json.loads((self.event_root / "force_stats.json").read_text())
        self.pad_std = np.asarray(st["pad_std"], np.float32)
        self.wrist_std = np.asarray(st["wrist_std"], np.float32)
        assert pad_transform in ("std", "slog"), pad_transform
        self.pad_transform = pad_transform
        if pad_transform == "slog":
            self.pad_slog_std = np.asarray(st["pad_slog_std"], np.float32)
        self.use_crop, self.crop_px, self.crop_res = use_crop, crop_px, crop_res
        self.crop_lead_s, self.bin_dt = crop_lead_s, bin_dt
        self._events: dict[str, dict] = {}

    def _ev(self, name: str) -> dict:
        e = self._events.get(name)
        if e is None:
            e = self._events[name] = dict(np.load(self.event_root / name / "events.npz"))
        return e

    def event_keys(self, ep: Path, t0: float) -> dict:
        ev = self._ev(Path(ep).name)
        fb = np.array([[first_bin(ev[f"{k}_t_{f}"], t0, self.bin_dt) for k in ("onset", "slip")]
                       for f in FINGERS], np.int64)
        pb = np.array([first_bin(ev[f"{k}_t"], t0, self.bin_dt) for k in ("close", "lift", "release")],
                      np.int64)
        pad, wrist = force_increments(ev, t0)
        return {"finger_bin": torch.from_numpy(fb), "phase_bin": torch.from_numpy(pb),
                "pad_fut": torch.from_numpy(self.pad_target(pad)),
                "wrist_fut": torch.from_numpy(wrist / self.wrist_std)}

    def pad_target(self, pad):
        if self.pad_transform == "slog":
            return (slog(pad, self.pad_std) / self.pad_slog_std).astype(np.float32)
        return (pad / self.pad_std).astype(np.float32)

    def pad_physical(self, y):
        """Inverse of pad_target: standardised target -> force increment in sensor units."""
        y = np.asarray(y, np.float64)
        if self.pad_transform == "slog":
            z = y * self.pad_slog_std
            return np.sign(z) * self.pad_std * np.expm1(np.abs(z))
        return y * self.pad_std

    def crop_keys(self, ep: Path, t0: float) -> dict:
        c = self._ep(ep)
        img = np.asarray(c.at(STREAM_CAMERA_SCENE, t0))
        p = np.asarray(c.at(STREAM_ARM_TCP_POSE, t0), np.float64)[:3]
        vel = np.asarray(c.at(STREAM_ARM_TCP_SPEED, t0), np.float64)[:3]
        H, W = img.shape[:2]
        crops, valid = [], []
        for q in (p, p + self.crop_lead_s * vel):
            uv, ok = self.cam.uv_norm(q)
            crops.append(crop_at(img, uv[0] * W, uv[1] * H, self.crop_px, self.crop_res))
            valid.append(float(ok))
        return {"crops": torch.from_numpy(np.stack(crops)),
                "crop_valid": torch.tensor(valid, dtype=torch.float32)}

    def sample(self, ep: Path, t0: float | None = None) -> dict:
        if t0 is None:
            lo, hi = self.valid_range(ep)
            t0 = float(self.rng.uniform(lo, hi))
        w = super().sample(ep, t0)
        w.update(self.event_keys(ep, t0))
        if self.use_crop:
            w.update(self.crop_keys(ep, t0))
        return w
