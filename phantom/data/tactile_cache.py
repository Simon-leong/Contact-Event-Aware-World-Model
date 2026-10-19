"""Per-episode tactile cache for the CAEWM tactile tokenizer and windows.

For each episode and finger: the five informative field channels of `fields_ds`
(deformation x/y, depth, shear x/y; the distributed-force channels are uncalibrated
and dropped) resized to TAC_HW, stored float16 as (T, 5, h, w), plus the stream
timestamps and per-frame derived quantities at the native rate (mask_frac, slip,
CoP from `derived.derive_timestep`, computed on the native 72x96 frame).

Layout: <cache_root>/<episode name>/{fields_left,fields_right}.npy (memory-mappable),
{ts,mask_frac,slip,cop}_{left,right}.npy, meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TAC_CH = (0, 1, 2, 3, 4)          # deformation2d x/y, depth, shear x/y
TAC_HW = (64, 80)
FINGERS = ("left", "right")


def _resize(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """(T, C, H, W) float32 -> (T, C, h, w) bilinear (align_corners=False)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x))
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False).numpy()


def build_episode(ep: Path, out_root: Path, hw_path: str) -> dict:
    import zarr
    from phantom.config.hardware import load_hardware
    from phantom.data.derived import derive_timestep
    hw = load_hardware(hw_path)
    out = out_root / ep.name
    done = out / "meta.json"
    if done.exists():
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    meta = {"episode": str(ep), "fingers": {}}
    for f in FINGERS:
        g = zarr.open(str(ep / f"tactile_{f}_fields_ds.zarr"), mode="r")
        ts = np.asarray(g["ts"][:], np.float64)
        raw = np.asarray(g["data"][:], np.float32)                  # (T, 72, 96, 8)
        mf = np.zeros(len(ts), np.float32)
        sl = np.zeros(len(ts), np.float32)
        cop = np.full((len(ts), 2), np.nan, np.float32)
        for i in range(len(ts)):
            d = derive_timestep(raw[i], raw[i - 1] if i else None,
                                float(ts[i] - ts[i - 1]) if i else 1.0, hw)
            mf[i], sl[i], cop[i] = d["mask_frac"], d["slip"], d["cop"]
        x = raw[..., list(TAC_CH)].transpose(0, 3, 1, 2)            # (T, 5, 72, 96)
        np.save(out / f"fields_{f}.npy", _resize(x, TAC_HW).astype(np.float16))
        np.save(out / f"ts_{f}.npy", ts)
        np.save(out / f"mask_frac_{f}.npy", mf)
        np.save(out / f"slip_{f}.npy", sl)
        np.save(out / f"cop_{f}.npy", cop)
        dt = np.diff(ts)
        meta["fingers"][f] = {"n": int(len(ts)), "rate_hz": float(1.0 / np.median(dt)) if len(dt) else 0.0}
    done.write_text(json.dumps(meta))
    return meta


class TactileCache:
    """Read-only access to one cached episode (memory-mapped)."""

    def __init__(self, cache_root: Path, ep_name: str):
        d = Path(cache_root) / ep_name
        self.fields = {f: np.load(d / f"fields_{f}.npy", mmap_mode="r") for f in FINGERS}
        self.ts = {f: np.load(d / f"ts_{f}.npy") for f in FINGERS}
        self.mask_frac = {f: np.load(d / f"mask_frac_{f}.npy") for f in FINGERS}
        self.slip = {f: np.load(d / f"slip_{f}.npy") for f in FINGERS}

    def index_at(self, finger: str, t: float) -> int:
        """Newest frame at or before t (causal), clamped to the stream."""
        ts = self.ts[finger]
        return int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))


def main(argv=None):
    import phantom.caewm.paths as CP
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(CP.TASKS))
    ap.add_argument("--out", default=str(CP.TAC_CACHE))
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[2] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    eps = sorted(p for p in Path(a.tasks).glob("*/ep_*") if p.is_dir())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    n = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for meta in pool.map(build_episode, eps, [out] * len(eps), [a.hardware] * len(eps), chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(eps)}", flush=True)
    print(f"cached {n} episodes -> {out}")


if __name__ == "__main__":
    main()
