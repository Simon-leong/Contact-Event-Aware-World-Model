"""Per-episode event and force cache for CAEWM V2 (layer 1).

Two label families (the "dual label" scheme):
  independent phase events, from the gripper and the arm only (no tactile input)
      close    gripper position first crosses half its stroke range while closing
      lift     after a close, the TCP has risen 3 cm above its height at the close
      release  after a close, the gripper position falls back below the threshold
  tactile-derived events, from FUTURE frames of the tactile cache (never an input)
      onset    per finger, frame-level contact appears (mask_frac > tau_contact_area)
      slip     per finger, debounced slip (two consecutive frames in contact with
               slip > tau_slip), the V1 definition
Force series (targets only): per-finger pad normal force |fz|, tangential |fxy| and
contact area from the fingertip wrench/area streams at their native rate, and the
arm wrist wrench low-passed at 2 Hz and resampled to 25 Hz.

    python -m phantom.caewm2.events [--out <runs_root>/../cache/events]

Layout: <out>/<episode>/events.npz, and <out>/force_stats.json with the standard
deviation of force increments over 0.5..3 s on the training split.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FINGERS = ("left", "right")
LIFT_DZ = 0.03
WRIST_HZ = 25.0
HORIZONS = tuple(0.5 * (k + 1) for k in range(6))


def _read(ep: Path, stream: str):
    import zarr
    g = zarr.open(str(ep / f"{stream}.zarr"), mode="r")
    return np.asarray(g["ts"][:], np.float64), np.asarray(g["data"][:], np.float32)


def phase_events(tg, pos, tp, z):
    """Close / lift / release times from the gripper position and TCP height."""
    lo, hi = float(np.percentile(pos, 2)), float(np.percentile(pos, 98))
    if hi - lo < 0.05:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    closed = pos > lo + 0.5 * (hi - lo)
    edges = np.diff(closed.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if closed[0]:
        starts = np.r_[0, starts]
    close_t, lift_t, release_t = [], [], []
    for s in starts:
        e = ends[ends > s]
        t_close = tg[s]
        t_open = tg[e[0]] if len(e) else np.inf
        close_t.append(t_close)
        if np.isfinite(t_open):
            release_t.append(t_open)
        z0 = np.interp(t_close, tp, z)
        up = np.flatnonzero((tp > t_close) & (tp < t_open) & (z > z0 + LIFT_DZ))
        if len(up):
            lift_t.append(tp[up[0]])
    return np.asarray(close_t), np.asarray(lift_t), np.asarray(release_t)


def build_episode(ep: Path, out_root: Path, tac_cache: Path, hw_path: str) -> str:
    from scipy.signal import butter, filtfilt
    from phantom.config.hardware import load_hardware
    out = out_root / ep.name / "events.npz"
    if out.exists():
        return ep.name
    d = load_hardware(hw_path).derived
    tg, g = _read(ep, "gripper")
    tp, pose = _read(ep, "arm_tcp_pose")
    close_t, lift_t, release_t = phase_events(tg, g[:, 0], tp, pose[:, 2])

    tf, ft = _read(ep, "arm_ft")
    b, a = butter(2, 2.0 / (0.5 / np.median(np.diff(tf))))
    ft_lp = filtfilt(b, a, ft, axis=0)
    tw = np.arange(tf[0], tf[-1], 1.0 / WRIST_HZ)
    wrist = np.stack([np.interp(tw, tf, ft_lp[:, i]) for i in range(6)], 1).astype(np.float32)

    arrs = {"close_t": close_t, "lift_t": lift_t, "release_t": release_t,
            "wrist_ts": tw, "wrist": wrist}
    tc = tac_cache / ep.name
    for f in FINGERS:
        tw_f, wr = _read(ep, f"tactile_{f}_wrench")
        ta, ar = _read(ep, f"tactile_{f}_area")
        pad = np.stack([np.abs(wr[:, 2]), np.linalg.norm(wr[:, :2], axis=1),
                        np.interp(tw_f, ta, ar.reshape(-1))], 1).astype(np.float32)
        arrs[f"pad_ts_{f}"], arrs[f"pad_{f}"] = tw_f, pad
        ts = np.load(tc / f"ts_{f}.npy")
        mf = np.load(tc / f"mask_frac_{f}.npy")
        sl = np.load(tc / f"slip_{f}.npy")
        con = mf > d.tau_contact_area
        hot = (sl > d.tau_slip) & con
        two = hot & np.r_[False, hot[:-1]]
        arrs[f"onset_t_{f}"] = ts[np.flatnonzero(con & ~np.r_[False, con[:-1]])]
        arrs[f"slip_t_{f}"] = ts[np.flatnonzero(two & ~np.r_[False, two[:-1]])]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrs)
    return ep.name


def value_at(ts, x, t):
    """Newest sample at or before t (causal), clamped."""
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return x[i]


def force_increments(ev, t0):
    """(pad (6, 2, 3), wrist (6, 6)) increments x(t0 + tau) - x(t0)."""
    taus = t0 + np.asarray(HORIZONS)
    pad = np.stack([value_at(ev[f"pad_ts_{f}"], ev[f"pad_{f}"], taus)
                    - value_at(ev[f"pad_ts_{f}"], ev[f"pad_{f}"], t0) for f in FINGERS], 1)
    wrist = value_at(ev["wrist_ts"], ev["wrist"], taus) - value_at(ev["wrist_ts"], ev["wrist"], t0)
    return pad.astype(np.float32), wrist.astype(np.float32)


def slog(x, scale):
    """Signed log compression of force increments, sign(x) log(1 + |x| / scale)."""
    return np.sign(x) * np.log1p(np.abs(x) / scale)


def force_stats(out_root: Path, train_names: list[str], n_eps: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pads, wrists = [], []
    for name in rng.choice(train_names, min(n_eps, len(train_names)), replace=False):
        ev = dict(np.load(out_root / name / "events.npz"))
        lo, hi = ev["wrist_ts"][0], ev["wrist_ts"][-1] - HORIZONS[-1]
        for t0 in rng.uniform(lo, hi, 20) if hi > lo else []:
            p, w = force_increments(ev, t0)
            pads.append(p)
            wrists.append(w)
    pads, wrists = np.stack(pads), np.stack(wrists)
    pad_std = pads.reshape(-1, 3).std(0)
    # pad forces are heavy-tailed and task-dependent (pressing a whiteboard eraser gives
    # 3-8x the variation of a grasp): the slog targets are compressed with pad_std as the
    # knee and standardised with pad_slog_std
    return {"pad_std": pad_std.tolist(), "wrist_std": wrists.reshape(-1, 6).std(0).tolist(),
            "pad_slog_std": slog(pads, pad_std).reshape(-1, 3).std(0).tolist(),
            "n_windows": int(len(pads))}


def main(argv=None):
    import phantom.caewm.paths as CP
    from phantom.train.common import manifest_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(CP.TASKS))
    ap.add_argument("--tac-cache", default=str(CP.TAC_CACHE))
    ap.add_argument("--out", default=str(CP.ARTEFACTS / "cache" / "events"))
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[2] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    eps = sorted(p for p in Path(a.tasks).glob("*/ep_*") if p.is_dir())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    n = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for _ in pool.map(build_episode, eps, [out] * len(eps), [Path(a.tac_cache)] * len(eps),
                          [a.hardware] * len(eps), chunksize=4):
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(eps)}", flush=True)
    train = [p.name for p in manifest_split(Path(a.tasks), "train")]
    stats = force_stats(out, train)
    (out / "force_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"cached {n} episodes -> {out}; force stats {stats}")


if __name__ == "__main__":
    main()
