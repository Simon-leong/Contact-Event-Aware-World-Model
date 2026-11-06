"""Candidate selection on a terminal_eval --dump-select JSON: the seeds of a window are the
K candidates. Endpoint error of the candidate each rule keeps.

  mean     the per-candidate average (what terminal_eval reports, = random choice)
  medoid   GT-free: the candidate whose xyz trajectory is closest to the others
  event    GT-free: the candidate whose own close step is most likely under the event
           readout's close-time distribution (averaged over the K candidates), ties by medoid
  event+lift  the same with the lift time added
  oracle   the best candidate by the GT error (upper bound only)

  python -m tools.v2_select_eval eval/<run>/stepXXXXXX/terminal_sensor_nfe2_select.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np

from phantom.train.common import close_index

ACTION_DT = 0.1          # hardware.nuc.yaml action_rate_hz 10
BIN_DT = 0.25            # caewm2 hazard bins
LIFT_MM = 10.0


def first_bin_pmf(cum: np.ndarray, t: float | None, horizon_s: float) -> float:
    """P(first event in the bin holding t) from cumulative probabilities per bin;
    t None = no event within horizon_s."""
    cum = np.clip(np.asarray(cum, np.float64), 0.0, 1.0)
    prev = np.concatenate([[0.0], cum[:-1]])
    if t is None:
        b = int(horizon_s / BIN_DT)                   # bins fully inside the horizon
        frac = horizon_s / BIN_DT - b
        c = prev[b] + frac * (cum[b] - prev[b]) if b < len(cum) else cum[-1]
        return 1.0 - c
    b = min(max(int(math.ceil(t / BIN_DT)) - 1, 0), len(cum) - 1)
    return cum[b] - prev[b]


def own_events(actions: np.ndarray) -> tuple[float | None, float | None]:
    """(close time, lift time) in s after the window start, from a predicted chunk."""
    ci = close_index(actions[:, 6])
    if ci is None:
        return None, None
    z = np.cumsum(actions[:, 2]) * 1000.0
    after = z[ci:]
    rise = np.nonzero(after - np.minimum.accumulate(after) >= LIFT_MM)[0]
    lift = (ci + int(rise[0]) + 1) * ACTION_DT if len(rise) else None
    return (ci + 1) * ACTION_DT, lift


def medoid_order(acts: list[np.ndarray]) -> np.ndarray:
    traj = np.stack([np.cumsum(a[:, :3], 0).ravel() for a in acts])
    d = np.linalg.norm(traj[:, None] - traj[None], axis=-1).sum(1)
    return np.argsort(d, kind="stable")


def select(cands: list[dict], rule: str) -> int:
    errs = [c["endpoint_err_mm"] for c in cands]
    if rule == "oracle":
        return int(np.argmin(errs))
    acts = [np.asarray(c["pr_actions"]) for c in cands]
    order = medoid_order(acts)
    if rule == "medoid":
        return int(order[0])
    cum = np.mean([np.asarray(c["phase_cum"]) for c in cands], 0)        # (12, 3)
    H = len(acts[0]) * ACTION_DT
    score = []
    for a in acts:
        tc, tl = own_events(a)
        s = math.log(max(first_bin_pmf(cum[:, 0], tc, H), 1e-6))
        if rule == "event+lift":
            s += math.log(max(first_bin_pmf(cum[:, 1], tl, H), 1e-6))
        score.append(s)
    score = np.asarray(score)
    best = np.flatnonzero(score >= score.max() - 1e-9)
    return int(next(i for i in order if i in best))


def boot(d: np.ndarray, n: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    args = ap.parse_args(argv)
    rows = json.load(open(args.dump))["rows"]
    win = collections.defaultdict(list)
    for r in rows:
        win[(r["episode"], round(float(r["t0"]), 4))].append(r)
    keys = sorted(win)
    task = [win[k][0]["task"] for k in keys]
    res = {"mean": np.array([np.mean([c["endpoint_err_mm"] for c in win[k]]) for k in keys])}
    for rule in ("medoid", "event", "event+lift", "oracle"):
        res[rule] = np.array([win[k][select(win[k], rule)]["endpoint_err_mm"] for k in keys])
    print(f"{len(keys)} windows x {len(win[keys[0]])} candidates")
    for rule, e in res.items():
        per = " ".join(f"{t[:6]} {e[[x == t for x in task]].mean():.1f}" for t in sorted(set(task)))
        print(f"{rule:10s} {e.mean():6.2f} mm  median {np.median(e):6.2f}  >60 {100 * (e > 60).mean():4.1f}%  | {per}")
    for a, b in (("event", "mean"), ("event", "medoid"), ("event+lift", "medoid"), ("medoid", "mean")):
        d = res[a] - res[b]
        lo, hi = boot(d)
        print(f"{a} - {b}: {d.mean():+.2f} mm  95% CI [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
