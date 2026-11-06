"""Endpoint-probe report on a terminal_eval --dump-traj JSON: the official per-candidate
numbers plus metrics that separate WHEN from WHERE and respect multimodality. The seeds of a
window are its K candidates.

  per-candidate     mean / median / > 60 mm of the endpoint error (the official protocol)
  energy score      mean_i |x_i - y| - 1/(2K^2) sum_ij |x_i - x_j| on the chunk endpoint, a
                    proper scoring rule for the K-sample predictive distribution (mm)
  minFDE@K          best candidate's endpoint error (mm)
  time-tolerant     min over |d| <= E steps of |x_i(H) - y(H + d)|: the endpoint error once the
                    demonstration is allowed to be d steps earlier or later; the arg-min |d| is
                    the timing error (reported in ms)
  hover             the candidate ends the chunk more than 20 mm above the demonstration's endpoint
  no close          the demonstration closes within the chunk, the candidate does not; under the
                    official anchor the demonstration closes on the chunk's last step, so this
                    counts late as well as missing closes
  high close        the candidate closes more than 20 mm above the demonstration's close

  python -m tools.v3_report run.json [--ref other.json] [--out report.json]
--ref adds a paired bootstrap that resamples whole episodes (one episode = one cluster).
"""
from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np

ACTION_DT_MS = 100.0
HIGH_CLOSE_MM = 20.0
HOVER_MM = 20.0


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def window_metrics(cands: list[dict]) -> dict:
    """Metrics of one window from its K candidate rows (all from the same window)."""
    y_ext = np.asarray(cands[0]["gt_xyz_ext"])                      # (H+E, 3) mm
    xs = np.stack([np.asarray(c["pr_xyz"])[-1] for c in cands])     # (K, 3)
    H = len(cands[0]["pr_xyz"])
    E = len(y_ext) - H
    y = y_ext[H - 1]
    d = np.linalg.norm(xs - y, axis=1)
    pair = np.linalg.norm(xs[:, None] - xs[None], axis=-1)
    es = float(d.mean() - pair.sum() / (2 * len(xs) ** 2))
    lo = max(0, H - 1 - E)
    shifts = np.arange(lo, H + E) - (H - 1)                          # d in [-E, E] where defined
    dist = np.linalg.norm(xs[:, None] - y_ext[lo:H + E][None], axis=-1)   # (K, 2E+1)
    tt = dist.min(1)
    dt = np.abs(shifts[dist.argmin(1)]) * ACTION_DT_MS
    no_close = [(_ok(c["gt_close_height_mm"]) and not _ok(c["pred_close_height_mm"])) for c in cands]
    hover = [c["z_end_err_mm"] > HOVER_MM for c in cands]
    high = [(_ok(c["gt_close_height_mm"]) and _ok(c["pred_close_height_mm"])
             and c["pred_close_height_mm"] - c["gt_close_height_mm"] > HIGH_CLOSE_MM) for c in cands]
    return {"episode": cands[0]["episode"], "task": cands[0]["task"], "endpoint": d, "es": es,
            "minfde": float(d.min()), "tt": tt, "dt_ms": dt, "hover": np.asarray(hover, float),
            "no_close": np.asarray(no_close, float),
            "high": np.asarray(high, float),
            "commit": np.asarray([c["commit_ratio"] for c in cands], float)}


def load(path: str) -> dict:
    rows = json.load(open(path))["rows"]
    win = collections.defaultdict(list)
    for r in rows:
        win[(r["episode"], round(float(r["t0"]), 4))].append(r)
    return {k: window_metrics(v) for k, v in win.items()}


def summarize(W: dict) -> dict:
    ws = list(W.values())
    e = np.concatenate([w["endpoint"] for w in ws])
    out = {"n_windows": len(ws), "n_candidates": int(e.size),
           "endpoint_mean": float(e.mean()), "endpoint_median": float(np.median(e)),
           "endpoint_gt60": float((e > 60).mean()),
           "energy_score": float(np.mean([w["es"] for w in ws])),
           "minfde": float(np.mean([w["minfde"] for w in ws])),
           "tt_endpoint_mean": float(np.mean(np.concatenate([w["tt"] for w in ws]))),
           "timing_err_ms": float(np.mean(np.concatenate([w["dt_ms"] for w in ws]))),
           "hover_rate": float(np.mean(np.concatenate([w["hover"] for w in ws]))),
           "no_close_rate": float(np.mean(np.concatenate([w["no_close"] for w in ws]))),
           "high_close_rate": float(np.mean(np.concatenate([w["high"] for w in ws]))),
           "commit_median": float(np.nanmedian(np.concatenate([w["commit"] for w in ws])))}
    tasks = sorted({w["task"] for w in ws})
    out["per_task"] = {t: {"endpoint_mean": float(np.mean(np.concatenate([w["endpoint"] for w in ws if w["task"] == t]))),
                           "tt_endpoint_mean": float(np.mean(np.concatenate([w["tt"] for w in ws if w["task"] == t]))),
                           "energy_score": float(np.mean([w["es"] for w in ws if w["task"] == t]))}
                       for t in tasks}
    return out


def paired_bootstrap(A: dict, B: dict, key: str, n: int = 5000, seed: int = 0) -> dict:
    """Mean of A - B over shared windows; CI by resampling whole episodes."""
    keys = sorted(set(A) & set(B))
    by_ep = collections.defaultdict(list)
    for k in keys:
        a, b = A[k][key], B[k][key]
        by_ep[A[k]["episode"]].append(float(np.mean(a)) - float(np.mean(b)))
    eps = sorted(by_ep)
    sums = np.array([sum(by_ep[e]) for e in eps])
    cnts = np.array([len(by_ep[e]) for e in eps])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eps), size=(n, len(eps)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"diff": float(sums.sum() / cnts.sum()), "ci95": [float(np.percentile(boots, 2.5)),
                                                             float(np.percentile(boots, 97.5))],
            "n_windows": len(keys), "n_episodes": len(eps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    W = load(args.run)
    rep = {"run": args.run, "summary": summarize(W)}
    s = rep["summary"]
    print(f"{args.run}\n  endpoint {s['endpoint_mean']:.2f} mm (median {s['endpoint_median']:.2f}, "
          f">60 {100 * s['endpoint_gt60']:.1f}%) | energy {s['energy_score']:.2f} | minFDE@K {s['minfde']:.2f}"
          f" | time-tolerant {s['tt_endpoint_mean']:.2f} (timing {s['timing_err_ms']:.0f} ms)"
          f" | hover {100 * s['hover_rate']:.1f}% | no close {100 * s['no_close_rate']:.1f}%"
          f" | high close {100 * s['high_close_rate']:.1f}%"
          f" | commit {s['commit_median']:.2f}")
    print("  " + " ".join(f"{t} {v['endpoint_mean']:.1f}/{v['tt_endpoint_mean']:.1f}" for t, v in s["per_task"].items()))
    if args.ref:
        R = load(args.ref)
        rep["vs_ref"] = {"ref": args.ref, **{k: paired_bootstrap(W, R, k) for k in ("endpoint", "tt", "es")}}
        for k, v in rep["vs_ref"].items():
            if isinstance(v, dict):
                print(f"  vs ref {k}: {v['diff']:+.2f} mm, 95% CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                      f" ({v['n_episodes']} episodes)")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
