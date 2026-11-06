"""Episode-clustered paired bootstrap of the terminal endpoint between two terminal_eval JSONs,
for all windows, Carton and the other three tasks; needs only endpoint_err_mm, so it also works
on files written without --dump-traj (reproduces tools.v3_report's endpoint bootstrap).

  python -m tools.v3_pair A.json B.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def windows(p, group=None):
    W = {}
    for r in json.loads(Path(p).read_text())["rows"]:
        if (group == "rest" and r["task"] == "Carton") or (group == "Carton" and r["task"] != "Carton"):
            continue
        W.setdefault((r["episode"], round(r["t0"], 3)), []).append(r["endpoint_err_mm"])
    return {k: float(np.mean(v)) for k, v in W.items()}


def boot(A, B, n=5000, seed=0):
    keys = sorted(set(A) & set(B))
    eps = sorted({k[0] for k in keys})
    s = np.array([sum(A[k] - B[k] for k in keys if k[0] == e) for e in eps])
    c = np.array([sum(1 for k in keys if k[0] == e) for e in eps])
    idx = np.random.default_rng(seed).integers(0, len(eps), (n, len(eps)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(argv=None) -> int:
    a, b = (sys.argv[1:] if argv is None else argv)[:2]
    for g in (None, "rest", "Carton"):
        m, lo, hi = boot(windows(a, g), windows(b, g))
        print(f"{g or 'all':7s} {m:+.2f} mm, 95% CI [{lo:+.2f}, {hi:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
