"""Per-task close decision cost matrices from counterfactual plans on the training subset.

C[j, k] is the mean two-step endpoint error (mm) when the close is decided at bin j (12 = NONE)
and the demonstration's first close is k. For k < 12 it is measured by giving the model the
demonstration's plan with only the close moved by d = j - k bins (|d| > 3 held at +-3) or decided
as NONE; for k = 12 (no close within 3 s) the two cases come from the windows of the natural
rule runs on the same subset, pooled over tasks.

  python -m tools.v3_cost_table --runs DIR --mode sensor --natural F1.json ... --out cost_sensor.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

NONE, D = 12, 3


def rows(p):
    return json.load(open(p))["rows"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--mode", required=True, choices=("sensor", "imagine"))
    ap.add_argument("--natural", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    shift = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in range(-D, D + 1):
        for r in rows(Path(a.runs) / f"{a.mode}_shift{d}.json"):
            if 0 <= r["plan_gt"][4] < NONE:
                shift[r["task"].lower()][d].append(r["endpoint_err_mm"])
    for r in rows(Path(a.runs) / f"{a.mode}_none.json"):
        if 0 <= r["plan_gt"][4] < NONE:
            shift[r["task"].lower()]["none"].append(r["endpoint_err_mm"])
    fa, tn = [], []
    for p in a.natural:
        for r in rows(p):
            if r["plan_gt"][4] == NONE:
                (tn if r["plan"][4] == NONE else fa).append(r["endpoint_err_mm"])
    c_fa = float(np.mean(fa)) if len(fa) >= 20 else float(np.mean(fa + tn))
    c_tn = float(np.mean(tn)) if len(tn) >= 20 else float(np.mean(fa + tn))
    cost, detail = {}, {}
    for task, s in shift.items():
        c = {d: float(np.mean(s[d])) for d in range(-D, D + 1)}
        c_none = float(np.mean(s["none"]))
        C = np.zeros((NONE + 1, NONE + 1))
        for k in range(NONE):
            for j in range(NONE):
                C[j, k] = c[int(np.clip(j - k, -D, D))]
            C[NONE, k] = c_none
        C[:NONE, NONE] = c_fa
        C[NONE, NONE] = c_tn
        cost[task] = C.round(4).tolist()
        detail[task] = {"by_shift": c, "none": c_none, "n": {str(d): len(s[d]) for d in list(range(-D, D + 1)) + ["none"]}}
        print(f"{task:10s} " + " ".join(f"{d:+d}:{c[d]:5.1f}" for d in range(-D, D + 1)) + f" none:{c_none:5.1f}  (n0={len(s[0])})")
    print(f"no close in the demonstration: decide a close {c_fa:.1f} (n={len(fa)}), decide NONE {c_tn:.1f} (n={len(tn)})")
    if "carton" in cost:
        cost.setdefault("Carton".lower(), cost["carton"])
    json.dump({"mode": a.mode, "cost": cost, "detail": detail, "no_close": {"close": c_fa, "none": c_tn}}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
