"""Deployed ACC readout: the contact gate and the closing veto of one plan, read at the first
sampling step and averaged over the draws of a window (the seeds of a cae_imagination_eval
rows file). The veto fuses the ACC contact probability with the contact the model predicts for
the first future slot, a * p_contact + (1 - a) * imagined; the weight a is chosen on the
training-subset rows (veto F1 at the deployed threshold 0.5) and applied unchanged to the
validation rows. The same procedure applies to HapticWAM rows.

  python -m tools.cae_imagination_eval --ckpt CK --touch sensor \
      --windows configs/eval/windows_train8x2_seed7.json --out train8.json      # -> train8_rows.json
  python -m tools.cae_imagination_eval --ckpt CK --touch sensor --out val.json  # -> val_rows.json
  python -m tools.v3_acc_deploy --train-rows train8_rows.json --val-rows val_rows.json --out acc.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from phantom.config.model import EVENT_IDX
from tools.haptic_imagination_eval import IN_CONTACT, _auroc

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _rows(p):
    d = json.load(open(p))
    return d["rows"] if isinstance(d, dict) else d


def per_window(rows):
    """(n_windows, 5): mean gate, mean ACC contact probability, share of draws imagining
    contact at the first slot, gate label, veto label."""
    W = collections.defaultdict(list)
    for r in rows:
        if r.get("p_evt") is not None:
            W[(r["episode"], round(r["t0"], 4))].append(r)
    out = []
    for v in W.values():
        out.append((np.mean([x["gate"] for x in v]),
                    np.mean([1.0 - x["p_evt"][EVENT_IDX["none"]] for x in v]),
                    np.mean([x["ev_pred"][0] in IN_CONTACT for x in v]),
                    v[0]["gate_label"] > 0.5, v[0]["ev_obs"][0] in IN_CONTACT))
    return np.asarray(out, float)


def f1(score, label, thr=0.5):
    p, y = score > thr, label.astype(bool)
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rows", required=True)
    ap.add_argument("--val-rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    T, V = per_window(_rows(a.train_rows)), per_window(_rows(a.val_rows))
    train_f1 = [f1(w * T[:, 1] + (1 - w) * T[:, 2], T[:, 4]) for w in GRID]
    w = float(GRID[int(np.argmax(train_f1))])
    veto = w * V[:, 1] + (1 - w) * V[:, 2]
    out = {"n_windows": len(V), "fusion_weight": w,
           "gate_auroc": _auroc(V[:, 0], V[:, 3].astype(bool)), "gate_f1": f1(V[:, 0], V[:, 3]),
           "veto_auroc": _auroc(veto, V[:, 4].astype(bool)), "veto_f1": f1(veto, V[:, 4]),
           "imagined_only_veto_f1": f1(V[:, 2], V[:, 4])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
