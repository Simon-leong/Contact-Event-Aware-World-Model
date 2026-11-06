"""ACC gate and closing-veto decision thresholds chosen on the training subset, the same procedure
for every model, then applied unchanged to the validation rows. The veto reads p_contact =
1 - p_evt[none] against the measured first-step event, the gate reads g against gate_label
(tools.haptic_imagination_eval.veto_signal, whose deployed thresholds are 0.5).

  python -m tools.v3_acc_calib --model NAME TRAIN_ROWS VAL_ROWS [--model ...] --out calib.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from phantom.config.model import EVENT_IDX
from tools.haptic_imagination_eval import IN_CONTACT, _auroc

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _rows(p):
    d = json.load(open(p))
    return d["rows"] if isinstance(d, dict) else d


def signals(rows):
    pc, lab, g, gl = [], [], [], []
    for r in rows:
        if r.get("p_evt") is None:
            continue
        pc.append(1.0 - float(r["p_evt"][EVENT_IDX["none"]]))
        lab.append(r["ev_obs"][0] in IN_CONTACT)
        if r.get("gate") is not None:
            g.append(float(r["gate"]))
            gl.append(float(r["gate_label"]) > 0.5)
    return {"veto": (np.asarray(pc), np.asarray(lab, bool)), "gate": (np.asarray(g), np.asarray(gl, bool))}


def f1(s, y, t):
    p = s > t
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "TRAIN_ROWS", "VAL_ROWS"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {}
    for name, tr, va in a.model:
        S_tr, S_va = signals(_rows(tr)), signals(_rows(va))
        out[name] = {}
        for k in ("gate", "veto"):
            st, yt = S_tr[k]
            sv, yv = S_va[k]
            ft = np.array([f1(st, yt, t) for t in GRID])
            t = float(GRID[int(ft.argmax())])
            out[name][k] = {"threshold_train": t, "train_f1_at_t": float(ft.max()), "train_n": int(st.size),
                            "val_f1_default_0.5": f1(sv, yv, 0.5), "val_f1_train_threshold": f1(sv, yv, t),
                            "val_auroc": _auroc(sv, yv), "val_n": int(sv.size)}
            o = out[name][k]
            print(f"{name:22s} {k:5s} t*={t:.2f} (train F1 {o['train_f1_at_t']:.3f}, n={o['train_n']}) | val F1 0.5: {o['val_f1_default_0.5']:.3f}  "
                  f"t*: {o['val_f1_train_threshold']:.3f} | val AUROC {o['val_auroc']:.3f}")
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
