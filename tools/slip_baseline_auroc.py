"""Slip anticipation of the official HapticWAM models on the fixed 248 windows.

Labels: CAEWM debounced slip (two consecutive native frames in contact with slip >
tau_slip) within (t0, t0+1] s on either finger — the same label cae_tactile_eval
uses for its window-level score. Score: the ACC event distribution's slip
probability p_evt[slip] logged by the official imagination eval (rows files).
"""
import json
import sys
from pathlib import Path

import numpy as np

import phantom.caewm.paths as CP
from phantom.caewm.windows import FUT_TAUS
from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.data.tactile_cache import FINGERS
from tools.cae_tactile_eval import ap, auroc

CACHE = CP.TAC_CACHE


def window_label(name, t0, d, horizon=1.0):
    lab = False
    for f in FINGERS:
        ts = np.load(CACHE / name / f"ts_{f}.npy")
        mf = np.load(CACHE / name / f"mask_frac_{f}.npy")
        sl = np.load(CACHE / name / f"slip_{f}.npy")
        hot = (sl > d.tau_slip) & (mf > d.tau_contact_area)
        two = hot & np.concatenate([[False], hot[:-1]])
        lab |= bool(two[(ts > t0) & (ts <= t0 + horizon)].any())
    return lab


def main():
    d = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml")).derived
    out = {}
    for rows_file in sys.argv[1:]:
        rows = json.loads(Path(rows_file).read_text())["rows"]
        y = np.array([window_label(r["episode"], r["t0"], d) for r in rows])
        s = np.array([r["p_evt"][EVENT_IDX["slip"]] for r in rows])
        out[Path(rows_file).stem] = {"n_rows": len(rows), "pos_rate": float(y.mean()),
                                     "slip_window_auroc_1s": auroc(y, s), "slip_window_ap_1s": ap(y, s)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
