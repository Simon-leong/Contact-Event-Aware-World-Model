"""Fairness check for the slip table: score the official HapticWAM models by their EventReadout
head's slip probability (instead of the ACC event distribution) on the fixed 248 windows, with
the official seeds and the same window-level label as slip_baseline_auroc (debounced slip within
1 s). Two readings: the +1 s step, and the max over the three future steps.

    python -m tools.official_head_slip_auroc --ckpt <official ckpt> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.train import common as C
from tools.cae_tactile_eval import ap as avg_prec, auroc
from tools.slip_baseline_auroc import window_label
import phantom.caewm.paths as CP
from tools.video_dump import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    a = p.parse_args()
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    pm, sampler, caewm = load(a.ckpt, hw, "cuda", torch.bfloat16)
    assert not caewm
    y, s1, smax = [], [], []
    for wi, w in enumerate(json.loads(Path(a.windows).read_text())["windows"]):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        b = C.to_device(C.collate_windows([sampler.sample(ep, t0)]), "cuda", torch.bfloat16)
        for k in list(b):
            if k == "events" or k.startswith("cpk_"):
                b[k] = torch.zeros_like(b[k])
        lab = window_label(ep.name, t0, hw.derived)
        for s in range(a.seeds):
            pm.rf._gen = torch.Generator().manual_seed(1000 * s + wi)
            with torch.no_grad():
                pr = torch.softmax(pm.rf.sample(b, nfe=1).event_logits_B_Tc_E[0].float(), -1)
            sl = pr[:, EVENT_IDX["slip"]].cpu().numpy()
            y.append(lab); s1.append(float(sl[0])); smax.append(float(sl.max()))
    y = np.asarray(y)
    out = {"ckpt": a.ckpt, "n_rows": len(y), "pos_rate": float(y.mean()),
           "head_slip_step1": {"auroc": auroc(y, np.asarray(s1)), "ap": avg_prec(y, np.asarray(s1))},
           "head_slip_max": {"auroc": auroc(y, np.asarray(smax)), "ap": avg_prec(y, np.asarray(smax))}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
