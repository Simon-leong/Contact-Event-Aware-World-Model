"""Replan latency of one checkpoint on the current GPU: rf.sample with 4 candidates
(k_seeds=4), 1 step, EMA weights, no compilation, batch built from one validation
window; median over --iters after --warmup. Official models get the previous replan's
package (the deploy path, no inner anticipation sample); CAEWM needs none.
`--refine N` additionally times CAEWM's visualisation path (video-only refinement)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
from phantom.train import common as C
import phantom.caewm.paths as CP
from tools.video_dump import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--touch", default="sensor")
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--k", type=int, default=4, help="candidates per replan (deploy default 4)")
    a = ap.parse_args()
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    pm, sampler, caewm = load(a.ckpt, hw, "cuda", torch.bfloat16)
    w = json.load(open(CP.EVAL_WINDOWS))["windows"][0]
    b = C.to_device(C.collate_windows([sampler.sample(CP.episode_path(w["episode"]), float(w["t0"]))]), "cuda", torch.bfloat16)
    for k in list(b):
        if k == "events" or k.startswith("cpk_"):
            b[k] = torch.zeros_like(b[k])
        elif k in ("tac_fut", "tac_slip", "tac_contact"):
            b.pop(k)
    prev = None if caewm else pm.rf.sample(b, nfe=1).cpk

    def run():
        if caewm:
            return pm.rf.sample(b, nfe=a.nfe, k_seeds=a.k, touch=a.touch == "sensor", video_refine_steps=a.refine)
        return pm.rf.sample(b, nfe=a.nfe, k_seeds=a.k, prev_cpk=prev)
    ts = []
    with torch.no_grad():
        for i in range(a.warmup + a.iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            run()
            torch.cuda.synchronize()
            if i >= a.warmup:
                ts.append(1000 * (time.perf_counter() - t))
    print(json.dumps({"ckpt": a.ckpt, "touch": a.touch, "refine": a.refine, "nfe": a.nfe, "k": a.k, "median_ms": float(np.median(ts)),
                      "p90_ms": float(np.percentile(ts, 90)), "gpu": torch.cuda.get_device_name()}))


if __name__ == "__main__":
    main()
