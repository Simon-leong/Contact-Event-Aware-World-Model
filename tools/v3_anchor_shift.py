"""How far layer 2's plan-indexed fingertip anchors move from the current anchor on the token
grid, measured on the demonstration's TCP trajectory (an upper bound on the model's own estimate)
over the evaluation windows; anchors beyond the action chunk are held at its end, as in
CaewmRectifiedFlow3._anchors.

  python -m tools.v3_anchor_shift --out eval/v3_final/anchor_shift.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import phantom.caewm.paths as CP
from phantom.caewm.windows import FUT_TAUS
from phantom.caewm3.geometry import TorchCamera
from phantom.data.schema import STREAM_ARM_TCP_POSE
from phantom.data.windows import _EpisodeCache


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap.add_argument("--hp", type=int, default=16)
    ap.add_argument("--wp", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--chunk-s", type=float, default=1.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cam = TorchCamera()
    taus = [min(t, a.chunk_s) for t in FUT_TAUS]
    D = []
    for w in json.load(open(a.windows))["windows"]:
        ep = _EpisodeCache(CP.episode_path(w["episode"]))
        P = np.stack([np.asarray(ep.at(STREAM_ARM_TCP_POSE, float(w["t0"]) + t), np.float64)[:3] for t in [0.0] + taus])
        uv, ok = cam.uv_norm(torch.from_numpy(P))
        if bool(ok.all()):
            tok = uv.double().numpy() * np.array([a.wp, a.hp])
            D.append(np.linalg.norm(tok[1:] - tok[0], axis=1))
    D = np.array(D)
    out = {"n_windows": len(D), "taus_s": taus, "grid": [a.hp, a.wp], "sigma_tokens": a.sigma,
           "median_shift_tokens": np.median(D, 0).tolist(), "p90_shift_tokens": np.percentile(D, 90, 0).tolist(),
           "mean_overlap_with_current": np.exp(-D ** 2 / (4 * a.sigma ** 2)).mean(0).tolist(),
           "frac_shift_gt_sigma_at_chunk_end": float((D[:, -1] > a.sigma).mean())}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
