"""Encode every cached tactile frame with the frozen tokenizer:
<out>/<episode>/lat_{left,right}.npy  (T, C_TOK, 8, 10) float16, standardised, R mirrored.

    python -m phantom.caewm.build_latent_cache --ae <tactile_ae.pt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import phantom.caewm.paths as CP
from phantom.data.tactile_cache import FINGERS
from phantom.model.tactile.tokenizer import HIST, CausalTactileAE

TAC = CP.TAC_CACHE


def load_ae(path, device="cpu") -> CausalTactileAE:
    ae = CausalTactileAE()
    ae.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return ae.to(device).eval().requires_grad_(False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True)
    ap.add_argument("--out", default=str(CP.LAT_CACHE))
    a = ap.parse_args(argv)
    ae = load_ae(a.ae, "cuda")
    out = Path(a.out)
    eps = sorted(p.name for p in TAC.iterdir() if (p / "meta.json").exists())
    for n, name in enumerate(eps):
        d = out / name
        if all((d / f"lat_{f}.npy").exists() for f in FINGERS):
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in FINGERS:
            x = torch.from_numpy(np.load(TAC / name / f"fields_{f}.npy").astype(np.float32)).cuda()
            T = x.shape[0]
            idx = (torch.arange(T)[:, None] + torch.arange(-HIST + 1, 1)[None]).clamp_min(0).cuda()
            zs = [ae.encode_raw(x[idx[s:s + 512]], mirror=(f == "right")) for s in range(0, T, 512)]
            np.save(d / f"lat_{f}.npy", torch.cat(zs).cpu().numpy().astype(np.float16))
        if n % 100 == 0:
            print(f"{n}/{len(eps)}", flush=True)
    print("done", len(eps))


if __name__ == "__main__":
    main()
