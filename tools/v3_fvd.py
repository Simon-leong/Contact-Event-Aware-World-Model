"""FVD of the imagined future on the 248 evaluation windows (StyleGAN-V I3D, as in
common_metrics_on_video_quality): the 12 generated frames (t0 + 0.25 ... 3 s at 4 fps) and the
real ones, short side resized to 224 and centre-cropped, scaled to [-1, 1], 400-d I3D logits
(return_features), Frechet distance of the two Gaussians. The sampling protocol is the one of
tools/video_dump.py (seed 1000, privileged contact package zeroed). A copy-last baseline (the
t0 frame held) is scored too.

  python -m tools.v3_fvd --ckpt a.pt --nfe 2 --touch sensor --out fvd.json
  python -m tools.v3_fvd compare a.feats.npz b.feats.npz     # paired bootstrap over windows
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

import phantom.caewm.paths as CP
from phantom.config.hardware import load_hardware
from phantom.model.sequence import FrameGroup
from phantom.train import common as C
from tools.video_dump import load

# StyleGAN-V I3D (i3d_torchscript.pt), downloaded separately
I3D = os.environ.get("CAEWM_I3D", str(CP.ARTEFACTS / "weights" / "i3d_torchscript.pt"))


def prep(v: torch.Tensor) -> torch.Tensor:
    """(T, 3, H, W) in [-1, 1] -> (3, T, 224, 224) in [-1, 1]."""
    T, _, H, W = v.shape
    s = 224 / min(H, W)
    v = F.interpolate(v.float(), size=(round(H * s), round(W * s)), mode="bilinear", align_corners=False)
    h, w = v.shape[-2:]
    y, x = (h - 224) // 2, (w - 224) // 2
    return v[:, :, y:y + 224, x:x + 224].clamp(-1, 1).permute(1, 0, 2, 3).contiguous()


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    m = scipy.linalg.sqrtm(s1 @ s2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * np.real(m)))


def frechet_eig(a: np.ndarray, b: np.ndarray) -> float:
    """frechet() with tr sqrt(S1 S2) from the eigenvalues of S1 S2 (fast, for the bootstrap)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ev = np.linalg.eigvals(s1 @ s2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2)
                 - 2 * np.sqrt(np.clip(ev.real, 0, None)).sum())


def compare(fa: str, fb: str, n: int = 500, seed: int = 0) -> dict:
    """FVD(a) - FVD(b) on the same windows; CI by resampling windows (the real set is resampled
    with the same indices, so both models are always scored against the same real videos)."""
    A, B = np.load(fa), np.load(fb)
    assert np.allclose(A["real"], B["real"], atol=1e-4), "the two runs used different windows"
    rng = np.random.default_rng(seed)
    N = len(A["real"])
    diffs = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        diffs.append(frechet_eig(A["model"][i], A["real"][i]) - frechet_eig(B["model"][i], B["real"][i]))
    return {"a": fa, "b": fb, "diff": frechet(A["model"], A["real"]) - frechet(B["model"], B["real"]),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "compare":
        print(json.dumps(compare(argv[1], argv[2]), indent=1))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--touch", default="sensor", choices=("sensor", "imagine"))
    ap.add_argument("--max-windows", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev, dt = "cuda", torch.bfloat16
    hw = load_hardware(a.hardware)
    pm, sampler, caewm = load(a.ckpt, hw, dev, dt)
    i3d = torch.jit.load(I3D).eval().to(dev)
    kw = dict(rescale=False, resize=False, return_features=True)
    lay = pm.rf.layout
    sl = slice(lay.frame_slice(FrameGroup.VIDEO_COND).start, lay.frame_slice(FrameGroup.VIDEO_GEN).stop)
    wins = json.loads(Path(a.windows).read_text())["windows"][:a.max_windows or None]
    feats = {"model": [], "real": [], "copy_last": []}
    for i, w in enumerate(wins):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        item = sampler.sample(ep, t0)
        batch = C.to_device(C.collate_windows([item]), dev, dt)
        for k in list(batch):
            if k == "events" or k.startswith("cpk_") or k in ("tac_fut", "tac_slip", "tac_contact"):
                batch.pop(k) if k.startswith("tac_") else batch.__setitem__(k, torch.zeros_like(batch[k]))
        pm.rf._gen = torch.Generator().manual_seed(1000)
        with torch.no_grad():
            p = (pm.rf.sample(batch, nfe=a.nfe, touch=a.touch == "sensor") if caewm
                 else pm.rf.sample(batch, nfe=a.nfe))
            frames = pm.rf.vae.decode(p.x_final_B_C_T_H_W[:, :, sl])[0].float().clamp(-1, 1)   # (3,13,H,W)
            real = item["video"].to(dev).float()                                              # (13,3,H,W)
            vids = {"model": frames.permute(1, 0, 2, 3)[1:], "real": real[1:],
                    "copy_last": real[:1].expand(len(real) - 1, -1, -1, -1)}
            for k, v in vids.items():
                feats[k].append(i3d(prep(v).unsqueeze(0), **kw)[0].float().cpu().numpy())
        if i % 20 == 0:
            print(f"{i}/{len(wins)}", flush=True)
    X = {k: np.stack(v) for k, v in feats.items()}
    res = {"ckpt": a.ckpt, "nfe": a.nfe, "touch": a.touch, "n": len(wins),
           "fvd": frechet(X["model"], X["real"]), "fvd_copy_last": frechet(X["copy_last"], X["real"])}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".feats.npz"), **X)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
