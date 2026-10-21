"""Dump decoded future frames for the video-quality comparison (stage 1 of 2).

Works for HapticWAM checkpoints and CAEWM checkpoints. For each window of a fixed
window file, samples once (seed 1000), decodes the conditioning + generated latents
through the Wan VAE and stores, as uint8, the predicted and real frames at
t0 + 1, 2, 3 s plus the real frame at t0 (the copy-last-frame reference), and the
real TCP pixel position at those times (gripper-region crops in stage 2).

    python tools/video_dump.py --ckpt ... --windows windows_val124x2_seed7.json \
        --out <dir> [--nfe 1] [--touch sensor|imagine] [--refine 10]
Stage 2: tools/video_quality_score.py (an env with lpips + scikit-image).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phantom.config.hardware import load_hardware
import phantom.caewm.paths as CP
from phantom.config.paths import load_paths
from phantom.data.schema import STREAM_ARM_TCP_POSE, NormStats
from phantom.model.sequence import FrameGroup
from phantom.train import common as C

PIX = (4, 8, 12)


def load(ckpt, hw, dev, dt):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcd = payload["configs"]["model"]
    ns = payload["norm_stats"]
    norm = NormStats(mean={k: np.asarray(v, np.float32) for k, v in ns["mean"].items()},
                     std={k: np.asarray(v, np.float32) for k, v in ns["std"].items()})
    wbr = C.wrench_baseline_rows_of(payload)
    if mcd.get("arch") == "caewm":
        from phantom.caewm.builder import build_caewm_model
        from phantom.caewm.config import CaewmConfig
        from phantom.caewm.windows import CaewmWindowSampler
        mc = CaewmConfig.from_dict(mcd)
        pm = build_caewm_model(hw, load_paths(), mc=mc, device=dev, dtype=dt, inference=True)
        sampler = CaewmWindowSampler(hw, pm.bb, norm, seed=0, lat_cache=str(CP.stored_or_default(mc.tactile_lat_cache, CP.LAT_CACHE)),
                                     tac_cache=str(CP.TAC_CACHE),
                                     dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(),
                                     wrench_baseline_rows=wbr)
    else:
        from phantom.config.model import PhantomModelConfig
        from phantom.data.windows import WindowSampler
        from phantom.train.builder import build_model
        mc = PhantomModelConfig.from_dict(mcd)
        pm = build_model(hw, load_paths(), student=mc.student, mc=mc, device=dev, dtype=dt,
                         inference=True)
        sampler = WindowSampler(hw, pm.bb, norm, student=mc.student, seed=0,
                                wrench_baseline_rows=wbr)
    C.load_phantom_checkpoint(Path(ckpt), pm.rf, hw=hw, load_ema=True, payload=payload)
    pm.rf.eval()
    return pm, sampler, mcd.get("arch") == "caewm"


def to_u8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--touch", default="sensor", choices=("sensor", "imagine"))
    ap.add_argument("--refine", type=int, default=0, help="CAEWM: video-only refinement steps")
    a = ap.parse_args(argv)
    dev, dt = "cuda", torch.bfloat16
    hw = load_hardware(a.hardware)
    pm, sampler, caewm = load(a.ckpt, hw, dev, dt)
    from phantom.caewm.geometry import Camera
    cam = Camera()
    wins = json.loads(Path(a.windows).read_text())["windows"]
    lay = pm.rf.layout
    sl = slice(lay.frame_slice(FrameGroup.VIDEO_COND).start, lay.frame_slice(FrameGroup.VIDEO_GEN).stop)
    pred, gt, uv, ok = [], [], [], []
    for i, w in enumerate(wins):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        item = sampler.sample(ep, t0)
        batch = C.to_device(C.collate_windows([item]), dev, dt)
        for k in list(batch):
            if k == "events" or k.startswith("cpk_") or k in ("tac_fut", "tac_slip", "tac_contact"):
                batch.pop(k) if k.startswith("tac_") else batch.__setitem__(k, torch.zeros_like(batch[k]))
        pm.rf._gen = torch.Generator().manual_seed(1000)
        with torch.no_grad():
            if caewm:
                p = pm.rf.sample(batch, nfe=a.nfe, touch=a.touch == "sensor", video_refine_steps=a.refine)
            else:
                p = pm.rf.sample(batch, nfe=a.nfe)
            frames = pm.rf.vae.decode(p.x_final_B_C_T_H_W[:, :, sl])[0]          # (3,13,H,W)
        pred.append(to_u8(frames[:, list(PIX)].permute(1, 0, 2, 3)))
        gt.append(to_u8(item["video"][[0, *PIX]]))
        c = sampler._ep(ep)
        rows = [cam.uv_norm(np.asarray(c.at(STREAM_ARM_TCP_POSE, t0 + s))[:3]) for s in (1.0, 2.0, 3.0)]
        uv.append(np.stack([r[0] for r in rows])); ok.append(np.array([r[1] for r in rows]))
        if i % 20 == 0:
            print(f"{i}/{len(wins)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz", pred=np.stack(pred), gt=np.stack(gt),
                        tcp_uv=np.stack(uv), tcp_ok=np.stack(ok),
                        terminal=np.array([bool(w.get("terminal")) for w in wins]))
    (out / "meta.json").write_text(json.dumps(vars(a) | {"n": len(wins), "caewm": caewm}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
