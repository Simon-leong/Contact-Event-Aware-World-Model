"""Score dumped future frames (stage 2 of 2; run in an env with lpips + scikit-image).

Full-frame PSNR / SSIM / LPIPS(alex) at t0+1/2/3 s, plus the same three inside a
96x96 crop centred on the real TCP pixel position (the gripper region), for the
model and for the copy-last-frame reference (real t0 frame).

    python tools/video_quality_score.py <dump_dir> [<dump_dir> ...]
"""
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

CROP = 96


def crop(img, u, v):
    H, W = img.shape[1:]
    cx, cy = int(round(u * W)), int(round(v * H))
    x0 = int(np.clip(cx - CROP // 2, 0, W - CROP)); y0 = int(np.clip(cy - CROP // 2, 0, H - CROP))
    return img[:, y0:y0 + CROP, x0:x0 + CROP]


def score(net, a, b):
    """a, b: (3,H,W) uint8."""
    fa, fb = a.transpose(1, 2, 0), b.transpose(1, 2, 0)
    ta = torch.from_numpy(a).float().div(127.5).sub(1).unsqueeze(0).cuda()
    tb = torch.from_numpy(b).float().div(127.5).sub(1).unsqueeze(0).cuda()
    return (psnr(fb, fa, data_range=255), ssim(fb, fa, channel_axis=2, data_range=255),
            float(net(ta, tb).item()))


def main():
    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    for d in map(Path, sys.argv[1:]):
        z = np.load(d / "frames.npz")
        pred, gt, uv, ok = z["pred"], z["gt"], z["tcp_uv"], z["tcp_ok"]
        res = {}
        for name, src in (("model", lambda i, k: pred[i, k]), ("copy_last", lambda i, k: gt[i, 0])):
            per = {}
            for k, sec in enumerate((1, 2, 3)):
                full, reg = [], []
                for i in range(len(gt)):
                    with torch.no_grad():
                        full.append(score(net, src(i, k), gt[i, k + 1]))
                        if ok[i, k]:
                            u, v = uv[i, k]
                            reg.append(score(net, crop(src(i, k), u, v), crop(gt[i, k + 1], u, v)))
                f, r = np.mean(full, 0), np.mean(reg, 0)
                per[f"+{sec}s"] = {"psnr": f[0], "ssim": f[1], "lpips": f[2],
                                   "grip_psnr": r[0], "grip_ssim": r[1], "grip_lpips": r[2],
                                   "n_grip": len(reg)}
            per["mean"] = {m: float(np.mean([per[s][m] for s in ("+1s", "+2s", "+3s")]))
                           for m in ("psnr", "ssim", "lpips", "grip_psnr", "grip_ssim", "grip_lpips")}
            res[name] = per
        res["n"] = int(len(gt))
        (d / "scores.json").write_text(json.dumps(res, indent=1, default=float))
        m, c = res["model"]["mean"], res["copy_last"]["mean"]
        print(f"{d.name}: PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f} LPIPS {m['lpips']:.3f} | "
              f"grip PSNR {m['grip_psnr']:.2f} SSIM {m['grip_ssim']:.3f} LPIPS {m['grip_lpips']:.3f} "
              f"|| copy-last PSNR {c['psnr']:.2f} LPIPS {c['lpips']:.3f}")


if __name__ == "__main__":
    main()
