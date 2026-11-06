"""Nearest-neighbour baseline for touch imagination without touch (hypothesis 3).

The query is what the model sees in the no-touch deployment, the current scene frame and the
robot state. The database holds training windows of the same task on a 0.5 s grid. Visual
features are DINOv3 ViT-L/16 (CLS and mean patch token, each L2-normalised), the state is the
normalised ur_state of the checkpoint's norm stats, and the distance is
    d = |f_q - f_db|^2 + lam * |s_q - s_db|^2 / dim(s).
The K nearest windows from distinct episodes predict the future fields at t0 + tau with their
own real fields at t_db + tau, giving K samples as the model's K seeds. lam is chosen by
leave-one-episode-out on the training set only. Scored as tools/cae_tactile_eval.py does
(pooled contact IoU, field MSE and skill vs persistence, energy-form CRPS).

  python -m tools.v3_knn_tactile --ckpt v2.pt --data .../tasks --out out_dir
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import phantom.caewm.paths as CP
from phantom.caewm.windows import FUT_TAUS
from phantom.data.schema import (STREAM_ARM_Q, STREAM_ARM_QD, STREAM_ARM_TCP_POSE,
                                 STREAM_ARM_TCP_SPEED, STREAM_CAMERA_SCENE, STREAM_GRIPPER, NormStats)
from phantom.data.tactile_cache import FINGERS, TactileCache
from phantom.train import common as C

TAU_DEPTH = 0.05
GRID_S = 0.5
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class Features:
    def __init__(self, dev):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(DINO, dtype=torch.bfloat16).to(dev).eval()
        self.n_reg = int(getattr(self.m.config, "num_register_tokens", 0))
        self.dev = dev
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(imgs)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        H = 224
        W = int(round(w * H / h / 16)) * 16
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).bfloat16()
        hs = self.m(pixel_values=x).last_hidden_state.float()
        cls, patch = hs[:, 0], hs[:, 1 + self.n_reg:].mean(1)
        f = torch.cat([torch.nn.functional.normalize(cls, dim=-1),
                       torch.nn.functional.normalize(patch, dim=-1)], -1) / np.sqrt(2)
        return f.cpu().numpy().astype(np.float32)


def state_at(c, t, norm) -> np.ndarray:
    ur = np.concatenate([c.at(STREAM_ARM_Q, t), c.at(STREAM_ARM_QD, t), c.at(STREAM_ARM_TCP_POSE, t),
                         c.at(STREAM_ARM_TCP_SPEED, t), c.at(STREAM_GRIPPER, t)]).astype(np.float32)
    return np.asarray(norm.normalize("ur_state", ur), np.float32)


def fields_at(tc: TactileCache, t: float) -> np.ndarray:
    """(2, 5, H, W) real fields at the newest frame <= t."""
    return np.stack([np.asarray(tc.fields[f][tc.index_at(f, t)], np.float32) for f in FINGERS])


def encode(sampler, feat, items, norm, bs=32):
    """items: list of (episode path, t) -> (visual features, states)."""
    F, S = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        imgs = [np.asarray(sampler._ep(ep).at(STREAM_CAMERA_SCENE, t)) for ep, t in chunk]
        F.append(feat(imgs))
        S += [state_at(sampler._ep(ep), t, norm) for ep, t in chunk]
        if i % (bs * 50) == 0:
            print(f"encoded {i}/{len(items)}", flush=True)
    return np.concatenate(F), np.stack(S)


def knn(fq, sq, fdb, sdb, ep_db, lam, k, exclude=None):
    """indices (k,) of the nearest database windows from distinct episodes."""
    d = ((fdb - fq) ** 2).sum(1) + lam * ((sdb - sq) ** 2).sum(1) / sq.shape[0]
    if exclude is not None:
        d = np.where(ep_db == exclude, np.inf, d)
    out, seen = [], set()
    for j in np.argsort(d):
        if ep_db[j] in seen or not np.isfinite(d[j]):
            continue
        out.append(j); seen.add(ep_db[j])
        if len(out) == k:
            break
    return out


def score(acc, X, gt, now):
    """X (K, 6, 2, 5, H, W) samples, gt (6, 2, 5, H, W), now (2, 5, H, W) real t0 field."""
    m_gt = np.abs(gt[:, :, 2]) > TAU_DEPTH
    ax = (1, 2, 3)
    for pr in X:
        m = np.abs(pr[:, :, 2]) > TAU_DEPTH
        acc["inter"] += (m & m_gt).sum(ax); acc["union"] += (m | m_gt).sum(ax)
        acc["se"] += ((pr - gt) ** 2).mean(axis=(1, 2, 3, 4)); acc["n"] += 1
        acc["p_se"] += ((now[None] - gt) ** 2).mean(axis=(1, 2, 3, 4))
    spread = np.mean([np.abs(X[j] - X[k]).mean(axis=(1, 2, 3, 4)) for j in range(len(X)) for k in range(len(X))], 0)
    acc["crps"] += np.abs(X - gt[None]).mean(axis=(0, 2, 3, 4, 5)) - 0.5 * spread
    acc["n_win"] += 1


def new_acc(S):
    return {"inter": np.zeros(S), "union": np.zeros(S), "se": np.zeros(S), "p_se": np.zeros(S),
            "crps": np.zeros(S), "n": 0, "n_win": 0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="any CAEWM checkpoint (norm stats only)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap.add_argument("--tac-cache", default=str(CP.TAC_CACHE))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--loo-queries", type=int, default=1500)
    ap.add_argument("--max-train-episodes", type=int, default=0, help="smoke tests only")
    ap.add_argument("--max-windows", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.data.windows import WindowSampler
    hw = load_hardware(a.hardware)
    ns = torch.load(a.ckpt, map_location="cpu", weights_only=False)["norm_stats"]
    norm = NormStats(mean={k: np.asarray(v, np.float32) for k, v in ns["mean"].items()},
                     std={k: np.asarray(v, np.float32) for k, v in ns["std"].items()})
    sampler = WindowSampler(hw, BackboneConfig(), norm, seed=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tac_root = Path(a.tac_cache)
    S = len(FUT_TAUS)
    tmax = max(FUT_TAUS)

    db_file = out / "db.npz"
    if db_file.exists():
        z = np.load(db_file, allow_pickle=True)
        items = [(Path(e), float(t)) for e, t in zip(z["ep"], z["t"])]
        fdb, sdb = z["f"], z["s"]
    else:
        items = []
        train_eps = C.manifest_split(Path(a.data), "train")
        for ep in train_eps[:a.max_train_episodes or None]:
            if not (tac_root / ep.name).exists():
                continue
            lo, hi = sampler.valid_range(ep)
            tc = TactileCache(tac_root, ep.name)
            t_end = min(float(tc.ts[f][-1]) for f in FINGERS) - tmax
            items += [(ep, float(t)) for t in np.arange(lo, min(hi, t_end), GRID_S)]
        feat = Features("cuda")
        fdb, sdb = encode(sampler, feat, items, norm)
        np.savez(db_file, ep=np.array([str(e) for e, _ in items]), t=np.array([t for _, t in items]), f=fdb, s=sdb)
    ep_db = np.array([e.name for e, _ in items])
    task_db = np.array([e.parent.name for e, _ in items])
    tcs = {}

    def tcache(name):
        if name not in tcs:
            tcs[name] = TactileCache(tac_root, name)
        return tcs[name]

    def predict(fq, sq, task, lam, exclude=None):
        sel = np.flatnonzero(task_db == task)
        nb = knn(fq, sq, fdb[sel], sdb[sel], ep_db[sel], lam, a.k, exclude)
        return np.stack([np.stack([fields_at(tcache(ep_db[sel[j]]), items[sel[j]][1] + tau) for tau in FUT_TAUS])
                         for j in nb])

    # lam by leave-one-episode-out on the training set only
    rng = np.random.default_rng(0)
    q_idx = rng.choice(len(items), size=min(a.loo_queries, len(items)), replace=False)
    loo = {}
    for lam in LAMS:
        acc = new_acc(S)
        for i in q_idx:
            ep, t = items[i]
            tc = tcache(ep.name)
            gt = np.stack([fields_at(tc, t + tau) for tau in FUT_TAUS])
            score(acc, predict(fdb[i], sdb[i], ep.parent.name, lam, exclude=ep.name), gt, fields_at(tc, t))
        loo[lam] = float(acc["crps"].sum() / (S * acc["n_win"]))
        print(f"LOO lam {lam}: CRPS {loo[lam]:.5f}", flush=True)
    lam = min(loo, key=loo.get)

    wins = json.loads(Path(a.windows).read_text())["windows"][:a.max_windows or None]
    q_items = [(CP.episode_path(w["episode"]), float(w["t0"])) for w in wins]
    feat = Features("cuda")
    fq, sq = encode(sampler, feat, q_items, norm)
    acc = new_acc(S)
    for (ep, t0), f_, s_ in zip(q_items, fq, sq):
        tc = TactileCache(tac_root, ep.name)
        gt = np.stack([fields_at(tc, t0 + tau) for tau in FUT_TAUS])
        score(acc, predict(f_, s_, ep.parent.name, lam), gt, fields_at(tc, t0))
    iou = acc["inter"] / np.maximum(acc["union"], 1)
    summary = {"method": "knn", "k": a.k, "lam": lam, "loo_crps": loo, "n_windows": len(wins),
               "db_windows": len(items),
               "per_slot": {f"+{tau:g}s": {"iou": iou[j], "mse": acc["se"][j] / acc["n"],
                                           "skill": 1 - acc["se"][j] / acc["p_se"][j],
                                           "crps": acc["crps"][j] / acc["n_win"]} for j, tau in enumerate(FUT_TAUS)},
               "iou_all": float(acc["inter"].sum() / acc["union"].sum()),
               "skill_all": float(1 - acc["se"].sum() / acc["p_se"].sum()),
               "crps_all": float(acc["crps"].sum() / (S * acc["n_win"]))}
    (out / "knn_tactile.json").write_text(json.dumps({"summary": summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
