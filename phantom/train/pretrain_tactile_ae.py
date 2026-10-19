"""Train the CAEWM causal tactile tokenizer (phantom/model/tactile/tokenizer.py)
on the train-split episodes of the tactile cache, then export latent statistics.

    python -m phantom.train.pretrain_tactile_ae [--out <runs_root>/tactile_ae]
Output: <out>/tactile_ae.pt {model, step, val} (EMA weights, stats buffers filled).
Gate: pooled contact-mask IoU of decoded x_t on the val split >= 0.85.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import phantom.caewm.paths as CP
from phantom.data.tactile_cache import FINGERS, TactileCache
from phantom.model.tactile.tokenizer import HIST, CausalTactileAE, contact_mask
from phantom.train.common import manifest_split

log = logging.getLogger("pretrain_tactile_ae")
DATA_ROOT = CP.TASKS
CACHE = CP.TAC_CACHE


def hist_indices(i: int) -> np.ndarray:
    return np.clip(np.arange(i - HIST + 1, i + 1), 0, None)


class FrameSet(torch.utils.data.Dataset):
    """Items = (episode, finger, frame); half the draws come from contact frames."""

    def __init__(self, split: str, n_items: int, seed: int = 0):
        self.names = [p.name for p in manifest_split(DATA_ROOT, split)]
        self.items, self.contact = [], []
        for e, n in enumerate(self.names):
            for fi, f in enumerate(FINGERS):
                mf = np.load(CACHE / n / f"mask_frac_{f}.npy")
                for i in range(len(mf)):
                    self.items.append((e, fi, i))
                    self.contact.append(mf[i] > 0.0)
        self.items = np.asarray(self.items)
        self.contact = np.asarray(self.contact)
        self.n_items, self.seed = n_items, seed
        self._caches: dict[int, TactileCache] = {}

    def cache(self, e: int) -> TactileCache:
        c = self._caches.get(e)
        if c is None:
            c = self._caches[e] = TactileCache(CACHE, self.names[e])
        return c

    def window(self, e, fi, i):
        f = FINGERS[fi]
        return np.asarray(self.cache(e).fields[f][hist_indices(i)], np.float32)

    def __len__(self):
        return self.n_items

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k))
        pool = np.flatnonzero(self.contact) if rng.random() < 0.5 else None
        j = rng.choice(pool) if pool is not None else rng.integers(len(self.items))
        e, fi, i = self.items[j]
        return torch.from_numpy(self.window(e, fi, i)), int(fi)


def prep(model, hist_raw, fi):
    x = model.norm(hist_raw)
    right = (fi == 1).view(-1, 1, 1, 1, 1)
    return torch.where(right, x.flip(-2), x)


def losses(model, hist_raw, fi, aug_sigma: float, kl_w: float):
    x = prep(model, hist_raw, fi)
    mu, logvar = model.encode_dist(x)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    keep = (torch.rand(z.shape[0], 1, 1, 1, device=z.device) < 0.5).float()
    z = z + keep * aug_sigma * mu.detach().std(dim=(0, 2, 3), keepdim=True) * torch.randn_like(z)
    xh, dxh = model.decode(z)
    tgt, dtgt = x[:, -1], x[:, -1] - x[:, -2]
    w = 1.0 + 4.0 * contact_mask(model.denorm(tgt)).unsqueeze(1).float()
    rec = (w * (xh - tgt) ** 2).mean()
    drec = (w * (dxh - dtgt) ** 2).mean()
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).mean()
    return rec + drec + kl_w * kl, {"rec": rec.item(), "drec": drec.item(), "kl": kl.item()}


@torch.no_grad()
def evaluate(model, ds: FrameSet, device, stride: int = 2, bs: int = 256):
    """Pooled contact IoU and raw MSE over every `stride`-th frame of every val episode,
    with the persistence reference (x_{t-1} as the reconstruction of x_t) for dx."""
    model.eval()
    inter = union = 0
    se = se_d = se_pers = n = 0.0
    idx = np.arange(0, len(ds.items), stride)
    for s in range(0, len(idx), bs):
        sel = ds.items[idx[s:s + bs]]
        h = torch.from_numpy(np.stack([ds.window(*r) for r in sel])).to(device)
        fi = torch.as_tensor(sel[:, 1], device=device)
        x = prep(model, h, fi)
        mu, _ = model.encode_dist(x)
        xh, dxh = model.decode(mu)
        xr, tr = model.denorm(xh), model.denorm(x[:, -1])
        m_hat, m = contact_mask(xr), contact_mask(tr)
        inter += (m_hat & m).sum().item()
        union += (m_hat | m).sum().item()
        se += ((xr - tr) ** 2).mean(dim=(1, 2, 3)).sum().item()
        dt = model.denorm(x[:, -1] - x[:, -2])
        se_d += ((model.denorm(dxh) - dt) ** 2).mean(dim=(1, 2, 3)).sum().item()
        se_pers += (dt ** 2).mean(dim=(1, 2, 3)).sum().item()
        n += len(sel)
    model.train()
    return {"iou": inter / max(union, 1), "mse": se / n, "dmse": se_d / n, "dmse_zero": se_pers / n}


@torch.no_grad()
def export_stats(model, ds: FrameSet, device, n_eps: int = 300, lag_s=(0.5, 1.0, 2.0, 3.0)):
    """z_mean/z_std over train frames (mirrored right), and dz_std of standardised
    latent increments z(t+lag) - z(t) on the same finger, pooled over lags."""
    model.eval()
    zs, dzs = [], []
    rng = np.random.default_rng(0)
    for e in rng.choice(len(ds.names), min(n_eps, len(ds.names)), replace=False):
        c = ds.cache(e)
        for fi, f in enumerate(FINGERS):
            T = len(c.ts[f])
            h = torch.from_numpy(np.stack([ds.window(e, fi, i) for i in range(T)])).to(device)
            x = prep(model, h, torch.full((T,), fi, device=device))
            mu = torch.cat([model.encode_dist(x[s:s + 512])[0] for s in range(0, T, 512)])
            zs.append(mu.cpu())
            ts = c.ts[f]
            for lag in lag_s:
                j = np.searchsorted(ts, ts + lag, side="right") - 1
                ok = np.flatnonzero(ts + lag <= ts[-1])
                if len(ok):
                    dzs.append((mu[j[ok]] - mu[ok]).cpu())
    z = torch.cat(zs)
    model.z_mean.copy_(z.mean(dim=(0, 2, 3)).to(device))
    model.z_std.copy_(z.std(dim=(0, 2, 3)).to(device))
    dz = torch.cat(dzs) / model.z_std.cpu().view(1, -1, 1, 1)
    model.dz_std.copy_(dz.std(dim=(0, 2, 3)).to(device))
    model.train()
    return {"z_std": model.z_std.tolist(), "dz_std": model.dz_std.tolist(), "n_frames": int(z.shape[0])}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CP.TAC_AE.parent))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-w", type=float, default=1e-4)
    ap.add_argument("--aug-sigma", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)

    train = FrameSet("train", a.steps * a.bs)
    val = FrameSet("val", 0)
    log.info("train frames %d (contact %.1f%%), val frames %d", len(train.items),
             100 * train.contact.mean(), len(val.items))

    model = CausalTactileAE().to(device)
    rng = np.random.default_rng(1)
    sample = np.stack([train.window(*train.items[j])[-1] for j in rng.integers(len(train.items), size=4000)])
    model.ch_std.copy_(torch.from_numpy(sample.std(axis=(0, 2, 3))).to(device))
    log.info("params %.2fM, ch_std %s", sum(p.numel() for p in model.parameters()) / 1e6,
             np.round(model.ch_std.cpu().numpy(), 4).tolist())
    ema = CausalTactileAE().to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + np.cos(np.pi * min(s / a.steps, 1.0))))
    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=4)
    t0, hist = time.time(), []
    for step, (h, fi) in enumerate(dl, start=1):
        h, fi = h.to(device, non_blocking=True), fi.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = losses(model, h, fi, a.aug_sigma, a.kl_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1e-3 if step > 1000 else 1.0)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        hist.append(parts)
        if step % 200 == 0:
            m = {k: np.mean([p[k] for p in hist]) for k in hist[0]}
            hist = []
            log.info("step %d  rec %.4f drec %.4f kl %.2f  %.2f it/s", step, m["rec"], m["drec"],
                     m["kl"], step / (time.time() - t0))
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(ema, val, device)
            log.info("VAL step %d  %s", step, json.dumps({k: round(x, 5) for k, x in v.items()}))
            torch.save({"model": ema.state_dict(), "step": step, "val": v}, out / "tactile_ae_last.pt")
        if step >= a.steps:
            break
    stats = export_stats(ema, train, device)
    v = evaluate(ema, val, device, stride=1)
    log.info("FINAL VAL %s", json.dumps(v))
    log.info("stats %s", json.dumps(stats))
    torch.save({"model": ema.state_dict(), "step": a.steps, "val": v, "stats": stats,
                "args": vars(a)}, out / "tactile_ae.pt")
    (out / "final_val.json").write_text(json.dumps({"val": v, "stats": stats}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
