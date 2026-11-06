"""CAEWM V2 event / force evaluation on the fixed window file, for V2, V1 and the official
models under the same labels.

Labels come from the V2 event cache (phantom/caewm2/events.py), first occurrence after t0:
  onset, slip    rising edges of contact / debounced slip on either finger (future tactile
                 frames, i.e. the sensor that is an input in sensor mode)
  close, lift, release   gripper position and TCP height (independent of touch)
A window is positive for (event, H) when the event first happens in (t0, t0 + H], H = 1, 2, 3 s.

Scores per model:
  caewm2    P(event by H) from the hazard heads; fingers combined as 1 - prod(1 - F_f)
  caewm, official   the event head's per-step distribution over (none, onset, hold, slip,
            release) at +1/+2/+3 s, read as the max over steps <= H and as the noisy-OR;
            V1 additionally its per-slot slip probability (noisy-OR over slots and fingers)
Metrics: AUROC, AP, Brier, ECE (10 equal-width bins), pooled over windows and seeds; for
caewm2 also the time-to-event error (first bin with F >= 0.5) and force skill against
persistence (zero increment) for the pad force / area and the wrist wrench.

--sweep adds the lead time at a fixed false-alarm rate: every validation episode is swept
at --stride s with one seed; the alarm score is P(event within 3 s); the threshold is set
so that --far of the sweep points with no event in (t0, t0 + 3] alarm; for every event
the lead is te - (first sweep point in [te - 3, te) whose score passes), 0 when missed.

    python -m tools.v2_event_eval --ckpt <ckpt> --out <json> [--touch imagine] [--sweep]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import phantom.caewm.paths as CP
import phantom.caewm2.paths as CP2
from phantom.caewm2.events import FINGERS
from phantom.caewm2.heads import cumulative
from phantom.caewm2.windows import first_bin
from phantom.config.model import EVENT_IDX
from phantom.train import common as C
from tools.cae_tactile_eval import ap as avg_prec, auroc
from tools.video_dump import load

HORIZONS = (1, 2, 3)
BIN_DT = 0.25
EVENTS = ("onset", "slip", "close", "lift", "release")
INDEPENDENT = ("close", "lift", "release")
TARGET_KEYS = ("tac_fut", "tac_slip", "tac_contact", "finger_bin", "phase_bin", "pad_fut", "wrist_fut")


def kbin(H):
    return int(round(H / BIN_DT)) - 1


def brier(y, s):
    return float(np.mean((np.asarray(s, float) - np.asarray(y, float)) ** 2))


def ece(y, s, n_bins=10):
    y, s = np.asarray(y, float), np.clip(np.asarray(s, float), 0, 1)
    idx = np.minimum((s * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(tot)


def metrics(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "auroc": auroc(y, s), "ap": avg_prec(y, s),
            "brier": brier(y, s), "ece": ece(y, s)}


class Labels:
    def __init__(self, root):
        self.root, self._c = Path(root), {}

    def ev(self, name):
        if name not in self._c:
            self._c[name] = dict(np.load(self.root / name / "events.npz"))
        return self._c[name]

    def times(self, name, e):
        ev = self.ev(name)
        if e in ("onset", "slip"):
            return np.sort(np.concatenate([ev[f"{e}_t_{f}"] for f in FINGERS]))
        return np.asarray(ev[f"{e}_t"], float)

    def first(self, name, t0):
        return {e: first_bin(self.times(name, e), t0, BIN_DT) for e in EVENTS}


def model_kind(pm):
    """caewm3 predictions carry every caewm2 field, so they are scored as caewm2."""
    arch = getattr(pm.mc, "arch", None)
    return {"caewm3": "caewm2"}.get(arch, arch) if arch in ("caewm", "caewm2", "caewm3") else "official"


def scores(kind, p):
    """-> {reading: {event: (B, len(HORIZONS)) array}}"""
    out = {}
    ks = [kbin(H) for H in HORIZONS]

    def hazard_reading(fin, ph):
        d = {}
        if fin is not None:
            anyf = 1 - (1 - cumulative(fin.float())).prod(2)                 # (B,12,2) fingers combined
            d.update(onset=anyf[:, ks, 0], slip=anyf[:, ks, 1])
        if ph is not None:
            Fp = cumulative(ph.float())                                      # (B,12,3)
            d.update(close=Fp[:, ks, 0], lift=Fp[:, ks, 1], release=Fp[:, ks, 2])
        return d

    if kind == "caewm2":
        for name, fin, ph in (("hazard", p.finger_hazard_B_12_2_2, p.phase_hazard_B_12_3),
                              ("mid_hazard", p.mid_finger_hazard_B_12_2_2, p.mid_phase_hazard_B_12_3)):
            d = hazard_reading(fin, ph)
            if d:
                out[name] = d
    pr = torch.softmax(p.event_logits_B_Tc_E.float(), -1)                     # (B,3,E)
    for name, red in (("head_max", lambda q, h: q[:, :h].max(1).values),
                      ("head_noisy_or", lambda q, h: 1 - (1 - q[:, :h]).prod(1))):
        out[name] = {e: torch.stack([red(pr[..., EVENT_IDX[e]], H) for H in HORIZONS], 1)
                     for e in ("onset", "slip", "release")}
    if kind in ("caewm", "caewm2"):
        sp = p.slip_prob_B_6_2.float()                                       # slots at 0.5..3 s
        out["slot_slip_noisy_or"] = {"slip": torch.stack(
            [1 - (1 - sp[:, :2 * H]).flatten(1).prod(1) for H in HORIZONS], 1)}
    return {r: {e: v.cpu().numpy() for e, v in d.items()} for r, d in out.items()}


def make_batch(sampler, kind, ep, t0s, dev):
    items = [sampler.sample(ep, t) for t in t0s]
    b = C.to_device(C.collate_windows(items), dev, torch.bfloat16)
    for k in list(b):
        if k in TARGET_KEYS:
            b.pop(k)
        elif k == "events" or k.startswith("cpk_"):
            b[k] = torch.zeros_like(b[k])
    return b, items


def run(pm, kind, b, touch, **kw):
    if kind == "official":
        return pm.rf.sample(b, nfe=1)
    return pm.rf.sample(b, nfe=1, touch=touch, **kw)


def fixed_windows(pm, kind, sampler, labels, wins, seeds, touch, dev):
    rows = {"first": [], "seed": [], "scores": {}, "force": []}
    for wi, w in enumerate(wins):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        b, (item,) = make_batch(sampler, kind, ep, [t0], dev)
        fb = labels.first(ep.name, t0)
        for s in range(seeds):
            pm.rf._gen = torch.Generator().manual_seed(1000 * s + wi)
            with torch.no_grad():
                p = run(pm, kind, b, touch)
            rows["first"].append(fb)
            rows["seed"].append(s)
            for r, d in scores(kind, p).items():
                for e, v in d.items():
                    rows["scores"].setdefault(r, {}).setdefault(e, []).append(v[0])
            if kind == "caewm2" and p.pad_B_6_2_3 is not None and p.finger_hazard_B_12_2_2 is not None:
                F = cumulative(p.finger_hazard_B_12_2_2.float())[0].cpu().numpy()
                rows["force"].append({
                    "pad": p.pad_B_6_2_3[0].float().cpu().numpy(), "pad_gt": item["pad_fut"].numpy(),
                    "pad_phys": sampler.pad_physical(p.pad_B_6_2_3[0].float().cpu().numpy()),
                    "pad_gt_phys": sampler.pad_physical(item["pad_fut"].numpy()),
                    "wrist": p.wrist_B_6_6[0].float().cpu().numpy(), "wrist_gt": item["wrist_fut"].numpy(),
                    "F": F, "finger_bin": item["finger_bin"].numpy()})
        if wi % 25 == 0:
            print(f"{wi}/{len(wins)}", flush=True)
    return rows


def summarize_fixed(rows):
    out = {}
    first = rows["first"]
    for r, d in rows["scores"].items():
        for e, v in d.items():
            v = np.asarray(v)
            for hi, H in enumerate(HORIZONS):
                y = np.array([f[e] <= kbin(H) for f in first])
                out.setdefault(e, {}).setdefault(f"{H}s", {})[r] = metrics(y, v[:, hi])
    for e in out:
        for H in out[e]:
            best = max(out[e][H], key=lambda r: -1 if np.isnan(out[e][H][r]["auroc"]) else out[e][H][r]["auroc"])
            out[e][H]["best_reading"] = best
    return out


def time_to_event(rows):
    """caewm2: predicted first bin with F >= 0.5 vs the true first bin, per finger event."""
    res = {}
    for j, e in enumerate(("onset", "slip")):
        err, hit, n = [], 0, 0
        for fr in rows["force"]:
            for f in range(2):
                k = int(fr["finger_bin"][f, j])
                if k >= 12:
                    continue
                n += 1
                cross = np.nonzero(fr["F"][:, f, j] >= 0.5)[0]
                if len(cross):
                    hit += 1
                    err.append(abs(int(cross[0]) - k) * BIN_DT)
        res[e] = {"n_events": n, "predicted_frac": hit / max(n, 1),
                  "mae_s": float(np.mean(err)) if err else float("nan"),
                  "median_s": float(np.median(err)) if err else float("nan")}
    return res


def force_skill(rows):
    if not rows["force"]:
        return {}
    pad = np.stack([r["pad"] for r in rows["force"]]); pad_gt = np.stack([r["pad_gt"] for r in rows["force"]])
    wr = np.stack([r["wrist"] for r in rows["force"]]); wr_gt = np.stack([r["wrist_gt"] for r in rows["force"]])

    def skill(p, g, axis):
        return (1 - ((p - g) ** 2).sum(axis) / np.maximum((g ** 2).sum(axis), 1e-12)).tolist()

    active = np.abs(pad_gt).sum((1, 2, 3)) > 0
    return {"horizons_s": [0.5 * (k + 1) for k in range(6)],
            "pad_skill_by_horizon": skill(pad, pad_gt, (0, 2, 3)),
            "pad_skill_by_channel": dict(zip(("fz", "fxy", "area"), skill(pad, pad_gt, (0, 1, 2)))),
            "pad_skill_active_windows": skill(pad[active], pad_gt[active], (0, 2, 3)) if active.any() else None,
            # in sensor units (comparable across pad target transforms)
            "pad_phys_skill_by_horizon": skill(np.stack([r["pad_phys"] for r in rows["force"]]),
                                               np.stack([r["pad_gt_phys"] for r in rows["force"]]), (0, 2, 3)),
            "pad_phys_skill_by_channel": dict(zip(("fz", "fxy", "area"), skill(
                np.stack([r["pad_phys"] for r in rows["force"]]),
                np.stack([r["pad_gt_phys"] for r in rows["force"]]), (0, 1, 2)))),
            "wrist_skill_by_horizon": skill(wr, wr_gt, (0, 2)),
            "wrist_skill_by_channel": dict(zip(("fx", "fy", "fz", "tx", "ty", "tz"), skill(wr, wr_gt, (0, 1))))}


def sweep(pm, kind, sampler, labels, episodes, stride, far, touch, dev, batch=8):
    per_ep = []
    for n, ep in enumerate(episodes):
        lo, hi = sampler.valid_range(ep)
        grid = np.arange(lo, hi, stride)
        sc = {}
        pm.rf._gen = torch.Generator().manual_seed(1000)
        for i in range(0, len(grid), batch):
            b, _ = make_batch(sampler, kind, ep, grid[i:i + batch].tolist(), dev)
            with torch.no_grad():
                p = run(pm, kind, b, touch)
            for r, d in scores(kind, p).items():
                for e, v in d.items():
                    sc.setdefault(r, {}).setdefault(e, []).append(v[:, -1])
        per_ep.append((ep.name, grid, {r: {e: np.concatenate(v) for e, v in d.items()} for r, d in sc.items()}))
        if n % 10 == 0:
            print(f"sweep {n}/{len(episodes)}", flush=True)
    out = {}
    readings = {(r, e) for _, _, sc in per_ep for r, d in sc.items() for e in d}
    for r, e in sorted(readings):
        neg, evs = [], []
        for name, grid, sc in per_ep:
            s = sc[r][e]
            te = labels.times(name, e)
            te = te[(te > grid[0]) & (te <= grid[-1] + 3.0)]
            nxt = np.array([((te > t) & (te <= t + 3.0)).any() for t in grid])
            neg.append(s[~nxt])
            evs.append((grid, s, te))
        neg = np.concatenate(neg)
        thr = float(np.quantile(neg, 1 - far)) if len(neg) else float("nan")
        leads = []
        for grid, s, te in evs:
            for t in te:
                m = (grid >= t - 3.0) & (grid < t) & (s >= thr)
                leads.append(float(t - grid[m][0]) if m.any() else 0.0)
        leads = np.asarray(leads)
        det = leads > 0
        out.setdefault(e, {})[r] = {
            "far": far, "threshold": thr, "n_events": int(len(leads)), "n_negative_points": int(len(neg)),
            "detection_rate": float(det.mean()) if len(leads) else float("nan"),
            "mean_lead_s": float(leads.mean()) if len(leads) else float("nan"),
            "median_lead_detected_s": float(np.median(leads[det])) if det.any() else float("nan")}
    return out


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    a.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    a.add_argument("--event-cache", default=str(CP2.EVENT_CACHE))
    a.add_argument("--seeds", type=int, default=4)
    a.add_argument("--touch", default="sensor", choices=("sensor", "imagine"))
    a.add_argument("--sweep", action="store_true")
    a.add_argument("--stride", type=float, default=0.25)
    a.add_argument("--far", type=float, default=0.10)
    a.add_argument("--max-episodes", type=int, default=0)
    a = a.parse_args(argv)
    from phantom.config.hardware import load_hardware
    hw = load_hardware(a.hardware)
    dev = "cuda"
    pm, sampler, _ = load(a.ckpt, hw, dev, torch.bfloat16)
    kind = model_kind(pm)
    labels = Labels(a.event_cache)
    wins = json.loads(Path(a.windows).read_text())["windows"]
    touch = a.touch == "sensor"
    rows = fixed_windows(pm, kind, sampler, labels, wins, a.seeds, touch, dev)
    res = {"ckpt": a.ckpt, "kind": kind, "touch": a.touch if kind != "official" else "official",
           "seeds": a.seeds, "n_windows": len(wins),
           "label_source": {e: ("gripper/TCP (independent of touch)" if e in INDEPENDENT
                                else "future tactile frames") for e in EVENTS},
           "fixed": summarize_fixed(rows)}
    if kind == "caewm2":
        res["time_to_event"] = time_to_event(rows)
        res["force"] = force_skill(rows)
    if a.sweep:
        eps = sorted({CP.episode_path(w["episode"]) for w in wins}, key=lambda p: p.name)
        if a.max_episodes:
            eps = eps[:a.max_episodes]
        res["sweep"] = sweep(pm, kind, sampler, labels, eps, a.stride, a.far, touch, dev)
        res["sweep_episodes"] = len(eps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({e: {H: {k: v for k, v in d.items() if k != "best_reading"} | {"best": d["best_reading"]}
                          for H, d in hs.items() if H == "1s"} for e, hs in res["fixed"].items()}, indent=0)[:3000])


if __name__ == "__main__":
    main()
