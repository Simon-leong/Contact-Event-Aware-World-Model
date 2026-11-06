"""CAEWM future-touch evaluation on a fixed window file.

For every window, samples (seed 1000+s, s < --seeds) in the chosen touch mode,
decodes the six future tactile latents (+0.5 ... +3 s) through the frozen tokenizer
decoder and compares with the real cached fields at the newest frame <= t0+tau:
  field MSE (5 channels, raw units) and its skill vs persistence (real t0 field held),
  contact-mask IoU (|depth| > tau_contact_depth), pooled over windows, for the model
  and for persistence, at +1/+2/+3 s and over all slots;
  field CRPS over the seeds (energy form), against the persistence absolute error;
  slip AUROC / AP over in-contact (slot, finger) entries with the debounced labels,
  and window-level AUROC of "slip within (t0, t0+1]".
The model's own current latent (imagined under --touch imagine, the real one under sensor),
decoded and held over the six slots, is scored the same way as "self_persist": it carries
exactly the model's information, whereas the real t0 field is a sensor prior.
Writes <out>/tactile_eval.json and per-window rows (for paired comparisons).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import phantom.caewm.paths as CP
from phantom.caewm.windows import FUT_TAUS
from phantom.data.tactile_cache import FINGERS, TactileCache
from phantom.train import common as C
from tools.video_dump import load

TAU_DEPTH = 0.05


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks for ties
        m = s == v; r[m] = r[m].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ap(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if not y.any():
        return float("nan")
    o = np.argsort(-s); y = y[o]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ckpt", required=True)
    ap_.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    ap_.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--nfe", type=int, default=1)
    ap_.add_argument("--seeds", type=int, default=4)
    ap_.add_argument("--touch", default="sensor", choices=("sensor", "imagine"))
    ap_.add_argument("--tac-cache", default=str(CP.TAC_CACHE))
    ap_.add_argument("--plan-rule", default=None, help="caewm3: decision rule (default: the checkpoint's)")
    ap_.add_argument("--no-anchors", action="store_true", help="caewm3: layer-2 ablation, see rf.sample")
    a = ap_.parse_args(argv)
    from phantom.config.hardware import load_hardware
    hw = load_hardware(a.hardware)
    pm, sampler, caewm = load(a.ckpt, hw, "cuda", torch.bfloat16)
    assert caewm, "cae_tactile_eval needs a CAEWM checkpoint"
    wins = json.loads(Path(a.windows).read_text())["windows"]
    S = len(FUT_TAUS)
    acc = {"inter": np.zeros(S), "union": np.zeros(S), "p_inter": np.zeros(S), "p_union": np.zeros(S),
           "se": np.zeros(S), "p_se": np.zeros(S), "n": 0,
           "crps": np.zeros(S), "p_crps": np.zeros(S), "n_win": 0,
           "s_inter": np.zeros(S), "s_union": np.zeros(S), "s_se": np.zeros(S), "s_crps": np.zeros(S)}
    slip_y, slip_s, win_y, win_s, rows = [], [], [], [], []
    for i, w in enumerate(wins):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        item = sampler.sample(ep, t0)
        lab_slip, lab_con = item["tac_slip"].numpy(), item["tac_contact"].numpy()
        batch = C.to_device(C.collate_windows([item]), "cuda", torch.bfloat16)
        for k in ("tac_fut", "tac_slip", "tac_contact"):
            batch.pop(k)
        tc = TactileCache(Path(a.tac_cache), ep.name)
        gt = np.stack([[np.asarray(tc.fields[f][tc.index_at(f, t0 + tau)], np.float32) for f in FINGERS]
                       for tau in FUT_TAUS])                                   # (6,2,5,H,W)
        now = np.stack([np.asarray(tc.fields[f][tc.index_at(f, t0)], np.float32) for f in FINGERS])
        seeds_pr, seeds_self = [], []
        for s in range(a.seeds):
            pm.rf._gen = torch.Generator().manual_seed(1000 + s)
            with torch.no_grad():
                p = pm.rf.sample(batch, nfe=a.nfe, touch=a.touch == "sensor", decode_tactile=True,
                                 plan_rule=a.plan_rule, anchors=not a.no_anchors)
            pr = p.fields_B_6_2_C_H_W[0].float().cpu().numpy()
            seeds_pr.append(pr)
            z_now = p.tac_obs_B_2_2_C_h_w[:, 1].float()
            ae = pm.rf.tac_ae
            with torch.no_grad():
                f_now = torch.stack([ae.decode_raw(z_now[:, 0], mirror=False)[0],
                                     ae.decode_raw(z_now[:, 1], mirror=True)[0]], 1)
            sp_ = np.repeat(f_now[0].float().cpu().numpy()[None], S, 0)       # (6,2,5,H,W)
            seeds_self.append(sp_)
            m_hat, m_gt, m_p = (np.abs(pr[:, :, 2]) > TAU_DEPTH, np.abs(gt[:, :, 2]) > TAU_DEPTH,
                                np.abs(now[None, :, 2]) > TAU_DEPTH)
            ax = (1, 2, 3)
            acc["inter"] += (m_hat & m_gt).sum(ax); acc["union"] += (m_hat | m_gt).sum(ax)
            acc["p_inter"] += (m_p & m_gt).sum(ax); acc["p_union"] += (m_p | m_gt).sum(ax)
            se = ((pr - gt) ** 2).mean(axis=(1, 2, 3, 4)); pse = ((now[None] - gt) ** 2).mean(axis=(1, 2, 3, 4))
            acc["se"] += se; acc["p_se"] += pse; acc["n"] += 1
            m_s = np.abs(sp_[:, :, 2]) > TAU_DEPTH
            acc["s_inter"] += (m_s & m_gt).sum(ax); acc["s_union"] += (m_s | m_gt).sum(ax)
            acc["s_se"] += ((sp_ - gt) ** 2).mean(axis=(1, 2, 3, 4))
            sp = p.slip_prob_B_6_2[0].float().cpu().numpy()
            sel = lab_con > 0.5
            slip_y += list(lab_slip[sel]); slip_s += list(sp[sel])
            win_y.append(bool(lab_slip[:2].max() > 0.5)); win_s.append(float(sp[:2].max()))
            rows.append({"episode": ep.name, "t0": t0, "seed": s, "terminal": bool(w.get("terminal")),
                         "mse": se.tolist(), "mse_persist": pse.tolist(),
                         "slip_prob": sp.tolist(), "slip_label": lab_slip.tolist(),
                         "contact_label": lab_con.tolist(),
                         "contact_prob": p.contact_prob_B_6_2[0].float().cpu().numpy().tolist()})
        # CRPS of the field over the seeds (energy form), per slot, pixel/channel mean
        X = np.stack(seeds_pr)                                                 # (K,6,2,5,H,W)
        spread = np.mean([np.abs(X[j] - X[k]).mean(axis=(1, 2, 3, 4))
                          for j in range(len(X)) for k in range(len(X))], 0)
        acc["crps"] += np.abs(X - gt[None]).mean(axis=(0, 2, 3, 4, 5)) - 0.5 * spread
        acc["p_crps"] += np.abs(now[None] - gt).mean(axis=(1, 2, 3, 4))
        Y = np.stack(seeds_self)
        s_spread = np.mean([np.abs(Y[j] - Y[k]).mean(axis=(1, 2, 3, 4))
                            for j in range(len(Y)) for k in range(len(Y))], 0)
        acc["s_crps"] += np.abs(Y - gt[None]).mean(axis=(0, 2, 3, 4, 5)) - 0.5 * s_spread
        acc["n_win"] += 1
        if i % 20 == 0:
            print(f"{i}/{len(wins)}", flush=True)
    iou, piou = acc["inter"] / np.maximum(acc["union"], 1), acc["p_inter"] / np.maximum(acc["p_union"], 1)
    mse, pmse = acc["se"] / acc["n"], acc["p_se"] / acc["n"]
    per = {f"+{tau:g}s": {"iou": iou[k], "iou_persist": piou[k], "mse": mse[k], "mse_persist": pmse[k],
                          "skill": 1 - mse[k] / pmse[k],
                          "crps": acc["crps"][k] / acc["n_win"],
                          "crps_persist": acc["p_crps"][k] / acc["n_win"],
                          "iou_self_persist": acc["s_inter"][k] / max(acc["s_union"][k], 1),
                          "crps_self_persist": acc["s_crps"][k] / acc["n_win"]} for k, tau in enumerate(FUT_TAUS)}
    summary = {"touch": a.touch, "nfe": a.nfe, "seeds": a.seeds, "n_windows": len(wins),
               "per_slot": per,
               "iou_all": float(acc["inter"].sum() / acc["union"].sum()),
               "iou_persist_all": float(acc["p_inter"].sum() / acc["p_union"].sum()),
               "skill_all": float(1 - acc["se"].sum() / acc["p_se"].sum()),
               "crps_all": float(acc["crps"].sum() / (S * acc["n_win"])),
               "crps_persist_all": float(acc["p_crps"].sum() / (S * acc["n_win"])),
               "iou_self_persist_all": float(acc["s_inter"].sum() / acc["s_union"].sum()),
               "skill_vs_self_persist_all": float(1 - acc["se"].sum() / acc["s_se"].sum()),
               "crps_self_persist_all": float(acc["s_crps"].sum() / (S * acc["n_win"])),
               "slip_auroc": auroc(slip_y, slip_s), "slip_ap": ap(slip_y, slip_s),
               "slip_pos_rate": float(np.mean(slip_y)) if slip_y else float("nan"),
               "slip_window_auroc_1s": auroc(win_y, win_s),
               "slip_window_pos_rate_1s": float(np.mean(win_y))}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tactile_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
