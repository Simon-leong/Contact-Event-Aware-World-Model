"""HapticWAM's offline contact-imagination benchmark (tools/haptic_imagination_eval.py)
run on a CAEWM checkpoint, so the numbers sit in the same table as the official models.

The CAEWM prediction is turned into the official contact package on the same latent
grid u_k = t0 + k s (k = 1..3, i.e. CAEWM's +1/+2/+3 s slots): the decoded fields are
resized to the package resolution for the contact mask, CoP and slip come from
`derived.derive_timestep` on the decoded field (its native-rate change dx gives the
previous frame), events from the event readout, ACC gate / p_evt from the ACC.
CAEWM does not model the distributed-force channel or the pad wrench, so d_fz is
reported as persistence (zero change) and the wrench is absent (NaN), exactly as the
scorer treats a package without them. Scoring, pooling and the event / onset / veto
statistics are the official functions, unchanged.

    python -m tools.cae_imagination_eval --ckpt <caewm ckpt> --touch sensor --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import phantom.eval.tactile_prediction as TP
from phantom.config.hardware import load_hardware
from phantom.data import derived as dv
from phantom.data.episode_store import EpisodeReader
from phantom.data.tactile_cache import TAC_CH
from phantom.data.windows import bilinear_resize
from phantom.train import common as C
from tools.haptic_imagination_eval import (contact_event_prf, onset_timing,
                                           onset_window_prf, veto_signal)
import phantom.caewm.paths as CP
from tools.video_dump import load

SLOTS = (1, 3, 5)                    # +1 s, +2 s, +3 s among CAEWM's six future slots
NATIVE_DT = 1.0 / 5.85


def to_package(p, hw):
    cph, cpw = hw.cpk_shape
    H, W = hw.recording.field_ds.hw
    tau = hw.derived.tau_contact_depth
    depth = dv.channel_slices(hw.tactile)["depth"].start
    x = p.fields_B_6_2_C_H_W[0].float().cpu().numpy()                 # (6,2,5,64,80)
    ae = p._dx
    F = x.shape[1]
    mask = np.zeros((len(SLOTS), F, cph, cpw), np.float32)
    cop = np.full((len(SLOTS), F, 2), np.nan, np.float32)
    slip = np.zeros((len(SLOTS), F), np.float32)
    for k, s in enumerate(SLOTS):
        for f in range(F):
            stack = np.zeros((H, W, hw.tactile.field_ch), np.float32)
            stack[..., list(TAC_CH)] = bilinear_resize(x[s, f].transpose(1, 2, 0), (H, W))
            prev = stack.copy()
            prev[..., list(TAC_CH)] -= bilinear_resize(ae[s, f].transpose(1, 2, 0), (H, W))
            d = dv.derive_timestep(stack, prev, NATIVE_DT, hw)
            cop[k, f], slip[k, f] = d["cop"], d["slip"]
            mask[k, f] = np.abs(bilinear_resize(stack, (cph, cpw))[..., depth]) > tau
    ev = torch.softmax(p.event_logits_B_Tc_E[0].float(), -1).cpu().numpy()
    return {"d_fz": np.zeros((len(SLOTS), F, cph, cpw), np.float32), "mask": mask,
            "cop": cop, "slip": slip, "event": ev}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hardware", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--touch", default="sensor", choices=("sensor", "imagine"))
    a = ap.parse_args(argv)
    hw = load_hardware(a.hardware)
    pm, sampler, caewm = load(a.ckpt, hw, "cuda", torch.bfloat16)
    assert caewm
    rf, ae = pm.rf, pm.rf.tac_ae
    Tc, latent_dt = pm.bb.t_video - 1, sampler.latent_dt
    plan = json.loads(Path(a.windows).read_text())["windows"]
    rows, per_window = [], []
    for wi, w in enumerate(plan):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        item = sampler.sample(ep, t0)
        obs = TP.observed_package(TP._Streams(EpisodeReader(ep), hw), hw, t0, latent_dt, Tc)
        if obs is None:
            continue
        batch = C.to_device(C.collate_windows([item]), "cuda", torch.bfloat16)
        for k in list(batch):
            if k == "events" or k.startswith("cpk_"):
                batch[k] = torch.zeros_like(batch[k])
            elif k in ("tac_fut", "tac_slip", "tac_contact"):
                batch.pop(k)
        seed_steps = []
        for s in range(a.seeds):
            rf._gen = torch.Generator().manual_seed(1000 * s + wi)
            with torch.no_grad():
                p = rf.sample(batch, nfe=a.nfe, touch=a.touch == "sensor", decode_tactile=True)
                zs = p.tac_fut_B_6_2_C_h_w[0].float()
                _, dl = ae.decode_raw(zs[:, 0], mirror=False)
                _, dr = ae.decode_raw(zs[:, 1], mirror=True)
                p._dx = torch.stack([dl, dr], 1).cpu().numpy()
            pkg = to_package(p, hw)
            st = TP.score_replan(pkg, obs)
            seed_steps.append({k: v[None] for k, v in st.items()})
            rows.append({
                "episode": ep.name, "t0": t0, "seed": s, "terminal": bool(w["terminal"]),
                "task": item.get("task", ""),
                "ev_pred": [int(k) for k in np.argmax(pkg["event"], axis=-1)],
                "ev_obs": [int(k) for k in obs["event"]],
                "p_evt": p.acc.p_evt[0].float().cpu().numpy().tolist(),
                "gate": float(p.acc.g[0]), "gate_label": float(item["gate_label"]),
                "obs_contact0": bool(obs["contact0"]),
                "obs_contact": [bool(x) for x in obs["contact"]]})
        per_window.append({"terminal": bool(w["terminal"]),
                           "steps": {k: np.concatenate([d[k] for d in seed_steps]) for k in seed_steps[0]}})
        if wi % 20 == 0:
            print(f"{wi}/{len(plan)}", flush=True)

    def pooled(sel):
        ch = [x["steps"] for x in per_window if sel(x)]
        pl = {k: np.concatenate([c[k] for c in ch]) for k in ch[0]}
        n = int(pl["dfz_se"].shape[0])
        return TP.summarize_steps(pl, Tc=Tc, n_total=n, n_scored=n)

    term = [r for r in rows if r["terminal"]]
    out = {"ckpt": a.ckpt, "touch": a.touch, "nfe": a.nfe, "seeds": a.seeds,
           "n_windows": len(per_window), "n_rows": len(rows),
           "pooled": pooled(lambda x: True), "pooled_terminal": pooled(lambda x: x["terminal"]),
           "contact_event_prf": contact_event_prf(rows),
           "contact_event_prf_terminal": contact_event_prf(term),
           "onset_window_prf": onset_window_prf(rows), "onset_timing": onset_timing(rows, latent_dt),
           "veto_signal": veto_signal(rows), "veto_signal_terminal": veto_signal(term)}

    def dflt(o):
        if isinstance(o, (np.floating, np.integer)):
            v = o.item()
            return None if isinstance(v, float) and not math.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        raise TypeError(type(o).__name__)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=dflt))
    Path(a.out).with_name(Path(a.out).stem + "_rows.json").write_text(json.dumps({"rows": rows}, default=dflt))
    p = out["pooled"]
    print(f"mask IoU {p['tpe_mask_iou']:.3f} / persist {p['tpe_mask_iou_persist']:.3f}  "
          f"event acc {p['tpe_event_acc']:.3f} / {p['tpe_event_acc_persist']:.3f}  "
          f"veto AUROC {out['veto_signal'].get('auroc')}  gate AUROC {out['veto_signal'].get('gate', {}).get('auroc')}")


if __name__ == "__main__":
    main()
