"""Event accuracy of an official HapticWAM checkpoint read from its EventReadout head
(the same kind of readout CAEWM's event accuracy uses), on the fixed 248 windows with
the official seeds (1000*s + window index) and no previous-replan package. The
official imagination eval scores the packed event band instead; this puts both models
on the same readout. Event / onset statistics are the official functions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import phantom.eval.tactile_prediction as TP
from phantom.config.hardware import load_hardware
from phantom.config.model import EVENT_IDX
from phantom.data.episode_store import EpisodeReader
from phantom.train import common as C
from tools.haptic_imagination_eval import contact_event_prf, onset_timing, onset_window_prf
import phantom.caewm.paths as CP
from tools.video_dump import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--windows", default=str(CP.EVAL_WINDOWS))
    a = ap.parse_args()
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    pm, sampler, caewm = load(a.ckpt, hw, "cuda", torch.bfloat16)
    assert not caewm
    Tc, ldt = pm.bb.t_video - 1, sampler.latent_dt
    rows, acc, accp = [], [], []
    for wi, w in enumerate(json.loads(Path(a.windows).read_text())["windows"]):
        ep, t0 = CP.episode_path(w["episode"]), float(w["t0"])
        obs = TP.observed_package(TP._Streams(EpisodeReader(ep), hw), hw, t0, ldt, Tc)
        if obs is None:
            continue
        b = C.to_device(C.collate_windows([sampler.sample(ep, t0)]), "cuda", torch.bfloat16)
        for k in list(b):
            if k == "events" or k.startswith("cpk_"):
                b[k] = torch.zeros_like(b[k])
        persist = np.full(Tc, EVENT_IDX["hold"] if obs["contact0"] else EVENT_IDX["none"])
        for s in range(a.seeds):
            pm.rf._gen = torch.Generator().manual_seed(1000 * s + wi)
            with torch.no_grad():
                p = pm.rf.sample(b, nfe=1)
            ev = p.event_logits_B_Tc_E[0].float().argmax(-1).cpu().numpy()
            acc += list(ev == obs["event"]); accp += list(persist == obs["event"])
            rows.append({"terminal": bool(w["terminal"]), "ev_pred": ev.tolist(),
                         "ev_obs": [int(x) for x in obs["event"]], "obs_contact0": bool(obs["contact0"]),
                         "obs_contact": [bool(x) for x in obs["contact"]]})
        if wi % 20 == 0:
            print(f"{wi}/248", flush=True)
    out = {"ckpt": a.ckpt, "event_acc_head": float(np.mean(acc)), "event_acc_persist": float(np.mean(accp)),
           "contact_event_prf": contact_event_prf(rows), "onset_window_prf": onset_window_prf(rows),
           "onset_timing": onset_timing(rows, ldt)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
