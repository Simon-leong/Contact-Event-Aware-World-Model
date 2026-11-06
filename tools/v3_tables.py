"""Markdown table rows in the V1/V2 result-document formats, one row per "Label=path" argument.

  python -m tools.v3_tables TABLE "Label=path" ...

  endpoint   terminal_eval JSON: mean, median, > 60 mm, per task (TABLE II of the V2 document)
  endpoint2  terminal_eval JSON (with trajectories): time-tolerant endpoint, timing error, energy
             score, minFDE@4, hover, high close, Carton, the other three tasks
  package    cae_imagination_eval JSON: mask IoU, mask fraction error, CoP error, slip error
  horizon    cae_imagination_eval JSON: mask IoU at 1, 2, 3 s
  events     cae_imagination_eval JSON: event acc., contact-event F1, onset F1, onset error
  acc        cae_imagination_eval JSON: gate AUROC / F1, veto AUROC / F1
  anticip    v2_event_eval JSON: 1 s anticipation from the mid-block hazard
  force      v2_event_eval JSON: pad (sensor units) and wrist skill at each horizon
  field      "Label=sensor_tactile_eval.json,imagine_tactile_eval.json": IoU, CRPS, skill
  video      video_quality_score scores.json, optionally ",fvd.json": PSNR SSIM LPIPS, gripper
"""
from __future__ import annotations

import json
import sys

import numpy as np


def _j(p):
    return json.load(open(p))


def _f(v, fmt):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def endpoint(p):
    from tools.v3_report import load, summarize
    s = summarize(load(p))
    t = s["per_task"]
    return [_f(s["endpoint_mean"], "{:.2f}"), _f(s["endpoint_median"], "{:.1f}"), _f(100 * s["endpoint_gt60"], "{:.1f}%")] + \
        [_f(t[k]["endpoint_mean"], "{:.1f}") for k in ("whiteboard", "waffles", "egg", "Carton")]


def endpoint2(p):
    from tools.v3_report import load, summarize
    W = load(p)
    s = summarize(W)
    rest = [w["endpoint"] for w in W.values() if w["task"] != "Carton"]
    return [_f(s["tt_endpoint_mean"], "{:.2f}"), _f(s["timing_err_ms"], "{:.0f}"), _f(s["energy_score"], "{:.2f}"),
            _f(s["minfde"], "{:.2f}"), _f(100 * s["hover_rate"], "{:.1f}%"), _f(100 * s["high_close_rate"], "{:.1f}%"),
            _f(s["per_task"]["Carton"]["endpoint_mean"], "{:.1f}"), _f(float(np.mean(rest)), "{:.2f}")]


def package(p):
    q = _j(p)["pooled"]
    return [_f(q["tpe_mask_iou"], "{:.3f}"), _f(q["tpe_mask_frac_err"], "{:.3f}"), _f(q["tpe_cop_err"], "{:.3f}"),
            _f(100 * q["tpe_slip_err"], "{:.2f}")]


def horizon(p):
    q = _j(p)["pooled"]
    return [_f(q[f"tpe_mask_iou_s{k}"], "{:.3f}") for k in (1, 2, 3)]


def events(p):
    d = _j(p)
    return [_f(d["pooled"]["tpe_event_acc"], "{:.3f}"), _f(d["contact_event_prf"]["f1"], "{:.3f}"),
            _f(d["onset_window_prf"]["f1"], "{:.3f}"), _f(d["onset_timing"]["mae_ms"], "{:.0f}")]


def acc(p):
    v = _j(p)["veto_signal"]
    return [_f(v["gate"]["auroc"], "{:.3f}"), _f(v["gate"]["f1"], "{:.3f}"), _f(v["auroc"], "{:.3f}"), _f(v["f1"], "{:.3f}")]


def anticip(p):
    f = _j(p)["fixed"]
    g = lambda e, k: f[e]["1s"]["mid_hazard"][k]
    return [_f(g("slip", "auroc"), "{:.3f}"), _f(g("slip", "ap"), "{:.3f}"), _f(g("slip", "ece"), "{:.3f}"),
            _f(g("onset", "auroc"), "{:.3f}"), _f(g("onset", "ap"), "{:.3f}"), _f(g("close", "auroc"), "{:.3f}"),
            _f(g("lift", "auroc"), "{:.3f}"), _f(g("release", "ap"), "{:.3f}")]


def force(p):
    f = _j(p)["force"]
    return [[_f(v, "{:.2f}") for v in f["pad_phys_skill_by_horizon"]], [_f(v, "{:.2f}") for v in f["wrist_skill_by_horizon"]]]


def field(p):
    out = []
    for q in p.split(","):
        s = _j(q)["summary"]
        out += [_f(s["iou_all"], "{:.3f}"), _f(s["crps_all"], "{:.4f}"), _f(s["skill_all"], "{:.3f}")]
    return out


def video(p):
    ps = p.split(",")
    m = _j(ps[0])["model"]["mean"]
    out = [_f(m[k], f) for k, f in (("psnr", "{:.2f}"), ("ssim", "{:.3f}"), ("lpips", "{:.3f}"),
                                    ("grip_psnr", "{:.2f}"), ("grip_ssim", "{:.3f}"), ("grip_lpips", "{:.3f}"))]
    if len(ps) > 1:
        out.append(_f(_j(ps[1])["fvd"], "{:.1f}"))
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fn = globals()[argv[0]]
    for a in argv[1:]:
        label, p = a.split("=", 1)
        try:
            r = fn(p)
        except (FileNotFoundError, KeyError) as e:
            print(f"| {label} | missing {e} |")
            continue
        if argv[0] == "force":
            print(f"| {label} | Fingertip pad (sensor units) | " + " | ".join(r[0]) + " |")
            print(f"| {label} | Wrist wrench | " + " | ".join(r[1]) + " |")
        else:
            print(f"| {label} | " + " | ".join(r) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
