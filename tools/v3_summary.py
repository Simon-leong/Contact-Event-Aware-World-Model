"""One row of gate metrics per evaluation directory (scripts/pilot_eval_v3.sh layout), with the
paired endpoint bootstrap against a reference directory of the same step when given.

  python -m tools.v3_summary DIR [DIR ...] [--ref REF_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def row(d: Path) -> dict:
    r = {"dir": str(d)}
    rep = _j(d / "terminal_sensor_nfe2.report.json")
    if rep:
        s = rep["summary"]
        r.update(endpoint=s["endpoint_mean"], gt60=100 * s["endpoint_gt60"], tt=s["tt_endpoint_mean"],
                 high=100 * s["high_close_rate"], hover=100 * s["hover_rate"], carton=s["per_task"]["Carton"]["endpoint_mean"])
    for p in ("gt", "unknown"):
        rp = _j(d / f"terminal_sensor_nfe2_plan{p}.report.json")
        if rp:
            r[f"endpoint_plan{p}"] = rp["summary"]["endpoint_mean"]
    rq = _j(d / "terminal_sensor_nfe2_q0.7.report.json")
    if rq:
        r.update(endpoint_q07=rq["summary"]["endpoint_mean"], carton_q07=rq["summary"]["per_task"]["Carton"]["endpoint_mean"])
    v = _j(d / "video_nfe1" / "scores.json")
    if v:
        m = v["model"]["mean"]
        r.update(psnr=m["psnr"], lpips=m["lpips"])
    im = _j(d / "imagination_sensor.json")
    if im:
        r.update(iou=im["pooled"]["tpe_mask_iou"], event_f1=im["contact_event_prf"]["f1"],
                 onset_f1=im["onset_window_prf"]["f1"], veto_auroc=im["veto_signal"]["auroc"],
                 gate_auroc=im["veto_signal"]["gate"]["auroc"])
    t = _j(d / "tactile_imagine" / "tactile_eval.json")
    if t:
        r.update(notouch_iou=t["summary"]["iou_all"], notouch_crps=t["summary"]["crps_all"])
    f = _j(d / "fvd_nfe1.json")
    if f:
        r["fvd"] = f["fvd"]
    return r


FMT = [("endpoint", "{:.2f}"), ("gt60", "{:.1f}"), ("tt", "{:.2f}"), ("high", "{:.1f}"), ("hover", "{:.1f}"),
       ("carton", "{:.1f}"), ("endpoint_q07", "{:.2f}"), ("carton_q07", "{:.1f}"),
       ("endpoint_plangt", "{:.2f}"), ("endpoint_planunknown", "{:.2f}"),
       ("psnr", "{:.2f}"), ("lpips", "{:.3f}"), ("iou", "{:.3f}"), ("event_f1", "{:.3f}"),
       ("onset_f1", "{:.3f}"), ("veto_auroc", "{:.3f}"), ("gate_auroc", "{:.3f}"),
       ("notouch_iou", "{:.3f}"), ("notouch_crps", "{:.4f}"), ("fvd", "{:.1f}")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args(argv)
    rows = [row(Path(d)) for d in a.dirs]
    keys = [k for k, _ in FMT if any(k in r for r in rows)]
    print("dir".ljust(28) + "".join(k[:11].rjust(12) for k in keys))
    for r in rows:
        print(Path(r["dir"]).parent.name[-14:].ljust(14) + Path(r["dir"]).name[-14:].ljust(14)
              + "".join((fmt.format(r[k]) if k in r else "-").rjust(12) for k, fmt in FMT if k in keys))
    if a.ref:
        from tools.v3_report import load, paired_bootstrap
        R = load(str(Path(a.ref) / "terminal_sensor_nfe2.json"))
        for d in a.dirs:
            W = load(str(Path(d) / "terminal_sensor_nfe2.json"))
            b = paired_bootstrap(W, R, "endpoint")
            print(f"{d}: endpoint vs ref {b['diff']:+.2f} mm, 95% CI [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
