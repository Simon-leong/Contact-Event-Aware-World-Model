"""Write release copies of a CAEWM checkpoint and its tactile tokenizer.

The optimizer state is dropped (inference, evaluation and fine-tuning from the weights do
not need it) and every absolute path recorded on the training machine is cleared, so the
loaders fall back to the locations derived from configs/paths*.yaml. The output mirrors
<runs_root>:

    <out>/runs/teacher/<run>/teacher_<step>.pt
    <out>/runs/tactile_ae/tactile_ae.pt

    python -m tools.export_release --ckpt <teacher ckpt> --ae <tactile_ae.pt> --out <dir>
"""
import argparse
from pathlib import Path

import torch


def scrub(o, cleared, at=""):
    if isinstance(o, dict):
        return {k: scrub(v, cleared, f"{at}.{k}") for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(scrub(v, cleared, f"{at}[{i}]") for i, v in enumerate(o))
    if isinstance(o, str) and o.startswith("/"):
        cleared.append(at)
        return ""
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", default="caewm_v1")
    a = ap.parse_args()
    out = Path(a.out) / "runs"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    rel = {k: v for k, v in ck.items() if k != "optimizer"}
    cleared = []
    for k in ("base_ckpt_path", "configs"):
        rel[k] = scrub(rel[k], cleared, k)
    dst = out / "teacher" / a.run_name / Path(a.ckpt).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rel, dst)
    print("checkpoint ->", dst, "| cleared", cleared)

    ae = torch.load(a.ae, map_location="cpu", weights_only=False)
    rel = {k: ae[k] for k in ("model", "step", "val", "stats")}
    dst = out / "tactile_ae" / "tactile_ae.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rel, dst)
    print("tokenizer  ->", dst)


if __name__ == "__main__":
    main()
