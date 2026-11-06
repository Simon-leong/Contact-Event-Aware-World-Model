"""Training-subset window plan with the evaluation windows' own rule (tools/haptic_imagination_eval
build_window_plan, 2 per episode, half in the pre-close band, plan seed 7) over every 8th episode of
the task-sorted training split (terminal_eval --episode-stride 8), for thresholds and rules that
must be chosen on training data only.

  python -m tools.v3_train_windows --ckpt CK --out configs/eval/windows_train8x2_seed7.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import phantom.caewm.paths as CP
from phantom.config.hardware import load_hardware
from phantom.train import common as C
from tools.haptic_imagination_eval import build_window_plan
from tools.video_dump import load



def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    _, sampler, _ = load(a.ckpt, hw, "cuda", torch.bfloat16)
    eps = sorted(C.manifest_split(CP.TASKS, "train"))[::a.stride]
    plan = build_window_plan(sampler, eps, per_episode=2, terminal_frac=0.5, seed=7)
    for w in plan:
        w["episode"] = str(Path(w["episode"]).relative_to(CP.TASKS))
    Path(a.out).write_text(json.dumps({"split": "train", "episode_stride": a.stride, "plan_seed": 7, "windows": plan}, indent=1))
    print(f"{len(plan)} windows over {len(eps)} training episodes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
