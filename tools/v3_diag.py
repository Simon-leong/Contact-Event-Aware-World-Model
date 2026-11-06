"""Phase-0 diagnostics of the three V3 hypotheses on V2 checkpoints (no training).

plan    terminal_eval with the mid-depth readout's hazards replaced by the step function of
        the demonstration's contact plan (first bins of onset / slip / close / lift / release),
        optionally shifted by --shift bins; --plan self keeps the model's own hazards and only
        logs how well its close / lift medians match the demonstration (hypothesis 1).
        Every argument after "--" goes to tools/terminal_eval.py.

  python -m tools.v3_diag plan --plan gt --shift 0 --log out_plan.json -- --ckpt ... --out ...

swap    writes a hybrid checkpoint that takes one part of the trainable weights from --donor
        and the rest from --base (hypothesis 2): early = LoRA of blocks 0..13, late = LoRA of
        blocks 14..27, lora = all LoRA, modules = every non-LoRA trainable module. The raw
        weights and their EMA are swapped together; the optimizer state is dropped.

  python -m tools.v3_diag swap --base a.pt --donor b.pt --part late --out hybrid.pt

gradcos gradients of the video and the action flow losses at the same noisy input, on
        training windows: their cosine and norm ratio over the shared LoRA parameters (all,
        blocks 0..13, blocks 14..27) and over the ACTION-token hidden states after blocks
        13 and 27 (hypothesis 2). Video decoupling and condition dropout are switched off
        so both losses see the same sample.

  python -m tools.v3_diag gradcos --ckpt a.pt --data ... --n 32 --out gc.json
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys

import torch

N_BINS = 12
BIG = 30.0


def step_logits(first_bin: torch.Tensor, shift: int) -> torch.Tensor:
    """first bins (B, ...) in [0, 12] (12 = not within 3 s) -> hazard logits (B, 12, ...) whose
    cumulative probability is 0 before the bin and 1 from it on."""
    fb = first_bin.long()
    fb = torch.where(fb < N_BINS, (fb + shift).clamp(0, N_BINS - 1), fb)
    bins = torch.arange(N_BINS, device=fb.device).view(1, N_BINS, *([1] * (fb.dim() - 1)))
    return torch.where(bins == fb.unsqueeze(1), BIG, -BIG).float()


def median_bin(logits: torch.Tensor) -> torch.Tensor:
    """hazard logits (B, 12, ...) -> first bin whose cumulative probability reaches 0.5 (12 = none)."""
    from phantom.caewm2.heads import cumulative
    cum = cumulative(logits.float(), dim=1)
    hit = cum >= 0.5
    first = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(hit[:, 0], N_BINS, dtype=torch.long))
    return first.long()


def run_plan(args, rest) -> None:
    import phantom.caewm2.heads as H
    from phantom.caewm2.rf import CaewmRectifiedFlow2

    state = {"bins": None, "first_call": False, "log": []}
    orig_fwd, orig_sample = H.EventForceHead.forward, CaewmRectifiedFlow2.sample

    def fwd(self, contact_hidden):
        out = orig_fwd(self, contact_hidden)
        if not getattr(self, "_v3_mid", False) or state["bins"] is None:
            return out
        fb, pb = state["bins"]
        if state["first_call"]:
            m = median_bin(out["phase_logits"])
            state["log"].append({"close_med": int(m[0, 0]), "lift_med": int(m[0, 1]),
                                 "close_gt": int(pb[0, 0]), "lift_gt": int(pb[0, 1])})
            state["first_call"] = False
        if args.plan == "gt":
            out = dict(out, finger_logits=step_logits(fb.to(out["finger_logits"].device), args.shift).to(out["finger_logits"].dtype),
                       phase_logits=step_logits(pb.to(out["phase_logits"].device), args.shift).to(out["phase_logits"].dtype))
        return out

    def sample(self, batch, **kw):
        self.net.phantom_mid_head._v3_mid = True
        state["bins"] = (batch["finger_bin"], batch["phase_bin"])
        state["first_call"] = True
        return orig_sample(self, batch, **kw)

    H.EventForceHead.forward = fwd
    CaewmRectifiedFlow2.sample = sample
    sys.argv = ["terminal_eval.py"] + rest
    try:
        runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("terminal_eval.py")), run_name="__main__")
    except SystemExit:
        pass
    finally:
        if args.log:
            json.dump({"plan": args.plan, "shift": args.shift, "calls": state["log"]}, open(args.log, "w"))


MID = 14


def _block(name: str):
    if ".blocks." not in name:
        return None
    return int(name.split(".blocks.")[1].split(".")[0])


def in_part(name: str, part: str) -> bool:
    is_lora = "lora_" in name
    b = _block(name)
    return {"early": is_lora and b is not None and b < MID,
            "late": is_lora and b is not None and b >= MID,
            "lora": is_lora,
            "modules": not is_lora}[part]


def run_swap(args) -> None:
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    assert base["configs"]["model"] == donor["configs"]["model"], "the two checkpoints differ in model config"
    n = 0
    for key in ("lora", "phantom_modules", "ema"):
        assert set(base[key]) == set(donor[key]), key
        for k in base[key]:
            if in_part(k, args.part):
                base[key][k] = donor[key][k]
                n += 1
    base.pop("optimizer", None)
    base["v3_swap"] = {"base": args.base, "donor": args.donor, "part": args.part, "tensors": n}
    torch.save(base, args.out)
    print(f"{args.out}: {n} tensors of part {args.part} from {args.donor}")


def cosine_and_ratio(gv, ga):
    """video / action gradient vectors -> (cosine, |g_video| / |g_action|)."""
    gv, ga = gv.double(), ga.double()
    nv, na = gv.norm(), ga.norm()
    return float(gv @ ga / (nv * na).clamp_min(1e-30)), float(nv / na.clamp_min(1e-30))


def run_gradcos(args) -> None:
    import numpy as np
    from phantom.config.hardware import load_hardware
    from phantom.model.sequence import FrameGroup
    from phantom.train import common as C
    from tools.video_dump import load

    hw = load_hardware(args.hardware)
    pm, sampler, _ = load(args.ckpt, hw, "cuda", torch.bfloat16)
    rf = pm.rf
    from dataclasses import replace
    rf.mc = replace(rf.mc, video_decouple_p=0.0, cond_dropout_p=0.0)
    names = [k for k, p in rf.named_parameters() if "lora_" in k]
    params = [p for k, p in rf.named_parameters() if "lora_" in k]
    for p in params:
        p.requires_grad_(True)
    early = torch.tensor([(_block(k) or 0) < MID for k in names])
    sizes = torch.tensor([p.numel() for p in params])
    seg = torch.repeat_interleave(early, sizes)
    sl_act = rf.layout.frame_slice(FrameGroup.ACTION)
    hidden = {}

    def hook(i):
        def f(_m, _inp, out):
            out.retain_grad()
            hidden[i] = out
        return f
    # after the last block the video loss no longer reaches the ACTION tokens (per-token final layer)
    hooks = [rf.net.blocks[i].register_forward_hook(hook(i)) for i in (MID - 1, 20, len(rf.net.blocks) - 2)]
    eps = C.manifest_split(__import__("pathlib").Path(args.data), "train")
    rng = np.random.default_rng(args.seed)
    rf._gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for j in range(args.n):
        ep = eps[int(rng.integers(len(eps)))]
        batch = C.to_device(C.collate_windows([sampler.sample(ep)]), "cuda", torch.bfloat16)
        parts = rf.training_step(batch)
        lv, la = parts["video_v_mse"], parts["action_v_mse"]
        hs = [hidden[i] for i in sorted(hidden)]
        gv = torch.autograd.grad(lv, params + hs, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(la, params + hs, allow_unused=True)

        def flat(gs, ps):
            return torch.cat([(g if g is not None else torch.zeros_like(p)).float().reshape(-1)
                              for g, p in zip(gs, ps)])
        P = len(params)
        fv, fa = flat(gv[:P], params).cpu(), flat(ga[:P], params).cpu()
        r = {"episode": ep.name, "loss_video": float(lv), "loss_action": float(la)}
        r["lora_all"] = cosine_and_ratio(fv, fa)
        r["lora_early"] = cosine_and_ratio(fv[seg], fa[seg])
        r["lora_late"] = cosine_and_ratio(fv[~seg], fa[~seg])
        for i, (hv, ha) in zip(sorted(hidden), zip(gv[P:], ga[P:])):
            hv = hv[:, sl_act].reshape(-1) if hv is not None else torch.zeros(1, device="cuda")
            ha = ha[:, sl_act].reshape(-1) if ha is not None else torch.zeros(1, device="cuda")
            r[f"hidden_block{i}"] = cosine_and_ratio(hv.cpu(), ha.cpu())
        rows.append(r)
        hidden.clear()
        print(j, {k: v for k, v in r.items() if k.startswith(("lora", "hidden"))}, flush=True)
    for h in hooks:
        h.remove()
    keys = [k for k in rows[0] if k.startswith(("lora", "hidden"))]
    summ = {k: {"cos_mean": float(np.mean([r[k][0] for r in rows])),
                "cos_frac_negative": float(np.mean([r[k][0] < 0 for r in rows])),
                "ratio_median": float(np.median([r[k][1] for r in rows]))} for k in keys}
    w = rf.mc.loss
    for v in summ.values():
        v["weighted_ratio_median"] = v["ratio_median"] * w.video / w.action
    json.dump({"ckpt": args.ckpt, "n": args.n, "loss_w": {"video": w.video, "action": w.action}, "summary": summ, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main() -> int:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--plan", choices=("gt", "self"), required=True)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--log", default="")
    s = sub.add_parser("swap")
    s.add_argument("--base", required=True)
    s.add_argument("--donor", required=True)
    s.add_argument("--part", choices=("early", "late", "lora", "modules"), required=True)
    s.add_argument("--out", required=True)
    g = sub.add_parser("gradcos")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--hardware", default=str(__import__("pathlib").Path(__file__).resolve().parents[1]
                                             / "configs/hardware.nuc.yaml"))
    g.add_argument("--n", type=int, default=32)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    args = ap.parse_args(own)
    {"plan": lambda: run_plan(args, rest), "swap": lambda: run_swap(args),
     "gradcos": lambda: run_gradcos(args)}[args.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
