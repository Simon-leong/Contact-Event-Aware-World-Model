"""CAEWM V3 tests (contact-plan factorisation)."""
from __future__ import annotations

import math

import numpy as np


def _row(ep, seed, pr_end, gt_ext, pred_close=100.0, gt_close=100.0, commit=1.0):
    H = 16
    pr = np.zeros((H, 3)); pr[-1] = pr_end
    return {"episode": ep, "task": "Carton", "t0": 1.0, "seed": seed,
            "endpoint_err_mm": float(np.linalg.norm(np.asarray(pr_end) - gt_ext[H - 1])),
            "z_end_err_mm": float(pr_end[2] - gt_ext[H - 1][2]),
            "pr_xyz": pr.tolist(), "gt_xyz_ext": gt_ext.tolist(),
            "pred_close_height_mm": pred_close, "gt_close_height_mm": gt_close, "commit_ratio": commit}


def test_v3_report_window_metrics():
    from tools.v3_report import window_metrics
    H, E = 16, 5
    gt = np.zeros((H + E, 3)); gt[:, 2] = -np.arange(1, H + E + 1) * 2.0      # descends 2 mm per step
    y = gt[H - 1]
    late = gt[H - 1 + 3]                                                      # where the demo is 3 steps later
    cands = [_row("e", 0, y, gt), _row("e", 1, late, gt, pred_close=float("nan")),
             _row("e", 2, y + [0, 0, 40.0], gt, pred_close=130.0), _row("e", 3, y, gt)]
    m = window_metrics(cands)
    assert np.allclose(m["endpoint"], [0, 6, 40, 0])
    assert np.allclose(m["tt"], [0, 0, 30, 0])          # 40 mm high = 20 steps early; only 5 (10 mm) are tolerated
    assert m["dt_ms"][1] == 300.0
    assert m["minfde"] == 0.0
    xs = np.array([y, late, y + [0, 0, 40.0], y])
    es = np.linalg.norm(xs - y, axis=1).mean() - np.linalg.norm(xs[:, None] - xs[None], axis=-1).sum() / 32
    assert math.isclose(m["es"], es)
    assert list(m["no_close"]) == [0, 1, 0, 0] and list(m["high"]) == [0, 0, 1, 0]
    assert list(m["hover"]) == [0, 0, 1, 0]             # ends 40 mm above the demonstration


def test_v3_report_bootstrap_clusters_episodes():
    from tools.v3_report import paired_bootstrap
    A = {("a", 1): {"episode": "a", "endpoint": np.array([2.0])}, ("b", 1): {"episode": "b", "endpoint": np.array([4.0])}}
    B = {("a", 1): {"episode": "a", "endpoint": np.array([1.0])}, ("b", 1): {"episode": "b", "endpoint": np.array([1.0])}}
    r = paired_bootstrap(A, B, "endpoint", n=200)
    assert r["diff"] == 2.0 and r["n_episodes"] == 2 and 1.0 <= r["ci95"][0] <= r["ci95"][1] <= 3.0


def test_v3_diag_step_logits_roundtrip():
    import torch
    from tools.v3_diag import median_bin, step_logits
    from phantom.caewm2.heads import cumulative
    fb = torch.tensor([[0, 5, 12], [11, 3, 12]])
    lg = step_logits(fb, 0)
    assert lg.shape == (2, 12, 3)
    assert torch.equal(median_bin(lg), fb)
    cum = cumulative(lg, dim=1)
    assert cum[0, 4, 1] < 1e-6 and cum[0, 5, 1] > 1 - 1e-6 and cum[0, :, 2].max() < 1e-6
    assert torch.equal(median_bin(step_logits(fb, 2)), torch.tensor([[2, 7, 12], [11, 5, 12]]))
    finger = torch.tensor([[[1, 12], [4, 9]]])                      # (B, finger, event)
    assert torch.equal(median_bin(step_logits(finger, 0)), finger)


# ---------------------------------------------------------------------------
# V3 model: contact plan (layer 1), plan-indexed bridge (layer 2), factor-wise adapters (layer 3)
import dataclasses
from pathlib import Path

import pytest
import torch

import phantom.caewm.paths as CP
import phantom.caewm2.paths as CP2
from phantom.config.paths import load_paths

EP = CP.TASKS / "egg" / "ep_egg_1785680901_002"
HW_YAML = str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml")


def test_plan_decide_and_resolve():
    from phantom.caewm3.schedule import DECIDE, NONE, UNKNOWN, decide, resolve
    from tools.v3_diag import median_bin, step_logits
    fb = torch.tensor([[0, 5, 12, 11, 3, 12, 7]])
    lg = step_logits(fb, 0)                                   # (1, 12, 7)
    assert torch.equal(decide(lg, "median"), fb)
    assert torch.equal(decide(lg, "median"), median_bin(lg))
    g = torch.Generator().manual_seed(0)
    s = decide(torch.randn(64, 12, 7), "sample", g)
    assert ((s >= 0) & (s <= NONE)).all() and len(s.unique()) > 3
    src = torch.tensor([[DECIDE, 4, UNKNOWN, NONE, DECIDE, 0, 2]])
    r = resolve(src, lg)
    assert r.tolist() == [[0, 4, UNKNOWN, NONE, 3, 0, 2]]


def test_plan_features():
    from phantom.caewm3.schedule import NONE, PLAN_FEAT_DIM, UNKNOWN, frame_edges_s, plan_features
    e = frame_edges_s(4, 4, 10.0)
    assert torch.allclose(e, torch.tensor([[0.1, 0.4], [0.5, 0.8], [0.9, 1.2], [1.3, 1.6]]))
    plan = torch.tensor([[5, NONE, UNKNOWN, 0, 0, 0, 0]])
    f = plan_features(plan, e, 0.25).view(1, 4, 7, 6)
    assert f.shape[-1] * 7 == PLAN_FEAT_DIM
    tau = 5.5 * 0.25                                          # bin 5 centre, 1.375 s
    assert torch.allclose(f[0, 0, 0, 0], torch.tanh(torch.tensor((tau - 0.1) / 0.25)))
    assert f[0, 3, 0, 1] < 0 < f[0, 3, 0, 0]                  # frame 3 starts before the event, ends after it
    assert f[0, :, 1, :4].abs().sum() == 0 and (f[0, :, 1, 4] == 1).all()     # NONE
    assert f[0, :, 2, :4].abs().sum() == 0 and (f[0, :, 2, 5] == 1).all()     # UNKNOWN


def test_torch_camera_matches_numpy():
    from phantom.caewm.geometry import Camera
    from phantom.caewm3.geometry import TorchCamera
    cam, tc = Camera(), TorchCamera()
    g = torch.Generator().manual_seed(0)
    P = torch.randn(50, 3, generator=g, dtype=torch.float64) * 0.3 + torch.tensor([0.4, -0.2, 0.2], dtype=torch.float64)
    uv, ok = tc.uv_norm(P)
    for i in range(len(P)):
        u_np, ok_np = cam.uv_norm(P[i].numpy())
        assert ok_np == bool(ok[i])
        if ok_np:
            assert torch.allclose(uv[i], torch.from_numpy(u_np), atol=1e-5)


def test_group_lora_routes_tokens():
    from phantom.caewm3.group_lora import GroupLoRALinear, TokenRouter
    T, tpf, D = 5, 6, 8
    frames = torch.tensor([False, False, True, True, False])
    router = TokenRouter(frames, tpf)
    lin = GroupLoRALinear(torch.nn.Linear(D, D, bias=False), router, 2, 2)
    x3 = torch.randn(2, T * tpf, D)
    x5 = x3.view(2, T, 2, 3, D)
    assert torch.equal(lin(x3), lin.base_layer(x3))            # B = 0: identity at init
    with torch.no_grad():
        lin.lora_B_action.normal_()
    d3 = (lin(x3) - lin.base_layer(x3)).abs().sum(-1)          # (2, T*tpf)
    act = frames.repeat_interleave(tpf)
    assert (d3[:, act] > 0).all() and (d3[:, ~act] == 0).all()
    d5 = (lin(x5) - lin.base_layer(x5)).abs().sum(-1)          # (2, T, 2, 3)
    assert (d5[:, frames] > 0).all() and (d5[:, ~frames] == 0).all()
    with torch.no_grad():
        lin.lora_B_action.zero_(); lin.lora_B_rest.normal_()
    d3 = (lin(x3) - lin.base_layer(x3)).abs().sum(-1)
    assert (d3[:, act] == 0).all() and (d3[:, ~act] > 0).all()
    ctx = torch.randn(2, 7, D)                                 # not a token sequence: shared only
    shared = GroupLoRALinear(torch.nn.Linear(D, D, bias=False), None, 2, 2)
    assert not hasattr(shared, "lora_A_action") and shared(ctx).shape == ctx.shape


def tiny_batch3(hw, bb):
    from tests.test_caewm2 import tiny_batch
    b = tiny_batch(hw, bb)
    b["tcp_xyz"] = torch.tensor([[0.45, -0.1, 0.25], [0.35, 0.05, 0.20]])
    for k in ("crops", "crop_valid"):
        b.pop(k)
    return b


@pytest.fixture(scope="module")
def tiny3():
    from phantom.caewm3.builder import build_caewm3_model
    from phantom.caewm3.config import Caewm3Config
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.config.model import LossWeights
    hw = load_hardware(HW_YAML)
    bb = dataclasses.replace(BackboneConfig.tiny(), res_h=256, res_w=320)
    mc = Caewm3Config(bridge_blocks=(0, 1), bridge_rank=32, mid_block=0, loss=LossWeights(video=1.0),
                      action_noise_per_strip=True, rope_time_mode="time_true", cond_dropout_p=0.0,
                      use_plan=True, use_plan_bridge=True, group_lora=True, lora_r_shared=4,
                      lora_r_private=4)
    pm = build_caewm3_model(hw, load_paths(), mc=mc, tiny=True, load_base=False, bb=bb)
    pm.rf.set_action_norm(torch.zeros(7), torch.full((7,), 0.01))
    return pm, tiny_batch3(hw, bb)


def _net_args3(pm, b):
    x0, cond, acc = pm.rf.build_x0(b)
    return dict(x_B_C_T_H_W=x0, timesteps_B_T=torch.full((2, 13), 500.0),
                crossattn_emb=pm.rf.text.get(2, b["text"]),
                condition_video_input_mask_B_C_T_H_W=pm.rf._cond_mask(cond, pm.layout).float(),
                action=b["prev_chunk"], acc_inputs=acc, layout=pm.layout,
                fps=torch.full((2,), 4.0), geo_G=pm.rf.geo_map(b, 2, "cpu"))


def test_v3_config_roundtrip(tiny3):
    from phantom.caewm3.config import Caewm3Config
    pm, _ = tiny3
    assert Caewm3Config.from_dict(dataclasses.asdict(pm.mc)) == pm.mc


def test_v3_zero_init_plan_and_lora_are_identity(tiny3):
    """At initialisation the plan embedding and every adapter output are zero, so the plan
    source does not matter and the factor-wise adapters equal the frozen base."""
    from phantom.caewm3.schedule import N_EVENTS, UNKNOWN, plan_of_bins
    pm, b = tiny3
    net = pm.rf.net
    pm.rf.eval()
    args = _net_args3(pm, b)
    gt = plan_of_bins(b["finger_bin"], b["phase_bin"])
    with torch.no_grad():
        o1 = net(**args, plan_src=gt)
        o2 = net(**args, plan_src=torch.full((2, N_EVENTS), UNKNOWN))
    assert torch.equal(o1.velocity_B_C_T_H_W, o2.velocity_B_C_T_H_W)
    assert torch.equal(o1.plan, gt)
    lin = [m for m in net.modules() if m.__class__.__name__ == "GroupLoRALinear"]
    assert len(lin) == 2 * 10 and sum(hasattr(m, "lora_A_action") for m in lin) == 2 * 8
    assert all(m.lora_B_shared.abs().sum() == 0 for m in lin)


def test_v3_train_step_grads_and_keys(tiny3):
    pm, b = tiny3
    pm.rf.train()
    mc0 = pm.rf.mc
    keys = []
    for anchor_p in (0.0, 1.0):
        pm.rf.mc = dataclasses.replace(mc0, touch_forcing_p=0.0, video_decouple_p=0.0, anchor_p=anchor_p)
        parts = pm.rf.training_step(b)
        keys.append(sorted(parts))
        assert torch.isfinite(parts["total"])
        parts["total"].backward()
        got = {n for n, p in pm.rf.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        for key in ("lora_B_shared", "lora_B_action", "lora_B_rest", "phantom_mid_head.",
                    "phantom_plan_embed.out", "phantom_bridges.0.v_out", "phantom_bridges.0.t_out"):
            assert any(key in n for n in got), (anchor_p, key)
        pm.rf.zero_grad(set_to_none=True)
    assert keys[0] == keys[1]                                  # DDP: same loss keys either way
    pm.rf.mc = mc0


def test_v3_anchor_prepass_has_no_grad(tiny3):
    pm, b = tiny3
    seen = []
    orig = pm.rf._anchors

    def spy(x_est, batch, B):
        seen.append(x_est.requires_grad)
        return orig(x_est, batch, B)
    pm.rf._anchors = spy
    try:
        pm.rf.train()
        pm.rf.mc = dataclasses.replace(pm.rf.mc, anchor_p=1.0)
        pm.rf.training_step(b)
    finally:
        del pm.rf._anchors
        pm.rf.mc = dataclasses.replace(pm.rf.mc, anchor_p=0.5)
    assert seen == [False]


def test_v3_anchors_follow_trajectory(tiny3):
    """Anchors are the projected fingertip along the de-normalised action estimate; a zero
    trajectory puts every future anchor on the current fingertip."""
    from phantom.caewm.geometry import Camera, tcp_gaussian
    from phantom.model.sequence import FrameGroup
    pm, b = tiny3
    rf = pm.rf
    x = torch.zeros(2, pm.layout.lat_c, pm.layout.t_total, pm.layout.lat_h, pm.layout.lat_w)
    zero = rf.a_pack.pack(torch.zeros(2, 16, 7))
    x[:, :, pm.layout.frame_slice(FrameGroup.ACTION)] = zero
    a = rf._anchors(x, b, 2)
    cam = Camera()
    uv, ok = cam.uv_norm(b["tcp_xyz"][0].numpy())
    G0 = tcp_gaussian(torch.tensor([uv]), torch.tensor([float(ok)]), rf.hp, rf.wp, pm.mc.bridge_sigma)[0]
    for k in ("con", "vid"):
        assert a[k].shape == (2, 3, rf.hp, rf.wp)
        assert torch.allclose(a[k][0, 0], G0, atol=1e-4) and torch.allclose(a[k][0, 2], G0, atol=1e-4)


def test_v3_sample_decides_then_keeps_plan(tiny3):
    from phantom.caewm3.schedule import UNKNOWN, plan_of_bins
    pm, b = tiny3
    pm.rf.eval()
    seen = []
    net = pm.rf.net
    orig = type(net).forward

    def spy(self, *a, **kw):
        seen.append(None if kw.get("plan_src") is None else kw["plan_src"].clone())
        return orig(self, *a, **kw)
    type(net).forward = spy
    try:
        p = pm.rf.sample(b, nfe=2, touch=False, k_seeds=2)
    finally:
        type(net).forward = orig
    assert p.plan_B_7.shape == (4, 7) and (p.plan_B_7 >= 0).all()
    assert (seen[0] == -2).all() and torch.equal(seen[1], p.plan_B_7)     # decide, then keep
    p = pm.rf.sample(b, nfe=2, touch=True, plan="gt")
    assert torch.equal(p.plan_B_7, plan_of_bins(b["finger_bin"], b["phase_bin"]))
    p = pm.rf.sample(b, nfe=1, touch=True, plan="unknown", decode_tactile=True)
    assert (p.plan_B_7 == UNKNOWN).all() and p.fields_B_6_2_C_H_W.shape == (2, 6, 2, 5, 64, 80)


def test_v3_real_pipeline_step_and_checkpoint(tiny3, tmp_path):
    from phantom.caewm3.builder import build_caewm3_model
    from phantom.caewm3.config import Caewm3Config
    from phantom.caewm3.windows import Caewm3WindowSampler
    from phantom.config.hardware import load_hardware
    from phantom.config.training import TeacherTrainConfig
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    if not (CP2.EVENT_CACHE / EP.name / "events.npz").exists():
        pytest.skip("event cache not built")
    pm, _ = tiny3
    hw = load_hardware(HW_YAML)
    s = Caewm3WindowSampler(hw, pm.bb, NormStats.identity(), seed=0, lat_cache=str(CP.LAT_CACHE),
                            tac_cache=str(CP.TAC_CACHE), dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(),
                            event_cache=str(CP2.EVENT_CACHE), use_crop=False)
    eps = [EP, next(p for p in sorted(CP.TASKS.glob("Carton/ep_*")) if p.is_dir())]
    ds = C.WindowDataset(CP.TASKS, s, windows_per_episode=2, episodes=eps, seed=0,
                         grasp_frac=0.3, photo_aug=1.0)
    b = C.collate_windows([ds[i] for i in range(2)])
    assert b["tcp_xyz"].shape == (2, 3) and "crops" not in b
    pm.rf.train()
    parts = pm.rf.training_step(C.to_device(b, "cpu"))
    assert all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()
    pm.rf.zero_grad(set_to_none=True)

    ema = C.EMA(pm.rf, 0.999)
    path = tmp_path / "ck.pt"
    C.save_phantom_checkpoint(path, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc, train_cfg=TeacherTrainConfig(),
                              step=1, norm_stats=NormStats.identity(), ema=ema)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["configs"]["model"]["arch"] == "caewm3"
    assert any("lora_A_action" in k for k in payload["lora"]) and any("lora_A_action" in k for k in payload["ema"])
    mc2 = Caewm3Config.from_dict(payload["configs"]["model"])
    assert mc2 == pm.mc
    pm2 = build_caewm3_model(hw, load_paths(), mc=mc2, tiny=True, load_base=False, bb=pm.bb)
    C.load_phantom_checkpoint(path, pm2.rf, hw=hw, payload=payload, load_ema=True)
    sd1, sd2 = pm.rf.state_dict(), pm2.rf.state_dict()
    for k in sd1:
        if "phantom_" in k or "lora_" in k:
            assert torch.equal(sd1[k], sd2[k]), k


def test_v3_self_plan_decided_at_t1(tiny3):
    """DECIDE rows are decided by a no-grad pass at t = 1 (the first sampling step), also when
    no anchors are drawn; the main pass then receives that resolved plan."""
    from phantom.caewm3.schedule import DECIDE
    pm, b = tiny3
    net, calls = pm.rf.net, []
    orig = type(net).forward

    def spy(self, *a, **kw):
        out = orig(self, *a, **kw)
        calls.append((float(kw["timesteps_B_T"].max()), kw["plan_src"].clone(), out.plan.clone()))
        return out
    mc0 = pm.rf.mc
    type(net).forward = spy
    try:
        pm.rf.train()
        pm.rf.mc = dataclasses.replace(mc0, plan_p_gt=0.0, plan_p_self=1.0, anchor_p=0.0,
                                       video_decouple_p=0.0)
        pm.rf.training_step(b)
    finally:
        type(net).forward = orig
        pm.rf.mc = mc0
    assert len(calls) == 2
    (t_pre, src_pre, plan_pre), (_, src_main, _) = calls
    assert t_pre == 1000.0 and (src_pre == DECIDE).all()
    assert torch.equal(src_main, plan_pre) and (src_main >= 0).all()


def test_v3_anchor_on_pinned_actions_uses_x0(tiny3):
    """On video-decoupled samples the ACTION frames are conditions: their trajectory is the
    demonstration's own chunk, not x - t v."""
    pm, b = tiny3
    seen = []
    orig = pm.rf._anchors

    def spy(x_est, batch, B):
        seen.append(pm.rf.a_pack.unpack(x_est[:, :, pm.layout.frame_slice(
            __import__("phantom.model.sequence", fromlist=["FrameGroup"]).FrameGroup.ACTION)].float()))
        return orig(x_est, batch, B)
    mc0 = pm.rf.mc
    pm.rf._anchors = spy
    try:
        pm.rf.train()
        pm.rf.mc = dataclasses.replace(mc0, anchor_p=1.0, video_decouple_p=1.0)
        pm.rf.training_step(b)
    finally:
        del pm.rf._anchors
        pm.rf.mc = mc0
    assert torch.allclose(seen[0], b["action_chunk"].float(), atol=2e-2)


def test_v3_camera_full_precision(tiny3):
    from phantom.caewm.geometry import Camera
    pm, _ = tiny3
    assert pm.rf.cam.K.dtype == torch.float64
    assert torch.equal(pm.rf.cam.K, torch.from_numpy(Camera().K))
    assert pm.rf.net._plan_edges.dtype == torch.float32


def test_plan_quantile_rule():
    from phantom.caewm3.schedule import decide
    lg = torch.full((1, 12, 1), -1.0)                          # constant hazard, F rises gradually
    bins = [int(decide(lg, r)[0, 0]) for r in ("q0.3", "median", "q0.7", "q0.9")]
    assert bins == sorted(bins) and bins[0] < bins[-1]
    assert int(decide(lg, "median")[0, 0]) == int(decide(lg, "q0.5")[0, 0])


def test_plan_mode_rule_picks_a_mode():
    """Bimodal first-event pmf (peaks at bins 2 and 9): the mode commits to the larger peak,
    the median falls between them."""
    from phantom.caewm3.schedule import decide
    pmf = torch.zeros(13); pmf[2], pmf[9], pmf[12] = 0.42, 0.48, 0.10
    surv = 1 - torch.cumsum(pmf[:12], 0)
    prev = torch.cat([torch.ones(1), surv[:-1]])
    h = (pmf[:12] / prev).clamp(1e-6, 1 - 1e-6)
    lg = torch.logit(h).view(1, 12, 1)
    assert int(decide(lg, "mode")[0, 0]) == 9
    assert int(decide(lg, "median")[0, 0]) == 9 or 2 < int(decide(lg, "median")[0, 0]) <= 9
    pmf2 = torch.zeros(13); pmf2[12] = 0.7; pmf2[4] = 0.3
    surv2 = 1 - torch.cumsum(pmf2[:12], 0); prev2 = torch.cat([torch.ones(1), surv2[:-1]])
    lg2 = torch.logit((pmf2[:12] / prev2).clamp(1e-6, 1 - 1e-6)).view(1, 12, 1)
    assert int(decide(lg2, "mode")[0, 0]) == 12


def test_v3_sample_redecide_on_second_step(tiny3):
    """--plan redecide: DECIDE on the first step and again on the second; given entries stay."""
    pm, b = tiny3
    pm.rf.eval()
    seen = []
    net = pm.rf.net
    orig = type(net).forward

    def spy(self, *a, **kw):
        seen.append(kw["plan_src"].clone())
        return orig(self, *a, **kw)
    type(net).forward = spy
    try:
        p = pm.rf.sample(b, nfe=3, touch=True, plan="redecide")
    finally:
        type(net).forward = orig
    assert (seen[0] == -2).all() and (seen[1] == -2).all() and (seen[2] >= 0).all()
    assert torch.equal(seen[2], p.plan_B_7)


def test_v3_sample_anchor_ablation(tiny3):
    pm, b = tiny3
    pm.rf.eval()
    calls = []
    orig = pm.rf._anchors
    pm.rf._anchors = lambda x, batch, B: calls.append(1) or orig(x, batch, B)
    try:
        pm.rf.sample(b, nfe=2, touch=True)
        n_on = len(calls)
        pm.rf.sample(b, nfe=2, touch=True, anchors=False)
    finally:
        del pm.rf._anchors
    assert n_on == 2 and len(calls) == n_on


def test_decision_spec_bayes_close_and_group():
    from phantom.caewm3.schedule import CLOSE, NONE, DecisionSpec, _pmf, decide
    from phantom.caewm2.heads import cumulative
    torch.manual_seed(0)
    lg = torch.randn(4, 12, 7)
    # a plain spec is the string rule
    assert torch.equal(decide(lg, DecisionSpec(base="q0.7")), decide(lg, "q0.7"))
    # group K: one plan per group, decided from the averaged cumulative
    d = decide(lg, DecisionSpec(base="median", group=2))
    assert torch.equal(d[0], d[1]) and torch.equal(d[2], d[3])
    cum = cumulative(lg.float(), dim=1).view(2, 2, 12, 7).mean(1)
    hit = cum >= 0.5
    ref = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(hit[:, 0], NONE, dtype=torch.float)).long()
    assert torch.equal(d[::2], ref)
    # 0-1 cost -> the close is the most probable first bin (NONE included); other events keep base
    C = (1 - torch.eye(13)).unsqueeze(0).expand(4, 13, 13)
    d = decide(lg, DecisionSpec(base="median", close_cost=C))
    pmf = _pmf(cumulative(lg.float(), dim=1))[..., CLOSE]
    assert torch.equal(d[:, CLOSE], pmf.argmax(1)) and torch.equal(d[:, :CLOSE], decide(lg, "median")[:, :CLOSE])
    # NONE made prohibitively expensive -> never decided
    C2 = C.clone(); C2[:, NONE, :NONE] = 1e6
    assert (decide(lg, DecisionSpec(base="median", close_cost=C2))[:, CLOSE] < NONE).all()


def test_v3_sample_close_shift_none_and_mean(tiny3):
    from phantom.caewm3.schedule import CLOSE, NONE, DecisionSpec, plan_of_bins
    pm, b = tiny3
    pm.rf.eval()
    gt = plan_of_bins(b["finger_bin"], b["phase_bin"])
    p = pm.rf.sample(b, nfe=1, touch=True, plan="gt", plan_close_shift=2)
    c = gt[:, CLOSE]; known = (c >= 0) & (c < NONE)
    assert torch.equal(p.plan_B_7[:, CLOSE], torch.where(known, (c + 2).clamp(0, NONE - 1), c))
    assert torch.equal(p.plan_B_7[:, :CLOSE], gt[:, :CLOSE])
    p = pm.rf.sample(b, nfe=1, touch=True, plan="gt", plan_close_none=True)
    assert (p.plan_B_7[:, CLOSE] == NONE).all()
    p = pm.rf.sample(b, nfe=2, touch=True, k_seeds=3, mean_action=True,
                     plan_rule=DecisionSpec(base="q0.8", group=3))
    a = p.actions_B_H_A.view(2, 3, *p.actions_B_H_A.shape[1:])
    assert torch.allclose(a, a[:, :1].expand_as(a))                        # one executed chunk
    pl = p.plan_B_7.view(2, 3, 7)
    assert (pl == pl[:, :1]).all()                                         # one shared plan
