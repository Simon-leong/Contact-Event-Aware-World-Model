"""CAEWM V2 unit tests: survival loss and hazard head, the V2 sampler on a real episode,
and a tiny-backbone (full 256x320 resolution, 2 blocks) train step / sample / zero-init
crop bridge / mid-depth gate (incl. activation-checkpoint recompute) checks."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import phantom.caewm.paths as CP
import phantom.caewm2.paths as CP2
from phantom.caewm2.heads import SUMMARY_DIM, EventForceHead, cumulative, hazard_nll, summary
from phantom.caewm2.windows import crop_at, first_bin
from phantom.config.paths import load_paths

EP = CP.TASKS / "egg" / "ep_egg_1785680901_002"


def test_first_bin():
    t = np.array([0.9, 1.6, 5.0])
    assert first_bin(t, 1.0, 0.25) == 2            # 1.6 - 1.0 = 0.6 -> bin 2
    assert first_bin(t, 0.9, 0.25) == 2            # event at t0 itself is not "after t0"
    assert first_bin(t, 5.0, 0.25) == 12
    assert first_bin(np.array([4.0]), 1.0, 0.25) == 11   # exactly t0 + 3 s
    assert first_bin(np.array([]), 1.0, 0.25) == 12
    assert first_bin(np.array([1.25]), 1.0, 0.25) == 0      # bins are right-closed
    assert first_bin(np.array([1.26]), 1.0, 0.25) == 1


def test_hazard_nll_matches_manual():
    lg = torch.randn(3, 12)
    k = torch.tensor([0, 5, 12])
    got = hazard_nll(lg, k, torch.ones(3))
    h = torch.sigmoid(lg)
    man = []
    for b in range(3):
        kb = int(k[b])
        ll = torch.log(1 - h[b, :kb]).sum()
        if kb < 12:
            ll = ll + torch.log(h[b, kb])
        man.append(-ll)
    assert torch.allclose(got, torch.stack(man).mean(), atol=1e-5)
    assert hazard_nll(lg, k, torch.zeros(3)).item() == 0.0
    F = cumulative(lg.unsqueeze(-1))[..., 0]
    assert (F[:, 1:] >= F[:, :-1] - 1e-7).all()
    surv = torch.cumprod(1 - h, 1)
    assert torch.allclose(F, 1 - surv, atol=1e-6)


def test_event_force_head_shapes_and_stopgrad():
    D, hp, wp = 32, 16, 20
    head = EventForceHead(D, hp, wp, width=16)
    x = torch.randn(2, 3, hp * wp, D)
    o = head(x)
    assert o["finger_logits"].shape == (2, 12, 2, 2)
    assert o["phase_logits"].shape == (2, 12, 3)
    assert o["pad"].shape == (2, 6, 2, 3) and o["wrist"].shape == (2, 6, 6)
    assert summary(o).shape == (2, SUMMARY_DIM)
    nograd = lambda m: m.weight.grad is None or not m.weight.grad.any()  # noqa: E731
    o["finger_logits"][..., 1].sum().backward()           # slip only
    assert nograd(head.pad) and nograd(head.onset) and not nograd(head.slip)
    head.zero_grad(set_to_none=True)
    o = head(x)
    o["pad"].sum().backward()
    assert nograd(head.onset) and nograd(head.phase) and not nograd(head.pad)
    mid = EventForceHead(D, hp, wp, width=16, force=False)(x)
    assert set(mid) == {"finger_logits", "phase_logits"}


def test_crop_at_padding():
    img = np.full((480, 640, 3), 200, np.uint8)
    c = crop_at(img, 0.0, 0.0, 160, 224)
    assert c.shape == (3, 224, 224) and c.dtype == np.uint8
    assert c[:, 0, 0].max() == 0 and c[:, -1, -1].min() == 200


@pytest.fixture(scope="module")
def real_sampler():
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.data.schema import NormStats
    from phantom.caewm2.windows import Caewm2WindowSampler
    if not (CP2.EVENT_CACHE / EP.name / "events.npz").exists():
        pytest.skip("event cache not built")
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    bb = BackboneConfig()
    s = Caewm2WindowSampler(hw, bb, NormStats.identity(), lat_cache=str(CP.LAT_CACHE),
                            tac_cache=str(CP.TAC_CACHE), dz_std=np.ones(16, np.float32),
                            event_cache=str(CP2.EVENT_CACHE))
    return s


def test_real_sampler_keys(real_sampler):
    s = real_sampler
    lo, hi = s.valid_range(EP)
    w = s.sample(EP, lo + 0.5 * (hi - lo))
    assert w["finger_bin"].shape == (2, 2) and w["finger_bin"].dtype == torch.int64
    assert w["phase_bin"].shape == (3,)
    assert ((w["finger_bin"] >= 0) & (w["finger_bin"] <= 12)).all()
    assert w["pad_fut"].shape == (6, 2, 3) and w["wrist_fut"].shape == (6, 6)
    assert torch.isfinite(w["pad_fut"]).all() and torch.isfinite(w["wrist_fut"]).all()
    assert w["crops"].shape == (2, 3, 224, 224) and w["crops"].dtype == torch.uint8
    assert w["crop_valid"].shape == (2,)
    w2 = s.sample(EP)                                         # t0 drawn by the sampler
    assert "finger_bin" in w2


def tiny_batch(hw, bb):
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    from phantom.train import common as C
    s = WindowSampler(hw, bb, NormStats.identity(), student=True)
    lo, hi = s.valid_range(EP)
    items = [s.sample(EP, lo + 0.3 * (hi - lo)), s.sample(EP, lo + 0.6 * (hi - lo))]
    b = C.collate_windows(items)
    g = torch.Generator().manual_seed(0)
    b["tac_obs"] = torch.randn(2, 2, 2, 16, 8, 10, generator=g)
    b["tac_fut"] = torch.randn(2, 6, 2, 16, 8, 10, generator=g)
    b["tac_slip"] = (torch.rand(2, 6, 2, generator=g) > 0.7).float()
    b["tac_contact"] = (torch.rand(2, 6, 2, generator=g) > 0.3).float()
    b["tcp_uv"] = torch.tensor([[0.5, 0.6], [0.3, 0.4]])
    b["tcp_valid"] = torch.tensor([1.0, 1.0])
    b["finger_bin"] = torch.tensor([[[3, 12], [5, 7]], [[12, 12], [0, 12]]])
    b["phase_bin"] = torch.tensor([[2, 8, 12], [12, 12, 12]])
    b["pad_fut"] = torch.randn(2, 6, 2, 3, generator=g)
    b["wrist_fut"] = torch.randn(2, 6, 6, generator=g)
    b["crops"] = torch.randint(0, 256, (2, 2, 3, 224, 224), generator=g, dtype=torch.uint8)
    b["crop_valid"] = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    return b


@pytest.fixture(scope="module")
def tiny():
    from phantom.caewm2.builder import build_caewm2_model
    from phantom.caewm2.config import Caewm2Config
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.config.model import LossWeights
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    bb = dataclasses.replace(BackboneConfig.tiny(), res_h=256, res_w=320)
    mc = Caewm2Config(bridge_blocks=(0, 1), bridge_rank=32, crop_blocks=(0,), crop_rank=32,
                      mid_block=0, loss=LossWeights(video=0.3), action_noise_per_strip=True,
                      rope_time_mode="time_true", cond_dropout_p=0.0)
    pm = build_caewm2_model(hw, load_paths(), mc=mc, tiny=True, load_base=False, bb=bb)
    return pm, tiny_batch(hw, bb)


def test_config_roundtrip(tiny):
    from phantom.caewm2.config import Caewm2Config
    pm, _ = tiny
    assert Caewm2Config.from_dict(dataclasses.asdict(pm.mc)) == pm.mc


def test_train_step_and_grads(tiny):
    pm, b = tiny
    pm.rf.train()
    mc0 = pm.rf.mc
    pm.rf.mc = dataclasses.replace(mc0, touch_forcing_p=1.0, video_decouple_p=1.0)
    parts = pm.rf.training_step(b)
    for k in ("action_v_mse", "tac_fut_v_mse", "tac_obs_v_mse", "event_ce", "hazard_finger",
              "hazard_phase", "force_pad", "force_wrist", "mid_hazard_phase"):
        assert parts[k].item() == 0, k
    assert parts["video_v_mse"].item() > 0
    pm.rf.mc = dataclasses.replace(mc0, touch_forcing_p=0.0, video_decouple_p=0.0)
    parts = pm.rf.training_step(b)
    assert torch.isfinite(parts["total"])
    for k in ("action_v_mse", "tac_fut_v_mse", "tac_obs_v_mse", "hazard_finger", "hazard_phase",
              "force_pad", "force_wrist", "mid_hazard_finger", "mid_hazard_phase"):
        assert parts[k].item() > 0, k
    parts["total"].backward()
    got = {n for n, p in pm.rf.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    for key in ("phantom_event_force.", "phantom_mid_head.", "phantom_mid_embed.",
                "phantom_crop_bridges.0.out", "phantom_tac_head", "lora_"):
        assert any(key in n for n in got), key
    assert any("phantom_bridges" in n and "t_out" in n for n in got)
    assert not any("v_out" in n for n, _ in pm.rf.named_parameters())     # GeoBridge2
    assert not any(p.requires_grad for p in pm.rf.tac_ae.parameters())
    from phantom.train.common import trainable_state_dicts
    lora, ph = trainable_state_dicts(pm.rf)
    assert not any("crop_enc" in k for k in list(lora) + list(ph))
    assert any(k.startswith("phantom_event_force.") for k in ph)
    pm.rf.zero_grad(set_to_none=True)
    pm.rf.mc = mc0


def _net_args(pm, b):
    x0, cond, acc = pm.rf.build_x0(b)
    tok, valid = pm.rf._crops(b, 2, "cpu", False)
    return dict(x_B_C_T_H_W=x0, timesteps_B_T=torch.full((2, 13), 500.0),
                crossattn_emb=pm.rf.text.get(2, b["text"]),
                condition_video_input_mask_B_C_T_H_W=pm.rf._cond_mask(cond, pm.layout).float(),
                action=b["prev_chunk"], acc_inputs=acc, layout=pm.layout,
                fps=torch.full((2,), 4.0), geo_G=pm.rf.geo_map(b, 2, "cpu"),
                crop_tokens=tok, crop_valid=valid)


def test_crop_bridge_zero_init_is_identity(tiny):
    pm, b = tiny
    net = pm.rf.net
    for br in net.phantom_crop_bridges:
        torch.nn.init.zeros_(br.out.weight); torch.nn.init.zeros_(br.out.bias)
    pm.rf.eval()
    args = _net_args(pm, b)
    with torch.no_grad():
        v1 = net(**args).velocity_B_C_T_H_W
        saved, net._crop_at = net._crop_at, {}
        v2 = net(**args).velocity_B_C_T_H_W
        net._crop_at = saved
    assert torch.equal(v1, v2)
    torch.manual_seed(0)
    for br in net.phantom_crop_bridges:
        torch.nn.init.normal_(br.out.weight, std=0.02)
    args["crop_valid"] = torch.zeros(2, 2)                       # only the null token left
    with torch.no_grad():
        assert torch.isfinite(net(**args).velocity_B_C_T_H_W).all()


def test_midgate_bias_consistent_under_recompute(tiny):
    """Blocks <= mid_block see no ACC bias, in the forward and in the backward recompute
    (training builds wrap every block in cosmos SAC); later blocks do see it."""
    pm, b = tiny
    net, holder = pm.rf.net, pm.rf.net.phantom_bias_holder
    calls = {}
    orig = holder.bias_for_block

    def spy(i):
        out = orig(i)
        calls.setdefault(i, []).append(None if out is None else out.detach().clone())
        return out

    holder.bias_for_block = spy
    lam0 = net.phantom_bias_lambdas.detach().clone()
    try:
        with torch.no_grad():
            net.phantom_bias_lambdas.fill_(0.5)
        pm.rf.train()
        out = net(**_net_args(pm, b))
        (out.velocity_B_C_T_H_W.float().pow(2).mean() + out.acc.g.float().sum()).backward()
        structural = holder.structural.detach().clone()
    finally:
        del holder.bias_for_block
        with torch.no_grad():
            net.phantom_bias_lambdas.copy_(lam0)
        pm.rf.zero_grad(set_to_none=True)
    mid = pm.mc.mid_block
    assert all(len(v) == 2 for v in calls.values())              # forward + recompute
    for i, (fwd, rec) in calls.items():
        assert torch.equal(fwd, rec), i
    for i in range(mid + 1):
        assert torch.equal(calls[i][0], structural), i
    late = calls[mid + 1][0]
    assert not torch.allclose(late, structural.expand_as(late))


def test_sample_modes(tiny):
    pm, b = tiny
    pm.rf.eval()
    for touch in (True, False):
        p = pm.rf.sample(b, nfe=1, touch=touch, video_refine_steps=2, k_seeds=2, decode_tactile=True)
        assert p.actions_B_H_A.shape == (4, 16, 7)
        assert p.tac_fut_B_6_2_C_h_w.shape == (4, 6, 2, 16, 8, 10)
        assert p.fields_B_6_2_C_H_W.shape == (4, 6, 2, 5, 64, 80)
        assert p.finger_hazard_B_12_2_2.shape == (4, 12, 2, 2)
        assert p.phase_hazard_B_12_3.shape == (4, 12, 3)
        assert p.pad_B_6_2_3.shape == (4, 6, 2, 3) and p.wrist_B_6_6.shape == (4, 6, 6)
        assert p.mid_phase_hazard_B_12_3.shape == (4, 12, 3)
        assert p.mid_finger_hazard_B_12_2_2.shape == (4, 12, 2, 2)
    p = pm.rf.sample(b, nfe=1, touch=True)
    assert torch.allclose(p.tac_obs_B_2_2_C_h_w, b["tac_obs"].float())


def test_real_pipeline_step_and_checkpoint(tiny, tmp_path):
    """train_teacher's caewm2 data path on real episodes (sampler -> WindowDataset ->
    collate -> training_step), then a checkpoint save / load round trip."""
    from phantom.caewm2.builder import build_caewm2_model
    from phantom.caewm2.config import Caewm2Config
    from phantom.caewm2.windows import Caewm2WindowSampler
    from phantom.config.hardware import load_hardware
    from phantom.config.training import TeacherTrainConfig
    from phantom.data.schema import NormStats
    from phantom.train import common as C
    if not (CP2.EVENT_CACHE / EP.name / "events.npz").exists():
        pytest.skip("event cache not built")
    pm, _ = tiny
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    s = Caewm2WindowSampler(hw, pm.bb, NormStats.identity(), seed=0, lat_cache=str(CP.LAT_CACHE),
                            tac_cache=str(CP.TAC_CACHE), dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(),
                            event_cache=str(CP2.EVENT_CACHE))
    eps = [EP, next(p for p in sorted(CP.TASKS.glob("Carton/ep_*")) if p.is_dir())]
    ds = C.WindowDataset(CP.TASKS, s, windows_per_episode=2, episodes=eps, seed=0,
                         grasp_frac=0.3, photo_aug=1.0)
    b = C.collate_windows([ds[i] for i in range(2)])
    for k, shape, dt in (("crops", (2, 2, 3, 224, 224), torch.uint8), ("finger_bin", (2, 2, 2), torch.int64),
                         ("phase_bin", (2, 3), torch.int64), ("pad_fut", (2, 6, 2, 3), torch.float32)):
        assert b[k].shape == shape and b[k].dtype == dt, k
    pm.rf.train()
    parts = pm.rf.training_step(C.to_device(b, "cpu"))
    assert all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()
    pm.rf.zero_grad(set_to_none=True)

    path = tmp_path / "ck.pt"
    C.save_phantom_checkpoint(path, pm.rf, hw=hw, bb=pm.bb, mc=pm.mc, train_cfg=TeacherTrainConfig(),
                              step=1, norm_stats=NormStats.identity())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["configs"]["model"]["arch"] == "caewm2"
    mc2 = Caewm2Config.from_dict(payload["configs"]["model"])
    assert mc2 == pm.mc
    pm2 = build_caewm2_model(hw, load_paths(), mc=mc2, tiny=True, load_base=False, bb=pm.bb)
    C.load_phantom_checkpoint(path, pm2.rf, hw=hw, payload=payload)
    sd1, sd2 = pm.rf.state_dict(), pm2.rf.state_dict()
    for k in sd1:
        if "phantom_" in k or "lora_" in k:
            assert torch.equal(sd1[k], sd2[k]), k


def test_v2_eval_metrics():
    from tools.v2_event_eval import ece, kbin, metrics, scores
    assert [kbin(H) for H in (1, 2, 3)] == [3, 7, 11]
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics(y, s)
    assert m["auroc"] == 1.0 and abs(m["brier"] - 0.025) < 1e-9
    assert abs(ece(y, s) - 0.15) < 1e-9

    class P:
        finger_hazard_B_12_2_2 = torch.full((2, 12, 2, 2), -20.0)
        phase_hazard_B_12_3 = torch.full((2, 12, 3), -20.0)
        mid_phase_hazard_B_12_3 = None
        mid_finger_hazard_B_12_2_2 = None
        event_logits_B_Tc_E = torch.zeros(2, 3, 5)
        slip_prob_B_6_2 = torch.zeros(2, 6, 2)
    P.finger_hazard_B_12_2_2[0, 5, 1, 1] = 20.0            # finger 1 slips in bin 5 for sample 0
    sc = scores("caewm2", P)
    assert sc["hazard"]["slip"].shape == (2, 3)
    assert np.allclose(sc["hazard"]["slip"][0], [0, 1, 1], atol=1e-6)
    assert np.allclose(sc["hazard"]["slip"][1], 0, atol=1e-6)
    assert np.allclose(sc["head_max"]["onset"], 0.2)

    class Q(P):                                            # L3 pilot: no final heads, mid readout only
        finger_hazard_B_12_2_2 = None
        phase_hazard_B_12_3 = None
        mid_finger_hazard_B_12_2_2 = torch.full((2, 12, 2, 2), -20.0)
        mid_phase_hazard_B_12_3 = torch.full((2, 12, 3), -20.0)
    sc = scores("caewm2", Q)
    assert "hazard" not in sc and set(sc["mid_hazard"]) == {"onset", "slip", "close", "lift", "release"}


def test_distill_step(tiny):
    """One-step self-distillation: the teacher pass runs on the snapshot taken at the first
    step, the student weights are restored exactly, and the loss trains the trainables."""
    pm, b = tiny
    rf = rf_ = pm.rf
    mc0 = rf.mc
    rf.mc = dataclasses.replace(mc0, distill_nfe=2, touch_forcing_p=0.5)
    rf.__dict__.pop("_distill_teacher", None)
    try:
        rf.train()
        before = {n: p.detach().clone() for n, p in rf._trainable().items()}
        parts = rf.training_step(b)
        assert "distill_action" in parts and "action_v_mse" not in parts
        assert all(torch.isfinite(v) for v in parts.values())
        assert parts["distill_action"].item() > 0                      # 1 step != 2 steps
        parts["total"].backward()
        after = rf._trainable()
        assert all(torch.equal(before[n], after[n]) for n in before)    # swap restored exactly
        got = {n for n, p in after.items() if p.grad is not None and p.grad.abs().sum() > 0}
        assert any("lora_" in n for n in got) and any("phantom_event_force." in n for n in got)
        rf.zero_grad(set_to_none=True)
        # the teacher stays the first-step snapshot when the student moves
        teacher = rf._teacher_snapshot()
        name = next(n for n in after if "lora_" in n)
        with torch.no_grad():
            after[name].add_(1.0)
        moved = after[name].detach().clone()
        with rf._teacher_weights():
            assert torch.equal(rf._trainable()[name], teacher[name])
        assert torch.equal(rf._trainable()[name], moved)
        with torch.no_grad():
            after[name].sub_(1.0)
        rf.eval()                                                       # validation uses the flow loss
        assert "action_v_mse" in rf.training_step(b)
    finally:
        rf_.mc = mc0
        rf_.__dict__.pop("_distill_teacher", None)
        rf_.zero_grad(set_to_none=True)


@pytest.fixture(scope="module")
def tiny_sc():
    from phantom.caewm2.builder import build_caewm2_model
    from phantom.caewm2.config import Caewm2Config
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.config.model import LossWeights
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    bb = dataclasses.replace(BackboneConfig.tiny(), res_h=256, res_w=320)
    mc = Caewm2Config(bridge_blocks=(0, 1), bridge_rank=32, crop_blocks=(0,), crop_rank=32,
                      mid_block=0, loss=LossWeights(video=0.3), action_noise_per_strip=True,
                      rope_time_mode="time_true", cond_dropout_p=0.0, use_shortcut=True)
    pm = build_caewm2_model(hw, load_paths(), mc=mc, tiny=True, load_base=False, bb=bb)
    return pm, tiny_batch(hw, bb)


def test_shortcut_zero_init_and_schedule(tiny_sc):
    pm, b = tiny_sc
    net = pm.rf.net
    assert net.phantom_step_emb.weight.shape[0] == pm.mc.shortcut_levels + 1
    pm.rf.eval()
    args = _net_args(pm, b)
    with torch.no_grad():                                   # zero init: the step index is a no-op
        v_fm = net(**args).velocity_B_C_T_H_W
        v_1 = net(**args, step_idx=torch.zeros(2, dtype=torch.long)).velocity_B_C_T_H_W
    assert torch.equal(v_fm, v_1)
    ts, m = pm.rf._step_schedule(1)
    assert m == 0 and ts.tolist() == [1.0, 0.0]
    ts, m = pm.rf._step_schedule(2)
    assert m == 1 and ts.tolist() == [1.0, 0.5, 0.0]
    ts, m = pm.rf._step_schedule(5)                         # not a power of two: flow velocity
    assert m is None and abs(float(ts[1]) - float(pm.rf._time_shift(torch.tensor(0.8)))) < 1e-6


def test_shortcut_step_and_sampling(tiny_sc):
    pm, b = tiny_sc
    rf = pm.rf
    # tiny builds keep cosmos' adaLN-Zero output layers at zero, so nothing added to the
    # timestep embedding gets a gradient (V1's frame-type embedding neither); the real base
    # weights are non-zero, emulated here
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in rf.net.named_parameters():
            if "adaln_modulation" in n and n.endswith("weight") and not p.any():
                p.normal_(0, 0.02)
    rf.train()
    every = rf.mc.shortcut_every
    rf.mc = dataclasses.replace(rf.mc, shortcut_every=10**9)    # flow branch
    flow_keys = set(rf.training_step(b))
    rf.zero_grad(set_to_none=True)
    rf.mc = dataclasses.replace(rf.mc, shortcut_every=1)        # force the self-consistency branch
    try:
        parts = rf.training_step(b)
    finally:
        rf.mc = dataclasses.replace(rf.mc, shortcut_every=every)
    # DDP ranks draw the branch independently and the loop all-reduces the logged keys
    assert set(parts) == flow_keys and parts["sc_active"].item() == 1.0
    assert "sc_action" in parts and all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()
    g = rf.net.phantom_step_emb.weight.grad
    assert g is not None and g.abs().sum() > 0
    rows = (g.abs().sum(1) > 0).nonzero().flatten().tolist()
    assert rows and max(rows) < rf.mc.shortcut_levels      # the trained step's row, not the flow row
    rf.zero_grad(set_to_none=True)
    rf.eval()
    for nfe in (1, 2, 5):
        p = rf.sample(b, nfe=nfe, touch=False)
        assert torch.isfinite(p.actions_B_H_A).all()


def test_pad_slog_roundtrip(real_sampler):
    from phantom.caewm2.windows import Caewm2WindowSampler
    s = real_sampler
    s2 = Caewm2WindowSampler(s.hw, s.bb, s.norm, lat_cache=str(CP.LAT_CACHE), tac_cache=str(CP.TAC_CACHE),
                             dz_std=np.ones(16, np.float32), event_cache=str(CP2.EVENT_CACHE),
                             pad_transform="slog")
    rng = np.random.default_rng(0)
    for scale in (0.1, 1.0, 10.0):
        pad = rng.normal(0, scale, (6, 2, 3)) * s.pad_std
        for smp in (s, s2):
            assert np.allclose(smp.pad_physical(smp.pad_target(pad)), pad, rtol=1e-4, atol=1e-3)
    big = np.full((6, 2, 3), 10.0) * s.pad_std            # a whiteboard-sized increment: 10 pad_std
    assert np.abs(s2.pad_target(big)).max() < 0.6 * np.abs(s.pad_target(big)).max()
    lo, hi = s2.valid_range(EP)
    w = s2.sample(EP, lo + 0.5 * (hi - lo))
    assert torch.isfinite(w["pad_fut"]).all()


@pytest.fixture(scope="module")
def tiny_mid():
    from phantom.caewm2.builder import build_caewm2_model
    from phantom.caewm2.config import Caewm2Config
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.config.model import LossWeights
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    bb = dataclasses.replace(BackboneConfig.tiny(), res_h=256, res_w=320)
    mc = Caewm2Config(bridge_blocks=(0, 1), bridge_rank=32, crop_blocks=(0,), crop_rank=32,
                      mid_block=0, loss=LossWeights(video=0.3), action_noise_per_strip=True,
                      rope_time_mode="time_true", cond_dropout_p=0.0, use_crop=False,
                      events_at="mid", pad_transform="slog", force_loss="huber",
                      video_decouple_p=0.0)            # decoupled samples carry no force supervision
    pm = build_caewm2_model(hw, load_paths(), mc=mc, tiny=True, load_base=False, bb=bb)
    return pm, tiny_batch(hw, bb)


def test_events_at_mid(tiny_mid):
    pm, b = tiny_mid
    rf = pm.rf
    assert not hasattr(rf, "phantom_event_force") and rf.net.phantom_mid_head.force
    rf.train()
    parts = rf.training_step(b)
    assert {"force_pad", "force_wrist", "mid_hazard_finger", "mid_hazard_phase"} <= set(parts)
    assert "hazard_finger" not in parts and all(torch.isfinite(v) for v in parts.values())
    parts["total"].backward()
    g = {n for n, p in rf.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("phantom_mid_head.pad" in n for n in g) and any("phantom_mid_head.wrist" in n for n in g)
    rf.zero_grad(set_to_none=True)
    rf.eval()
    p = rf.sample(b, nfe=1, touch=True)
    assert p.pad_B_6_2_3.shape == (2, 6, 2, 3) and p.wrist_B_6_6.shape == (2, 6, 6)
    assert torch.equal(p.finger_hazard_B_12_2_2, p.mid_finger_hazard_B_12_2_2)


def test_v2_select_eval_rules():
    from tools.v2_select_eval import first_bin_pmf, own_events, select
    cum = np.array([0.0, 0.1, 0.8, 0.9, 0.95, 0.97, 0.98] + [0.99] * 5)
    assert abs(first_bin_pmf(cum, 0.6, 1.6) - 0.7) < 1e-9            # 0.6 s -> bin 2
    assert abs(first_bin_pmf(cum, None, 1.6) - (1 - (0.97 + 0.4 * 0.01))) < 1e-9
    def chunk(close_at):
        a = np.zeros((16, 7)); a[:, 2] = -0.005
        if close_at is not None:
            a[close_at:, 6] = 0.8
            a[close_at + 2:, 2] = 0.01                                 # lift after the close
        return a
    tc, tl = own_events(chunk(5))
    assert abs(tc - 0.6) < 1e-9 and tl is not None and tl > tc
    assert own_events(chunk(None)) == (None, None)
    phase = np.stack([cum, cum, cum], 1).tolist()
    cands = [{"pr_actions": chunk(c).tolist(), "phase_cum": phase, "endpoint_err_mm": e}
             for c, e in ((None, 90.0), (5, 10.0), (12, 40.0))]
    assert select(cands, "event") == 1                                 # close in the likely bin
    assert select(cands, "oracle") == 1
