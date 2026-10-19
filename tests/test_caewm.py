"""CAEWM unit tests: packing round trip, layout, and a tiny-backbone (full 256x320
resolution, 2 blocks) train step / sample / zero-init bridge equivalence."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phantom.caewm.packing import pack_slots, to_common_diff, to_fingers, unpack_slots
import phantom.caewm.paths as CP
from phantom.config.paths import load_paths

EP = CP.TASKS / "egg" / "ep_egg_1785680901_002"


def test_pack_roundtrip():
    z = torch.randn(3, 6, 2, 16, 8, 10)
    x = pack_slots(z)
    assert x.shape == (3, 16, 3, 32, 40)
    assert torch.allclose(unpack_slots(x), z)
    assert torch.allclose(to_fingers(to_common_diff(z)), z, atol=1e-6)
    noise = pack_slots(torch.randn(256, 6, 2, 16, 8, 10))
    assert abs(unpack_slots(noise).std().item() - 1.0) < 0.01


@pytest.fixture(scope="module")
def tiny():
    from phantom.caewm.builder import build_caewm_model
    from phantom.caewm.config import CaewmConfig, default_caewm_loss
    from phantom.config.backbone import BackboneConfig
    from phantom.config.hardware import load_hardware
    from phantom.data.schema import NormStats
    from phantom.data.windows import WindowSampler
    from phantom.train import common as C
    hw = load_hardware(str(Path(__file__).resolve().parents[1] / "configs/hardware.nuc.yaml"))
    bb = dataclasses.replace(BackboneConfig.tiny(), res_h=256, res_w=320)
    mc = CaewmConfig(bridge_blocks=(0, 1), bridge_rank=32, loss=default_caewm_loss(),
                     action_noise_per_strip=True, rope_time_mode="time_true", cond_dropout_p=0.0)
    pm = build_caewm_model(hw, load_paths(), mc=mc, tiny=True, load_base=False, bb=bb)
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
    return pm, b


def test_layout(tiny):
    pm, _ = tiny
    lay = pm.layout
    assert lay.t_total == 13
    from phantom.model.sequence import FrameGroup
    assert lay.frame_slice(FrameGroup.OBS_MECH) == slice(4, 5)
    assert lay.tactile_frames_T().sum() == 4


def test_train_step_and_grads(tiny):
    pm, b = tiny
    pm.rf.train()
    pm.rf.mc = dataclasses.replace(pm.rf.mc, touch_forcing_p=1.0, video_decouple_p=1.0)
    parts = pm.rf.training_step(b)
    assert parts["action_v_mse"].item() == 0 and parts["tac_fut_v_mse"].item() == 0
    assert parts["tac_obs_v_mse"].item() == 0 and parts["video_v_mse"].item() > 0
    pm.rf.mc = dataclasses.replace(pm.rf.mc, touch_forcing_p=0.0, video_decouple_p=0.0)
    parts = pm.rf.training_step(b)
    assert torch.isfinite(parts["total"])
    assert all(parts[k].item() > 0 for k in ("action_v_mse", "tac_fut_v_mse", "tac_obs_v_mse"))
    parts["total"].backward()
    got = {n for n, p in pm.rf.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("phantom_tac_head" in n for n in got)
    assert any("phantom_bridges" in n and "t_out" in n for n in got)
    assert any("lora_" in n for n in got)
    assert not any(p.requires_grad for p in pm.rf.tac_ae.parameters())
    pm.rf.zero_grad(set_to_none=True)


def test_bridge_zero_init_is_identity(tiny):
    pm, b = tiny
    net = pm.rf.net
    for br in net.phantom_bridges:
        torch.nn.init.zeros_(br.t_out.weight); torch.nn.init.zeros_(br.t_out.bias)
        torch.nn.init.zeros_(br.v_out.weight); torch.nn.init.zeros_(br.v_out.bias)
    pm.rf.eval()
    x0, cond, acc = pm.rf.build_x0(b)
    args = dict(x_B_C_T_H_W=x0, timesteps_B_T=torch.full((2, 13), 500.0), crossattn_emb=pm.rf.text.get(2, b["text"]),
                condition_video_input_mask_B_C_T_H_W=pm.rf._cond_mask(cond, pm.layout).float(),
                action=b["prev_chunk"], acc_inputs=acc, layout=pm.layout,
                fps=torch.full((2,), 4.0), geo_G=pm.rf.geo_map(b, 2, "cpu"))
    with torch.no_grad():
        v1 = net(**args).velocity_B_C_T_H_W
        saved, net._bridge_at = net._bridge_at, {}
        v2 = net(**args).velocity_B_C_T_H_W
        net._bridge_at = saved
    assert torch.equal(v1, v2)


def test_sample_modes(tiny):
    pm, b = tiny
    pm.rf.eval()
    for touch in (True, False):
        p = pm.rf.sample(b, nfe=1, touch=touch, video_refine_steps=2, k_seeds=2, decode_tactile=True)
        assert p.actions_B_H_A.shape == (4, 16, 7)
        assert p.tac_fut_B_6_2_C_h_w.shape == (4, 6, 2, 16, 8, 10)
        assert p.fields_B_6_2_C_H_W.shape == (4, 6, 2, 5, 64, 80)
        assert p.slip_prob_B_6_2.shape == (4, 6, 2)
    p = pm.rf.sample(b, nfe=1, touch=True)
    assert torch.allclose(p.tac_obs_B_2_2_C_h_w, b["tac_obs"].float())
