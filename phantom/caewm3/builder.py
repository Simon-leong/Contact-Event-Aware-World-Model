"""build_caewm3_model: the V2 build path with CaewmDiT3 / CaewmRectifiedFlow3 and, with
group_lora, the factor-wise adapters in place of the shared peft LoRA."""
from __future__ import annotations

import logging

import torch

from phantom.backbone import loader as bl
from phantom.backbone.text_embedding import TextEmbeddingProvider
from phantom.backbone.vae import make_vae
from phantom.config.backbone import BackboneConfig
from phantom.train.builder import PhantomModel

log = logging.getLogger(__name__)


def build_caewm3_model(hw, paths, *, mc, tiny=False, load_base=True, lora=True,
                       device="cpu", dtype=torch.float32, inference=False, bb=None,
                       sac_mode=None) -> PhantomModel:
    from phantom.caewm.build_latent_cache import load_ae
    from phantom.caewm.layout import CaewmLayout
    from phantom.caewm3.group_lora import inject_group_lora
    from phantom.caewm3.geometry import TorchCamera
    from phantom.caewm3.rf import CaewmRectifiedFlow3
    from phantom.caewm3.schedule import frame_edges_s
    from phantom.model.sequence import FrameGroup
    from phantom.model.hht.hht import HHT
    from phantom.model.tactile.tokenizer import CausalTactileAE

    bb = bb or (BackboneConfig.tiny() if tiny else BackboneConfig())
    layout = CaewmLayout.build(bb, mc, hw)
    log.info("layout:\n%s", layout.describe())
    bl.setup_cosmos(paths)
    from phantom.caewm3.dit import CaewmDiT3
    net = bl.build_phantom_net(bb, mc, hw, layout, paths, device=device, dtype=dtype,
                               inference=inference, net_cls=CaewmDiT3, sac_mode=sac_mode)
    if load_base and not tiny:
        bl.load_base_weights(net, paths.cosmos_checkpoint, verify=bb)
    if lora and mc.group_lora:
        inject_group_lora(net, layout, mc.lora.target_modules, mc.lora_r_shared, mc.lora_r_private)
        n = sum(p.numel() for name, p in net.named_parameters() if "lora_" in name)
        log.info("factor-wise LoRA: shared r=%d, private r=%d (ACTION / rest), %.1f M params",
                 mc.lora_r_shared, mc.lora_r_private, n / 1e6)
    elif lora:
        bl.inject_lora(net, mc)
    bl.set_trainable(net)
    vae = make_vae(bb, paths, fake=tiny, device=device, dtype=dtype)
    text = TextEmbeddingProvider(bb, paths if not tiny else None, fake=tiny,
                                 device=device, dtype=dtype)
    hht = HHT(hw, bb, mc, vae, student=True).to(device=device, dtype=dtype)
    if mc.tactile_ae or not tiny:
        import phantom.caewm.paths as CP
        ae_path = CP.stored_or_default(mc.tactile_ae, CP.TAC_AE)
        if not ae_path.exists():
            raise FileNotFoundError(f"tactile tokenizer not found at {ae_path} (checkpoint records "
                                    f"{mc.tactile_ae!r}); place tactile_ae.pt there or set runs_root")
        ae = load_ae(ae_path, device)
    else:
        ae = CausalTactileAE().to(device).eval().requires_grad_(False)
    rf = CaewmRectifiedFlow3(net, hht, hw, bb, mc, layout, vae, text, ae).to(device=device, dtype=dtype)
    rf.phantom_event_head.float()
    rf.phantom_sigma_head.float()
    rf.phantom_tac_head.float()
    net.phantom_mid_head.float()
    if mc.use_plan:
        net.phantom_plan_embed.float()
    # full-precision geometry and timing, after the cast of the whole model to dtype
    rf.cam = TorchCamera().to(device)
    rf.act_mean.data, rf.act_std.data = rf.act_mean.float(), rf.act_std.float()
    if mc.use_plan:
        sl = layout.frame_slice(FrameGroup.ACTION)
        net._plan_edges = frame_edges_s(sl.stop - sl.start, layout.actions_per_frame,
                                        hw.control.action_rate_hz).to(device)
    n_train = sum(p.numel() for p in rf.parameters() if p.requires_grad)
    log.info("CaewmRectifiedFlow3 built: %.1f M trainable params (plan %s, plan bridge %s, group lora %s)",
             n_train / 1e6, mc.use_plan, mc.use_plan_bridge, mc.group_lora)
    return PhantomModel(rf=rf, layout=layout, bb=bb, mc=mc, hw=hw)
