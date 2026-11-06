"""build_caewm2_model: the V1 build path with CaewmDiT2 / CaewmRectifiedFlow2 and the frozen
crop encoder attached (tiny builds get a random-token stand-in of the same shape)."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from phantom.backbone import loader as bl
from phantom.backbone.text_embedding import TextEmbeddingProvider
from phantom.backbone.vae import make_vae
from phantom.config.backbone import BackboneConfig
from phantom.train.builder import PhantomModel

log = logging.getLogger(__name__)


class FakeCropEncoder(nn.Module):
    dim = 1024

    def forward(self, crops):
        B, V = crops.shape[:2]
        return torch.randn(B, V, 196, self.dim, device=crops.device, dtype=torch.bfloat16)


def build_caewm2_model(hw, paths, *, mc, tiny=False, load_base=True, lora=True,
                       device="cpu", dtype=torch.float32, inference=False, bb=None,
                       sac_mode=None) -> PhantomModel:
    from phantom.caewm.build_latent_cache import load_ae
    from phantom.caewm.layout import CaewmLayout
    from phantom.caewm2.rf import CaewmRectifiedFlow2
    from phantom.model.tactile.tokenizer import CausalTactileAE

    bb = bb or (BackboneConfig.tiny() if tiny else BackboneConfig())
    layout = CaewmLayout.build(bb, mc, hw)
    log.info("layout:\n%s", layout.describe())
    bl.setup_cosmos(paths)
    from phantom.caewm2.dit import CaewmDiT2
    net = bl.build_phantom_net(bb, mc, hw, layout, paths, device=device, dtype=dtype,
                               inference=inference, net_cls=CaewmDiT2, sac_mode=sac_mode)
    if load_base and not tiny:
        bl.load_base_weights(net, paths.cosmos_checkpoint, verify=bb)
    if lora:
        bl.inject_lora(net, mc)
    bl.set_trainable(net)
    vae = make_vae(bb, paths, fake=tiny, device=device, dtype=dtype)
    text = TextEmbeddingProvider(bb, paths if not tiny else None, fake=tiny,
                                 device=device, dtype=dtype)
    from phantom.model.hht.hht import HHT
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
    crop_enc = None
    if mc.use_crop:
        if tiny:
            crop_enc = FakeCropEncoder()
        else:
            from phantom.caewm2.bridge import CropEncoder
            crop_enc = CropEncoder(mc.dino, device)
    rf = CaewmRectifiedFlow2(net, hht, hw, bb, mc, layout, vae, text, ae, crop_enc).to(
        device=device, dtype=dtype)
    rf.phantom_event_head.float()
    rf.phantom_sigma_head.float()
    rf.phantom_tac_head.float()
    if mc.use_events and mc.events_at == "final":
        rf.phantom_event_force.float()
    if mc.use_midgate:
        net.phantom_mid_head.float()
    n_train = sum(p.numel() for p in rf.parameters() if p.requires_grad)
    log.info("CaewmRectifiedFlow2 built: %.1f M trainable params (events %s, crop %s, midgate %s)",
             n_train / 1e6, mc.use_events, mc.use_crop, mc.use_midgate)
    return PhantomModel(rf=rf, layout=layout, bb=bb, mc=mc, hw=hw)
