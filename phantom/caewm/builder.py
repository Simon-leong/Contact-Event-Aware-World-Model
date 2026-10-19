"""build_caewm_model: CaewmLayout -> CaewmDiT (Cosmos base + LoRA) -> proprio-only
HHT -> CaewmRectifiedFlow with the frozen tactile tokenizer attached."""
from __future__ import annotations

import logging

import torch

from phantom.backbone import loader as bl
from phantom.backbone.text_embedding import TextEmbeddingProvider
from phantom.backbone.vae import make_vae
from phantom.config.backbone import BackboneConfig
from phantom.train.builder import PhantomModel

log = logging.getLogger(__name__)


def build_caewm_model(hw, paths, *, mc, tiny=False, load_base=True, lora=True,
                      device="cpu", dtype=torch.float32, inference=False, bb=None) -> PhantomModel:
    from phantom.caewm.build_latent_cache import load_ae
    from phantom.caewm.layout import CaewmLayout
    from phantom.caewm.rf import CaewmRectifiedFlow
    from phantom.model.tactile.tokenizer import CausalTactileAE

    bb = bb or (BackboneConfig.tiny() if tiny else BackboneConfig())
    layout = CaewmLayout.build(bb, mc, hw)
    log.info("layout:\n%s", layout.describe())
    bl.setup_cosmos(paths)
    from phantom.caewm.dit import CaewmDiT
    net = bl.build_phantom_net(bb, mc, hw, layout, paths, device=device, dtype=dtype,
                               inference=inference, net_cls=CaewmDiT)
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
    ae = (load_ae(mc.tactile_ae, device) if mc.tactile_ae
          else CausalTactileAE().to(device).eval().requires_grad_(False))
    rf = CaewmRectifiedFlow(net, hht, hw, bb, mc, layout, vae, text, ae).to(device=device, dtype=dtype)
    rf.phantom_event_head.float()
    rf.phantom_sigma_head.float()
    rf.phantom_tac_head.float()
    n_train = sum(p.numel() for p in rf.parameters() if p.requires_grad)
    log.info("CaewmRectifiedFlow built: %.1f M trainable params", n_train / 1e6)
    return PhantomModel(rf=rf, layout=layout, bb=bb, mc=mc, hw=hw)
