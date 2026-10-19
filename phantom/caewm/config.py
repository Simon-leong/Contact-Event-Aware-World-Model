"""CAEWM model configuration: PhantomModelConfig plus the CAEWM-only fields."""
from __future__ import annotations

from dataclasses import dataclass, fields

from phantom.config.model import LossWeights, PhantomModelConfig


@dataclass(frozen=True)
class CaewmConfig(PhantomModelConfig):
    arch: str = "caewm"
    tactile_ae: str = ""                 # frozen tokenizer checkpoint (pretrain_tactile_ae)
    tactile_lat_cache: str = ""          # per-frame standardised latents of every episode
    touch_forcing_p: float = 0.5         # p(tactile obs pinned) per training sample
    video_decouple_p: float = 0.3        # p(forward-dynamics sample: video-only noise)
    bridge_blocks: tuple = (4, 10, 16, 22)
    bridge_rank: int = 512
    bridge_sigma: float = 1.5            # TCP Gaussian std in token units
    loss_tac_fut: float = 1.0
    loss_tac_obs: float = 0.5
    loss_slip: float = 0.5
    loss_tac_contact: float = 0.5
    focal_gamma: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "CaewmConfig":
        base = PhantomModelConfig.from_dict(d)
        kw = {f.name: getattr(base, f.name) for f in fields(PhantomModelConfig)}
        for f in fields(cls):
            if f.name not in kw and f.name in d:
                kw[f.name] = tuple(d[f.name]) if f.name == "bridge_blocks" else d[f.name]
        return cls(**kw)


def default_caewm_loss() -> LossWeights:
    return LossWeights(video=1.0)
