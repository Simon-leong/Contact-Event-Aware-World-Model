"""CAEWM V2 configuration: the V1 config plus one switch per upgraded layer.

    layer 1  use_events   event hazards (onset, slip, close, lift, release) and force heads
    layer 2  use_crop     future-aware high-resolution fingertip crops for the tactile frames
    layer 3  use_midgate  mid-depth event readout that drives ACC for the later blocks
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from phantom.caewm.config import CaewmConfig
from phantom.config.model import PhantomModelConfig

TUPLE_FIELDS = ("bridge_blocks", "crop_blocks")


@dataclass(frozen=True)
class Caewm2Config(CaewmConfig):
    arch: str = "caewm2"
    video_decouple_p: float = 0.15
    bridge_rank: int = 256
    use_events: bool = True
    use_crop: bool = True
    use_midgate: bool = True
    event_cache: str = ""
    # layer 1 placement: "final" reads events and forces from the last block's CONTACT
    # hidden (first V2 pilot); "mid" puts the force regression into the mid-depth readout
    # and drops the final head, leaving the last blocks to generation
    events_at: str = "final"
    pad_transform: str = "std"           # "std": increment / pad_std; "slog": see events.slog
    force_loss: str = "mse"              # "mse" or "huber" (delta 1, standardised units)
    hazard_bins: int = 12
    bin_dt: float = 0.25
    loss_hazard: float = 1.0
    loss_mid_hazard: float = 0.5
    loss_force: float = 0.5
    mid_block: int = 13
    crop_px: int = 160
    crop_res: int = 224
    crop_lead_s: float = 0.5
    crop_blocks: tuple = (10, 22)
    crop_rank: int = 256
    dino: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    distill_nfe: int = 0                 # > 0: one-step self-distillation stage, teacher steps
    # layer 3, trained for one-step inference (shortcut models, Frans et al. ICLR 2025):
    # the net also sees the step size; 1/shortcut_every of the samples train a step of
    # 2^-m (m < shortcut_levels) to match two half steps of the same model
    use_shortcut: bool = False
    shortcut_every: int = 8
    shortcut_levels: int = 7

    @classmethod
    def from_dict(cls, d: dict) -> "Caewm2Config":
        base = PhantomModelConfig.from_dict(d)
        kw = {f.name: getattr(base, f.name) for f in fields(PhantomModelConfig)}
        for f in fields(cls):
            if f.name not in kw and f.name in d:
                kw[f.name] = tuple(d[f.name]) if f.name in TUPLE_FIELDS else d[f.name]
        return cls(**kw)
