"""CAEWM V3 configuration: the V2 final configuration plus one switch per layer of the
contact-plan factorisation p(a, c, v | o) = sum_pi p(pi | o) p(a | pi, o) p(c, v | a, pi, o).

    layer 1  use_plan         decide the contact plan from the mid-depth hazards, condition
                              the ACTION tokens on it
    layer 2  use_plan_bridge  anchor every future frame at the fingertip of its own time on
                              the model's estimated trajectory
    layer 3  group_lora       shared + per-factor adapters (ACTION / rest)
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from phantom.caewm2.config import TUPLE_FIELDS, Caewm2Config
from phantom.config.model import PhantomModelConfig


@dataclass(frozen=True)
class Caewm3Config(Caewm2Config):
    arch: str = "caewm3"
    use_crop: bool = False
    events_at: str = "mid"
    pad_transform: str = "slog"
    force_loss: str = "huber"
    # layer 1: where the plan of a training sample comes from (the rest is UNKNOWN)
    use_plan: bool = False
    plan_p_gt: float = 0.5
    plan_p_self: float = 0.3
    plan_rule: str = "median"
    # layer 2: share of training samples whose future frames are anchored on the trajectory
    # estimated by a no-grad pass at t = 1 (the first sampling step); the rest use V1's
    # current-fingertip anchor, as the first sampling step does
    use_plan_bridge: bool = False
    anchor_p: float = 0.5
    # layer 3
    group_lora: bool = False
    lora_r_shared: int = 8
    lora_r_private: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "Caewm3Config":
        base = PhantomModelConfig.from_dict(d)
        kw = {f.name: getattr(base, f.name) for f in fields(PhantomModelConfig)}
        for f in fields(cls):
            if f.name not in kw and f.name in d:
                kw[f.name] = tuple(d[f.name]) if f.name in TUPLE_FIELDS else d[f.name]
        return cls(**kw)
