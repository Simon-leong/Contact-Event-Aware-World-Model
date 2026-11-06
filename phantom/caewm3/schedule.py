"""Layer 1, from predicting contact to deciding it: the contact plan.

A plan pi assigns each of the seven contact events of a window (onset and slip of both
fingers, close, lift, release) the 0.25 s bin of its first occurrence within 3 s, or NONE
(not within 3 s), or UNKNOWN (no decision). The hazards of the mid-depth readout give the
distribution p(pi | o); the plan is decided from it (median or sample) or taken from the
demonstration (the oracle, also the teacher-forced plan of training) and enters the ACTION
tokens as per-frame time-to-event features, so every action frame knows where it stands in
the plan.

Encoding: for each event and each ACTION frame, the signed time from the frame's first and
last action step to the event (bin centre), squashed at two scales (0.25 s and 1 s), set to
zero when the event is NONE or UNKNOWN, plus the NONE and UNKNOWN flags.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from phantom.caewm2.heads import N_BINS, cumulative

N_EVENTS = 7                      # order: see plan_of_bins
NONE = N_BINS
UNKNOWN = -1
DECIDE = -2                       # placeholder: decide from the model's own hazards
FEATS_PER_EVENT = 6
PLAN_FEAT_DIM = N_EVENTS * FEATS_PER_EVENT


def plan_of_bins(finger_bin: torch.Tensor, phase_bin: torch.Tensor) -> torch.Tensor:
    """finger_bin (B, 2 fingers, 2 [onset, slip]), phase_bin (B, 3) -> plan (B, 7) long, in the
    order onset L, slip L, onset R, slip R, close, lift, release."""
    return torch.cat([finger_bin.long().flatten(1), phase_bin.long()], 1)


def event_logits(mid_out: dict) -> torch.Tensor:
    """Readout dict -> hazard logits (B, 12, 7) in plan order."""
    f = mid_out["finger_logits"]                                         # (B, 12, 2, 2)
    return torch.cat([f.flatten(2), mid_out["phase_logits"]], 2)


CLOSE = 4                         # plan index of the gripper close


@dataclass(frozen=True)
class DecisionSpec:
    """A decision rule beyond the plain strings. `base` decides every event as a string rule does;
    with `close_cost` (G, 13, 13), cost of deciding bin j (12 = NONE) when the first close is k,
    the close is the Bayes decision argmin_j sum_k p(k) C[j, k] under the readout's first-event
    distribution; with `group` K > 1 the plan is decided once per group of K consecutive samples
    from their averaged first-event distribution (the K draws of one window share one plan)."""
    base: str = "median"
    close_cost: torch.Tensor | None = None
    group: int = 1


def _pmf(cum: torch.Tensor) -> torch.Tensor:
    """(B, 12, E) cumulative -> (B, 13, E) first-event pmf, NONE last."""
    return torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1], 1 - cum[:, -1:]], 1)


def _decide_cum(cum: torch.Tensor, rule: str, gen: torch.Generator | None = None) -> torch.Tensor:
    if rule == "mode":
        # the most probable first-event bin (or NONE): commits to one mode of a multimodal
        # first-event distribution, where the median falls between the modes
        return _pmf(cum).argmax(1).long()
    if rule == "median":
        u = torch.full_like(cum[:, 0], 0.5)
    elif rule.startswith("q"):
        u = torch.full_like(cum[:, 0], float(rule[1:]))
    elif rule == "sample":
        u = torch.rand(cum[:, 0].shape, generator=gen).to(cum.device)
    else:
        raise ValueError(rule)
    hit = cum >= u.unsqueeze(1)
    first = hit.float().argmax(1)
    return torch.where(hit.any(1), first, torch.full_like(first, NONE)).long()


def _decide_spec(cum: torch.Tensor, spec: DecisionSpec, gen) -> torch.Tensor:
    K = spec.group
    if K > 1:
        cum = cum.view(cum.shape[0] // K, K, *cum.shape[1:]).mean(1)
    plan = _decide_cum(cum, spec.base, gen)
    if spec.close_cost is not None:
        risk = torch.einsum("gk,gjk->gj", _pmf(cum)[..., CLOSE], spec.close_cost.to(cum))
        plan[:, CLOSE] = risk.argmin(1)
    return plan.repeat_interleave(K, 0) if K > 1 else plan


def decide(logits: torch.Tensor, rule, gen: torch.Generator | None = None) -> torch.Tensor:
    """Hazard logits (B, 12, E) -> first bins (B, E) in [0, 12]: the median of the first-event
    time (first bin whose cumulative probability reaches 1/2), the q-quantile ("q0.65"), the
    mode (most probable first-event bin, NONE included), a sample from it (inverse CDF, NONE
    with the survival probability), or a DecisionSpec."""
    cum = cumulative(logits.float(), dim=1)                               # (B, 12, E)
    if isinstance(rule, DecisionSpec):
        return _decide_spec(cum, rule, gen)
    return _decide_cum(cum, rule, gen)


def resolve(plan_src: torch.Tensor, logits: torch.Tensor, rule: str = "median",
            gen: torch.Generator | None = None) -> torch.Tensor:
    """plan_src (B, 7): given bins in [0, 12], UNKNOWN, or DECIDE -> the plan, with every DECIDE
    entry replaced by the decision from the readout's hazards (no gradient)."""
    if not (plan_src == DECIDE).any():
        return plan_src
    own = decide(logits.detach(), rule, gen)
    return torch.where(plan_src == DECIDE, own, plan_src)


def frame_edges_s(n_frames: int, actions_per_frame: int, rate_hz: float) -> torch.Tensor:
    """(n_frames, 2) times (s after t0) of the first and last action step of each ACTION frame."""
    j = torch.arange(n_frames, dtype=torch.float32)
    first = (j * actions_per_frame + 1) / rate_hz
    return torch.stack([first, first + (actions_per_frame - 1) / rate_hz], 1)


def plan_features(plan: torch.Tensor, edges: torch.Tensor, bin_dt: float) -> torch.Tensor:
    """plan (B, 7) -> (B, n_frames, PLAN_FEAT_DIM) float features."""
    known = (plan >= 0) & (plan < NONE)
    tau = (plan.clamp(0, N_BINS - 1).float() + 0.5) * bin_dt                # (B, 7)
    d = tau[:, None, :, None] - edges.to(plan.device)[None, :, None, :]     # (B, F, 7, 2)
    sq = torch.cat([torch.tanh(d / 0.25), torch.tanh(d / 1.0)], -1)        # (B, F, 7, 4)
    sq = sq * known[:, None, :, None].float()
    flags = torch.stack([(plan == NONE).float(), (plan == UNKNOWN).float()], -1)   # (B, 7, 2)
    flags = flags[:, None].expand(-1, edges.shape[0], -1, -1)
    return torch.cat([sq, flags], -1).flatten(2)


class PlanEmbed(nn.Module):
    """Per-ACTION-frame plan features -> a vector added to that frame's tokens; the output
    layer is zero-initialised, so at step 0 the network ignores the plan."""

    def __init__(self, dim: int, width: int = 256):
        super().__init__()
        self.inp = nn.Linear(PLAN_FEAT_DIM, width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats):                                              # (B, F, PLAN_FEAT_DIM)
        return self.out(torch.nn.functional.silu(self.inp(feats)))
