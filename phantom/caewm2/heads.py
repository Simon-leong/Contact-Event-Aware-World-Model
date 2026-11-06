"""Layer 1 heads: discrete-time event hazards and force increments, read from the
hidden states of the future-touch frames.

The 3 s horizon is split into N_BINS = 12 bins of 0.25 s. Each future slot (0.5 s)
covers two bins. A hazard h_k is the probability that the event first happens in
bin k given that it has not happened before; the probability that it happens by
bin k is F_k = 1 - prod_{j<=k} (1 - h_j).

Hierarchy (stop-gradient between levels, after HiTac-WAM): onset and phase hazards
come from the slot features; force increments see the detached cumulative onset
and phase probabilities; slip hazards see the detached force prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phantom.caewm.packing import slot_quadrant_tokens

N_BINS = 12
FINGER_EVENTS = ("onset", "slip")
PHASE_EVENTS = ("close", "lift", "release")
SUMMARY_BINS = (1, 3, 7, 11)                     # cumulative probability by 0.5, 1, 2, 3 s
SUMMARY_DIM = len(SUMMARY_BINS) * (2 * len(FINGER_EVENTS) + len(PHASE_EVENTS))
HAZARD_PRIOR_LOGIT = -4.0
SLIP_PRIOR_LOGIT = -5.5


def cumulative(h_logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Hazard logits -> probability that the event happened by each bin."""
    log_surv = torch.cumsum(F.logsigmoid(-h_logits.float()), dim=dim)
    return 1.0 - log_surv.exp()


class _Causal(nn.Module):
    """Residual causal convolution over the 6 future slots."""

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3)

    def forward(self, h):                                   # (N, S, W)
        c = self.conv(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2)
        return h + F.silu(c)


class EventForceHead(nn.Module):
    """CONTACT hidden (B, 3, hp*wp, D) -> dict of
         finger_logits (B, 12, 2 fingers, 2 [onset, slip])
         phase_logits  (B, 12, 3 [close, lift, release])
         pad           (B, 6, 2, 3) standardised pad force increments [|fz|, |fxy|, area]
         wrist         (B, 6, 6) standardised wrist wrench increments
    With force=False only the hazards are produced (the mid-depth readout)."""

    def __init__(self, dim: int, hp: int, wp: int, width: int = 256, force: bool = True):
        super().__init__()
        self.hp, self.wp, self.force = hp, wp, force
        self.norm = nn.LayerNorm(dim)
        self.finger_in = nn.Linear(2 * dim, width)
        self.phase_in = nn.Linear(2 * dim, width)
        self.finger_t, self.phase_t = _Causal(width), _Causal(width)
        self.onset = nn.Linear(width, 2)
        self.phase = nn.Linear(width, 2 * len(PHASE_EVENTS))
        if force:
            self.pad = nn.Linear(width + 2 + len(PHASE_EVENTS), 3)
            self.wrist = nn.Linear(width + 2 + len(PHASE_EVENTS), 6)
            self.slip = nn.Linear(width + 3, 2)
        else:
            self.slip = nn.Linear(width, 2)
        # prior init (measured per-bin hazards on training windows: onset/phase 0.01-0.02, slip
        # 0.003), so the survival loss starts near its base-rate value rather than at
        # 12 log 2 per censored series
        for m, b in ((self.onset, HAZARD_PRIOR_LOGIT), (self.phase, HAZARD_PRIOR_LOGIT),
                     (self.slip, SLIP_PRIOR_LOGIT)):
            nn.init.constant_(m.bias, b)

    def forward(self, contact_hidden):
        q = self.norm(slot_quadrant_tokens(contact_hidden.float(), self.hp, self.wp))  # (B,6,2,D)
        B, S = q.shape[:2]
        c, d = q[:, :, 0], q[:, :, 1]
        hf = torch.stack([torch.cat([c, d], -1), torch.cat([c, -d], -1)], 2)          # (B,6,2,2D)
        hf = F.silu(self.finger_in(hf))
        hf = self.finger_t(hf.transpose(1, 2).reshape(B * 2, S, -1)).reshape(B, 2, S, -1).transpose(1, 2)
        hg = self.phase_t(F.silu(self.phase_in(torch.cat([c, d], -1))))                 # (B,6,W)

        onset = self.onset(hf).reshape(B, S, 2, 2).permute(0, 1, 3, 2).reshape(B, 2 * S, 2)
        phase = self.phase(hg).reshape(B, S, 2, len(PHASE_EVENTS)).reshape(B, 2 * S, -1)
        out = {"phase_logits": phase}
        if not self.force:
            slip = self.slip(hf).reshape(B, S, 2, 2).permute(0, 1, 3, 2).reshape(B, 2 * S, 2)
            out["finger_logits"] = torch.stack([onset, slip], -1)
            return out
        cum_on = cumulative(onset)[:, 1::2].detach()                                    # (B,6,2) by slot end
        cum_ph = cumulative(phase)[:, 1::2].detach()                                    # (B,6,3)
        own_other = torch.stack([cum_on, cum_on.flip(-1)], -1)                          # (B,6,2,2)
        pad = self.pad(torch.cat([hf, own_other,
                                  cum_ph.unsqueeze(2).expand(-1, -1, 2, -1)], -1))      # (B,6,2,3)
        wrist = self.wrist(torch.cat([hg, cum_on, cum_ph], -1))                         # (B,6,6)
        slip = self.slip(torch.cat([hf, pad.detach()], -1))
        slip = slip.reshape(B, S, 2, 2).permute(0, 1, 3, 2).reshape(B, 2 * S, 2)
        out.update(finger_logits=torch.stack([onset, slip], -1), pad=pad, wrist=wrist)
        return out


def summary(out: dict) -> torch.Tensor:
    """(B, SUMMARY_DIM): cumulative probabilities by 0.5/1/2/3 s for onset and slip of
    both fingers and the three phase events."""
    fin = cumulative(out["finger_logits"])[:, list(SUMMARY_BINS)]                       # (B,4,2,2)
    ph = cumulative(out["phase_logits"])[:, list(SUMMARY_BINS)]                         # (B,4,3)
    return torch.cat([fin.flatten(1), ph.flatten(1)], 1)


def hazard_nll(logits: torch.Tensor, first_bin: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Discrete-time survival negative log-likelihood.
    logits (B, K, ...) with K bins on dim 1; first_bin (B, ...) long in [0, K], K means
    the event did not happen within the horizon (censored); weight (B, ...) >= 0."""
    K = logits.shape[1]
    lg = logits.float().movedim(1, -1)                                                  # (B, ..., K)
    k = first_bin.long().unsqueeze(-1)
    bins = torch.arange(K, device=lg.device)
    nll = -(F.logsigmoid(-lg) * (bins < k)).sum(-1) - (F.logsigmoid(lg) * (bins == k)).sum(-1)
    w = weight.float()
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def masked_huber(pred, target, weight, delta: float = 1.0):
    """Huber loss (delta in target units), weight (B,) per sample."""
    h = F.huber_loss(pred.float(), target.float(), reduction="none", delta=delta).flatten(1).mean(1)
    w = weight.float()
    return (h * w).sum() / w.sum().clamp_min(1e-6)


def masked_mse(pred, target, weight):
    """Mean squared error, weight (B,) per sample."""
    se = ((pred.float() - target.float()) ** 2).flatten(1).mean(1)
    w = weight.float()
    return (se * w).sum() / w.sum().clamp_min(1e-6)
