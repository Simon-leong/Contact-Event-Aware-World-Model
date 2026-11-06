"""Layer 3, factor-wise parameters: a low-rank adapter shared by every token plus one private
adapter for the ACTION tokens and one for all other tokens, on every LoRA target of the
DiT blocks. Attention still runs over the whole sequence, so the factors condition on each
other through attention and no longer through one shared set of weights.

Which tokens are ACTION tokens is fixed by the layout (frame-major (t h w) order in the
attention projections, (B, T, H, W, D) in the MLP), so the routing is a static function of
the input shape: a block recomputed in backward (activation checkpointing) routes exactly
as in forward. The cross-attention key / value projections read the text context, which
belongs to no factor, and get the shared adapter only.

Parameter names keep the lora_ prefix, so trainability, the optimizer groups, the EMA and
checkpoint saving treat them like the peft adapters they replace.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

SHARED_ONLY = ("cross_attn.k_proj", "cross_attn.v_proj")


class TokenRouter:
    """ACTION-frame mask of the current layout, shared by all adapters of one net."""

    def __init__(self, is_action_frame: torch.Tensor, tokens_per_frame: int):
        self.set(is_action_frame, tokens_per_frame)

    def set(self, is_action_frame: torch.Tensor, tokens_per_frame: int):
        self.frames = is_action_frame.bool().cpu()
        self.tpf = int(tokens_per_frame)
        self._cache = {}

    def mask(self, x: torch.Tensor) -> torch.Tensor | None:
        """Broadcastable (.., 1) bool mask of the ACTION tokens of x, None if x is not a
        token sequence of this layout."""
        T = len(self.frames)
        key = (x.dim(), x.shape[1], x.device)
        if key not in self._cache:
            if x.dim() == 5 and x.shape[1] == T:
                m = self.frames.view(1, T, 1, 1, 1)
            elif x.dim() == 3 and x.shape[1] == T * self.tpf:
                m = self.frames.repeat_interleave(self.tpf).view(1, -1, 1)
            else:
                m = None
            self._cache[key] = None if m is None else m.to(x.device)
        return self._cache[key]


class GroupLoRALinear(nn.Module):
    """base(x) + B_s A_s x + [action token] B_a A_a x + [other token] B_r A_r x."""

    def __init__(self, base: nn.Linear, router: TokenRouter | None, r_shared: int, r_private: int):
        super().__init__()
        self.base_layer = base
        self.router = router
        self.r_s, self.r_p = r_shared, (r_private if router is not None else 0)
        din, dout = base.in_features, base.out_features
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A_shared = nn.Parameter(torch.empty(self.r_s, din, device=dev, dtype=dt))
        self.lora_B_shared = nn.Parameter(torch.zeros(dout, self.r_s, device=dev, dtype=dt))
        if self.r_p:
            self.lora_A_action = nn.Parameter(torch.empty(self.r_p, din, device=dev, dtype=dt))
            self.lora_B_action = nn.Parameter(torch.zeros(dout, self.r_p, device=dev, dtype=dt))
            self.lora_A_rest = nn.Parameter(torch.empty(self.r_p, din, device=dev, dtype=dt))
            self.lora_B_rest = nn.Parameter(torch.zeros(dout, self.r_p, device=dev, dtype=dt))
        self.reset_lora()

    def reset_lora(self):
        """peft's default: A Kaiming-uniform, B zero, so the adapter starts as the identity."""
        for n in ("shared", "action", "rest"):
            A = getattr(self, f"lora_A_{n}", None)
            if A is not None:
                with torch.no_grad():
                    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    getattr(self, f"lora_B_{n}").zero_()

    def forward(self, x):
        y = self.base_layer(x)
        xs = x.to(self.lora_A_shared.dtype)
        m = self.router.mask(x) if self.r_p else None
        if m is None:
            return y + (xs @ self.lora_A_shared.t() @ self.lora_B_shared.t()).to(y.dtype)
        h_s = xs @ self.lora_A_shared.t()
        h_a = xs @ self.lora_A_action.t()
        h_r = xs @ self.lora_A_rest.t()
        mf = m.to(h_s.dtype)
        h = torch.cat([h_s, h_a * mf, h_r * (1 - mf)], -1)
        Bm = torch.cat([self.lora_B_shared, self.lora_B_action, self.lora_B_rest], 1)
        return y + (h @ Bm.t()).to(y.dtype)


def inject_group_lora(net, layout, targets: str, r_shared: int, r_private: int) -> TokenRouter:
    """Replace every target nn.Linear inside net.blocks by a GroupLoRALinear; returns the
    router (kept on the net as net._v3_router so a changed layout can update it)."""
    from phantom.model.sequence import FrameGroup
    is_act = torch.from_numpy(layout.group_of_frame() == list(FrameGroup).index(FrameGroup.ACTION))
    router = TokenRouter(is_act, layout.tokens_per_frame)
    suffixes = tuple(targets.split(","))
    todo = []
    for name, mod in net.blocks.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(suffixes):
            todo.append(name)
    assert todo, f"no LoRA target among {suffixes}"
    for name in todo:
        parent_name, _, child = name.rpartition(".")
        parent = net.blocks.get_submodule(parent_name)
        base = getattr(parent, child)
        shared_only = name.endswith(SHARED_ONLY)
        setattr(parent, child, GroupLoRALinear(base, None if shared_only else router, r_shared, r_private))
    net._v3_router = router
    return router
