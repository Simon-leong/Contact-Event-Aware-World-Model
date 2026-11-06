"""CaewmDiT3: the V2 network (CaewmDiT2) organised by the contact plan.

layer 1 (use_plan)        after the mid-depth readout (block mc.mid_block) the contact plan is
                          resolved (given, unknown, or decided from the readout's own hazards)
                          and its per-frame time-to-event features are added to the ACTION
                          tokens through a zero-initialised embedding. The ACC and the V2
                          summary embedding keep reading the soft hazards.
layer 2 (use_plan_bridge) GeoBridge3 after mc.bridge_blocks: every future CONTACT frame and
                          every generated video frame is anchored at the projected fingertip
                          of its own time along the model's estimated trajectory.
layer 3 (group_lora)      factor-wise adapters (phantom/caewm3/group_lora.py), routed by the
                          layout; the router is refreshed from the layout of every call.
The forward re-implements CaewmDiT2.forward with these changes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.amp as amp
import torch.nn as nn

from phantom.caewm.bridge import GeoBridge
from phantom.caewm.dit import ACC_SUMMARY_DIM
from phantom.caewm2.dit import Caewm2NetOutput, CaewmDiT2, MidGateBiasHolder
from phantom.caewm2.heads import SUMMARY_DIM, EventForceHead, summary
from phantom.caewm3.bridge import GeoBridge3
from phantom.caewm3.schedule import (DECIDE, N_EVENTS, PlanEmbed, event_logits, frame_edges_s,
                                     plan_features, resolve)
from phantom.model.phantom_dit import PhantomDiT
from phantom.model.sequence import FrameGroup


@dataclass
class Caewm3NetOutput(Caewm2NetOutput):
    plan: torch.Tensor | None = None


class CaewmDiT3(CaewmDiT2):
    def __init__(self, *, layout, mc, bb, hw, **cosmos_kwargs):
        assert mc.use_midgate and not mc.use_crop and not mc.use_shortcut, \
            "V3 builds on the V2 final configuration (mid-depth readout, no crops, no shortcut)"
        acc_dim = ACC_SUMMARY_DIM + SUMMARY_DIM
        PhantomDiT.__init__(self, layout=layout, mc=mc, bb=bb, hw=hw,
                            cpk_summary_dim=acc_dim, **cosmos_kwargs)
        D = self.model_channels
        hp = layout.lat_h // self.patch_spatial
        wp = layout.lat_w // self.patch_spatial
        geo = GeoBridge3 if mc.use_plan_bridge else GeoBridge
        self.phantom_bridges = nn.ModuleList(geo(D, mc.bridge_rank) for _ in mc.bridge_blocks)
        self._bridge_at = {b: i for i, b in enumerate(mc.bridge_blocks)}
        self.phantom_bias_holder.__class__ = MidGateBiasHolder
        self.phantom_bias_holder.first_acc_block = mc.mid_block + 1
        self.phantom_mid_head = EventForceHead(D, hp, wp, width=128, force=mc.events_at == "mid")
        self.phantom_mid_embed = nn.Linear(SUMMARY_DIM, D)
        nn.init.zeros_(self.phantom_mid_embed.weight)
        nn.init.zeros_(self.phantom_mid_embed.bias)
        if mc.use_plan:
            self.phantom_plan_embed = PlanEmbed(D)
            n_act = layout.frame_slice(FrameGroup.ACTION).stop - layout.frame_slice(FrameGroup.ACTION).start
            self.register_buffer("_plan_edges", frame_edges_s(n_act, layout.actions_per_frame,
                                                              hw.control.action_rate_hz), persistent=False)

    def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                condition_video_input_mask_B_C_T_H_W, action=None, acc_inputs=None,
                layout=None, fps=None, padding_mask=None, return_contact_hidden=True,
                crossattn_projected=False, geo_G=None, geo_future=None, plan_src=None,
                plan_rule="median", plan_gen=None, **kwargs) -> Caewm3NetOutput:
        layout = layout or self.layout
        mc = self.mc
        B = x_B_C_T_H_W.shape[0]
        router = getattr(self, "_v3_router", None)
        if router is not None:
            router.set(torch.from_numpy(layout.group_of_frame() == list(FrameGroup).index(FrameGroup.ACTION)),
                       layout.tokens_per_frame)
        net_dtype = self.x_embedder.proj[1].weight.dtype \
            if hasattr(self.x_embedder, "proj") else next(self.parameters()).dtype
        x_B_C_T_H_W = x_B_C_T_H_W.to(net_dtype)
        crossattn_emb = crossattn_emb.to(net_dtype)
        device, in_dtype = x_B_C_T_H_W.device, x_B_C_T_H_W.dtype
        assert x_B_C_T_H_W.shape[2] == layout.t_total

        x_B_C_T_H_W = torch.cat(
            [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        if timesteps_B_T.shape[1] == 1:
            timesteps_B_T = timesteps_B_T.expand(-1, layout.t_total)
        if padding_mask is None:
            padding_mask = torch.zeros(B, 1, x_B_C_T_H_W.shape[-2], x_B_C_T_H_W.shape[-1],
                                       device=device, dtype=in_dtype)
        x_B_T_H_W_D, rope_standard, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        rope_emb = (rope_standard if mc.rope_time_mode == "append"
                    else self._phantom_rope(layout, fps, device))
        if self.use_crossattn_projection and not crossattn_projected:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        context_input = crossattn_emb

        a_emb, a_emb3 = self._action_adaln_embeddings(action, layout, B, device, torch.float32)
        group_ids = torch.from_numpy(layout.group_of_frame()).to(device)
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = (t_embedding_B_T_D + a_emb
                                 + self.phantom_frame_type_emb(group_ids).unsqueeze(0))
            adaln_lora_B_T_3D = (adaln_lora_B_T_3D + a_emb3
                                 + self.phantom_frame_type_emb_3d(group_ids).unsqueeze(0))
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        acc_out, mid_out, plan = None, None, None
        holder = self.phantom_bias_holder
        holder.clear()
        holder.structural = self._structural(layout, device)
        z_vis = None
        if acc_inputs is not None:
            z_vis = x_B_T_H_W_D[:, layout.frame_slice(FrameGroup.VIDEO_COND)].mean(dim=(1, 2, 3)).float()

        if geo_G is None:
            geo_G = torch.zeros(B, x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3], device=device)
        sl_con = layout.frame_slice(FrameGroup.CONTACT)
        sl_act = layout.frame_slice(FrameGroup.ACTION)
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_embedding_B_T_D, context_input,
                rope_emb_L_1_1_D=rope_emb, adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb)
            j = self._bridge_at.get(i)
            if j is not None:
                if mc.use_plan_bridge:
                    x_B_T_H_W_D = self.phantom_bridges[j](x_B_T_H_W_D, geo_G, layout, geo_future)
                else:
                    x_B_T_H_W_D = self.phantom_bridges[j](x_B_T_H_W_D, geo_G, layout)
            if i == mc.mid_block:
                mid_out = self.phantom_mid_head(x_B_T_H_W_D[:, sl_con].flatten(2, 3))
                s = summary(mid_out)
                if acc_inputs is not None:
                    acc_in = replace(acc_inputs, prev_cpk_summary_B_S=torch.cat(
                        [acc_inputs.prev_cpk_summary_B_S.float(), s], 1).to(acc_inputs.prev_cpk_summary_B_S.dtype))
                    acc_out = self._set_acc(acc_in, z_vis, layout, holder)
                e = self.phantom_mid_embed(s.to(self.phantom_mid_embed.weight.dtype)).to(x_B_T_H_W_D.dtype)
                tgt = torch.zeros(x_B_T_H_W_D.shape[1], device=device, dtype=x_B_T_H_W_D.dtype)
                tgt[sl_con] = 1
                tgt[sl_act] = 1
                x_B_T_H_W_D = x_B_T_H_W_D + tgt.view(1, -1, 1, 1, 1) * e[:, None, None, None, :]
                if mc.use_plan:
                    src = (plan_src.to(device).long() if plan_src is not None
                           else torch.full((B, N_EVENTS), DECIDE, device=device, dtype=torch.long))
                    plan = resolve(src, event_logits(mid_out), plan_rule, plan_gen)
                    f = plan_features(plan, self._plan_edges, mc.bin_dt)
                    ep = self.phantom_plan_embed(f.to(self.phantom_plan_embed.inp.weight.dtype))
                    delta = torch.zeros(B, x_B_T_H_W_D.shape[1], 1, 1, x_B_T_H_W_D.shape[-1],
                                        device=device, dtype=x_B_T_H_W_D.dtype)
                    delta[:, sl_act] = ep.to(x_B_T_H_W_D.dtype)[:, :, None, None, :]
                    x_B_T_H_W_D = x_B_T_H_W_D + delta

        contact_hidden = None
        if return_contact_hidden and layout.has(FrameGroup.CONTACT):
            contact_hidden = x_B_T_H_W_D[:, sl_con].flatten(2, 3)
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D,
                                       adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        velocity = self.unpatchify(x_B_T_H_W_O)
        return Caewm3NetOutput(velocity_B_C_T_H_W=velocity, acc=acc_out,
                               contact_hidden_B_Tc_S_D=contact_hidden, mid=mid_out, plan=plan)
