"""CaewmDiT2: the V1 backbone wrapper (CaewmDiT) with the V2 layers.

layer 3 (use_midgate)  after block mc.mid_block a light EventForceHead reads hazards from
                       the CONTACT frames; their cumulative-probability summary is appended
                       to the ACC input, so the ACC gate that biases attention on the haptic
                       tokens is computed there and applied to the later blocks only, and a
                       zero-initialised embedding of the same summary is added to the
                       CONTACT and ACTION tokens. Earlier blocks run without the ACC bias.
layer 2 (use_crop)     CropBridge after the blocks in mc.crop_blocks; GeoBridge2 (touch ->
                       image only) after mc.bridge_blocks. Without crops the V1 GeoBridge.
The forward re-implements CaewmDiT.forward with these changes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.amp as amp
import torch.nn as nn

from phantom.caewm.bridge import GeoBridge
from phantom.caewm.dit import ACC_SUMMARY_DIM, CaewmDiT
from phantom.caewm2.bridge import CropBridge, GeoBridge2
from phantom.caewm2.heads import SUMMARY_DIM, EventForceHead, summary
from phantom.model.attention_bias import BiasHolder, acc_key_bias
from phantom.model.phantom_dit import PhantomDiT, PhantomNetOutput
from phantom.model.sequence import FrameGroup

CROP_KV_DIM = 1024


@dataclass
class Caewm2NetOutput(PhantomNetOutput):
    mid: dict | None = None


class MidGateBiasHolder(BiasHolder):
    """Blocks before first_acc_block get the structural bias only, also when their attention
    is recomputed in backward (cosmos SAC) after the ACC key has been set: the recompute
    must replay the forward's ops exactly, and SDPA picks its kernel by whether the mask
    requires grad."""
    first_acc_block: int = 0

    def bias_for_block(self, block_idx: int):
        if block_idx < self.first_acc_block:
            return self.structural
        return super().bias_for_block(block_idx)


class CaewmDiT2(CaewmDiT):
    def __init__(self, *, layout, mc, bb, hw, **cosmos_kwargs):
        acc_dim = ACC_SUMMARY_DIM + (SUMMARY_DIM if mc.use_midgate else 0)
        PhantomDiT.__init__(self, layout=layout, mc=mc, bb=bb, hw=hw,
                            cpk_summary_dim=acc_dim, **cosmos_kwargs)
        D = self.model_channels
        hp = layout.lat_h // self.patch_spatial
        wp = layout.lat_w // self.patch_spatial
        geo = GeoBridge2 if mc.use_crop else GeoBridge
        self.phantom_bridges = nn.ModuleList(geo(D, mc.bridge_rank) for _ in mc.bridge_blocks)
        self._bridge_at = {b: i for i, b in enumerate(mc.bridge_blocks)}
        if mc.use_crop:
            self.phantom_crop_bridges = nn.ModuleList(
                CropBridge(D, CROP_KV_DIM, mc.crop_rank) for _ in mc.crop_blocks)
            self._crop_at = {b: i for i, b in enumerate(mc.crop_blocks)}
        if mc.use_shortcut:
            # index m: a step of 2^-m; index shortcut_levels: the flow-matching velocity
            self.phantom_step_emb = nn.Embedding(mc.shortcut_levels + 1, D)
            nn.init.zeros_(self.phantom_step_emb.weight)
        if mc.use_midgate:
            self.phantom_bias_holder.__class__ = MidGateBiasHolder
            self.phantom_bias_holder.first_acc_block = mc.mid_block + 1
            self.phantom_mid_head = EventForceHead(
                D, hp, wp, width=128, force=mc.use_events and mc.events_at == "mid")
            self.phantom_mid_embed = nn.Linear(SUMMARY_DIM, D)
            nn.init.zeros_(self.phantom_mid_embed.weight)
            nn.init.zeros_(self.phantom_mid_embed.bias)

    def _set_acc(self, acc_inputs, z_vis, layout, holder):
        acc_out = self.phantom_acc(acc_inputs, z_vis)
        beta = self.phantom_acc.beta_of_events(acc_out.p_evt)
        holder.acc_key = acc_key_bias(layout, acc_out.g, beta)
        holder.lambdas = list(self.phantom_bias_lambdas)
        return acc_out

    def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                condition_video_input_mask_B_C_T_H_W, action=None, acc_inputs=None,
                layout=None, fps=None, padding_mask=None, return_contact_hidden=True,
                crossattn_projected=False, geo_G=None, crop_tokens=None, crop_valid=None,
                step_idx=None, **kwargs) -> Caewm2NetOutput:
        layout = layout or self.layout
        mc = self.mc
        B = x_B_C_T_H_W.shape[0]
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
            if mc.use_shortcut:
                if step_idx is None:
                    step_idx = torch.full((B,), mc.shortcut_levels, device=device, dtype=torch.long)
                t_embedding_B_T_D = t_embedding_B_T_D + self.phantom_step_emb(step_idx).unsqueeze(1)
            adaln_lora_B_T_3D = (adaln_lora_B_T_3D + a_emb3
                                 + self.phantom_frame_type_emb_3d(group_ids).unsqueeze(0))
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        acc_out, mid_out = None, None
        holder = self.phantom_bias_holder
        holder.clear()
        holder.structural = self._structural(layout, device)
        z_vis = None
        if acc_inputs is not None:
            z_vis = x_B_T_H_W_D[:, layout.frame_slice(FrameGroup.VIDEO_COND)].mean(dim=(1, 2, 3)).float()
            if not mc.use_midgate:
                acc_out = self._set_acc(acc_inputs, z_vis, layout, holder)


        if geo_G is None:
            geo_G = torch.zeros(B, x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3], device=device)
        use_crop = mc.use_crop and crop_tokens is not None
        sl_con = layout.frame_slice(FrameGroup.CONTACT)
        sl_act = layout.frame_slice(FrameGroup.ACTION)
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_embedding_B_T_D, context_input,
                rope_emb_L_1_1_D=rope_emb, adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb)
            j = self._bridge_at.get(i)
            if j is not None:
                x_B_T_H_W_D = self.phantom_bridges[j](x_B_T_H_W_D, geo_G, layout)
            if use_crop:
                j = self._crop_at.get(i)
                if j is not None:
                    x_B_T_H_W_D = self.phantom_crop_bridges[j](x_B_T_H_W_D, crop_tokens, crop_valid, layout)
            if mc.use_midgate and i == mc.mid_block:
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

        contact_hidden = None
        if return_contact_hidden and layout.has(FrameGroup.CONTACT):
            contact_hidden = x_B_T_H_W_D[:, sl_con].flatten(2, 3)
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D,
                                       adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        velocity = self.unpatchify(x_B_T_H_W_O)
        return Caewm2NetOutput(velocity_B_C_T_H_W=velocity, acc=acc_out,
                               contact_hidden_B_Tc_S_D=contact_hidden, mid=mid_out)
