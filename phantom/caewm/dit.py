"""CaewmDiT: PhantomDiT with
  - tactile frames (OBS_MECH, CONTACT) placed on RoPE rows [Hp, 2Hp), so they no
    longer share spatial positions with image rows they have no relation to;
  - GeoBridge residuals after the blocks in mc.bridge_blocks;
  - the ACC visual summary pooled from the conditioning frame only (the tactile-obs
    frame may be noised in the no-sensor mode).
The forward re-implements PhantomDiT.forward with those three changes.
"""
from __future__ import annotations

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from phantom.caewm.bridge import GeoBridge
from phantom.model.attention_bias import acc_key_bias
from phantom.model.phantom_dit import PhantomDiT, PhantomNetOutput
from phantom.model.sequence import FrameGroup

ACC_SUMMARY_DIM = 2 * 2 * 16 * 2 + 1     # (slot x common/diff x C_TOK) mean & rms + touch flag


class CaewmDiT(PhantomDiT):
    def __init__(self, *, layout, mc, bb, hw, **cosmos_kwargs):
        super().__init__(layout=layout, mc=mc, bb=bb, hw=hw,
                         cpk_summary_dim=ACC_SUMMARY_DIM, **cosmos_kwargs)
        D = self.model_channels
        self.phantom_bridges = nn.ModuleList(GeoBridge(D, mc.bridge_rank) for _ in mc.bridge_blocks)
        self._bridge_at = {b: i for i, b in enumerate(mc.bridge_blocks)}

    def _phantom_rope(self, layout, fps, device):
        key = ("caewm", layout.slots, layout.drop_video, str(device),
               None if fps is None else float(fps.flatten()[0]))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        pe = self.pos_embedder
        Hp = layout.lat_h // self.patch_spatial
        Wp = layout.lat_w // self.patch_spatial
        t_pos = torch.from_numpy(layout.rope_frame_positions(self.mc.rope_time_mode)).float().to(device)
        h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.float().to(device))
        w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.float().to(device))
        t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.float().to(device))
        if pe.enable_fps_modulation and fps is not None:
            t_pos = t_pos / float(fps.flatten()[0]) * pe.base_fps
        T = len(t_pos)
        h_off = torch.from_numpy(layout.tactile_frames_T()).float().to(device) * Hp
        seq_h = torch.arange(Hp, dtype=torch.float32, device=device)[None] + h_off[:, None]   # (T,Hp)
        seq_w = torch.arange(Wp, dtype=torch.float32, device=device)
        half_t = torch.outer(t_pos, t_freqs)
        half_h = seq_h[..., None] * h_freqs                                                 # (T,Hp,dh)
        half_w = torch.outer(seq_w, w_freqs)
        em = torch.cat([
            half_t[:, None, None, :].expand(-1, Hp, Wp, -1),
            half_h[:, :, None, :].expand(-1, -1, Wp, -1),
            half_w[None, None, :, :].expand(T, Hp, -1, -1),
        ] * 2, dim=-1)
        rope = rearrange(em, "t h w d -> (t h w) 1 1 d").float()
        self._rope_cache[key] = rope
        return rope

    def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                condition_video_input_mask_B_C_T_H_W, action=None, acc_inputs=None,
                layout=None, fps=None, padding_mask=None, return_contact_hidden=True,
                crossattn_projected=False, geo_G=None, **kwargs) -> PhantomNetOutput:
        layout = layout or self.layout
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
        rope_emb = (rope_standard if self.mc.rope_time_mode == "append"
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

        acc_out = None
        holder = self.phantom_bias_holder
        holder.clear()
        holder.structural = self._structural(layout, device)
        if acc_inputs is not None:
            z_vis = x_B_T_H_W_D[:, layout.frame_slice(FrameGroup.VIDEO_COND)].mean(dim=(1, 2, 3)).float()
            acc_out = self.phantom_acc(acc_inputs, z_vis)
            beta = self.phantom_acc.beta_of_events(acc_out.p_evt)
            holder.acc_key = acc_key_bias(layout, acc_out.g, beta)
            holder.lambdas = list(self.phantom_bias_lambdas)

        if geo_G is None:
            geo_G = torch.zeros(B, x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3], device=device)
        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_embedding_B_T_D, context_input,
                rope_emb_L_1_1_D=rope_emb, adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb)
            j = self._bridge_at.get(i)
            if j is not None:
                x_B_T_H_W_D = self.phantom_bridges[j](x_B_T_H_W_D, geo_G, layout)

        contact_hidden = None
        if return_contact_hidden and layout.has(FrameGroup.CONTACT):
            contact_hidden = x_B_T_H_W_D[:, layout.frame_slice(FrameGroup.CONTACT)].flatten(2, 3)
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D,
                                       adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        velocity = self.unpatchify(x_B_T_H_W_O)
        return PhantomNetOutput(velocity_B_C_T_H_W=velocity, acc=acc_out,
                                contact_hidden_B_Tc_S_D=contact_hidden)
