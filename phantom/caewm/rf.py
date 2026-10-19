"""CaewmRectifiedFlow: the CAEWM joint rectified flow.

Sequence content (layout.py):
  OBS_MECH  — tactile-tokenizer latents at t0-0.5 s and t0 (common/diff, packed);
  CONTACT   — future tactile latent increments z(t0+tau) - z(t0), tau = 0.5..3 s,
              divided by the tokenizer's dz_std (packed);
  the video/proprio/action frames are unchanged from HapticWAM.

Group-wise noise (one model, one training run):
  Touch Forcing   — per sample, with p = touch_forcing_p the tactile obs is pinned
                    (sensor mode); otherwise it is noised with the joint t and
                    supervised (imagine mode). Replaces teacher->student distillation.
  Video decoupling — with p = video_decouple_p a sample pins future actions and
                    future touch (and the tactile obs) to ground truth and noises
                    only the future video at its own t (forward dynamics).
Deployment: 1-step joint sampling for actions + touch; for visualisation the
predicted actions/touch are pinned and the video alone is refined for N steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from phantom.caewm.geometry import tcp_gaussian
from phantom.caewm.heads import TactileEventHead, focal_bce, masked_bce
from phantom.caewm.packing import pack_slots, to_common_diff, to_fingers, unpack_slots
from phantom.model.ace import losses as L
from phantom.model.acc import AccInputs
from phantom.model.rf import PhantomPrediction, PhantomRectifiedFlow, _expand_acc
from phantom.model.sequence import FrameGroup

N_FUT = 6


@dataclass
class CaewmPrediction(PhantomPrediction):
    tac_obs_B_2_2_C_h_w: torch.Tensor | None = None     # standardised z, fingers (R mirrored)
    tac_fut_B_6_2_C_h_w: torch.Tensor | None = None     # standardised absolute z(t0+tau)
    slip_prob_B_6_2: torch.Tensor | None = None
    contact_prob_B_6_2: torch.Tensor | None = None
    fields_B_6_2_C_H_W: torch.Tensor | None = None      # decoded raw fields (R un-mirrored)


class CaewmRectifiedFlow(PhantomRectifiedFlow):
    def __init__(self, net, hht, hw, bb, mc, layout, vae, text, tac_ae):
        super().__init__(net, hht, hw, bb, mc, layout, vae, text)
        self.__dict__["tac_ae"] = tac_ae          # frozen, not registered (like the VAE)
        self.hp = layout.lat_h // bb.patch_spatial
        self.wp = layout.lat_w // bb.patch_spatial
        self.phantom_tac_head = TactileEventHead(net.model_channels, self.hp, self.wp)

    # ------------------------------------------------------------------
    def _tac_noise(self, B, n_slots, dev, dt):
        z = torch.randn(B, n_slots, 2, self.layout.lat_c, self.layout.lat_h // 4,
                        self.layout.lat_w // 4, generator=self._gen)
        return pack_slots(z).to(dev, dt)

    def _noise(self, shape, layout, dev, dt):
        B = shape[0]
        eps = torch.randn(shape, generator=self._gen).to(dev, dt)
        if self.mc.action_noise_per_strip:
            eps[:, :, layout.frame_slice(FrameGroup.ACTION)] = self._action_strip_noise(B, dev, dt)
        eps[:, :, layout.frame_slice(FrameGroup.OBS_MECH)] = self._tac_noise(B, 2, dev, dt)
        eps[:, :, layout.frame_slice(FrameGroup.CONTACT)] = self._tac_noise(B, N_FUT, dev, dt)
        return eps

    @staticmethod
    def tac_summary(tac_obs, touch_B):
        cd = to_common_diff(tac_obs.float())                       # (B,2,2,C,h,w)
        s = torch.cat([cd.mean(dim=(-2, -1)).flatten(1),
                       cd.pow(2).mean(dim=(-2, -1)).sqrt().flatten(1)], dim=1)
        t = touch_B.float().view(-1, 1)
        return torch.cat([s * t, t], dim=1)

    def geo_map(self, batch, B, dev):
        if "tcp_uv" not in batch:
            return torch.zeros(B, self.hp, self.wp, device=dev)
        return tcp_gaussian(batch["tcp_uv"].to(dev), batch["tcp_valid"].to(dev),
                            self.hp, self.wp, self.mc.bridge_sigma)

    # ------------------------------------------------------------------
    def build_x0(self, batch, layout=None, *, encode_gen=True, null_video_cond=False, touch=None):
        """-> x0, cond_B_T (bool), acc_inputs. touch (B,) bool: tactile obs observed."""
        layout = layout or self.layout
        dev, dt = self.device, self.dtype
        video = batch["video"].to(dev, dt)
        B = video.shape[0]
        pix = video if encode_gen else video[:, :1]
        vid_lat = self.vae.encode(pix.permute(0, 2, 1, 3, 4)).to(dt)
        x0 = torch.zeros(B, layout.lat_c, layout.t_total, layout.lat_h, layout.lat_w,
                         device=dev, dtype=dt)
        x0[:, :, layout.frame_slice(FrameGroup.VIDEO_COND)] = (
            self._null_cond_latent(B) if null_video_cond else vid_lat[:, :, :1])
        if layout.has(FrameGroup.VIDEO_GEN) and encode_gen:
            x0[:, :, layout.frame_slice(FrameGroup.VIDEO_GEN)] = vid_lat[:, :, 1:]
        obs_batch = {k: v.to(dev, dt) for k, v in batch.items()
                     if k in ("wrist", "ur_state") and torch.is_tensor(v)}
        x0[:, :, layout.frame_slice(FrameGroup.OBS_PROPRIO)] = \
            self.hht.obs_frames(obs_batch)[FrameGroup.OBS_PROPRIO].to(dt)

        tac_obs = batch["tac_obs"].to(dev).float()                       # (B,2,2,C,h,w)
        x0[:, :, layout.frame_slice(FrameGroup.OBS_MECH)] = pack_slots(to_common_diff(tac_obs)).to(dt)
        if "tac_fut" in batch:
            x0[:, :, layout.frame_slice(FrameGroup.CONTACT)] = \
                pack_slots(to_common_diff(batch["tac_fut"].to(dev).float())).to(dt)
        if "action_chunk" in batch:
            x0[:, :, layout.frame_slice(FrameGroup.ACTION)] = \
                self.a_pack.pack(batch["action_chunk"].to(dev)).to(dt)

        touch = torch.ones(B, dtype=torch.bool, device=dev) if touch is None else touch.to(dev)
        cond = torch.from_numpy(layout.cond_mask_T()).to(dev).reshape(1, -1).repeat(B, 1)
        cond[:, layout.frame_slice(FrameGroup.OBS_MECH)] = touch.view(B, 1)
        acc_inputs = AccInputs(
            wrist_feat_B_D=self.hht.wrist_feature(obs_batch).to(dt),
            intent_B_H_A=batch["prev_chunk"].to(dev, dt),
            prev_cpk_summary_B_S=self.tac_summary(tac_obs, touch).to(dt),
            react_score_B=None)
        return x0, cond, acc_inputs

    @staticmethod
    def _null_obs_batch(batch):
        out = PhantomRectifiedFlow._null_obs_batch(batch)
        if "tac_obs" in out:
            out["tac_obs"] = torch.zeros_like(out["tac_obs"])
        return out

    @staticmethod
    def _cond_mask(cond_B_T, layout):
        B = cond_B_T.shape[0]
        return cond_B_T.view(B, 1, -1, 1, 1).expand(B, 1, -1, layout.lat_h, layout.lat_w)

    # ------------------------------------------------------------------
    def training_step(self, batch):
        layout, mc = self.layout, self.mc
        drop_cond = (mc.cond_dropout_p > 0 and self.training
                     and float(torch.rand((), generator=self._gen)) < mc.cond_dropout_p)
        if drop_cond:
            batch = self._null_obs_batch(batch)
        B = batch["video"].shape[0]
        touch = torch.rand(B, generator=self._gen) < mc.touch_forcing_p
        if drop_cond:
            touch[:] = False
        decouple = torch.rand(B, generator=self._gen) < mc.video_decouple_p
        x0, cond, acc_inputs = self.build_x0(batch, null_video_cond=drop_cond, touch=touch)
        dev, dt = x0.device, x0.dtype
        touch, decouple = touch.to(dev), decouple.to(dev)

        sl_vid = layout.frame_slice(FrameGroup.VIDEO_GEN)
        t = self._sample_t(B, dev).to(dt)
        t_vid = torch.where(decouple, self._sample_t(B, dev).to(dt), t)
        t_B_T = t.reshape(B, 1).expand(B, layout.t_total).clone()
        t_B_T[:, sl_vid] = t_vid.reshape(B, 1)
        for g in (FrameGroup.OBS_MECH, FrameGroup.CONTACT, FrameGroup.ACTION):
            cond[:, layout.frame_slice(g)] |= decouple.view(B, 1)
        t_B_T[cond] = 0.0
        cond_mask = self._cond_mask(cond, layout).to(dt)

        eps = self._noise(x0.shape, layout, dev, dt)
        t_frame = t_B_T.reshape(B, 1, layout.t_total, 1, 1).to(dt)
        x_t = (1 - t_frame) * x0 + t_frame * eps
        x_t = torch.where(cond_mask.bool(), x0, x_t)
        text = [""] * B if drop_cond else batch.get("text")
        out = self.net(
            x_B_C_T_H_W=x_t, timesteps_B_T=t_B_T * 1000.0,
            crossattn_emb=self.text.get(B, text),
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            action=batch["prev_chunk"].to(dev, dt), acc_inputs=acc_inputs, layout=layout,
            fps=torch.full((B,), self.bb.fps, device=dev), geo_G=self.geo_map(batch, B, dev))
        v_pred, v_tgt = out.velocity_B_C_T_H_W, eps - x0

        joint = (~decouple).float()
        act_w = batch.get("action_weight")
        act_w = joint if act_w is None else joint * torch.as_tensor(act_w, device=dev).reshape(-1).float()
        con_w = batch.get("contact_weight")
        con_w = joint if con_w is None else joint * torch.as_tensor(con_w, device=dev).reshape(-1).float()
        logits = self.phantom_tac_head(out.contact_hidden_B_Tc_S_D)            # (B,6,2,2)
        slip_lab, con_lab = batch["tac_slip"].to(dev), batch["tac_contact"].to(dev)
        head_w = con_w.view(B, 1, 1)
        parts = {
            "action_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.ACTION, act_w,
                                                 channels=slice(0, layout.actions_per_frame)),
            "video_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.VIDEO_GEN),
            "tac_fut_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.CONTACT, con_w),
            "tac_obs_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.OBS_MECH,
                                                  (~touch).float() * joint),
            "slip_focal": focal_bce(logits[..., 0], slip_lab, con_lab * head_w, mc.focal_gamma),
            "tac_contact_bce": masked_bce(logits[..., 1], con_lab, head_w.expand_as(con_lab)),
            "event_ce": L.event_ce(self.phantom_event_head(out.contact_hidden_B_Tc_S_D),
                                   batch["events"].to(dev)),
        }
        if out.acc is not None:
            parts.update(L.acc_losses(out.acc, batch["gate_label"].to(dev),
                                      batch["events"][:, 0].to(dev), mc.acc.alpha_entropy_weight))
        w = mc.loss
        zero = torch.zeros((), device=dev)
        parts["total"] = (w.action * parts["action_v_mse"] + w.video * parts["video_v_mse"]
                          + mc.loss_tac_fut * parts["tac_fut_v_mse"]
                          + mc.loss_tac_obs * parts["tac_obs_v_mse"]
                          + mc.loss_slip * parts["slip_focal"]
                          + mc.loss_tac_contact * parts["tac_contact_bce"]
                          + w.event * parts["event_ce"]
                          + w.gate_bce * parts.get("acc_gate_bce", zero)
                          + w.event * parts.get("acc_event_ce", zero))
        return parts

    # ------------------------------------------------------------------
    def _euler(self, x, x0, cond, layout, steps, ctx, prev_chunk, acc_inputs, fps, G):
        B = x.shape[0]
        cond_mask = self._cond_mask(cond, layout).to(x.dtype)
        x = torch.where(cond_mask.bool(), x0, x)
        ts = self._time_shift(torch.linspace(1.0, 0.0, steps + 1)).to(x.device, x.dtype)
        out = None
        for i in range(steps):
            t_B_T = ts[i].expand(B, layout.t_total).clone()
            t_B_T[cond] = 0.0
            out = self.net(x_B_C_T_H_W=x, timesteps_B_T=t_B_T * 1000.0, crossattn_emb=ctx,
                           crossattn_projected=True, condition_video_input_mask_B_C_T_H_W=cond_mask,
                           action=prev_chunk, acc_inputs=acc_inputs, layout=layout, fps=fps, geo_G=G)
            x = x + (ts[i + 1] - ts[i]) * out.velocity_B_C_T_H_W
            x = torch.where(cond_mask.bool(), x0, x)
        return x, out

    @torch.no_grad()
    def sample(self, batch, *, nfe=None, touch: bool = True, video_refine_steps: int = 0,
               k_seeds: int = 1, decode_tactile: bool = False, pin_tac_fut_zero: bool = False,
               **unused) -> CaewmPrediction:
        nfe = nfe or self.mc.nfe
        layout = self.layout
        B0 = batch["video"].shape[0]
        touch_B = torch.full((B0,), bool(touch), dtype=torch.bool)
        x0, cond, acc_inputs = self.build_x0(batch, layout, encode_gen=False, touch=touch_B)
        if pin_tac_fut_zero:        # diagnostic: future touch pinned to persistence (zero increment)
            x0[:, :, layout.frame_slice(FrameGroup.CONTACT)] = 0
            cond[:, layout.frame_slice(FrameGroup.CONTACT)] = True
        prev_chunk = batch["prev_chunk"].to(self.device, self.dtype)
        texts = batch.get("text")
        G = self.geo_map(batch, B0, self.device)
        tac_obs_in = batch["tac_obs"].to(self.device).float()
        k = max(1, int(k_seeds))
        if k > 1:
            x0, cond, G = (v.repeat_interleave(k, 0) for v in (x0, cond, G))
            acc_inputs = _expand_acc(acc_inputs, k)
            prev_chunk = prev_chunk.repeat_interleave(k, 0)
            tac_obs_in = tac_obs_in.repeat_interleave(k, 0)
            if isinstance(texts, (list, tuple)):
                texts = [t for t in texts for _ in range(k)]
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype
        ctx = self.text.get(B, texts).to(dt)
        if getattr(self.net, "use_crossattn_projection", False):
            ctx = self.net.crossattn_proj(ctx)
        fps = torch.full((B,), self.bb.fps, device=dev)

        x = self._noise(x0.shape, layout, dev, dt)
        x, out = self._euler(x, x0, cond, layout, nfe, ctx, prev_chunk, acc_inputs, fps, G)
        logits = self.phantom_tac_head(out.contact_hidden_B_Tc_S_D)
        event_logits = self.phantom_event_head(out.contact_hidden_B_Tc_S_D)

        if video_refine_steps > 0:
            sl = layout.frame_slice(FrameGroup.VIDEO_GEN)
            cond2 = torch.ones_like(cond)
            cond2[:, sl] = False
            xr = x.clone()
            xr[:, :, sl] = torch.randn(xr[:, :, sl].shape, generator=self._gen).to(dev, dt)
            x, _ = self._euler(xr, x, cond2, layout, video_refine_steps, ctx, prev_chunk,
                               acc_inputs, fps, G)

        ae = self.tac_ae
        obs = (tac_obs_in if touch else
               to_fingers(unpack_slots(x[:, :, layout.frame_slice(FrameGroup.OBS_MECH)].float())))
        delta = to_fingers(unpack_slots(x[:, :, layout.frame_slice(FrameGroup.CONTACT)].float()))
        fut = obs[:, 1:2] + delta * ae.dz_std.view(1, 1, 1, -1, 1, 1)
        fields = None
        if decode_tactile:
            zs = fut.flatten(0, 1)                                       # (B*6, 2, C, h, w)
            xl, _ = ae.decode_raw(zs[:, 0].float(), mirror=False)
            xr_, _ = ae.decode_raw(zs[:, 1].float(), mirror=True)
            fields = torch.stack([xl, xr_], 1).reshape(B, N_FUT, 2, *xl.shape[1:])
        actions = self.a_pack.unpack(x[:, :, layout.frame_slice(FrameGroup.ACTION)].float())
        zeros = torch.zeros(B, layout.t_video_gen, 1, device=dev)
        return CaewmPrediction(
            actions_B_H_A=actions, cpk=None, event_logits_B_Tc_E=event_logits,
            log_sigma_B_Tc_K=zeros, governor_sigma_B_Tc=zeros[..., 0], acc=out.acc,
            x_final_B_C_T_H_W=x, tac_obs_B_2_2_C_h_w=obs, tac_fut_B_6_2_C_h_w=fut,
            slip_prob_B_6_2=torch.sigmoid(logits[..., 0]),
            contact_prob_B_6_2=torch.sigmoid(logits[..., 1]), fields_B_6_2_C_H_W=fields)
