"""CaewmRectifiedFlow2: the V1 joint rectified flow with the V2 layers.

Changes to the V1 training step (phantom/caewm/rf.py):
  * video decoupling uses mc.video_decouple_p (0.15 in V2) and the video loss weight comes
    from mc.loss.video (0.3 in V2);
  * fixes: the event cross-entropy is masked out on video-decoupled samples, whose
    CONTACT frames hold the ground truth; when the conditions are dropped for
    classifier-free training the tactile observation is no longer supervised;
  * layer 1: EventForceHead on the final CONTACT hidden, trained with the discrete-time
    survival likelihood (onset/slip per finger from future tactile frames, close/lift/
    release from the gripper and arm) and with force-increment regression;
  * layer 3: the mid-depth readout of CaewmDiT2 is trained with the same hazard loss;
  * layer 2: scene crops are encoded by the frozen CropEncoder and fed to the net
    (photometric jitter on the crops during training, dropped with the conditions);
  * layer 3, trained for one-step inference (mc.use_shortcut): shortcut-model
    self-consistency on 1/shortcut_every of the samples, see shortcut_step;
  * alternative one-step stage (mc.distill_nfe > 0): self-distillation of the one-step
    sampler from the model's own distill_nfe-step samples, see distill_step.
The V1 TactileEventHead (per-slot slip / contact) is kept, so V1 tools keep working.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from phantom.caewm.heads import focal_bce, masked_bce
from phantom.caewm.packing import to_fingers, unpack_slots
from phantom.caewm.rf import N_FUT, CaewmPrediction, CaewmRectifiedFlow
from phantom.caewm2.heads import EventForceHead, cumulative, hazard_nll, masked_huber, masked_mse
from phantom.model.ace import losses as L
from phantom.model.rf import _expand_acc
from phantom.model.sequence import FrameGroup


@dataclass
class Caewm2Prediction(CaewmPrediction):
    finger_hazard_B_12_2_2: torch.Tensor | None = None   # logits, [onset, slip]
    phase_hazard_B_12_3: torch.Tensor | None = None      # logits, [close, lift, release]
    pad_B_6_2_3: torch.Tensor | None = None              # standardised increments
    wrist_B_6_6: torch.Tensor | None = None
    mid_phase_hazard_B_12_3: torch.Tensor | None = None
    mid_finger_hazard_B_12_2_2: torch.Tensor | None = None


def weighted_ce(logits_B_T_E, target_B_T, weight_B):
    ce = F.cross_entropy(logits_B_T_E.float().flatten(0, 1), target_B_T.flatten(0, 1),
                         reduction="none").view(target_B_T.shape).mean(1)
    w = weight_B.float()
    return (ce * w).sum() / w.sum().clamp_min(1e-6)


def weighted_acc_losses(acc, gate_label_B, event_next_B, weight_B, alpha_entropy_weight=0.0):
    """L.acc_losses with a per-sample weight: on video-decoupled samples the mid-depth
    readout that feeds the ACC sees the pinned ground-truth future touch."""
    w = weight_B.float()
    den = w.sum().clamp_min(1e-6)
    bce = F.binary_cross_entropy(acc.g.float().clamp(1e-6, 1 - 1e-6), gate_label_B.float(),
                                 reduction="none").reshape(-1)
    ce = F.cross_entropy(acc.event_logits.float(), event_next_B, reduction="none")
    out = {"acc_gate_bce": (bce * w).sum() / den, "acc_event_ce": (ce * w).sum() / den}
    if alpha_entropy_weight > 0:
        a = acc.alpha.float().clamp(1e-6, 1 - 1e-6)
        ent = (a * a.log() + (1 - a) * (1 - a).log()).reshape(len(w), -1).mean(1)
        out["acc_alpha_entropy"] = alpha_entropy_weight * (ent * w).sum() / den
    return out


class CaewmRectifiedFlow2(CaewmRectifiedFlow):
    def __init__(self, net, hht, hw, bb, mc, layout, vae, text, tac_ae, crop_enc=None):
        super().__init__(net, hht, hw, bb, mc, layout, vae, text, tac_ae)
        self.__dict__["crop_enc"] = crop_enc          # frozen, not registered
        if mc.use_events and mc.events_at == "final":
            self.phantom_event_force = EventForceHead(net.model_channels, self.hp, self.wp)

    # ------------------------------------------------------------------
    def _crops(self, batch, B, dev, drop):
        if not (self.mc.use_crop and self.crop_enc is not None and "crops" in batch):
            return None, None
        crops = batch["crops"].to(dev)
        if self.training:
            x = crops.float() / 255.0
            gain = 1.0 + 0.3 * (2 * torch.rand(B, 1, 1, 1, 1, generator=self._gen) - 1)
            con = 1.0 + 0.25 * (2 * torch.rand(B, 1, 1, 1, 1, generator=self._gen) - 1)
            ch = 1.0 + 0.08 * (2 * torch.rand(B, 1, 3, 1, 1, generator=self._gen) - 1)
            crops = (((x - 0.5) * con.to(dev) + 0.5) * gain.to(dev) * ch.to(dev)).clamp(0, 1)
        tok = self.crop_enc(crops)
        valid = batch["crop_valid"].to(dev).float()
        if drop:
            valid = torch.zeros_like(valid)
        return tok, valid

    # ------------------------------------------------------------------
    def training_step(self, batch):
        layout, mc = self.layout, self.mc
        if mc.distill_nfe > 0 and self.training:
            return self.distill_step(batch)
        if (mc.use_shortcut and self.training
                and float(torch.rand((), generator=self._gen)) < 1.0 / mc.shortcut_every):
            return self.shortcut_step(batch)
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
        crop_tok, crop_valid = self._crops(batch, B, dev, drop_cond)
        out = self.net(
            x_B_C_T_H_W=x_t, timesteps_B_T=t_B_T * 1000.0,
            crossattn_emb=self.text.get(B, text),
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            action=batch["prev_chunk"].to(dev, dt), acc_inputs=acc_inputs, layout=layout,
            fps=torch.full((B,), self.bb.fps, device=dev), geo_G=self.geo_map(batch, B, dev),
            crop_tokens=crop_tok, crop_valid=crop_valid)
        v_pred, v_tgt = out.velocity_B_C_T_H_W, eps - x0

        joint = (~decouple).float()
        act_w = batch.get("action_weight")
        act_w = joint if act_w is None else joint * torch.as_tensor(act_w, device=dev).reshape(-1).float()
        con_w = batch.get("contact_weight")
        con_w = joint if con_w is None else joint * torch.as_tensor(con_w, device=dev).reshape(-1).float()
        obs_w = (~touch).float() * joint * (0.0 if drop_cond else 1.0)
        parts = {
            "action_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.ACTION, act_w,
                                                 channels=slice(0, layout.actions_per_frame)),
            "video_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.VIDEO_GEN),
            "tac_fut_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.CONTACT, con_w),
            "tac_obs_v_mse": L.group_velocity_mse(v_pred, v_tgt, layout, FrameGroup.OBS_MECH, obs_w),
        }
        parts.update(self._head_parts(out, batch, con_w, joint))
        w = mc.loss
        parts["total"] = (w.action * parts["action_v_mse"] + w.video * parts["video_v_mse"]
                          + mc.loss_tac_fut * parts["tac_fut_v_mse"]
                          + mc.loss_tac_obs * parts["tac_obs_v_mse"]
                          + self._head_total(parts))
        return self._same_keys(parts, shortcut=False)

    FLOW_KEYS = ("action_v_mse", "video_v_mse", "tac_fut_v_mse", "tac_obs_v_mse")
    SHORTCUT_KEYS = ("sc_action", "sc_video", "sc_tac_fut", "sc_tac_obs")

    def _same_keys(self, parts, shortcut):
        """With shortcut training the two branches are drawn independently on every rank,
        while the training loop all-reduces the logged losses as one tensor over the sorted
        key set: both branches must log the same keys. The branch that did not run logs
        zeros and sc_active tells them apart (sc mean = sc_x / sc_active in the logs)."""
        if not self.mc.use_shortcut:
            return parts
        zero = torch.zeros((), device=parts["total"].device)
        for k in (self.FLOW_KEYS if shortcut else self.SHORTCUT_KEYS):
            parts.setdefault(k, zero)
        parts["sc_active"] = zero + float(shortcut)
        return parts

    def _head_parts(self, out, batch, con_w, joint):
        """Losses of the readouts on the CONTACT hidden (and the ACC), per-sample weights
        con_w (contact supervision) and joint (not video-decoupled)."""
        mc = self.mc
        B, dev = con_w.shape[0], con_w.device
        logits = self.phantom_tac_head(out.contact_hidden_B_Tc_S_D)            # (B,6,2,2)
        slip_lab, con_lab = batch["tac_slip"].to(dev), batch["tac_contact"].to(dev)
        head_w = con_w.view(B, 1, 1)
        parts = {
            "slip_focal": focal_bce(logits[..., 0], slip_lab, con_lab * head_w, mc.focal_gamma),
            "tac_contact_bce": masked_bce(logits[..., 1], con_lab, head_w.expand_as(con_lab)),
            "event_ce": weighted_ce(self.phantom_event_head(out.contact_hidden_B_Tc_S_D),
                                    batch["events"].to(dev), con_w),
        }
        if out.acc is not None:
            parts.update(weighted_acc_losses(out.acc, batch["gate_label"].to(dev),
                                             batch["events"][:, 0].to(dev), joint,
                                             mc.acc.alpha_entropy_weight))
        fb, pb = batch["finger_bin"].to(dev), batch["phase_bin"].to(dev)
        fin_w = con_w.view(B, 1, 1).expand(fb.shape)
        ph_w = joint.view(B, 1).expand(pb.shape)
        force_fn = masked_huber if mc.force_loss == "huber" else masked_mse
        if mc.use_events and mc.events_at == "final":
            ef = self.phantom_event_force(out.contact_hidden_B_Tc_S_D)
            parts["hazard_finger"] = hazard_nll(ef["finger_logits"], fb, fin_w)
            parts["hazard_phase"] = hazard_nll(ef["phase_logits"], pb, ph_w)
            parts["force_pad"] = force_fn(ef["pad"], batch["pad_fut"].to(dev), con_w)
            parts["force_wrist"] = force_fn(ef["wrist"], batch["wrist_fut"].to(dev), joint)
        if mc.use_events and mc.events_at == "mid":
            parts["force_pad"] = force_fn(out.mid["pad"], batch["pad_fut"].to(dev), con_w)
            parts["force_wrist"] = force_fn(out.mid["wrist"], batch["wrist_fut"].to(dev), joint)
        if out.mid is not None:
            parts["mid_hazard_finger"] = hazard_nll(out.mid["finger_logits"], fb, fin_w)
            parts["mid_hazard_phase"] = hazard_nll(out.mid["phase_logits"], pb, ph_w)
        return parts

    def _head_total(self, parts):
        mc, w = self.mc, self.mc.loss
        zero = torch.zeros((), device=parts["event_ce"].device)
        return (mc.loss_slip * parts["slip_focal"]
                + mc.loss_tac_contact * parts["tac_contact_bce"]
                + w.event * parts["event_ce"]
                + w.gate_bce * parts.get("acc_gate_bce", zero)
                + w.event * parts.get("acc_event_ce", zero)
                + mc.loss_hazard * (parts.get("hazard_finger", zero) + parts.get("hazard_phase", zero))
                + mc.loss_force * (parts.get("force_pad", zero) + parts.get("force_wrist", zero))
                + mc.loss_mid_hazard * (parts.get("mid_hazard_finger", zero)
                                        + parts.get("mid_hazard_phase", zero)))

    # ------------------------------------------------------------------
    # one-step self-distillation (progressive distillation, Salimans & Ho 2022, on the
    # rectified flow): the teacher is the model as loaded for this stage (the main run's
    # weights), integrated with distill_nfe Euler steps; the student takes one step from
    # the same noise and is trained to land where the teacher landed. Teacher and student
    # share the frozen Cosmos base, so the teacher is only a snapshot of the trainable
    # parameters, swapped in for its no-grad pass.
    def _trainable(self):
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def _teacher_snapshot(self):
        t = self.__dict__.get("_distill_teacher")
        if t is None:
            t = {n: p.detach().clone() for n, p in self._trainable().items()}
            self.__dict__["_distill_teacher"] = t
        return t

    @contextmanager
    def _teacher_weights(self):
        params, teacher = self._trainable(), self._teacher_snapshot()
        with torch.no_grad():
            saved = {n: p.detach().clone() for n, p in params.items()}
            for n, p in params.items():
                p.copy_(teacher[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(saved[n])

    def distill_step(self, batch):
        layout, mc = self.layout, self.mc
        B = batch["video"].shape[0]
        touch = torch.rand(B, generator=self._gen) < mc.touch_forcing_p
        text = batch.get("text")
        crop = self._crops(batch, B, self.device, False)
        with torch.no_grad(), self._teacher_weights():
            x0_t, cond, acc_t = self.build_x0(batch, encode_gen=False, touch=touch)
            dev, dt = x0_t.device, x0_t.dtype
            eps = self._noise(x0_t.shape, layout, dev, dt)
            prev_chunk = batch["prev_chunk"].to(dev, dt)
            fps = torch.full((B,), self.bb.fps, device=dev)
            G = self.geo_map(batch, B, dev)
            ctx = self.text.get(B, text).to(dt)
            if getattr(self.net, "use_crossattn_projection", False):
                ctx = self.net.crossattn_proj(ctx)
            x_tea, _ = self._euler(eps.clone(), x0_t, cond, layout, mc.distill_nfe, ctx, prev_chunk,
                                   acc_t, fps, G, crop)
        x0, cond_s, acc_inputs = self.build_x0(batch, encode_gen=False, touch=touch)
        cond_mask = self._cond_mask(cond_s, layout).to(dt)
        x1 = torch.where(cond_mask.bool(), x0, eps)
        t_B_T = torch.ones(B, layout.t_total, device=dev, dtype=dt)
        t_B_T[cond_s] = 0.0
        out = self.net(
            x_B_C_T_H_W=x1, timesteps_B_T=t_B_T * 1000.0, crossattn_emb=self.text.get(B, text),
            condition_video_input_mask_B_C_T_H_W=cond_mask, action=prev_chunk,
            acc_inputs=acc_inputs, layout=layout, fps=fps, geo_G=G,
            crop_tokens=crop[0], crop_valid=crop[1])
        x_stu = x1 - out.velocity_B_C_T_H_W                      # one Euler step, t = 1 -> 0
        ones = torch.ones(B, device=dev)
        act_w = batch.get("action_weight")
        act_w = ones if act_w is None else torch.as_tensor(act_w, device=dev).reshape(-1).float()
        con_w = batch.get("contact_weight")
        con_w = ones if con_w is None else torch.as_tensor(con_w, device=dev).reshape(-1).float()
        obs_w = (~touch).float().to(dev)
        parts = {
            "distill_action": L.group_velocity_mse(x_stu, x_tea, layout, FrameGroup.ACTION, act_w,
                                                   channels=slice(0, layout.actions_per_frame)),
            "distill_video": L.group_velocity_mse(x_stu, x_tea, layout, FrameGroup.VIDEO_GEN),
            "distill_tac_fut": L.group_velocity_mse(x_stu, x_tea, layout, FrameGroup.CONTACT, con_w),
            "distill_tac_obs": L.group_velocity_mse(x_stu, x_tea, layout, FrameGroup.OBS_MECH, obs_w),
        }
        # the readouts are trained on the hidden state of this same one-step pass, which is
        # the pass they are read from at deployment
        parts.update(self._head_parts(out, batch, con_w, ones))
        w = mc.loss
        parts["total"] = (w.action * parts["distill_action"] + w.video * parts["distill_video"]
                          + mc.loss_tac_fut * parts["distill_tac_fut"]
                          + mc.loss_tac_obs * parts["distill_tac_obs"]
                          + self._head_total(parts))
        return parts

    # ------------------------------------------------------------------
    # shortcut self-consistency (kvfrans/shortcut-models, targets_shortcut.py), in this
    # code's time convention (t = 1 noise, t = 0 data, x_{t-d} = x_t - d v): a step of
    # d = 2^-m from a grid time t is trained to equal the mean of two half steps of the
    # same model (no grad); the smallest level's half steps use the flow-matching index.
    def shortcut_step(self, batch):
        layout, mc = self.layout, self.mc
        B = batch["video"].shape[0]
        touch = torch.rand(B, generator=self._gen) < mc.touch_forcing_p
        m = torch.randint(0, mc.shortcut_levels, (B,), generator=self._gen)
        d = 0.5 ** m.float()
        k = (torch.rand(B, generator=self._gen) * (2.0 ** m.float())).floor()     # 0 .. 2^m - 1
        t = 1.0 - k * d                                                             # grid, t - d >= 0
        x0, cond, acc_inputs = self.build_x0(batch, touch=touch)
        dev, dt = x0.device, x0.dtype
        m, d, t = m.to(dev), d.to(dev, dt), t.to(dev, dt)
        half_idx = torch.where(m + 1 >= mc.shortcut_levels, torch.full_like(m, mc.shortcut_levels), m + 1)
        cond_mask = self._cond_mask(cond, layout).to(dt)
        pin = cond_mask.bool()
        eps = self._noise(x0.shape, layout, dev, dt)
        x_t = torch.where(pin, x0, (1 - t.view(B, 1, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1, 1) * eps)
        text = batch.get("text")
        crop = self._crops(batch, B, dev, False)
        kw = dict(action=batch["prev_chunk"].to(dev, dt), acc_inputs=acc_inputs, layout=layout,
                  fps=torch.full((B,), self.bb.fps, device=dev), geo_G=self.geo_map(batch, B, dev),
                  crop_tokens=crop[0], crop_valid=crop[1], condition_video_input_mask_B_C_T_H_W=cond_mask)

        def times(tt):
            t_B_T = tt.view(B, 1).expand(B, layout.t_total).clone()
            t_B_T[cond] = 0.0
            return t_B_T * 1000.0

        with torch.no_grad():
            ctx = self.text.get(B, text)
            v1 = self.net(x_B_C_T_H_W=x_t, timesteps_B_T=times(t), crossattn_emb=ctx, step_idx=half_idx,
                          **kw).velocity_B_C_T_H_W
            x_mid = torch.where(pin, x0, (x_t - 0.5 * d.view(B, 1, 1, 1, 1) * v1).clamp(-4, 4))
            v2 = self.net(x_B_C_T_H_W=x_mid, timesteps_B_T=times(t - 0.5 * d), crossattn_emb=ctx,
                          step_idx=half_idx, **kw).velocity_B_C_T_H_W
            v_tgt = (0.5 * (v1 + v2)).clamp(-4, 4)
        out = self.net(x_B_C_T_H_W=x_t, timesteps_B_T=times(t), crossattn_emb=self.text.get(B, text),
                       step_idx=m, **kw)
        v = out.velocity_B_C_T_H_W
        ones = torch.ones(B, device=dev)
        act_w = batch.get("action_weight")
        act_w = ones if act_w is None else torch.as_tensor(act_w, device=dev).reshape(-1).float()
        con_w = batch.get("contact_weight")
        con_w = ones if con_w is None else torch.as_tensor(con_w, device=dev).reshape(-1).float()
        obs_w = (~touch).float().to(dev)
        parts = {
            "sc_action": L.group_velocity_mse(v, v_tgt, layout, FrameGroup.ACTION, act_w,
                                              channels=slice(0, layout.actions_per_frame)),
            "sc_video": L.group_velocity_mse(v, v_tgt, layout, FrameGroup.VIDEO_GEN),
            "sc_tac_fut": L.group_velocity_mse(v, v_tgt, layout, FrameGroup.CONTACT, con_w),
            "sc_tac_obs": L.group_velocity_mse(v, v_tgt, layout, FrameGroup.OBS_MECH, obs_w),
        }
        parts.update(self._head_parts(out, batch, con_w, ones))
        w = mc.loss
        parts["total"] = (w.action * parts["sc_action"] + w.video * parts["sc_video"]
                          + mc.loss_tac_fut * parts["sc_tac_fut"] + mc.loss_tac_obs * parts["sc_tac_obs"]
                          + self._head_total(parts))
        return self._same_keys(parts, shortcut=True)

    def _step_schedule(self, steps):
        """Sampling grid and the net's step index: a shortcut model samples a power-of-two step
        count on the uniform grid with the step-size index of 1/steps; any other count (and a
        model without shortcut training) uses the flow velocity on the time-shifted grid."""
        if self.mc.use_shortcut:
            m = int(round(math.log2(steps)))
            if 2 ** m == steps and m < self.mc.shortcut_levels:
                return torch.linspace(1.0, 0.0, steps + 1), m
        return self._time_shift(torch.linspace(1.0, 0.0, steps + 1)), None

    # ------------------------------------------------------------------
    def _euler(self, x, x0, cond, layout, steps, ctx, prev_chunk, acc_inputs, fps, G, crop=(None, None)):
        B = x.shape[0]
        cond_mask = self._cond_mask(cond, layout).to(x.dtype)
        x = torch.where(cond_mask.bool(), x0, x)
        ts, m = self._step_schedule(steps)
        ts = ts.to(x.device, x.dtype)
        step_idx = None if m is None else torch.full((B,), m, device=x.device, dtype=torch.long)
        out = None
        for i in range(steps):
            t_B_T = ts[i].expand(B, layout.t_total).clone()
            t_B_T[cond] = 0.0
            out = self.net(x_B_C_T_H_W=x, timesteps_B_T=t_B_T * 1000.0, crossattn_emb=ctx,
                           crossattn_projected=True, condition_video_input_mask_B_C_T_H_W=cond_mask,
                           action=prev_chunk, acc_inputs=acc_inputs, layout=layout, fps=fps, geo_G=G,
                           crop_tokens=crop[0], crop_valid=crop[1], step_idx=step_idx)
            x = x + (ts[i + 1] - ts[i]) * out.velocity_B_C_T_H_W
            x = torch.where(cond_mask.bool(), x0, x)
        return x, out

    @torch.no_grad()
    def sample(self, batch, *, nfe=None, touch: bool = True, video_refine_steps: int = 0,
               k_seeds: int = 1, decode_tactile: bool = False, pin_tac_fut_zero: bool = False,
               **unused) -> Caewm2Prediction:
        nfe = nfe or self.mc.nfe
        layout = self.layout
        B0 = batch["video"].shape[0]
        touch_B = torch.full((B0,), bool(touch), dtype=torch.bool)
        x0, cond, acc_inputs = self.build_x0(batch, layout, encode_gen=False, touch=touch_B)
        if pin_tac_fut_zero:
            x0[:, :, layout.frame_slice(FrameGroup.CONTACT)] = 0
            cond[:, layout.frame_slice(FrameGroup.CONTACT)] = True
        prev_chunk = batch["prev_chunk"].to(self.device, self.dtype)
        texts = batch.get("text")
        G = self.geo_map(batch, B0, self.device)
        tac_obs_in = batch["tac_obs"].to(self.device).float()
        crop_tok, crop_valid = self._crops(batch, B0, self.device, False)
        k = max(1, int(k_seeds))
        if k > 1:
            x0, cond, G = (v.repeat_interleave(k, 0) for v in (x0, cond, G))
            acc_inputs = _expand_acc(acc_inputs, k)
            prev_chunk = prev_chunk.repeat_interleave(k, 0)
            tac_obs_in = tac_obs_in.repeat_interleave(k, 0)
            if crop_tok is not None:
                crop_tok, crop_valid = crop_tok.repeat_interleave(k, 0), crop_valid.repeat_interleave(k, 0)
            if isinstance(texts, (list, tuple)):
                texts = [t for t in texts for _ in range(k)]
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype
        ctx = self.text.get(B, texts).to(dt)
        if getattr(self.net, "use_crossattn_projection", False):
            ctx = self.net.crossattn_proj(ctx)
        fps = torch.full((B,), self.bb.fps, device=dev)

        x = self._noise(x0.shape, layout, dev, dt)
        x, out = self._euler(x, x0, cond, layout, nfe, ctx, prev_chunk, acc_inputs, fps, G,
                             (crop_tok, crop_valid))
        logits = self.phantom_tac_head(out.contact_hidden_B_Tc_S_D)
        event_logits = self.phantom_event_head(out.contact_hidden_B_Tc_S_D)
        if self.mc.use_events and self.mc.events_at == "final":
            ef = self.phantom_event_force(out.contact_hidden_B_Tc_S_D)
        elif self.mc.use_events and self.mc.events_at == "mid":
            ef = out.mid                    # layer 1 is read from the mid-depth readout
        else:
            ef = None

        if video_refine_steps > 0:
            sl = layout.frame_slice(FrameGroup.VIDEO_GEN)
            cond2 = torch.ones_like(cond)
            cond2[:, sl] = False
            xr = x.clone()
            xr[:, :, sl] = torch.randn(xr[:, :, sl].shape, generator=self._gen).to(dev, dt)
            x, _ = self._euler(xr, x, cond2, layout, video_refine_steps, ctx, prev_chunk,
                               acc_inputs, fps, G, (crop_tok, crop_valid))

        ae = self.tac_ae
        obs = (tac_obs_in if touch else
               to_fingers(unpack_slots(x[:, :, layout.frame_slice(FrameGroup.OBS_MECH)].float())))
        delta = to_fingers(unpack_slots(x[:, :, layout.frame_slice(FrameGroup.CONTACT)].float()))
        fut = obs[:, 1:2] + delta * ae.dz_std.view(1, 1, 1, -1, 1, 1)
        fields = None
        if decode_tactile:
            zs = fut.flatten(0, 1)
            xl, _ = ae.decode_raw(zs[:, 0].float(), mirror=False)
            xr_, _ = ae.decode_raw(zs[:, 1].float(), mirror=True)
            fields = torch.stack([xl, xr_], 1).reshape(B, N_FUT, 2, *xl.shape[1:])
        actions = self.a_pack.unpack(x[:, :, layout.frame_slice(FrameGroup.ACTION)].float())
        zeros = torch.zeros(B, layout.t_video_gen, 1, device=dev)
        return Caewm2Prediction(
            actions_B_H_A=actions, cpk=None, event_logits_B_Tc_E=event_logits,
            log_sigma_B_Tc_K=zeros, governor_sigma_B_Tc=zeros[..., 0], acc=out.acc,
            x_final_B_C_T_H_W=x, tac_obs_B_2_2_C_h_w=obs, tac_fut_B_6_2_C_h_w=fut,
            slip_prob_B_6_2=torch.sigmoid(logits[..., 0]),
            contact_prob_B_6_2=torch.sigmoid(logits[..., 1]), fields_B_6_2_C_H_W=fields,
            finger_hazard_B_12_2_2=None if ef is None else ef["finger_logits"],
            phase_hazard_B_12_3=None if ef is None else ef["phase_logits"],
            pad_B_6_2_3=None if ef is None else ef["pad"],
            wrist_B_6_6=None if ef is None else ef["wrist"],
            mid_phase_hazard_B_12_3=None if out.mid is None else out.mid["phase_logits"],
            mid_finger_hazard_B_12_2_2=None if out.mid is None else out.mid["finger_logits"])
