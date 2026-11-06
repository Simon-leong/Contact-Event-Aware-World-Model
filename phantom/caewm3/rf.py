"""CaewmRectifiedFlow3: the V2 rectified flow trained and sampled along the contact-plan
factorisation.

Training (on top of the V2 final recipe):
  * layer 1: each sample's plan is the demonstration's (plan_p_gt), the readout's own median
    decision (plan_p_self) or UNKNOWN (the rest), so the plan-conditioned action is learned
    from the true plan and also sees the plans it will be given at inference;
  * layer 2: with probability anchor_p the future frames are anchored on the trajectory the
    model itself estimates from pure noise (a no-grad pass at t = 1 with the same noise and
    plan source, i.e. exactly the first sampling step, whose plan the main pass then keeps);
    otherwise, and always when the conditions are dropped, on the current fingertip. The
    ground-truth trajectory is never used;
  * layer 3: the factor-wise adapters need nothing here; the video loss weight is set by the
    launch (1.0 with group_lora).
Sampling, decide then refine: the first step decides the plan from the readout (or takes the
oracle / UNKNOWN plan) and gives the first action estimate; every later step keeps that plan
and anchors the future frames on the latest action estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from phantom.caewm.geometry import tcp_gaussian
from phantom.caewm.packing import to_fingers, unpack_slots
from phantom.caewm.rf import N_FUT
from phantom.caewm.windows import FUT_TAUS
from phantom.caewm2.rf import Caewm2Prediction, CaewmRectifiedFlow2
from phantom.caewm3.geometry import TorchCamera
from phantom.caewm3.schedule import CLOSE, DECIDE, N_EVENTS, NONE, UNKNOWN, plan_of_bins
from phantom.model.ace import losses as L
from phantom.model.rf import _expand_acc
from phantom.model.sequence import FrameGroup


@dataclass
class Caewm3Prediction(Caewm2Prediction):
    plan_B_7: torch.Tensor | None = None


class CaewmRectifiedFlow3(CaewmRectifiedFlow2):
    def __init__(self, net, hht, hw, bb, mc, layout, vae, text, tac_ae):
        super().__init__(net, hht, hw, bb, mc, layout, vae, text, tac_ae, None)
        self.cam = TorchCamera()
        A = hw.control.action_dim
        self.register_buffer("act_mean", torch.zeros(A), persistent=False)
        self.register_buffer("act_std", torch.ones(A), persistent=False)
        self._act_norm_set = False
        fpl = (bb.frames_pix - 1) // max(layout.t_video_gen, 1)
        n_con = layout.frame_slice(FrameGroup.CONTACT).stop - layout.frame_slice(FrameGroup.CONTACT).start
        spf = len(FUT_TAUS) // n_con
        self._tau_con = [sum(FUT_TAUS[spf * k: spf * (k + 1)]) / spf for k in range(n_con)]
        self._tau_vid = [(g * fpl + (fpl + 1) / 2) / bb.fps for g in range(layout.t_video_gen)]

    def set_action_norm(self, mean, std):
        """Action normalisation of the data (the checkpoint's norm stats), for de-normalising
        the action estimate into TCP displacements."""
        self.act_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(-1)[: len(self.act_mean)])
        self.act_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(-1)[: len(self.act_std)])
        self._act_norm_set = True

    # ------------------------------------------------------------------
    def _train_plan_src(self, batch, B, dev):
        mc = self.mc
        if not mc.use_plan:
            return None
        gt = plan_of_bins(batch["finger_bin"], batch["phase_bin"]).to(dev)
        cw = batch.get("contact_weight")
        if cw is not None:
            no_touch = (torch.as_tensor(cw, device=dev).reshape(-1) <= 0).view(B, 1)
            gt[:, :4] = torch.where(no_touch, torch.full_like(gt[:, :4], UNKNOWN), gt[:, :4])
        u = torch.rand(B, generator=self._gen).to(dev).view(B, 1)
        return torch.where(u < mc.plan_p_gt, gt,
                           torch.where(u < mc.plan_p_gt + mc.plan_p_self,
                                       torch.full_like(gt, DECIDE), torch.full_like(gt, UNKNOWN)))

    def _anchors(self, x_est, batch, B):
        """Action estimate (in x space) -> future anchor Gaussians {con (B,Kc,hp,wp), vid (B,Kv,hp,wp)}
        at the fingertip positions of each future frame's time along the estimated trajectory
        (held at the chunk end beyond it)."""
        assert self._act_norm_set, "set_action_norm() before anchoring on the trajectory"
        dev = x_est.device
        a = self.a_pack.unpack(x_est[:, :, self.layout.frame_slice(FrameGroup.ACTION)].float())
        a = a * self.act_std + self.act_mean                                       # (B,H,A)
        p0 = batch["tcp_xyz"].to(dev).float().view(B, 1, 3)
        pos = torch.cat([p0, p0 + torch.cumsum(a[..., :3], 1)], 1)                # (B,H+1,3), t = i / rate
        H, rate = a.shape[1], self.hw.control.action_rate_hz

        def at(taus):
            s = (torch.tensor(taus, device=dev) * rate).clamp(0, H)
            i0 = s.floor().long().clamp(max=H - 1)
            f = (s - i0.float()).view(1, -1, 1)
            return pos[:, i0] * (1 - f) + pos[:, i0 + 1] * f                       # (B,K,3)

        out = {}
        for k, taus in (("con", self._tau_con), ("vid", self._tau_vid)):
            P = at(taus)
            uv, ok = self.cam.uv_norm(P.reshape(-1, 3))
            out[k] = tcp_gaussian(uv, ok, self.hp, self.wp, self.mc.bridge_sigma).view(B, len(taus), self.hp, self.wp)
        return out

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
        ctx = self.text.get(B, text)
        G = self.geo_map(batch, B, dev)
        fps = torch.full((B,), self.bb.fps, device=dev)
        plan_src = self._train_plan_src(batch, B, dev)
        geo_future = None
        anchor = (mc.use_plan_bridge and float(torch.rand((), generator=self._gen)) < mc.anchor_p
                  and not drop_cond)
        decide = plan_src is not None and bool((plan_src == DECIDE).any())
        if anchor or decide:
            # the first sampling step, without gradient: pure noise on every generated frame.
            # Its plan replaces DECIDE (deciding at the main pass's t would read a future touch
            # that is already nearly clean at small t), its action estimate gives the anchors
            with torch.no_grad():
                t_pre = torch.ones_like(t_B_T)
                t_pre[cond] = 0.0
                x_pre = torch.where(cond_mask.bool(), x0, eps)
                pre = self.net(
                    x_B_C_T_H_W=x_pre, timesteps_B_T=t_pre * 1000.0, crossattn_emb=ctx,
                    condition_video_input_mask_B_C_T_H_W=cond_mask,
                    action=batch["prev_chunk"].to(dev, dt), acc_inputs=acc_inputs, layout=layout,
                    fps=fps, geo_G=G, plan_src=plan_src, plan_rule=mc.plan_rule, plan_gen=self._gen)
                if anchor:
                    x_est = torch.where(cond_mask.bool(), x0, x_pre - pre.velocity_B_C_T_H_W)
                    geo_future = self._anchors(x_est, batch, B)
                if pre.plan is not None:
                    plan_src = pre.plan
        out = self.net(
            x_B_C_T_H_W=x_t, timesteps_B_T=t_B_T * 1000.0, crossattn_emb=ctx,
            condition_video_input_mask_B_C_T_H_W=cond_mask,
            action=batch["prev_chunk"].to(dev, dt), acc_inputs=acc_inputs, layout=layout,
            fps=fps, geo_G=G, geo_future=geo_future, plan_src=plan_src, plan_rule=mc.plan_rule,
            plan_gen=self._gen)
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
        return parts

    # ------------------------------------------------------------------
    def _euler3(self, x, x0, cond, layout, steps, ctx, prev_chunk, acc_inputs, fps, G, batch,
                plan_src, plan_rule, redecide=False, anchors=True):
        B = x.shape[0]
        cond_mask = self._cond_mask(cond, layout).to(x.dtype)
        x = torch.where(cond_mask.bool(), x0, x)
        ts, _ = self._step_schedule(steps)
        ts = ts.to(x.device, x.dtype)
        out, geo_future, plan = None, None, plan_src
        for i in range(steps):
            t_B_T = ts[i].expand(B, layout.t_total).clone()
            t_B_T[cond] = 0.0
            out = self.net(x_B_C_T_H_W=x, timesteps_B_T=t_B_T * 1000.0, crossattn_emb=ctx,
                           crossattn_projected=True, condition_video_input_mask_B_C_T_H_W=cond_mask,
                           action=prev_chunk, acc_inputs=acc_inputs, layout=layout, fps=fps, geo_G=G,
                           geo_future=geo_future, plan_src=plan, plan_rule=plan_rule, plan_gen=self._gen)
            if out.plan is not None:
                plan = out.plan                         # decided on the first step, kept after
                if redecide and i == 0:
                    # decided once more on the second step, from the readout that now sees the
                    # model's own first estimate of the future touch
                    plan = torch.where(plan_src == DECIDE, plan_src, plan)
            if self.mc.use_plan_bridge and anchors:
                x_est = torch.where(cond_mask.bool(), x0, x - ts[i] * out.velocity_B_C_T_H_W)
                geo_future = self._anchors(x_est, batch, B)
            x = x + (ts[i + 1] - ts[i]) * out.velocity_B_C_T_H_W
            x = torch.where(cond_mask.bool(), x0, x)
        return x, out, plan

    @torch.no_grad()
    def sample(self, batch, *, nfe=None, touch: bool = True, video_refine_steps: int = 0,
               k_seeds: int = 1, decode_tactile: bool = False, pin_tac_fut_zero: bool = False,
               plan: str = "decide", plan_rule=None, anchors: bool = True,
               plan_close_shift: int = 0, plan_close_none: bool = False, mean_action: bool = False,
               **unused) -> Caewm3Prediction:
        """anchors=False keeps layer 2 at the current-fingertip anchor on every step (the
        inference-time ablation of the plan-indexed bridge).
        plan: "decide" (the model's own decision, by plan_rule, default the config's),
        "gt" (the demonstration's plan, the oracle of the error decomposition) or "unknown"."""
        nfe = nfe or self.mc.nfe
        layout = self.layout
        B0 = batch["video"].shape[0]
        touch_B = torch.full((B0,), bool(touch), dtype=torch.bool)
        x0, cond, acc_inputs = self.build_x0(batch, layout, encode_gen=False, touch=touch_B)
        if pin_tac_fut_zero:
            x0[:, :, layout.frame_slice(FrameGroup.CONTACT)] = 0
            cond[:, layout.frame_slice(FrameGroup.CONTACT)] = True
        dev0 = x0.device
        plan_src = None
        if self.mc.use_plan:
            if plan == "gt":
                plan_src = plan_of_bins(batch["finger_bin"], batch["phase_bin"]).to(dev0)
                # counterfactual close time for the decision cost table: the demonstration's
                # plan with only its close moved by plan_close_shift bins, or decided as NONE
                c = plan_src[:, CLOSE]
                known = (c >= 0) & (c < NONE)
                c = torch.where(known, (c + plan_close_shift).clamp(0, NONE - 1), c)
                plan_src[:, CLOSE] = torch.where(known, torch.full_like(c, NONE), c) if plan_close_none else c
            else:
                plan_src = torch.full((B0, N_EVENTS), DECIDE if plan in ("decide", "redecide") else UNKNOWN,
                                      device=dev0, dtype=torch.long)
        prev_chunk = batch["prev_chunk"].to(self.device, self.dtype)
        texts = batch.get("text")
        G = self.geo_map(batch, B0, self.device)
        tac_obs_in = batch["tac_obs"].to(self.device).float()
        k = max(1, int(k_seeds))
        bat = batch
        if k > 1:
            x0, cond, G = (v.repeat_interleave(k, 0) for v in (x0, cond, G))
            acc_inputs = _expand_acc(acc_inputs, k)
            prev_chunk = prev_chunk.repeat_interleave(k, 0)
            tac_obs_in = tac_obs_in.repeat_interleave(k, 0)
            if plan_src is not None:
                plan_src = plan_src.repeat_interleave(k, 0)
            if "tcp_xyz" in batch:
                bat = dict(batch, tcp_xyz=batch["tcp_xyz"].repeat_interleave(k, 0))
            if isinstance(texts, (list, tuple)):
                texts = [t for t in texts for _ in range(k)]
        B = x0.shape[0]
        dev, dt = x0.device, x0.dtype
        ctx = self.text.get(B, texts).to(dt)
        if getattr(self.net, "use_crossattn_projection", False):
            ctx = self.net.crossattn_proj(ctx)
        fps = torch.full((B,), self.bb.fps, device=dev)

        x = self._noise(x0.shape, layout, dev, dt)
        x, out, plan_used = self._euler3(x, x0, cond, layout, nfe, ctx, prev_chunk, acc_inputs, fps, G,
                                         bat, plan_src, plan_rule or self.mc.plan_rule,
                                         redecide=(plan == "redecide"), anchors=anchors)
        logits = self.phantom_tac_head(out.contact_hidden_B_Tc_S_D)
        event_logits = self.phantom_event_head(out.contact_hidden_B_Tc_S_D)
        ef = out.mid

        if video_refine_steps > 0:
            sl = layout.frame_slice(FrameGroup.VIDEO_GEN)
            cond2 = torch.ones_like(cond)
            cond2[:, sl] = False
            xr = x.clone()
            xr[:, :, sl] = torch.randn(xr[:, :, sl].shape, generator=self._gen).to(dev, dt)
            x, _, _ = self._euler3(xr, x, cond2, layout, video_refine_steps, ctx, prev_chunk,
                                   acc_inputs, fps, G, bat, plan_used, plan_rule or self.mc.plan_rule)

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
        if mean_action and k > 1:
            # the K draws share one plan, so p(a | plan, o) is unimodal and the executed chunk
            # is its posterior mean
            actions = actions.view(B0, k, *actions.shape[1:]).mean(1).repeat_interleave(k, 0)
        zeros = torch.zeros(B, layout.t_video_gen, 1, device=dev)
        return Caewm3Prediction(
            actions_B_H_A=actions, cpk=None, event_logits_B_Tc_E=event_logits,
            log_sigma_B_Tc_K=zeros, governor_sigma_B_Tc=zeros[..., 0], acc=out.acc,
            x_final_B_C_T_H_W=x, tac_obs_B_2_2_C_h_w=obs, tac_fut_B_6_2_C_h_w=fut,
            slip_prob_B_6_2=torch.sigmoid(logits[..., 0]),
            contact_prob_B_6_2=torch.sigmoid(logits[..., 1]), fields_B_6_2_C_H_W=fields,
            finger_hazard_B_12_2_2=ef["finger_logits"], phase_hazard_B_12_3=ef["phase_logits"],
            pad_B_6_2_3=ef["pad"], wrist_B_6_6=ef["wrist"],
            mid_phase_hazard_B_12_3=ef["phase_logits"], mid_finger_hazard_B_12_2_2=ef["finger_logits"],
            plan_B_7=plan_used)
