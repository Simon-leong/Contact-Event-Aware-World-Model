"""Build a CAEWM model of any version ('caewm', 'caewm2', 'caewm3') and its window sampler
from a checkpoint payload, for the evaluation tools."""
from __future__ import annotations

import phantom.caewm.paths as CP

CAEWM_ARCHS = ("caewm", "caewm2", "caewm3")


def build_from_payload(payload, hw, paths, norm, *, device, dtype, tiny=False, wbr=0):
    mcd = payload["configs"]["model"]
    if mcd.get("arch") != "caewm3":
        from phantom.caewm2.load import build_from_payload as build_v12
        return build_v12(payload, hw, paths, norm, device=device, dtype=dtype, tiny=tiny, wbr=wbr)
    import phantom.caewm2.paths as CP2
    from phantom.caewm3.builder import build_caewm3_model
    from phantom.caewm3.config import Caewm3Config
    from phantom.caewm3.windows import Caewm3WindowSampler
    mc = Caewm3Config.from_dict(mcd)
    pm = build_caewm3_model(hw, paths, mc=mc, tiny=tiny, device=device, dtype=dtype, inference=True)
    pm.rf.set_action_norm(norm.mean["action"], norm.std["action"])
    sampler = Caewm3WindowSampler(
        hw, pm.bb, norm, dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(),
        event_cache=str(CP.stored_or_default(mc.event_cache, CP2.EVENT_CACHE)), use_crop=False,
        crop_px=mc.crop_px, crop_res=mc.crop_res, crop_lead_s=mc.crop_lead_s, bin_dt=mc.bin_dt,
        pad_transform=mc.pad_transform, seed=0,
        lat_cache=str(CP.stored_or_default(mcd.get("tactile_lat_cache"), CP.LAT_CACHE)),
        tac_cache=str(CP.TAC_CACHE), wrench_baseline_rows=wbr)
    return pm, sampler, mc
