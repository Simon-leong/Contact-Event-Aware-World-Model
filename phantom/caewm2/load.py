"""Build a CAEWM model (V1 'caewm' or V2 'caewm2') and its window sampler from a
checkpoint payload, for the evaluation tools."""
from __future__ import annotations

import phantom.caewm.paths as CP

CAEWM_ARCHS = ("caewm", "caewm2")


def build_from_payload(payload, hw, paths, norm, *, device, dtype, tiny=False, wbr=0):
    mcd = payload["configs"]["model"]
    arch = mcd.get("arch")
    common = dict(seed=0, lat_cache=str(CP.stored_or_default(mcd.get("tactile_lat_cache"), CP.LAT_CACHE)),
                  tac_cache=str(CP.TAC_CACHE), wrench_baseline_rows=wbr)
    if arch == "caewm":
        from phantom.caewm.builder import build_caewm_model
        from phantom.caewm.config import CaewmConfig
        from phantom.caewm.windows import CaewmWindowSampler
        mc = CaewmConfig.from_dict(mcd)
        pm = build_caewm_model(hw, paths, mc=mc, tiny=tiny, device=device, dtype=dtype, inference=True)
        sampler = CaewmWindowSampler(hw, pm.bb, norm, dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(), **common)
        return pm, sampler, mc
    assert arch == "caewm2", arch
    import phantom.caewm2.paths as CP2
    from phantom.caewm2.builder import build_caewm2_model
    from phantom.caewm2.config import Caewm2Config
    from phantom.caewm2.windows import Caewm2WindowSampler
    mc = Caewm2Config.from_dict(mcd)
    pm = build_caewm2_model(hw, paths, mc=mc, tiny=tiny, device=device, dtype=dtype, inference=True)
    sampler = Caewm2WindowSampler(
        hw, pm.bb, norm, dz_std=pm.rf.tac_ae.dz_std.cpu().numpy(),
        event_cache=str(CP.stored_or_default(mc.event_cache, CP2.EVENT_CACHE)), use_crop=mc.use_crop,
        crop_px=mc.crop_px, crop_res=mc.crop_res, crop_lead_s=mc.crop_lead_s, bin_dt=mc.bin_dt,
        pad_transform=mc.pad_transform, **common)
    return pm, sampler, mc
