"""TCP -> scene-camera projection and the token-grid Gaussian used by GeoBridge.

Camera: configs/sim/waffles_camera_estimate.json (K and T_camera_cv_from_ur_base,
640x480, ~17 px RMSE; checked across tasks and dates). TCP positions are in the UR
base frame, the same frame the delta-EE actions live in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CAM_JSON = Path(__file__).resolve().parents[2] / "configs/sim/waffles_camera_estimate.json"


class Camera:
    def __init__(self, path: Path = CAM_JSON):
        cam = json.loads(Path(path).read_text())
        self.K = np.asarray(cam["K"], np.float64)
        self.T = np.asarray(cam["T_camera_cv_from_ur_base"], np.float64)
        self.wh = (640.0, 480.0)

    def uv_norm(self, p_base) -> tuple[np.ndarray, bool]:
        """(3,) base-frame point -> (u/W, v/H) and whether it is in view."""
        pc = self.T[:3, :3] @ np.asarray(p_base, np.float64) + self.T[:3, 3]
        if pc[2] <= 1e-3:
            return np.zeros(2, np.float32), False
        uv = (self.K @ pc)[:2] / pc[2]
        n = (uv / np.asarray(self.wh)).astype(np.float32)
        return n, bool(np.all(n >= 0.0) and np.all(n <= 1.0))


def tcp_gaussian(uv_B_2: torch.Tensor, valid_B: torch.Tensor, hp: int, wp: int,
                 sigma: float) -> torch.Tensor:
    """(B, hp, wp) Gaussian with peak 1 at the projected TCP, zero when not in view."""
    dev = uv_B_2.device
    cy = uv_B_2[:, 1].float() * hp - 0.5
    cx = uv_B_2[:, 0].float() * wp - 0.5
    yy = torch.arange(hp, device=dev, dtype=torch.float32).view(1, hp, 1)
    xx = torch.arange(wp, device=dev, dtype=torch.float32).view(1, 1, wp)
    g = torch.exp(-((yy - cy.view(-1, 1, 1)) ** 2 + (xx - cx.view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))
    return g * valid_B.float().view(-1, 1, 1)
