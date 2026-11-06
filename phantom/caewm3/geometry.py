"""Batched torch version of phantom/caewm/geometry.py's Camera.uv_norm, for the fingertip
anchors that layer 2 computes from the model's own trajectory estimate on the GPU."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from phantom.caewm.geometry import CAM_JSON


class TorchCamera(nn.Module):
    def __init__(self, path: Path = CAM_JSON):
        super().__init__()
        cam = json.loads(Path(path).read_text())
        T = torch.tensor(cam["T_camera_cv_from_ur_base"], dtype=torch.float64)
        self.register_buffer("K", torch.tensor(cam["K"], dtype=torch.float64), persistent=False)
        self.register_buffer("R", T[:3, :3].clone(), persistent=False)
        self.register_buffer("t", T[:3, 3].clone(), persistent=False)
        self.register_buffer("wh", torch.tensor([640.0, 480.0], dtype=torch.float64), persistent=False)

    def uv_norm(self, p_N_3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 3) base-frame points -> (N, 2) normalised (u/W, v/H) float32 and (N,) in view."""
        pc = p_N_3.double() @ self.R.t() + self.t
        front = pc[:, 2] > 1e-3
        uv = (pc @ self.K.t())[:, :2] / pc[:, 2:3].clamp_min(1e-3)
        n = uv / self.wh
        ok = front & (n >= 0).all(1) & (n <= 1).all(1)
        n = torch.where(front[:, None], n, torch.zeros_like(n))
        return n.float(), ok
