import math

import torch
import torch.nn as nn

from .config import EnvConfig


def criterion_emulator(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(pred[:, :2], target[:, :2]) + 10.0 * nn.MSELoss()(
        pred[:, 2:4], target[:, 2:4]
    )


def criterion_controller(
    actions: torch.Tensor,
    traj: torch.Tensor,
    target_pos: torch.Tensor,
    cfg: EnvConfig,
) -> torch.Tensor:
    x = traj[..., 0]
    y = traj[..., 1]
    theta0 = traj[..., 2]
    theta1 = traj[..., 3]

    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    soft_boundary = 3.0
    jack_thresh = math.pi / 2.5

    # Kept for notebook parity (currently unused directly).
    _ = actions

    region_pen = (
        torch.relu(x - xmax + soft_boundary).pow(2).sum()
        + torch.relu(-x + xmin + soft_boundary).pow(2).sum()
        + torch.relu(y - ymax + soft_boundary).pow(2).sum()
        + torch.relu(-y + ymin + soft_boundary).pow(2).sum()
    )
    jack_pen = torch.relu(torch.abs(theta0 - theta1) - jack_thresh).pow(2).sum()
    process_pen = region_pen + 10.0 * jack_pen

    final_dest_pen = (
        (x[..., -1] - target_pos[0]).pow(2).sum()
        + (y[..., -1] - target_pos[1]).pow(2).sum()
        + (theta0[..., -1] - target_pos[2]).pow(2).sum()
        + (theta1[..., -1] - target_pos[3]).pow(2).sum()
    )
    return process_pen + final_dest_pen
