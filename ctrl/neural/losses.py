from typing import Any

import torch
import torch.nn as nn

CONTROLLER_BOUNDARY_MARGIN = 2.0
CONTROLLER_BOUNDARY_WEIGHT = 0.0


def _trailer_xy_from_traj(traj: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Trailer axle position for trajectories with last dim [x, y, theta0, theta1]."""
    x = traj[..., 0]
    theta1 = traj[..., 3]
    return x - cfg.hitch_length * torch.cos(theta1), traj[..., 1] - cfg.hitch_length * torch.sin(theta1)


def criterion_emulator(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted one-step emulator loss (angles weighted higher)."""
    mse = nn.MSELoss()
    return mse(pred[:, :2], target[:, :2]) + 10.0 * mse(pred[:, 2:4], target[:, 2:4])


def controller_loss_terms(
    actions: torch.Tensor,
    traj: torch.Tensor,
    target_pos: torch.Tensor,
    cfg: Any,
) -> dict[str, torch.Tensor]:
    """Compute controller optimization terms from rollout trajectory."""
    theta0 = traj[..., 2]
    theta1 = traj[..., 3]
    trailer_x, trailer_y = _trailer_xy_from_traj(traj, cfg)
    wrapped_delta = torch.atan2(torch.sin(theta0 - theta1), torch.cos(theta0 - theta1)).abs()

    # Kept for API parity; currently not used in total objective.
    _ = actions

    final_x_pen = (trailer_x[..., -1] - target_pos[0]).pow(2).sum()
    final_y_pen = (trailer_y[..., -1] - target_pos[1]).pow(2).sum()
    final_theta0_pen = (theta0[..., -1]).pow(2).sum()
    final_theta1_pen = (theta1[..., -1]).pow(2).sum()

    delta_thresh = torch.deg2rad(torch.tensor(88.0, device=traj.device, dtype=traj.dtype))
    jackknife_pen = torch.where(
        wrapped_delta >= delta_thresh,
        (wrapped_delta - delta_thresh).pow(2),
        torch.zeros_like(wrapped_delta),
    ).sum()

    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    boundary_pen = CONTROLLER_BOUNDARY_WEIGHT * (
        torch.relu((xmin + CONTROLLER_BOUNDARY_MARGIN) - trailer_x).pow(2)
        + torch.relu(trailer_x - (xmax - CONTROLLER_BOUNDARY_MARGIN)).pow(2)
        + torch.relu((ymin + CONTROLLER_BOUNDARY_MARGIN) - trailer_y).pow(2)
        + torch.relu(trailer_y - (ymax - CONTROLLER_BOUNDARY_MARGIN)).pow(2)
    ).mean()

    total_pen = final_x_pen + final_y_pen + final_theta0_pen + final_theta1_pen + jackknife_pen
    return {
        "total": total_pen,
        "final_x": final_x_pen,
        "final_y": final_y_pen,
        "final_theta0": final_theta0_pen,
        "final_theta1": final_theta1_pen,
        "jackknife": jackknife_pen,
        "boundary": boundary_pen,
    }


def criterion_controller(
    actions: torch.Tensor,
    traj: torch.Tensor,
    target_pos: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Controller scalar objective used for optimization."""
    return controller_loss_terms(actions, traj, target_pos, cfg)["total"]
