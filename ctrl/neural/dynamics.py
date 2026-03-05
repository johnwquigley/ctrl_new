import math

import torch

from .config import EnvConfig


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def step(x_c: float, y_c: float, theta0: float, theta1: float, phi: float, cfg: EnvConfig):
    x_c_new = x_c + cfg.truck_speed * math.cos(theta0)
    y_c_new = y_c + cfg.truck_speed * math.sin(theta0)
    theta0_new = theta0 + cfg.truck_speed / cfg.wheelbase * math.tan(phi)
    theta1_new = theta1 + cfg.truck_speed / cfg.hitch_length * math.sin(theta0 - theta1)
    return x_c_new, y_c_new, theta0_new, theta1_new


def step_batch_for_traj(em_input: torch.Tensor, cfg: EnvConfig) -> torch.Tensor:
    # Input shape: (..., 5) with [phi, x, y, theta0, theta1]
    phi_batch = em_input[..., 0]
    x_c = em_input[..., 1]
    y_c = em_input[..., 2]
    theta0 = em_input[..., 3]
    theta1 = em_input[..., 4]

    x_c_new = x_c + cfg.truck_speed * torch.cos(theta0)
    y_c_new = y_c + cfg.truck_speed * torch.sin(theta0)
    theta0_new = theta0 + cfg.truck_speed / cfg.wheelbase * torch.tan(phi_batch)
    theta1_new = theta1 + cfg.truck_speed / cfg.hitch_length * torch.sin(theta0 - theta1)
    return torch.stack((x_c_new, y_c_new, theta0_new, theta1_new), dim=-1)
