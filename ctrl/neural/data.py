import math
import random
from typing import Any

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def _deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def _step_scalar(
    x_c: float, y_c: float, theta0: float, theta1: float, phi: float, cfg: Any
) -> tuple[float, float, float, float]:
    """One-step truck dynamics in scalar form."""
    return (
        x_c + cfg.truck_speed * math.cos(theta0),
        y_c + cfg.truck_speed * math.sin(theta0),
        theta0 + cfg.truck_speed / cfg.wheelbase * math.tan(phi),
        theta1 + cfg.truck_speed / cfg.hitch_length * math.sin(theta0 - theta1),
    )


def create_emulator_dataset_rollout(
    episodes: int,
    cfg: Any,
    min_phi_deg: float = -80.0,
    max_phi_deg: float = 80.0,
) -> TensorDataset:
    """Sample one-step emulator pairs: [u, state] -> next_state."""
    inputs = []
    outputs = []
    for _ in tqdm(range(episodes), desc="Creating emulator rollout dataset"):
        x_c = random.uniform(*cfg.env_x_range)
        y_c = random.uniform(*cfg.env_y_range)
        theta0 = _deg2rad(random.uniform(*cfg.theta0_range_deg))
        theta1 = theta0 + _deg2rad(random.uniform(*cfg.theta1_range_deg))
        phi = _deg2rad(random.uniform(min_phi_deg, max_phi_deg))
        inputs.append([phi, x_c, y_c, theta0, theta1])
        outputs.append(_step_scalar(x_c, y_c, theta0, theta1, phi, cfg))

    return TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(outputs, dtype=torch.float32),
    )


def create_controller_dataset(episodes: int, cfg: Any) -> TensorDataset:
    """Sample controller initial states from cfg controller ranges."""
    x_range = cfg.controller_x_init_final_range
    y_range = cfg.controller_y_init_final_range
    theta0_range_deg = cfg.controller_theta0_init_final_range_deg
    delta_range_deg = cfg.controller_delta_final_range_deg

    inputs = []
    for _ in tqdm(range(episodes), desc="Creating controller dataset"):
        x_c = random.uniform(*x_range)
        y_c = random.uniform(*y_range)
        theta0 = _deg2rad(random.uniform(*theta0_range_deg))
        theta1 = theta0 + _deg2rad(random.uniform(*delta_range_deg))
        inputs.append([x_c, y_c, theta0, theta1])

    return TensorDataset(torch.tensor(inputs, dtype=torch.float32))
