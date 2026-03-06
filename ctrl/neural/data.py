import random
import math
from typing import Any

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _step_scalar(
    x_c: float, y_c: float, theta0: float, theta1: float, phi: float, cfg: Any
) -> tuple[float, float, float, float]:
    x_c_new = x_c + cfg.truck_speed * math.cos(theta0)
    y_c_new = y_c + cfg.truck_speed * math.sin(theta0)
    theta0_new = theta0 + cfg.truck_speed / cfg.wheelbase * math.tan(phi)
    theta1_new = theta1 + cfg.truck_speed / cfg.hitch_length * math.sin(theta0 - theta1)
    return x_c_new, y_c_new, theta0_new, theta1_new


def create_emulator_dataset_rollout(
    episodes: int,
    rollout_length: int,
    cfg: Any,
    min_phi_deg: float = -80.0,
    max_phi_deg: float = 80.0,
) -> TensorDataset:
    rollout_env_x_range = (
        cfg.env_x_range[0] + rollout_length,
        cfg.env_x_range[1] - rollout_length,
    )
    rollout_env_y_range = (
        cfg.env_y_range[0] + rollout_length,
        cfg.env_y_range[1] - rollout_length,
    )

    rollout_dataset_inputs = []
    rollout_dataset_outputs = []
    rollout_directions = []

    for _ in tqdm(range(episodes), desc="Creating emulator rollout dataset"):
        x_c = random.uniform(*rollout_env_x_range)
        y_c = random.uniform(*rollout_env_y_range)
        theta0 = _deg2rad(random.uniform(*cfg.theta0_range_deg))
        theta1 = _deg2rad(random.uniform(*cfg.theta1_range_deg)) + theta0

        cur_directions = [-1000.0] * rollout_length
        rollout_dataset_inputs.append([x_c, y_c, theta0, theta1])

        for i in range(rollout_length):
            phi = _deg2rad(random.uniform(min_phi_deg, max_phi_deg))
            cur_directions[i] = phi
            x_c, y_c, theta0, theta1 = _step_scalar(x_c, y_c, theta0, theta1, phi, cfg)

        rollout_dataset_outputs.append([x_c, y_c, theta0, theta1])
        rollout_directions.append(cur_directions)

    rollout_directions_t = torch.tensor(rollout_directions, dtype=torch.float32)
    rollout_inputs_t = torch.tensor(rollout_dataset_inputs, dtype=torch.float32)
    rollout_outputs_t = torch.tensor(rollout_dataset_outputs, dtype=torch.float32)

    combined_inputs = torch.cat((rollout_directions_t, rollout_inputs_t), dim=1)
    return TensorDataset(combined_inputs, rollout_outputs_t)


def create_controller_dataset(
    episodes: int, cfg: Any, difficulty: float = 1.0
) -> TensorDataset:
    def _scaled_range(bounds, difficulty: float):
        lo, hi = bounds
        d = max(0.0, min(1.0, float(difficulty)))
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * d
        return (center - half, center + half)

    x_range = getattr(cfg, "controller_x_init_range", cfg.env_x_range)
    y_range = getattr(cfg, "controller_y_init_range", cfg.env_y_range)
    theta0_range_deg = _scaled_range(cfg.theta0_range_deg, difficulty)
    # Keep initial articulation away from jackknife: |theta1 - theta0| <= 45 deg.
    theta1_diff_range_deg = (-10.0, 10.0)

    dataset_inputs = []
    desc = f"Creating controller dataset (difficulty={difficulty:.2f})"
    for _ in tqdm(range(episodes), desc=desc):
        x_c = random.uniform(*x_range)
        y_c = random.uniform(*y_range)
        theta0 = _deg2rad(random.uniform(*theta0_range_deg))
        theta1 = _deg2rad(random.uniform(*theta1_diff_range_deg)) + theta0
        dataset_inputs.append([x_c, y_c, theta0, theta1])

    dataset_inputs_t = torch.tensor(dataset_inputs, dtype=torch.float32)
    return TensorDataset(dataset_inputs_t)
