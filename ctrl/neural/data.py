import random

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .config import EnvConfig
from .dynamics import deg2rad, step


def create_emulator_dataset_rollout(
    episodes: int,
    rollout_length: int,
    cfg: EnvConfig,
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
        theta0 = deg2rad(random.uniform(*cfg.theta0_range_deg))
        theta1 = deg2rad(random.uniform(*cfg.theta1_range_deg)) + theta0

        cur_directions = [-1000.0] * rollout_length
        rollout_dataset_inputs.append([x_c, y_c, theta0, theta1])

        for i in range(rollout_length):
            phi = deg2rad(random.uniform(min_phi_deg, max_phi_deg))
            cur_directions[i] = phi
            x_c, y_c, theta0, theta1 = step(x_c, y_c, theta0, theta1, phi, cfg)

        rollout_dataset_outputs.append([x_c, y_c, theta0, theta1])
        rollout_directions.append(cur_directions)

    rollout_directions_t = torch.tensor(rollout_directions, dtype=torch.float32)
    rollout_inputs_t = torch.tensor(rollout_dataset_inputs, dtype=torch.float32)
    rollout_outputs_t = torch.tensor(rollout_dataset_outputs, dtype=torch.float32)

    combined_inputs = torch.cat((rollout_directions_t, rollout_inputs_t), dim=1)
    return TensorDataset(combined_inputs, rollout_outputs_t)


def create_controller_dataset(episodes: int, cfg: EnvConfig) -> TensorDataset:
    dataset_inputs = []
    for _ in tqdm(range(episodes), desc="Creating controller dataset"):
        x_c = random.uniform(*cfg.env_x_range)
        y_c = random.uniform(*cfg.env_y_range)
        theta0 = deg2rad(random.uniform(*cfg.theta0_range_deg))
        theta1 = deg2rad(random.uniform(*cfg.theta1_range_deg)) + theta0
        dataset_inputs.append([x_c, y_c, theta0, theta1])

    dataset_inputs_t = torch.tensor(dataset_inputs, dtype=torch.float32)
    return TensorDataset(dataset_inputs_t)
