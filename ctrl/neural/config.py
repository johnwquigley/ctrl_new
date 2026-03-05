from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class EnvConfig:
    env_x_range: Tuple[float, float] = (-10.0, 40.0)
    env_y_range: Tuple[float, float] = (-15.0, 15.0)
    theta0_range_deg: Tuple[float, float] = (-180.0, 180.0)
    theta1_range_deg: Tuple[float, float] = (-180.0, 180.0)
    truck_speed: float = -1.0
    wheelbase: float = 1.0
    hitch_length: float = 4.0


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
