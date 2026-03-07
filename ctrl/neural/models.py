import torch
import torch.nn as nn


class ResidualEmulator(nn.Module):
    def __init__(self, hidden_size: int = 100):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x[..., 1:]


class CheatEmulator(nn.Module):
    # Same signature as ResidualEmulator:
    # input  [u, x, y, theta0, theta1]
    # output [x_next, y_next, theta0_next, theta1_next]
    # But NN only consumes [u, theta0, theta1].
    def __init__(self, hidden_size: int = 100):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x[..., 0:1]
        theta0 = x[..., 3:4]
        theta1 = x[..., 4:5]
        dyn_in = torch.cat((u, theta0, theta1), dim=-1)
        delta = self.block(dyn_in)
        state = x[..., 1:]
        return state + delta


class TruckController(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PhysicsTruckEmulator(nn.Module):
    # Deterministic one-step truck dynamics.
    # Input:  [phi, x, y, theta0, theta1]
    # Output: [x_next, y_next, theta0_next, theta1_next]
    def __init__(
        self,
        truck_speed: float = -0.1,
        wheelbase: float = 1.0,
        hitch_length: float = 4.0,
    ):
        super().__init__()
        self.v = float(truck_speed)
        self.l = float(wheelbase)
        self.d = float(hitch_length)

    def forward(self, em_input: torch.Tensor) -> torch.Tensor:
        phi = em_input[..., 0]
        x = em_input[..., 1]
        y = em_input[..., 2]
        theta0 = em_input[..., 3]
        theta1 = em_input[..., 4]

        x_next = x + self.v * torch.cos(theta0)
        y_next = y + self.v * torch.sin(theta0)
        theta0_next = theta0 + self.v / self.l * torch.tan(phi)
        theta1_next = theta1 + self.v / self.d * torch.sin(theta0 - theta1)
        return torch.stack((x_next, y_next, theta0_next, theta1_next), dim=-1)
