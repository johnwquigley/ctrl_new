import torch
import torch.nn as nn

class Emulator(nn.Module):
    # input  [u, x, y, theta0, theta1]
    # output [x_next, y_next, theta0_next, theta1_next]
    # NN only consumes [u, theta0, theta1], and use residual connection
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


class PhysicsTruckModel(nn.Module):
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
        phi, x, y, theta0, theta1 = (em_input[..., i] for i in range(5))
        x_next = x + self.v * torch.cos(theta0)
        y_next = y + self.v * torch.sin(theta0)
        theta0_next = theta0 + self.v / self.l * torch.tan(phi)
        theta1_next = theta1 + self.v / self.d * torch.sin(theta0 - theta1)
        return torch.stack((x_next, y_next, theta0_next, theta1_next), dim=-1)


class TruckController(nn.Module):
    def __init__(self, hidden_size: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)