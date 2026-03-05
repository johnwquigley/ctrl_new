from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EnvConfig
from .losses import criterion_controller, criterion_emulator


def train_rollout(
    emulator: torch.nn.Module,
    rollout_length: int,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> float:
    emulator.train()
    average_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in tqdm(
            dataloader,
            desc=f"Training emulator rollout length {rollout_length}, epoch {epoch + 1}/{epochs}",
        ):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            actions = inputs[:, :rollout_length]
            current_states = inputs[:, rollout_length:]

            for i in range(rollout_length):
                current_action = actions[:, i : i + 1]
                step_input = torch.cat((current_action, current_states), dim=1)
                current_states = emulator(step_input)

            loss = criterion_emulator(current_states, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        average_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Rollout Training Loss: {average_loss:.6f}")

    return average_loss


@torch.no_grad()
def test_rollout(
    emulator: torch.nn.Module,
    rollout_length: int,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    emulator.eval()
    total_loss = 0.0

    for inputs, targets in tqdm(
        dataloader, desc=f"Testing emulator rollout length {rollout_length}"
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)

        actions = inputs[:, :rollout_length]
        current_states = inputs[:, rollout_length:]

        for i in range(rollout_length):
            current_action = actions[:, i : i + 1]
            step_input = torch.cat((current_action, current_states), dim=1)
            current_states = emulator(step_input)

        loss = criterion_emulator(current_states, targets)
        total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    print(f"Rollout Test Loss ({rollout_length} steps): {average_loss:.6f}")
    return average_loss


def train_controller(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_position: torch.Tensor,
    cfg: EnvConfig,
    device: torch.device,
    epochs: int = 10,
    rollout_steps: int = 40,
) -> float:
    average_loss = 0.0
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs in tqdm(
            dataloader, desc=f"Training Controller Epoch {epoch + 1}/{epochs}"
        ):
            initial_state = inputs[0].to(device)
            traj, actions = model(initial_state, steps=rollout_steps)
            loss = criterion_controller(actions, traj, target_position, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * initial_state.size(0)

        average_loss = total_loss / len(dataloader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs}, Controller Training Loss: {average_loss:.6f}"
        )

    return average_loss
