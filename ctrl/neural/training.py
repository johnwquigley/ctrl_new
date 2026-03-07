from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import controller_loss_terms, criterion_emulator


def _trailer_xy_batch(state: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    # Same geometry as ctrl.truck_data_gen.trailer_xy, vectorized for [B, 4] state.
    x = state[:, 0]
    theta1 = state[:, 3]
    trailer_x = x - cfg.hitch_length * torch.cos(theta1)
    trailer_y = state[:, 1] - cfg.hitch_length * torch.sin(theta1)
    return trailer_x, trailer_y


def _is_valid_batch(state: torch.Tensor, cfg: Any) -> torch.Tensor:
    jackknifed, in_box = _state_validity_terms(state, cfg)
    return (~jackknifed) & in_box


def _state_validity_terms(state: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    # state: [B, 4] -> [x, y, theta0, theta1]
    theta0 = state[:, 2]
    theta1 = state[:, 3]

    # Match truck_backer_upper jackknife logic: |wrapped(theta0 - theta1)| > 90 deg
    delta = torch.atan2(torch.sin(theta0 - theta1), torch.cos(theta0 - theta1))
    jackknifed = delta.abs() > (torch.pi / 2.0)

    # Tail-only in-box check using trailer axle position.
    trailer_x, trailer_y = _trailer_xy_batch(state, cfg)

    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    in_box = (
        (trailer_x >= xmin)
        & (trailer_x <= xmax)
        & (trailer_y >= ymin)
        & (trailer_y <= ymax)
    )

    return jackknifed, in_box


def _is_success_batch(state: torch.Tensor, cfg: Any) -> torch.Tensor:
    trailer_x, trailer_y = _trailer_xy_batch(state, cfg)
    success_radius = float(getattr(cfg, "controller_success_radius", 0.01))
    return (trailer_x.pow(2) + trailer_y.pow(2)) <= (success_radius**2)


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
    controller: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_position: torch.Tensor,
    cfg: Any,
    device: torch.device,
    emulator: torch.nn.Module,
    epochs: int = 10,
    max_rollout_steps: int = 400,
    on_batch_end: Optional[Callable[[dict[str, list[float]]], Optional[bool]]] = None,
    on_epoch_end: Optional[Callable[[dict[str, list[float]]], None]] = None,
) -> dict[str, list[float]]:
    average_loss = 0.0
    ckpt_dir = Path("ctrl/pth/neural")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "controller_latest.pth"

    history = {
        "batch_total": [],
        "batch_final_x": [],
        "batch_final_y": [],
        "batch_final_theta0": [],
        "batch_final_theta1": [],
        "batch_jackknife": [],
        "batch_final_dist_lt_0p1_ratio": [],
        "batch_final_dist_lt_1p0_ratio": [],
        "batch_rollout_steps": [],
        "batch_final_alive_ratio": [],
        "batch_stop_reason_jackknife_ratio": [],
        "batch_stop_reason_oob_ratio": [],
        "batch_stop_reason_success_ratio": [],
        "batch_stop_reason_timeout_ratio": [],
        "batch_index": [],
        "total": [],
        "final_x": [],
        "final_y": [],
        "final_theta0": [],
        "final_theta1": [],
        "jackknife": [],
    }

    emulator.eval()
    for p in emulator.parameters():
        p.requires_grad_(False)

    controller.train()

    should_stop = False
    for epoch in range(epochs):
        total_loss = 0.0
        total_weight = 0
        total_rollout_steps = 0.0
        total_final_alive = 0.0
        total_final_x = 0.0
        total_final_y = 0.0
        total_final_theta0 = 0.0
        total_final_theta1 = 0.0
        total_jackknife = 0.0
        for inputs in tqdm(
            dataloader,
            desc=f"Training Controller Epoch {epoch + 1}/{epochs}",
            leave=False,
        ):
            initial_state = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            initial_state = initial_state.to(device)
            batch_size = initial_state.size(0)

            current_state = initial_state
            traj = []
            actions = []
            alive = _is_valid_batch(current_state, cfg) & (~_is_success_batch(current_state, cfg))
            step = 0
            while step < max_rollout_steps and alive.any():
                raw_action = controller(current_state)
                # Keeps steering bounded in [-pi/2, pi/2] for stable rollouts.
                # action = torch.atan(raw_action)
                action = raw_action.clamp(-torch.pi / 4, torch.pi / 4)
                step_input = torch.cat((action, current_state), dim=-1)
                proposed_next_state = emulator(step_input)
                alive_f = alive.unsqueeze(-1)

                # Once a sample terminates, keep it parked at terminal state.
                next_state = torch.where(alive_f, proposed_next_state, current_state)
                action = torch.where(alive_f, action, torch.zeros_like(action))

                actions.append(action)
                traj.append(next_state)
                current_state = next_state
                alive = alive & _is_valid_batch(current_state, cfg) & (
                    ~_is_success_batch(current_state, cfg)
                )
                step += 1

            if not traj:
                continue

            traj = torch.stack(traj, dim=1)
            actions = torch.stack(actions, dim=1)
            terms = controller_loss_terms(actions, traj, target_position, cfg)
            loss = terms["total"]
            jackknifed_end, in_box_end = _state_validity_terms(current_state, cfg)
            success_end = _is_success_batch(current_state, cfg)
            trailer_x_end, trailer_y_end = _trailer_xy_batch(current_state, cfg)
            final_dist = torch.sqrt(
                (trailer_x_end - target_position[0]).pow(2)
                + (trailer_y_end - target_position[1]).pow(2)
            )
            final_dist_lt_0p1_ratio = (final_dist < 0.1).float().mean().item()
            final_dist_lt_1p0_ratio = (final_dist < 1.0).float().mean().item()
            terminated = ~alive
            timed_out = alive & (step >= max_rollout_steps)
            stop_jackknife = terminated & jackknifed_end
            stop_oob = terminated & (~jackknifed_end) & (~in_box_end)
            stop_success = terminated & (~jackknifed_end) & in_box_end & success_end
            stop_timeout = timed_out
            reason_denom = max(1, batch_size)
            stop_jackknife_ratio = stop_jackknife.float().sum().item() / reason_denom
            stop_oob_ratio = stop_oob.float().sum().item() / reason_denom
            stop_success_ratio = stop_success.float().sum().item() / reason_denom
            stop_timeout_ratio = stop_timeout.float().sum().item() / reason_denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history["batch_total"].append(loss.item() / max(1, batch_size))
            history["batch_final_x"].append(terms["final_x"].item() / max(1, batch_size))
            history["batch_final_y"].append(terms["final_y"].item() / max(1, batch_size))
            history["batch_final_theta0"].append(
                terms["final_theta0"].item() / max(1, batch_size)
            )
            history["batch_final_theta1"].append(
                terms["final_theta1"].item() / max(1, batch_size)
            )
            history["batch_jackknife"].append(
                terms["jackknife"].item() / max(1, batch_size)
            )
            history["batch_final_dist_lt_0p1_ratio"].append(final_dist_lt_0p1_ratio)
            history["batch_final_dist_lt_1p0_ratio"].append(final_dist_lt_1p0_ratio)
            history["batch_rollout_steps"].append(float(step))
            history["batch_final_alive_ratio"].append(alive.float().mean().item())
            history["batch_stop_reason_jackknife_ratio"].append(stop_jackknife_ratio)
            history["batch_stop_reason_oob_ratio"].append(stop_oob_ratio)
            history["batch_stop_reason_success_ratio"].append(stop_success_ratio)
            history["batch_stop_reason_timeout_ratio"].append(stop_timeout_ratio)
            history["batch_index"].append(len(history["batch_index"]) + 1)
            if on_batch_end is not None:
                stop_now = on_batch_end(history)
                if bool(stop_now):
                    should_stop = True
            total_loss += loss.item() * batch_size
            total_final_x += terms["final_x"].item() * batch_size
            total_final_y += terms["final_y"].item() * batch_size
            total_final_theta0 += terms["final_theta0"].item() * batch_size
            total_final_theta1 += terms["final_theta1"].item() * batch_size
            total_jackknife += terms["jackknife"].item() * batch_size
            total_weight += batch_size
            total_rollout_steps += step * batch_size
            total_final_alive += alive.float().mean().item() * batch_size
            if should_stop:
                break

        average_loss = total_loss / len(dataloader.dataset)
        avg_steps = total_rollout_steps / max(1, total_weight)
        final_alive_ratio = total_final_alive / max(1, total_weight)
        avg_final_x = total_final_x / len(dataloader.dataset)
        avg_final_y = total_final_y / len(dataloader.dataset)
        avg_final_theta0 = total_final_theta0 / len(dataloader.dataset)
        avg_final_theta1 = total_final_theta1 / len(dataloader.dataset)
        avg_jackknife = total_jackknife / len(dataloader.dataset)

        history["total"].append(average_loss)
        history["final_x"].append(avg_final_x)
        history["final_y"].append(avg_final_y)
        history["final_theta0"].append(avg_final_theta0)
        history["final_theta1"].append(avg_final_theta1)
        history["jackknife"].append(avg_jackknife)
        if on_epoch_end is not None:
            on_epoch_end(history)
        print(
            "Epoch "
            f"{epoch + 1}/{epochs}, "
            f"Controller Training Loss: {average_loss:.6f}, "
            f"Avg Rollout Steps: {avg_steps:.1f}, "
            f"Final Alive Ratio: {final_alive_ratio:.3f}"
        )
        epoch_ckpt_path = ckpt_dir / f"controller_epoch_{epoch + 1:03d}.pth"
        torch.save(controller.state_dict(), epoch_ckpt_path)
        torch.save(controller.state_dict(), latest_ckpt_path)
        print(f"saved controller ckpt: {epoch_ckpt_path}")
        if should_stop:
            break

    return history
