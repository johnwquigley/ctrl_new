from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import controller_loss_terms, criterion_emulator

BATCH_KEYS = (
    "batch_total",
    "batch_final_x",
    "batch_final_y",
    "batch_final_theta0",
    "batch_final_theta1",
    "batch_jackknife",
    "batch_final_dist_lt_0p1_ratio",
    "batch_final_dist_lt_1p0_ratio",
    "batch_rollout_steps",
    "batch_final_alive_ratio",
    "batch_stop_reason_jackknife_ratio",
    "batch_stop_reason_oob_ratio",
    "batch_stop_reason_success_ratio",
    "batch_stop_reason_timeout_ratio",
    "batch_index",
)
EPOCH_BATCH_KEY_MAP = {
    "batch_total": "total",
    "batch_final_x": "final_x",
    "batch_final_y": "final_y",
    "batch_final_theta0": "final_theta0",
    "batch_final_theta1": "final_theta1",
    "batch_jackknife": "jackknife",
    "batch_final_dist_lt_0p1_ratio": "final_dist_lt_0p1_ratio",
    "batch_final_dist_lt_1p0_ratio": "final_dist_lt_1p0_ratio",
    "batch_rollout_steps": "rollout_steps",
    "batch_final_alive_ratio": "final_alive_ratio",
    "batch_stop_reason_jackknife_ratio": "stop_reason_jackknife_ratio",
    "batch_stop_reason_oob_ratio": "stop_reason_oob_ratio",
    "batch_stop_reason_success_ratio": "stop_reason_success_ratio",
    "batch_stop_reason_timeout_ratio": "stop_reason_timeout_ratio",
}
EPOCH_KEYS = tuple(EPOCH_BATCH_KEY_MAP.values())


def _new_history() -> dict[str, list[float]]:
    """Allocate per-batch and per-epoch metric buffers."""
    return {k: [] for k in (*BATCH_KEYS, *EPOCH_KEYS)}


def _new_epoch_history() -> dict[str, float]:
    """Allocate per-epoch accumulators used during controller training."""
    return {key: 0.0 for key in EPOCH_KEYS}


def _extract_initial_state(batch: Any, device: torch.device) -> torch.Tensor:
    """Normalize dataloader batches to the initial-state tensor."""
    initial_state = batch[0] if isinstance(batch, (tuple, list)) else batch
    return initial_state.to(device)


def _trailer_xy_batch(state: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Trailer axle position for a batch of [x, y, theta0, theta1]."""
    x = state[:, 0]
    theta1 = state[:, 3]
    return x - cfg.hitch_length * torch.cos(theta1), state[:, 1] - cfg.hitch_length * torch.sin(theta1)


def _state_status_terms(
    state: torch.Tensor, cfg: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return jackknife, in-box, success, and active masks."""
    theta0 = state[:, 2]
    theta1 = state[:, 3]
    delta = torch.atan2(torch.sin(theta0 - theta1), torch.cos(theta0 - theta1))
    jackknifed = delta.abs() > (torch.pi / 2.0)

    trailer_x, trailer_y = _trailer_xy_batch(state, cfg)
    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    in_box = (trailer_x >= xmin) & (trailer_x <= xmax) & (trailer_y >= ymin) & (trailer_y <= ymax)

    success_radius = float(cfg.controller_success_radius)
    success = (trailer_x.pow(2) + trailer_y.pow(2)) <= (success_radius**2)
    return jackknifed, in_box, success, (~jackknifed) & in_box & (~success)


def _clamp_controller_action(action: torch.Tensor) -> torch.Tensor:
    """Project steering commands into the controller action bounds."""
    return action.clamp(-torch.pi / 4, torch.pi / 4)


def _rollout_controller_batch(
    controller: torch.nn.Module,
    emulator: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: Any,
    max_rollout_steps: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Run one closed-loop rollout batch and return trajectory, actions, final state, alive mask, and steps."""
    current_state = initial_state
    traj_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    _, _, _, alive = _state_status_terms(current_state, cfg)
    step = 0

    while step < max_rollout_steps and alive.any():
        action = _clamp_controller_action(controller(current_state))
        step_input = torch.cat((action, current_state), dim=-1)
        proposed_next_state = emulator(step_input)
        alive_f = alive.unsqueeze(-1)

        current_state = torch.where(alive_f, proposed_next_state, current_state)
        action = torch.where(alive_f, action, torch.zeros_like(action))

        action_steps.append(action)
        traj_steps.append(current_state)
        _, _, _, active = _state_status_terms(current_state, cfg)
        alive = alive & active
        step += 1

    if not traj_steps:
        return None, None, current_state, alive, step

    traj = torch.stack(traj_steps, dim=1)
    actions = torch.stack(action_steps, dim=1)
    return traj, actions, current_state, alive, step


def _compute_batch_history_values(
    terms: dict[str, torch.Tensor],
    final_state: torch.Tensor,
    alive: torch.Tensor,
    target_position: torch.Tensor,
    cfg: Any,
    step: int,
    max_rollout_steps: int,
    batch_size: int,
) -> dict[str, float]:
    """Compute the batch history snapshot expected by notebook callbacks."""
    jackknifed_end, in_box_end, success_end, _ = _state_status_terms(final_state, cfg)
    trailer_x_end, trailer_y_end = _trailer_xy_batch(final_state, cfg)
    final_dist = torch.sqrt(
        (trailer_x_end - target_position[0]).pow(2)
        + (trailer_y_end - target_position[1]).pow(2)
    )

    terminated = ~alive
    timed_out = alive & (step >= max_rollout_steps)
    stop_jackknife = terminated & jackknifed_end
    stop_success = terminated & (~jackknifed_end) & success_end
    stop_oob = terminated & (~jackknifed_end) & (~success_end) & (~in_box_end)
    stop_timeout = timed_out
    denom = max(1, batch_size)

    return {
        "batch_total": terms["total"].item() / denom,
        "batch_final_x": terms["final_x"].item() / denom,
        "batch_final_y": terms["final_y"].item() / denom,
        "batch_final_theta0": terms["final_theta0"].item() / denom,
        "batch_final_theta1": terms["final_theta1"].item() / denom,
        "batch_jackknife": terms["jackknife"].item() / denom,
        "batch_final_dist_lt_0p1_ratio": (final_dist < 0.1).float().mean().item(),
        "batch_final_dist_lt_1p0_ratio": (final_dist < 1.0).float().mean().item(),
        "batch_rollout_steps": float(step),
        "batch_final_alive_ratio": alive.float().mean().item(),
        "batch_stop_reason_jackknife_ratio": stop_jackknife.float().sum().item() / denom,
        "batch_stop_reason_oob_ratio": stop_oob.float().sum().item() / denom,
        "batch_stop_reason_success_ratio": stop_success.float().sum().item() / denom,
        "batch_stop_reason_timeout_ratio": stop_timeout.float().sum().item() / denom,
    }


def _append_batch_history(history: dict[str, list[float]], batch_values: dict[str, float]) -> None:
    """Append one batch snapshot while preserving notebook-visible history keys."""
    for key in BATCH_KEYS:
        if key == "batch_index":
            history[key].append(len(history[key]) + 1)
        else:
            history[key].append(batch_values[key])


def _update_epoch_history(
    epoch_history: dict[str, float],
    batch_values: dict[str, float],
    batch_size: int,
) -> None:
    """Accumulate per-epoch controller metrics from one batch."""
    for batch_key, epoch_key in EPOCH_BATCH_KEY_MAP.items():
        epoch_history[epoch_key] += batch_values[batch_key] * batch_size


def _append_epoch_history(
    history: dict[str, list[float]],
    epoch_history: dict[str, float],
    epoch_weight: int,
) -> None:
    """Append epoch aggregates with the existing public history layout."""
    denom = max(1, epoch_weight)
    for key in EPOCH_KEYS:
        history[key].append(epoch_history[key] / denom)


def _format_controller_log(label: str, metrics: dict[str, float]) -> str:
    """Format controller metrics in the same two-line style for batch and epoch summaries."""
    return (
        f"[{label}] "
        f"jackknife={metrics['stop_reason_jackknife_ratio']:.3f} "
        f"oob={metrics['stop_reason_oob_ratio']:.3f} "
        f"success={metrics['stop_reason_success_ratio']:.3f} "
        f"alive={metrics['final_alive_ratio']:.3f} "
        "\n"
        f"loss={metrics['total']:.4f} "
        f"final_x={metrics['final_x']:.4f} "
        f"final_y={metrics['final_y']:.4f} "
        f"final_theta0={metrics['final_theta0']:.4f} "
        f"final_theta1={metrics['final_theta1']:.4f} "
        f"jackknife_pen={metrics['jackknife']:.4f} "
        f"steps={metrics['rollout_steps']:.1f}"
    )


def train_rollout(
    emulator: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> float:
    """Train one-step emulator on [u, state] -> next_state."""
    emulator.train()
    average_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Training emulator one-step, epoch {epoch + 1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion_emulator(emulator(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        average_loss = total_loss / len(dataloader.dataset)
        print(f"[Rollout Train {epoch + 1}/{epochs}] loss={average_loss:.6f}")

    return average_loss


@torch.no_grad()
def test_rollout(
    emulator: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate one-step emulator loss."""
    emulator.eval()
    total_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc="Testing emulator one-step"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        total_loss += criterion_emulator(emulator(inputs), targets).item() * inputs.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    print(f"[Rollout Test] loss={average_loss:.6f}")
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
) -> dict[str, list[float]] | int:
    """Train controller in closed-loop against frozen emulator dynamics."""
    ckpt_dir = Path("ctrl/pth/neural")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "controller_latest.pth"

    history = _new_history()

    emulator.eval()
    for p in emulator.parameters():
        p.requires_grad_(False)

    controller.train()


    for epoch in range(epochs):
        epoch_history = _new_epoch_history()
        epoch_weight = 0
        progress = tqdm(dataloader, desc=f"Training Controller Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            # Get the starting point x0
            initial_state = _extract_initial_state(batch, device)
            batch_size = initial_state.size(0)

            # Rollout according to the current controller, and collect the states of each trial in the batch
            traj, actions, final_state, alive, step = _rollout_controller_batch(
                controller=controller,
                emulator=emulator,
                initial_state=initial_state,
                cfg=cfg,
                max_rollout_steps=max_rollout_steps,
            )
            if traj is None or actions is None:
                continue # Sometimes its a dud, then go again

            # Get a loss dictionary based on the actions we took, and the trajectories we followed
            terms = controller_loss_terms(actions, traj, target_position, cfg)
            loss = terms["total"]
            # Compute some batch-level statistics from this rollout
            batch_values = _compute_batch_history_values(
                terms=terms,
                final_state=final_state,
                alive=alive,
                target_position=target_position,
                cfg=cfg,
                step=step,
                max_rollout_steps=max_rollout_steps,
                batch_size=batch_size,
            )

            # Gradient updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Do some batch-level logging
            _append_batch_history(history, batch_values)
            _update_epoch_history(epoch_history, batch_values, batch_size)
            epoch_weight += batch_size

            if on_batch_end is not None:
                if on_batch_end(history):
                    return -1  # Early exit signal for notebook callbacks.

        # Do some epoch-level logging
        _append_epoch_history(
            history=history,
            epoch_history=epoch_history,
            epoch_weight=epoch_weight,
        )

        if on_epoch_end is not None:
            on_epoch_end(history)

        epoch_metrics = {key: history[key][-1] for key in EPOCH_KEYS}
        print(_format_controller_log(f"Epoch {epoch + 1}/{epochs}", epoch_metrics))

        # Save epoch checkpoint and update latest checkpoint after each epoch.
        epoch_ckpt_path = ckpt_dir / f"controller_epoch_{epoch + 1:03d}.pth"
        torch.save(controller.state_dict(), epoch_ckpt_path)
        torch.save(controller.state_dict(), latest_ckpt_path)
        print(f"[Checkpoint] saved {epoch_ckpt_path}")
    return history
