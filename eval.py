import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
from ctrl.neural.models import TruckController


@dataclass
class EvalConfig:
    env_x_range: Tuple[float, float] = (0.0, 40.0)
    env_y_range: Tuple[float, float] = (-15.0, 15.0)
    truck_speed: float = -0.1
    wheelbase: float = 1.0
    hitch_length: float = 4.0
    success_radius: float = 0.01
    max_steps: int = 400


def create_train_configs_tbu(
    x_cab_range: Tuple[float, float] = (10.0, 35.0),
    y_cab_range_abs: Tuple[float, float] = (2.0, 7.0),
    cab_angle_range_abs: Tuple[float, float] = (10.0, 180.0),
    cab_trailer_angle_diff_range_abs: Tuple[float, float] = (10.0, 45.0),
    num_lessons: int = 10,
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    n = num_lessons - 1
    configs: Dict[int, Dict[str, Tuple[float, float]]] = {}
    x_first, x_final = x_cab_range
    y_first, y_final = y_cab_range_abs
    th0_first, th0_final = cab_angle_range_abs
    dth_first, dth_final = cab_trailer_angle_diff_range_abs
    x_lower = x_first
    denom = max(1, n - 1)
    for i in range(1, n + 1):
        t = (i - 1) / float(denom)
        x_upper = x_first + (x_final - x_first) * t
        y_upper = y_first + (y_final - y_first) * t
        th0_upper = th0_first + (th0_final - th0_first) * t
        dth_upper = dth_first + (dth_final - dth_first) * t
        configs[i] = {
            "x_range": (x_lower, x_upper),
            "y_range": (-y_upper, y_upper),
            "theta0_range_deg": (-th0_upper, th0_upper),
            "delta_range_deg": (-dth_upper, dth_upper),
        }
        x_lower = x_upper
    configs[n + 1] = {
        "x_range": (x_first, x_upper),
        "y_range": (-y_upper, y_upper),
        "theta0_range_deg": (-th0_upper, th0_upper),
        "delta_range_deg": (-dth_upper, dth_upper),
    }
    return configs


def trailer_xy(state: torch.Tensor, cfg: EvalConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    x = state[..., 0]
    y = state[..., 1]
    theta1 = state[..., 3]
    tx = x - cfg.hitch_length * torch.cos(theta1)
    ty = y - cfg.hitch_length * torch.sin(theta1)
    return tx, ty


def is_jackknifed(state: torch.Tensor) -> torch.Tensor:
    theta0 = state[..., 2]
    theta1 = state[..., 3]
    delta = torch.atan2(torch.sin(theta0 - theta1), torch.cos(theta0 - theta1))
    return delta.abs() > (torch.pi / 2.0)


def in_box_tail(state: torch.Tensor, cfg: EvalConfig) -> torch.Tensor:
    tx, ty = trailer_xy(state, cfg)
    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    return (tx >= xmin) & (tx <= xmax) & (ty >= ymin) & (ty <= ymax)


def is_success(state: torch.Tensor, cfg: EvalConfig) -> torch.Tensor:
    tx, ty = trailer_xy(state, cfg)
    return (tx.pow(2) + ty.pow(2)) <= (cfg.success_radius**2)


def final_dist(state: torch.Tensor, cfg: EvalConfig) -> torch.Tensor:
    tx, ty = trailer_xy(state, cfg)
    return torch.sqrt(tx.pow(2) + ty.pow(2))


def step_physics(state: torch.Tensor, phi: torch.Tensor, cfg: EvalConfig) -> torch.Tensor:
    x = state[..., 0]
    y = state[..., 1]
    theta0 = state[..., 2]
    theta1 = state[..., 3]
    phi = phi.reshape(-1)
    v = cfg.truck_speed
    l = cfg.wheelbase
    d = cfg.hitch_length
    x_next = x + v * torch.cos(theta0)
    y_next = y + v * torch.sin(theta0)
    theta0_next = theta0 + v / l * torch.tan(phi)
    theta1_next = theta1 + v / d * torch.sin(theta0 - theta1)
    return torch.stack((x_next, y_next, theta0_next, theta1_next), dim=-1)


def sample_initial_state(stage_cfg: Dict[str, Tuple[float, float]], cfg: EvalConfig) -> torch.Tensor:
    while True:
        x = random.uniform(*stage_cfg["x_range"])
        y = random.uniform(*stage_cfg["y_range"])
        theta0 = math.radians(random.uniform(*stage_cfg["theta0_range_deg"]))
        delta = math.radians(random.uniform(*stage_cfg["delta_range_deg"]))
        theta1 = theta0 + delta
        s = torch.tensor([x, y, theta0, theta1], dtype=torch.float32)
        if (not bool(is_jackknifed(s))) and bool(in_box_tail(s, cfg)):
            return s


def sample_initial_phi_rad() -> float:
    # Approximate truck_backer_upper truncated normal steering init.
    while True:
        deg = random.gauss(0.0, 35.0)
        if -70.0 <= deg <= 70.0:
            return math.radians(deg)


def load_controller(
    path: str, device: torch.device, inference_mode: Literal["student", "me"]
) -> nn.Module:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        if inference_mode == "student":
            model = nn.Sequential(
                nn.Linear(5, 100),
                nn.GELU(),
                nn.Linear(100, 100),
                nn.GELU(),
                nn.Linear(100, 1),
            )
        elif inference_mode == "me":
            model = TruckController(state_dim=4)
        else:
            raise ValueError(f"Unsupported inference_mode={inference_mode}")
        model.load_state_dict(obj)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    model.to(device)
    model.eval()
    return model


def evaluate_stage(
    model: nn.Module,
    stage_cfg: Dict[str, Tuple[float, float]],
    cfg: EvalConfig,
    num_samples: int,
    device: torch.device,
    inference_mode: Literal["student", "me"],
) -> Dict[str, float]:
    steps_list = []
    dists = []
    stop_jack = 0
    stop_oob = 0
    stop_success = 0
    stop_timeout = 0
    phi_clip = math.radians(45.0)
    with torch.no_grad():
        for _ in range(num_samples):
            state = sample_initial_state(stage_cfg, cfg).to(device)
            phi_prev = torch.tensor([sample_initial_phi_rad()], dtype=torch.float32, device=device)
            steps = 0
            while steps < cfg.max_steps:
                alive = (not bool(is_jackknifed(state))) and bool(in_box_tail(state, cfg)) and (not bool(is_success(state, cfg)))
                if not alive:
                    break
                if inference_mode == "student":
                    ctrl_in = torch.cat((phi_prev, state), dim=0).unsqueeze(0)  # [1, 5]
                elif inference_mode == "me":
                    ctrl_in = state.unsqueeze(0)  # [1, 4]
                else:
                    raise ValueError(f"Unsupported inference_mode={inference_mode}")
                phi_next = model(ctrl_in).reshape(-1)[0]
                if inference_mode == "student":
                    # Student behavior: apply previous action, then shift in predicted action.
                    phi_apply = phi_prev
                elif inference_mode == "me":
                    # behavior: apply predicted action in the same step.
                    phi_next = torch.clamp(phi_next, -phi_clip, phi_clip)
                    phi_apply = phi_next.reshape(1)
                state = step_physics(state.unsqueeze(0), phi_apply, cfg).squeeze(0)
                phi_prev = phi_next.reshape(1)
                steps += 1

            jk = bool(is_jackknifed(state))
            ib = bool(in_box_tail(state, cfg))
            sc = bool(is_success(state, cfg))
            if steps >= cfg.max_steps and (not jk) and ib and (not sc):
                stop_timeout += 1
            elif jk:
                stop_jack += 1
            elif not ib:
                stop_oob += 1
            elif sc:
                stop_success += 1
            else:
                stop_timeout += 1

            steps_list.append(steps)
            dists.append(float(final_dist(state, cfg).item()))

    d = torch.tensor(dists, dtype=torch.float32)
    n = float(max(1, num_samples))
    return {
        "samples": float(num_samples),
        "avg_steps": float(sum(steps_list) / max(1, len(steps_list))),
        "avg_final_dist": float(d.mean().item()),
        "final_dist_lt_0p1": float((d < 0.1).float().mean().item()),
        "final_dist_lt_1p0": float((d < 1.0).float().mean().item()),
        "stop_jackknife": stop_jack / n,
        "stop_oob": stop_oob / n,
        "stop_success": stop_success / n,
        "stop_timeout": stop_timeout / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student controller on TBU curriculum stages.")
    parser.add_argument("--checkpoint", type=str, default="student_models/controllers/controller_lesson_10.pth")
    parser.add_argument("--samples-per-stage", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--success-radius", type=float, default=0.1)
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["student", "me"],
        default="student",
        help="student: 1-step lag action application; me: apply predicted action immediately.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = EvalConfig(
        max_steps=args.max_steps,
        success_radius=args.success_radius,
    )

    model = load_controller(args.checkpoint, device, args.inference_mode)
    curriculum = create_train_configs_tbu(num_lessons=10)

    print(f"checkpoint={args.checkpoint}")
    print(f"samples_per_stage={args.samples_per_stage} max_steps={cfg.max_steps} success_radius={cfg.success_radius}")
    print(f"inference_mode={args.inference_mode}")
    print("")

    all_denom = 0.0
    accum = {
        "avg_steps": 0.0,
        "avg_final_dist": 0.0,
        "final_dist_lt_0p1": 0.0,
        "final_dist_lt_1p0": 0.0,
        "stop_jackknife": 0.0,
        "stop_oob": 0.0,
        "stop_success": 0.0,
        "stop_timeout": 0.0,
    }

    for stage in range(1, 11):
        res = evaluate_stage(
            model=model,
            stage_cfg=curriculum[stage],
            cfg=cfg,
            num_samples=args.samples_per_stage,
            device=device,
            inference_mode=args.inference_mode,
        )
        n = res["samples"]
        all_denom += n
        for k in accum:
            accum[k] += res[k] * n
        print(
            f"stage={stage:02d} samples={int(n)} "
            f"avg_steps={res['avg_steps']:.1f} avg_final_dist={res['avg_final_dist']:.3f} "
            f"final_dist<0.1={res['final_dist_lt_0p1']:.3f} final_dist<1.0={res['final_dist_lt_1p0']:.3f} "
            f"stop(jack/oob/success/timeout)="
            f"{res['stop_jackknife']:.3f}/{res['stop_oob']:.3f}/{res['stop_success']:.3f}/{res['stop_timeout']:.3f}"
        )

    print("")
    print("overall:")
    print(
        f"samples={int(all_denom)} "
        f"avg_steps={accum['avg_steps']/all_denom:.1f} avg_final_dist={accum['avg_final_dist']/all_denom:.3f} "
        f"final_dist<0.1={accum['final_dist_lt_0p1']/all_denom:.3f} "
        f"final_dist<1.0={accum['final_dist_lt_1p0']/all_denom:.3f} "
        f"stop(jack/oob/success/timeout)="
        f"{accum['stop_jackknife']/all_denom:.3f}/"
        f"{accum['stop_oob']/all_denom:.3f}/"
        f"{accum['stop_success']/all_denom:.3f}/"
        f"{accum['stop_timeout']/all_denom:.3f}"
    )


if __name__ == "__main__":
    main()
