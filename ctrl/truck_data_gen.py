import os
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass

from local_optim import LBFGS

PI = torch.pi


@dataclass(frozen=True)
class EnvConfig:
    env_x_range: tuple[float, float] = (0.0, 50.0)
    env_y_range: tuple[float, float] = (-15.0, 15.0)
    theta0_range_deg: tuple[float, float] = (-180.0, 180.0)
    theta1_range_deg: tuple[float, float] = (-180.0, 180.0)
    truck_speed: float = -0.1
    wheelbase: float = 1.0
    hitch_length: float = 4.0


CFG = EnvConfig()


def truck_dynamics(x, u):
    l = CFG.wheelbase
    d = CFG.hitch_length
    v = CFG.truck_speed

    x_pos, y_pos, theta_cab, theta_truck = x
    phi = u[0] if u.ndim else u

    f = torch.zeros(4, dtype=x.dtype, device=x.device)
    f[0] = v * torch.cos(theta_cab)
    f[1] = v * torch.sin(theta_cab)
    f[2] = v / l * torch.tan(phi)
    f[3] = v / d * torch.sin(theta_cab - theta_truck)
    return f


def integrate(f, x0, u, dt=1.0):
    n_steps = len(u)
    x = [0] * (n_steps + 1)
    x[0] = x0
    for n in range(n_steps):
        x[n + 1] = x[n] + f(x[n], u[n]) * dt
    return torch.stack(x)


def trailer_xy(x):
    d = CFG.hitch_length
    x_pos, y_pos, _, theta_truck = x
    xt = x_pos - d * torch.cos(theta_truck)
    yt = y_pos - d * torch.sin(theta_truck)
    return torch.stack((xt, yt))


def cost_truck(y, x, u):
    n_steps = x.shape[0] - 1
    w_process_angle = 100.0
    w_final_pos = 10.0
    w_final_angle = 200.0

    _, _, theta_cab, theta_truck = x.T

    b = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for n in range(n_steps + 1):
        alpha = theta_truck[n] - theta_cab[n]
        alpha_limit = 75 * PI / 180
        violation = torch.relu(torch.abs(alpha) - alpha_limit)
        b = b + w_process_angle * violation.pow(2)

    xt, yt = trailer_xy(x[-1])
    c = w_final_pos * (xt.pow(2) + yt.pow(2)) + w_final_angle * (
        theta_cab[-1].pow(2) + theta_truck[-1].pow(2)
    )

    return b + c


def cost_truck_with_action_cost(y, x, u):
    n_steps = x.shape[0] - 1
    w_process_angle = 100.0
    w_final_pos = 10.0
    w_final_angle = 200.0
    w_action = 600.0
    alpha_limit = 75 * PI / 180
    u_limit = 35 * PI / 180

    _, _, theta_cab, theta_truck = x.T

    a = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for n in range(len(u)):
        u_violation = torch.relu(torch.abs(u[n].squeeze()) - u_limit)
        a = a + w_action * u_violation.pow(2)

    b = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for n in range(n_steps + 1):
        alpha = theta_truck[n] - theta_cab[n]
        violation = torch.relu(torch.abs(alpha) - alpha_limit)
        b = b + w_process_angle * violation.pow(2)

    xt, yt = trailer_xy(x[-1])
    c = w_final_pos * (xt.pow(2) + yt.pow(2)) + w_final_angle * (
        theta_cab[-1].pow(2) + theta_truck[-1].pow(2)
    )

    return a + b + c


def error_truck(x, y):
    xt, yt = trailer_xy(x[-1])
    return torch.norm(torch.stack((xt - y[0], yt - y[1], x[-1][3] - y[2])))


def lbfgs_projected(e, u0, projected=False, u_limit_deg=45.0, outer_steps=1):
    u0 = torch.nn.Parameter(u0)
    u_lim = float(u_limit_deg) * PI / 180.0
    opt = LBFGS(
        (u0,),
        line_search_fn="strong_wolfe",
        max_iter=200,
        projected=projected,
        project_min=-u_lim,
        project_max=u_lim,
    )

    def closure():
        e_u0 = e(u0)
        opt.zero_grad()
        e_u0.backward()
        return e_u0

    for _ in range(outer_steps):
        opt.step(closure)

    return u0.detach().clone()


def compute_u_hat(e, n_steps, c=1, projected=False, u_limit_deg=45.0, outer_steps=1):
    u = torch.zeros(n_steps, c)
    return lbfgs_projected(
        e,
        u,
        projected=projected,
        u_limit_deg=u_limit_deg,
        outer_steps=outer_steps,
    )


def detect_jackknife(x, limit_deg=90.0):
    theta_c = x[:, 2]
    theta_t = x[:, 3]
    alpha = (theta_t - theta_c + PI) % (2 * PI) - PI
    limit = limit_deg * PI / 180.0
    return torch.any(torch.abs(alpha) > limit)


def _plan_truck_projected(x0, y, n_steps, cost_fn, u_limit_deg=45.0):
    def e(u):
        x = integrate(truck_dynamics, x0, u)
        return cost_fn(y, x, u)

    return compute_u_hat(
        e,
        n_steps,
        c=1,
        projected=True,
        u_limit_deg=u_limit_deg,
        outer_steps=1,
    )


def _run_one_sample_worker(args):
    # Keep one intra-op thread per worker to avoid oversubscription.
    torch.set_num_threads(1)

    (
        x0,
        n_list,
        y_goal,
        eps,
        terminal_filter,
        u_limit_deg,
        use_action_cost,
    ) = args

    cost_fn = cost_truck_with_action_cost if use_action_cost else cost_truck
    results = []

    for n_steps in n_list:
        u = _plan_truck_projected(
            x0,
            y_goal,
            n_steps,
            cost_fn=cost_fn,
            u_limit_deg=u_limit_deg,
        )
        x_traj = integrate(truck_dynamics, x0, u)
        err = error_truck(x_traj, y_goal).item()
        jackknife = detect_jackknife(x_traj).item()

        results.append(
            {
                "N": n_steps,
                "err": err,
                "jackknife": jackknife,
                "u": u.detach().cpu(),
                "x_traj": x_traj.detach().cpu(),
            }
        )

    valid = [r for r in results if r["err"] <= terminal_filter]
    if len(valid) == 0:
        n_star = float("nan")
        failure = 1
    else:
        valid = sorted(valid, key=lambda r: r["err"])
        non_jackknife = [r for r in valid if not r["jackknife"]]
        if len(non_jackknife) == 0:
            n_star = float("nan")
            failure = 2
        else:
            best = non_jackknife[0]
            if best["err"] < eps:
                n_star = float(best["N"])
                failure = 0
            else:
                n_star = float("nan")
                failure = 1

    return {
        "x0": x0.detach().cpu(),
        "results": results,
        "N_star": n_star,
        "failure_type": failure,
    }


def generate_data_fast(
    num_samples=20,
    max_workers=14,
    seed=0,
    eps=0.5,
    terminal_filter=10.0,
    n_list=None,
    u_limit_deg=45.0,
    use_action_cost=True,
    out_path=None,
):
    torch.manual_seed(seed)
    y_goal = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32)

    x0_list = []
    for _ in range(num_samples):
        alpha0 = torch.empty(1).uniform_(-PI, PI).item()
        x0 = torch.tensor(
            (
                torch.empty(1).uniform_(*CFG.env_x_range).item(),
                torch.empty(1).uniform_(*CFG.env_y_range).item(),
                alpha0,
                torch.empty(1).uniform_(-PI / 6, PI / 6).item() + alpha0,
            ),
            dtype=torch.float32,
        )
        x0_list.append(x0)

    worker_args = [
        (x0, n_list, y_goal, eps, terminal_filter, u_limit_deg, use_action_cost)
        for x0 in x0_list
    ]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        sample_out = list(ex.map(_run_one_sample_worker, worker_args))

    x_raw = []
    n_star = []
    failure_type = []
    rollouts = []

    for out in sample_out:
        x0_cpu = out["x0"]
        results = out["results"]

        x_raw.append(x0_cpu)
        n_star.append(out["N_star"])
        failure_type.append(out["failure_type"])
        rollouts.append({"x0": x0_cpu, "candidates": results})

    x_raw = torch.stack(x_raw).float()
    n_star = torch.tensor(n_star, dtype=torch.float32)
    failure_type = torch.tensor(failure_type, dtype=torch.long)

    torch.save(
        {
            "X_raw": x_raw,
            "N_star": n_star,
            "failure_type": failure_type,
            "rollouts": rollouts,
        },
        out_path,
    )

    print("saved samples:", len(x_raw))
    print("success:", (failure_type == 0).sum().item())
    print("terminal_fail:", (failure_type == 1).sum().item())
    print("jackknife_fail:", (failure_type == 2).sum().item())
    print("saved to:", out_path)

    return {
        "X_raw": x_raw,
        "N_star": n_star,
        "failure_type": failure_type,
        "rollouts": rollouts,
    }
