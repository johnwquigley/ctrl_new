import torch
from torch import nn, optim, pi as PI

SEED = 0
NUM_SAMPLES = 500
EPS = 0.5
TERMINAL_FILTER = 10.0
N_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]
Y_GOAL = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32)
JACKKNIFE_LIMIT_DEG = 90.0

OPTIMIZER_CONFIG = {
    "name": "LBFGS",
    "line_search_fn": "strong_wolfe",
    "max_iter": 200,
}

HYPERPARAMS = {
    "w_process_angle": 100.0,
    "w_final_pos": 10.0,
    "w_final_angle": 200.0,
    "delta_theta_limit_deg": 75.0,
}

def truck_dynamics(x, u):
    l = 1
    d = 4
    v = -1.0

    x_pos, y_pos, theta_cab, theta_truck = x
    phi = u[0] if u.ndim else u

    f = torch.zeros(4, dtype=x.dtype, device=x.device)
    f[0] = v * torch.cos(theta_cab)
    f[1] = v * torch.sin(theta_cab)
    f[2] = v / l * torch.tan(phi)
    f[3] = v / d * torch.sin(theta_cab - theta_truck)
    return f


def integrate(f, x0, u, dt=1):
    n_steps = len(u)
    x = [0] * (n_steps + 1)
    x[0] = x0
    for n in range(n_steps):
        x[n + 1] = x[n] + f(x[n], u[n]) * dt
    return torch.stack(x)


def run_lbfgs(e, u0, optimizer_cfg):
    u0 = nn.Parameter(u0)
    opt = optim.LBFGS(
        (u0,),
        line_search_fn=optimizer_cfg["line_search_fn"],
        max_iter=optimizer_cfg["max_iter"],
    )

    def closure():
        e_u0 = e(u0)
        opt.zero_grad()
        e_u0.backward()
        return e_u0

    opt.step(closure)
    return u0.detach()


def compute_u_opt(e, n_steps, control_dim=1, optimizer_cfg=OPTIMIZER_CONFIG):
    u = torch.zeros(n_steps, control_dim)
    return run_lbfgs(e, u, optimizer_cfg)


def trailer_xy(x):
    d = 4
    x_pos, y_pos, _, theta_truck = x
    xt = x_pos - d * torch.cos(theta_truck)
    yt = y_pos - d * torch.sin(theta_truck)
    return torch.stack((xt, yt))


def cost_truck(y, x, hp):
    n_steps = x.shape[0] - 1
    _, _, theta_cab, theta_truck = x.T

    delta_theta_limit = hp["delta_theta_limit_deg"] * PI / 180

    process_cost = x.new_tensor(0.0)
    for n in range(n_steps + 1):
        delta_theta = theta_truck[n] - theta_cab[n]
        violation = torch.relu(torch.abs(delta_theta) - delta_theta_limit)
        process_cost = process_cost + hp["w_process_angle"] * violation.pow(2)

    xt, yt = trailer_xy(x[-1])
    final_cost = hp["w_final_pos"] * (xt.pow(2) + yt.pow(2)) + hp["w_final_angle"] * (
        theta_cab[-1].pow(2) + theta_truck[-1].pow(2)
    )
    return process_cost + final_cost


def error_truck(x, y):
    xt, yt = trailer_xy(x[-1])
    return torch.norm(torch.stack((xt - y[0], yt - y[1], x[-1][3] - y[2])))


def plan_truck(x0, y, n_steps, hp=HYPERPARAMS, optimizer_cfg=OPTIMIZER_CONFIG):
    def e(u):
        x = integrate(truck_dynamics, x0, u)
        return cost_truck(y, x, hp)

    return compute_u_opt(e, n_steps, control_dim=1, optimizer_cfg=optimizer_cfg)


def detect_jackknife(x, limit_deg=JACKKNIFE_LIMIT_DEG):
    theta_c = x[:, 2]
    theta_t = x[:, 3]
    delta_theta = (theta_t - theta_c + torch.pi) % (2 * torch.pi) - torch.pi
    limit = limit_deg * torch.pi / 180.0
    return torch.any(torch.abs(delta_theta) > limit)


torch.manual_seed(SEED)

x_raw_list = []
n_star_list = []
failure_type_list = []
rollouts = []

for _ in range(NUM_SAMPLES):
    delta_theta0 = torch.empty(1).uniform_(-torch.pi, torch.pi).item()
    x0 = torch.tensor(
        (
            torch.empty(1).uniform_(2.0, 38.0).item(),
            torch.empty(1).uniform_(-18.0, 18.0).item(),
            delta_theta0,
            torch.empty(1).uniform_(-torch.pi / 6, torch.pi / 6).item() + delta_theta0,
        ),
        dtype=torch.float32,
    )

    results = []
    for n_steps in N_LIST:
        u = plan_truck(x0, Y_GOAL, n_steps, hp=HYPERPARAMS, optimizer_cfg=OPTIMIZER_CONFIG)
        x_traj = integrate(truck_dynamics, x0, u)
        err = float(error_truck(x_traj, Y_GOAL).item())
        jackknife = bool(detect_jackknife(x_traj).item())

        results.append(
            {
                "N": int(n_steps),
                "err": err,
                "jackknife": jackknife,
                "u": u.detach().cpu(),
                "x_traj": x_traj.detach().cpu(),
            }
        )

    rollouts.append({"x0": x0.detach().cpu(), "candidates": results})

    valid = [r for r in results if r["err"] <= TERMINAL_FILTER]
    x_raw_list.append(x0)

    if not valid:
        n_star_list.append(float("nan"))
        failure_type_list.append(1)
        continue

    valid = sorted(valid, key=lambda r: r["err"])
    non_jackknife = [r for r in valid if not r["jackknife"]]

    if not non_jackknife:
        n_star_list.append(float("nan"))
        failure_type_list.append(2)
        continue

    best = non_jackknife[0]
    if best["err"] < EPS:
        n_star_list.append(float(best["N"]))
        failure_type_list.append(0)
    else:
        n_star_list.append(float("nan"))
        failure_type_list.append(1)


X_raw = torch.stack(x_raw_list).float()
N_star = torch.tensor(n_star_list, dtype=torch.float32)
failure_type = torch.tensor(failure_type_list, dtype=torch.long)

payload = {
    "metadata": {
        "optimizer": OPTIMIZER_CONFIG,
        "hyperparameters": HYPERPARAMS,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "eps": EPS,
        "terminal_filter": TERMINAL_FILTER,
        "N_list": N_LIST,
        "y_goal": Y_GOAL.detach().cpu(),
        "jackknife_limit_deg": JACKKNIFE_LIMIT_DEG,
    },
    "X_raw": X_raw,
    "N_star": N_star,
    "failure_type": failure_type,
    "rollouts": rollouts,
}

torch.save(payload, "truck_regression_data_verbose.pt")

print("saved samples:", len(X_raw))
print("success:", (failure_type == 0).sum().item())
print("terminal_fail:", (failure_type == 1).sum().item())
print("jackknife_fail:", (failure_type == 2).sum().item())


# usage:
# data = torch.load("truck_regression_data.pt")
# print(data["metadata"])
# X_raw = data["X_raw"].float()
# N_star = data["N_star"].float()
# failure_type = data["failure_type"]
# rollouts = data["rollouts"]
