import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML
import torch

__all__ = [
    'plot_truck_xu',
    'plot_truck',
    'plot_truck_fixed_view',
    'plot_signal',
    'plot_truck_cost_design',
    'plot_multi_us',
    'plot_ncheck_vs_squared_distance',
    'plot_failure_free_square',
    'plot_failure_free_rectangle',
]


def trailer_xy(x):
    d = 4.0
    x_pos, y_pos, _, theta_truck = x.T
    xt = x_pos - d * torch.cos(theta_truck)
    yt = y_pos - d * torch.sin(theta_truck)
    return xt, yt


def plot_truck_xu(x, u, title=None):
    """
    Plot truck states and steering in stacked layout.

    Top: positions
    Middle: angles
    Bottom: steering
    """
    fig = plt.figure(figsize=(10, 8))

    x_pos, y_pos, theta_cab, theta_truck = x.T
    xt, yt = trailer_xy(x)
    delta_theta = theta_truck - theta_cab
    theta_cab_deg = torch.rad2deg(theta_cab)
    theta_truck_deg = torch.rad2deg(theta_truck)
    delta_theta_deg = torch.rad2deg(delta_theta)
    u_deg = torch.rad2deg(u.squeeze())

    def _sym_ylim(series, pad=1.05):
        ymax = max(float(torch.max(torch.abs(s)).item()) for s in series)
        return -pad * ymax, pad * ymax

    def _fixed_or_dynamic_ylim(series, fixed_abs, pad=1.05):
        max_abs = max(float(torch.max(torch.abs(s)).item()) for s in series)
        fixed_abs = float(fixed_abs)
        if max_abs <= fixed_abs:
            return -fixed_abs, fixed_abs
        return _sym_ylim(series, pad=pad)

    def _shade_abs_limit(ax, limit_deg, ylo, yhi):
        ax.axhspan(limit_deg, yhi, hatch='///', facecolor='none',
                   edgecolor='C3', linewidth=0.0, alpha=0.6)
        ax.axhspan(ylo, -limit_deg, hatch='///', facecolor='none',
                   edgecolor='C3', linewidth=0.0, alpha=0.6)

    def _set_deg_yticks(ax, ylo, yhi, step=15.0):
        start = step * np.ceil(ylo / step)
        stop = step * np.floor(yhi / step)
        ticks = np.arange(start, stop + 1e-6, step)
        if ticks.size == 0 and ylo <= 0.0 <= yhi:
            ticks = np.array([0.0])
        ax.set_yticks(ticks)

    with plt.style.context(['dark_background', 'bmh']):
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])

        # --- Positions ---
        ax0 = plt.subplot(gs[0])
        ax0.set_facecolor('black')
        plt.plot(xt, label='x')
        plt.plot(yt, '--', label='y')
        plt.ylabel('Trailer Position')
        plt.grid(True)
        ax0.legend(loc='upper right', fontsize=8, framealpha=0.6, facecolor='k')

        # --- Angles ---
        ax1 = plt.subplot(gs[1])
        ax1.set_facecolor('black')
        plt.plot(theta_cab_deg, label='theta_cab (deg)')
        plt.plot(theta_truck_deg, '--', label='theta_truck (deg)')
        plt.plot(delta_theta_deg, ':', label='delta_theta (deg)')
        ylo, yhi = _fixed_or_dynamic_ylim([theta_cab_deg, theta_truck_deg, delta_theta_deg], fixed_abs=120.0)
        ax1.set_ylim(ylo, yhi)
        _set_deg_yticks(ax1, ylo, yhi, step=30.0)
        _shade_abs_limit(ax1, 90.0, ylo, yhi)
        plt.ylabel('Angle (deg)')
        plt.grid(True)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.6, facecolor='k')

        # --- Steering ---
        ax2 = plt.subplot(gs[2])
        ax2.set_facecolor('black')
        plt.stem(
            torch.arange(len(u)).numpy(),
            u_deg.detach().numpy(),
            basefmt='none'
        )
        ylo, yhi = _fixed_or_dynamic_ylim([u_deg], fixed_abs=60.0)
        ax2.set_ylim(ylo, yhi)
        _set_deg_yticks(ax2, ylo, yhi, step=15.0)
        _shade_abs_limit(ax2, 45.0, ylo, yhi)
        plt.ylabel('u (deg)')
        plt.xlabel('Time step')
        plt.grid(True)

    if title is not None:
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()
    plt.show()
    return gs


def plot_truck(coords, y_target=None, car=True, save_path=None):
    # Keep original style settings
    plt.style.use(['dark_background', 'bmh'])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.set_facecolor('black')

    # Constants for drawing (matching your dynamics l=1, d=4)
    l, d = 1.0, 4.0
    cab_w, tr_w = 1.0, 1.0

    delta_theta_warn = np.pi / 2   # jackknife warning threshold

    # Setup axis limits based on trajectory
    pad = 5
    ax.set_xlim(coords[:, 0].min().item() - pad, coords[:, 0].max().item() + pad)
    ax.set_ylim(coords[:, 1].min().item() - pad, coords[:, 1].max().item() + pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Always show target marker; default to origin if no target is provided
    if y_target is None:
        target_x, target_y = 0.0, 0.0
    else:
        target_x, target_y = float(y_target[0]), float(y_target[1])
    ax.scatter(target_x, target_y, marker='x', color='darkgray',
               s=60, zorder=10, label='Target')

    # Initialize Patches using original color palette
    cab_patch = patches.Polygon([[0, 0]], color='C2', alpha=1.0, zorder=5)
    trailer_patch = patches.Polygon([[0, 0]], color='C0', alpha=1.0, zorder=4)
    ax.add_patch(cab_patch)
    ax.add_patch(trailer_patch)

    # Trace line for trajectory
    ax.plot(coords[:, 0], coords[:, 1], 'w--', alpha=0.2, lw=1)

    # Jackknife warning text
    warn_text = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        color='red',
        fontsize=12,
        fontweight='bold',
        va='top'
    )

    def get_poly(cx, cy, angle, length, width, is_trailer=False):
        # Anchor at hitch (cx, cy)
        x_off = -length if is_trailer else 0
        rect = np.array([
            [x_off, -width / 2], [x_off + length, -width / 2],
            [x_off + length, width / 2], [x_off, width / 2]
        ])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return (rect @ rot.T) + np.array([cx, cy])

    def update(frame):
        state = coords[frame].numpy()
        x, y, th_c, th_t = state[0], state[1], state[2], state[3]

        # Update polygons
        cab_patch.set_xy(get_poly(x, y, th_c, l, cab_w))
        trailer_patch.set_xy(get_poly(x, y, th_t, d, tr_w, is_trailer=True))

        # Jackknife detection
        delta_theta = th_c - th_t
        if abs(delta_theta) > delta_theta_warn:
            trailer_patch.set_color('red')
            warn_text.set_text('JACKKNIFED!')
        else:
            trailer_patch.set_color('C0')
            warn_text.set_text('')

        return cab_patch, trailer_patch, warn_text

    # Create Animation
    anim = animation.FuncAnimation(fig, update,
                                   frames=len(coords),
                                   blit=True,
                                   interval=50)

    plt.close()  # Prevent static plot showing up
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)

    return HTML(anim.to_jshtml())


def plot_truck_fixed_view(
    coords,
    cfg,
    y_target=None,
    car=True,
    save_path=None,
    play_speed: float = 1.0,
    pad_ratio: float = 0.05,
):
    plt.style.use(['dark_background', 'bmh'])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.set_facecolor('black')

    l, d = 1.0, 4.0
    cab_w, tr_w = 1.0, 1.0
    delta_theta_warn = np.pi / 2

    xmin, xmax = cfg.env_x_range
    ymin, ymax = cfg.env_y_range
    xpad = (xmax - xmin) * float(pad_ratio)
    ypad = (ymax - ymin) * float(pad_ratio)
    ax.set_xlim(float(xmin) - xpad, float(xmax) + xpad)
    ax.set_ylim(float(ymin) - ypad, float(ymax) + ypad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if y_target is None:
        target_x, target_y = 0.0, 0.0
    else:
        target_x, target_y = float(y_target[0]), float(y_target[1])
    ax.scatter(target_x, target_y, marker='x', color='darkgray',
               s=60, zorder=10, label='Target')

    cab_patch = patches.Polygon([[0, 0]], color='C2', alpha=1.0, zorder=5)
    trailer_patch = patches.Polygon([[0, 0]], color='C0', alpha=1.0, zorder=4)
    ax.add_patch(cab_patch)
    ax.add_patch(trailer_patch)
    ax.plot(coords[:, 0], coords[:, 1], 'w--', alpha=0.2, lw=1)

    warn_text = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        color='red',
        fontsize=12,
        fontweight='bold',
        va='top'
    )

    def get_poly(cx, cy, angle, length, width, is_trailer=False):
        x_off = -length if is_trailer else 0
        rect = np.array([
            [x_off, -width / 2], [x_off + length, -width / 2],
            [x_off + length, width / 2], [x_off, width / 2]
        ])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return (rect @ rot.T) + np.array([cx, cy])

    def update(frame):
        state = coords[frame].numpy()
        x, y, th_c, th_t = state[0], state[1], state[2], state[3]

        cab_patch.set_xy(get_poly(x, y, th_c, l, cab_w))
        trailer_patch.set_xy(get_poly(x, y, th_t, d, tr_w, is_trailer=True))

        if abs(th_c - th_t) > delta_theta_warn:
            trailer_patch.set_color('red')
            warn_text.set_text('JACKKNIFED!')
        else:
            trailer_patch.set_color('C0')
            warn_text.set_text('')

        return cab_patch, trailer_patch, warn_text

    speed = max(0.05, float(play_speed))
    interval_ms = max(1, int(round(50.0 / speed)))
    gif_fps = max(1, int(round(20.0 * speed)))
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(coords),
        blit=True,
        interval=interval_ms,
    )

    plt.close()
    if save_path:
        anim.save(save_path, writer='pillow', fps=gif_fps)

    return HTML(anim.to_jshtml())


def plot_signal(signal, label='signal', labels=None, is_angle=False):
    """
    Plot a 1D tensor over time.

    signal : tensor shape (N,)
    label  : string for title and legend
    is_angle : bool, if True interpret signal as radians and plot in degrees
    """
    signals = list(signal) if isinstance(signal, (list, tuple)) else [signal]
    signals = [s.detach() for s in signals]
    if is_angle:
        signals = [torch.rad2deg(s) for s in signals]
    if labels is None:
        if len(signals) == 1:
            labels = [label]
        else:
            labels = [f'{label} ({i+1})' for i in range(len(signals))]
    if len(labels) != len(signals):
        raise ValueError('labels must match number of signals')

    with plt.style.context(['dark_background', 'bmh']):
        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        for i, s in enumerate(signals):
            t = torch.arange(len(s))
            ax.plot(t, s, label=labels[i])
        ax.set_title(f'{label} over time')
        ax.set_xlabel('Time step')
        ax.set_ylabel(f'{label} (deg)' if is_angle else label)
        if is_angle:
            max_abs = max(float(torch.max(torch.abs(s)).item()) for s in signals)
            preset = [15.0, 30.0, 45.0, 60.0]
            lim = next((v for v in preset if max_abs <= v), None)
            if lim is None:
                lim = 15.0 * np.ceil(max_abs / 15.0)
            ax.set_ylim(-lim, lim)
            ax.set_yticks(np.arange(-lim, lim + 1e-6, 15.0))
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.6, facecolor='k')
    plt.show()


def plot_truck_cost_design(
    w_process_angle=100.0,
    w_final_pos=10.0,
    w_final_angle=200.0,
    w_action=400.0,
    delta_theta_limit=75.0 * np.pi / 180.0,
    u_limit=35.0 * np.pi / 180.0,
):
    """
    Plot cost and gradient charts for the four truck cost terms:
    1) ReLU-then-squared process-angle penalty
    2) Final position penalty (single-coordinate view)
    3) Final cab/trailer angle-squared penalty
    4) ReLU-then-squared action penalty
    """
    delta_theta_xlim_deg = (-100.0, 100.0)
    action_xlim_deg = (-60.0, 60.0)
    ypad = 1.05
    delta_theta_init_deg = 45.0
    delta_theta_shade_limit_deg = 90.0
    u_init_deg = 30.0
    action_shade_limit_deg = 45.0

    def _visible_mask_from_radians(rad_grid, xlim_deg):
        deg = np.rad2deg(rad_grid)
        return (deg >= xlim_deg[0]) & (deg <= xlim_deg[1])

    def _positive_ylim_from_visible(y, mask):
        vmax = float(np.max(y[mask]))
        if vmax <= 0.0:
            vmax = 1.0
        return 0.0, ypad * vmax

    def _symmetric_ylim_from_visible(y, mask):
        vmax = float(np.max(np.abs(y[mask])))
        if vmax <= 0.0:
            vmax = 1.0
        return -ypad * vmax, ypad * vmax

    def _shade_abs_region(ax, x_deg, ylo, yhi, abs_limit_deg):
        mask = np.abs(x_deg) > abs_limit_deg
        ax.fill_between(
            x_deg, ylo, yhi, where=mask, hatch='///',
            facecolor='none', edgecolor='C3', linewidth=0.0
        )

    def _set_angle_xticks(ax, xlim_deg):
        lo, hi = xlim_deg
        start = 15.0 * np.ceil(lo / 15.0)
        stop = 15.0 * np.floor(hi / 15.0)
        ticks = np.arange(start, stop + 1e-6, 15.0)
        ax.set_xticks(ticks)

    def _set_action_xticks(ax, xlim_deg):
        lo, hi = xlim_deg
        start = 15.0 * np.ceil(lo / 15.0)
        stop = 15.0 * np.floor(hi / 15.0)
        ticks = np.arange(start, stop + 1e-6, 15.0)
        ax.set_xticks(ticks)

    # Shared grids
    delta_theta_grid = np.linspace(-np.pi, np.pi, 801)
    theta_grid = np.linspace(-np.pi, np.pi, 801)
    u_grid = np.linspace(-np.pi, np.pi, 801)
    pos_grid = np.linspace(-40.0, 40.0, 801)

    # Initialization regions
    delta_theta_init = np.deg2rad(delta_theta_init_deg)
    u_init = np.deg2rad(u_init_deg)

    # Process-angle cost
    violation = np.maximum(np.abs(delta_theta_grid) - delta_theta_limit, 0.0)
    process_cost = w_process_angle * violation**2
    process_grad = 2.0 * w_process_angle * violation * np.sign(delta_theta_grid)

    # Final position cost (single-coordinate profile)
    pos_cost = w_final_pos * pos_grid**2
    pos_grad = 2.0 * w_final_pos * pos_grid

    # Final angle cost
    final_angle_cost = w_final_angle * theta_grid**2
    final_angle_grad = 2.0 * w_final_angle * theta_grid

    # Action cost
    u_violation = np.maximum(np.abs(u_grid) - u_limit, 0.0)
    action_cost = w_action * u_violation**2
    action_grad = 2.0 * w_action * u_violation * np.sign(u_grid)

    with plt.style.context(['dark_background', 'bmh']):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
        fig.patch.set_facecolor('black')
        for ax in axes.ravel():
            ax.set_facecolor('black')

        title_fs = 16
        label_fs = 11
        tick_fs = 9
        suptitle_fs = 14

        # 1) Position cost (leftmost)
        ax = axes[0, 0]
        ax.plot(pos_grid, pos_cost, color='C0')
        ax.set_title('Final Trailer Position Cost', fontsize=title_fs)
        ax.set_xlabel('position coordinate (m)', fontsize=label_fs)
        ax.set_ylabel('cost', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(pos_grid, pos_grad, color='C1')
        ax.set_title('Final Trailer Position Gradient', fontsize=title_fs)
        ax.set_xlabel('position coordinate (m)', fontsize=label_fs)
        ax.set_ylabel('gradient', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)

        # 2) Process angle
        ax = axes[0, 1]
        delta_theta_deg = np.rad2deg(delta_theta_grid)
        ax.plot(delta_theta_deg, process_cost, color='C0')
        delta_theta_mask = _visible_mask_from_radians(delta_theta_grid, delta_theta_xlim_deg)
        ymin, ymax = _positive_ylim_from_visible(process_cost, delta_theta_mask)
        ax.set_ylim(ymin, ymax)
        _shade_abs_region(ax, delta_theta_deg, ymin, ymax, delta_theta_shade_limit_deg)
        ax.axvline(np.rad2deg(delta_theta_init), color='mediumorchid', ls='--', lw=1.2)
        ax.axvline(-np.rad2deg(delta_theta_init), color='mediumorchid', ls='--', lw=1.2)
        ax.set_title('Process Angle Cost', fontsize=title_fs)
        ax.set_xlabel('delta_theta = theta_truck - theta_cab (deg)', fontsize=label_fs)
        ax.set_ylabel('cost', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(delta_theta_xlim_deg)
        _set_angle_xticks(ax, delta_theta_xlim_deg)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(delta_theta_deg, process_grad, color='C1')
        gbot, gtop = _symmetric_ylim_from_visible(process_grad, delta_theta_mask)
        ax.set_ylim(gbot, gtop)
        _shade_abs_region(ax, delta_theta_deg, gbot, gtop, delta_theta_shade_limit_deg)
        ax.axvline(np.rad2deg(delta_theta_init), color='mediumorchid', ls='--', lw=1.2)
        ax.axvline(-np.rad2deg(delta_theta_init), color='mediumorchid', ls='--', lw=1.2)
        ax.set_title('Process Angle Gradient', fontsize=title_fs)
        ax.set_xlabel('delta_theta = theta_truck - theta_cab (deg)', fontsize=label_fs)
        ax.set_ylabel('gradient', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(delta_theta_xlim_deg)
        _set_angle_xticks(ax, delta_theta_xlim_deg)
        ax.grid(True, alpha=0.3)

        # 3) Final angle
        axes[0, 2].plot(np.rad2deg(theta_grid), final_angle_cost, color='C0')
        theta_mask = _visible_mask_from_radians(theta_grid, delta_theta_xlim_deg)
        ymin, ymax = _positive_ylim_from_visible(final_angle_cost, theta_mask)
        axes[0, 2].set_ylim(ymin, ymax)
        axes[0, 2].set_title('Final Cab/Trailer Angle Cost', fontsize=title_fs)
        axes[0, 2].set_xlabel('theta (deg)', fontsize=label_fs)
        axes[0, 2].set_ylabel('cost', fontsize=label_fs)
        axes[0, 2].tick_params(labelsize=tick_fs)
        axes[0, 2].set_xlim(delta_theta_xlim_deg)
        _set_angle_xticks(axes[0, 2], delta_theta_xlim_deg)
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 2].plot(np.rad2deg(theta_grid), final_angle_grad, color='C1')
        gbot, gtop = _symmetric_ylim_from_visible(final_angle_grad, theta_mask)
        axes[1, 2].set_ylim(gbot, gtop)
        axes[1, 2].set_title('Final Cab/Trailer Angle Gradient', fontsize=title_fs)
        axes[1, 2].set_xlabel('theta (deg)', fontsize=label_fs)
        axes[1, 2].set_ylabel('gradient', fontsize=label_fs)
        axes[1, 2].tick_params(labelsize=tick_fs)
        axes[1, 2].set_xlim(delta_theta_xlim_deg)
        _set_angle_xticks(axes[1, 2], delta_theta_xlim_deg)
        axes[1, 2].grid(True, alpha=0.3)

        # 4) Action
        ax = axes[0, 3]
        u_deg = np.rad2deg(u_grid)
        ax.plot(u_deg, action_cost, color='C0')
        u_mask = _visible_mask_from_radians(u_grid, action_xlim_deg)
        ymin, ytop = _positive_ylim_from_visible(action_cost, u_mask)
        ax.set_ylim(ymin, ytop)
        _shade_abs_region(ax, u_deg, ymin, ytop, action_shade_limit_deg)
        ax.axvline(np.rad2deg(u_init), color='mediumorchid', ls='--', lw=1.2)
        ax.axvline(-np.rad2deg(u_init), color='mediumorchid', ls='--', lw=1.2)
        ax.set_title('Action Cost', fontsize=title_fs)
        ax.set_xlabel('u (deg)', fontsize=label_fs)
        ax.set_ylabel('cost', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(action_xlim_deg)
        _set_action_xticks(ax, action_xlim_deg)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 3]
        ax.plot(u_deg, action_grad, color='C1')
        gbot, gtop = _symmetric_ylim_from_visible(action_grad, u_mask)
        ax.set_ylim(gbot, gtop)
        _shade_abs_region(ax, u_deg, gbot, gtop, action_shade_limit_deg)
        ax.axvline(np.rad2deg(u_init), color='mediumorchid', ls='--', lw=1.2)
        ax.axvline(-np.rad2deg(u_init), color='mediumorchid', ls='--', lw=1.2)
        ax.set_title('Action Gradient', fontsize=title_fs)
        ax.set_xlabel('u (deg)', fontsize=label_fs)
        ax.set_ylabel('gradient', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xlim(action_xlim_deg)
        _set_action_xticks(ax, action_xlim_deg)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Truck Cost Design: Cost and Gradient Charts', fontsize=suptitle_fs)
        plt.show()
        return fig, axes


def plot_multi_us(pt_path, k=10, title=None):
    """
    Plot steering sequences u for first k samples (best candidate per sample),
    with steering limits at +-60 deg.

    pt_path: path to .pt file containing key "rollouts"
    title: optional plot title
    """
    data = torch.load(pt_path, map_location='cpu')
    rollouts = data['rollouts']
    u_lim_deg = 60.0

    with plt.style.context(['dark_background', 'bmh']):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_facecolor('black')

        n = min(k, len(rollouts))
        for i in range(n):
            candidates = rollouts[i]['candidates']
            best = min(candidates, key=lambda c: c['err'])
            u_deg = np.rad2deg(best['u'].squeeze().cpu().numpy())
            ax.plot(np.arange(len(u_deg)), u_deg, alpha=0.85, lw=1.6, label=f'sample {i}')

        ax.axhline(+45, color='red', linestyle=':', lw=2, label='+45 deg')
        ax.axhline(-45, color='red', linestyle=':', lw=2, label='-45 deg')
        ax.set_ylim(-u_lim_deg, u_lim_deg)

        plot_title = title if title is not None else f'Steering Controls for First {n} Samples'
        ax.set_title(plot_title)
        ax.set_xlabel('Time step')
        ax.set_ylabel('u (deg)')
        ax.set_yticks(np.arange(-u_lim_deg, u_lim_deg + 1e-6, 15))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2, fontsize=8, facecolor='k')
        plt.tight_layout()
    plt.show()


def plot_ncheck_vs_squared_distance(pt_path='truck_regression_data.pt'):
    """
    Fit N_star = w * (x^2 + y^2) + b on successful samples and plot.
    """
    data = torch.load(pt_path, map_location='cpu')
    x_raw = data['X_raw'].float()
    n_star = data['N_star'].float()
    failure_type = data['failure_type']

    mask_success = failure_type == 0
    x = x_raw[mask_success, 0]
    y = x_raw[mask_success, 1]
    feature = (x**2 + y**2).unsqueeze(1)
    n_star_success = n_star[mask_success]

    x_design = torch.cat([feature, torch.ones_like(feature)], dim=1)
    w = torch.linalg.lstsq(x_design, n_star_success).solution
    pred = x_design @ w

    rmse = torch.sqrt(torch.mean((pred - n_star_success)**2))
    ss_tot = torch.sum((n_star_success - torch.mean(n_star_success))**2)
    ss_res = torch.sum((n_star_success - pred)**2)
    r2 = 1 - ss_res / ss_tot

    print('w:', w[0].item())
    print('b:', w[1].item())
    print('RMSE:', rmse.item())
    print('R^2:', r2.item())

    idx = torch.argsort(feature.squeeze())

    with plt.style.context(['dark_background', 'bmh']):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('k')
        ax.set_facecolor('k')
        ax.scatter(feature.numpy(), n_star_success.numpy(), label='success')
        ax.plot(feature[idx].numpy(), pred[idx].detach().numpy(), label='fit')
        ax.set_xlabel('squared distance feature')
        ax.set_ylabel('N_check')
        ax.legend(loc='upper right', facecolor='k')
        plt.tight_layout()
    plt.show()

    return {
        'w': w[0].item(),
        'b': w[1].item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
    }


def plot_failure_free_square(v_success, v_failure, bounds, title='Success/Failure Points with Failure-Free Square'):
    """
    Plot success/failure points with inscribed square from a failure-free rectangle.

    v_success: tensor shape (Ns, 2) with [x, y]
    v_failure: tensor shape (Nf, 2) with [x, y]
    bounds: (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = [float(v) for v in bounds]
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    side = min(xmax - xmin, ymax - ymin)
    sq_x = cx - side / 2.0
    sq_y = cy - side / 2.0

    all_points = torch.cat([v_success, v_failure], dim=0)
    x = all_points[:, 0]
    y = all_points[:, 1]

    with plt.style.context(['dark_background', 'bmh']):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('k')
        ax.set_facecolor('k')
        ax.scatter(v_success[:, 0].numpy(), v_success[:, 1].numpy(), s=14, c='green', alpha=0.7, label='success')
        ax.scatter(v_failure[:, 0].numpy(), v_failure[:, 1].numpy(), s=14, c='red', alpha=0.7, label='failure')
        square = patches.Rectangle((sq_x, sq_y), side, side, fill=False, ec='yellow', lw=2.5, ls='-')
        ax.add_patch(square)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', facecolor='k')

        x_pad = 0.05 * (float(x.max() - x.min()) + 1e-6)
        y_pad = 0.05 * (float(y.max() - y.min()) + 1e-6)
        ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
        ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
        plt.tight_layout()
    plt.show()

    return side


def plot_failure_free_rectangle(v_success, v_failure, bounds, title='Success/Failure Points with Failure-Free Rectangle'):
    """
    Plot success/failure points with a failure-free rectangle.

    v_success: tensor shape (Ns, 2) with [x, y]
    v_failure: tensor shape (Nf, 2) with [x, y]
    bounds: (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = [float(v) for v in bounds]

    all_points = torch.cat([v_success, v_failure], dim=0)
    x = all_points[:, 0]
    y = all_points[:, 1]

    with plt.style.context(['dark_background', 'bmh']):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('k')
        ax.set_facecolor('k')
        ax.scatter(v_success[:, 0].numpy(), v_success[:, 1].numpy(), s=14, c='green', alpha=0.7, label='success')
        ax.scatter(v_failure[:, 0].numpy(), v_failure[:, 1].numpy(), s=14, c='red', alpha=0.7, label='failure')
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec='yellow', lw=2.5, ls='-')
        ax.add_patch(rect)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', facecolor='k')

        x_pad = 0.05 * (float(x.max() - x.min()) + 1e-6)
        y_pad = 0.05 * (float(y.max() - y.min()) + 1e-6)
        ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
        ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
        plt.tight_layout()
    plt.show()

    return (xmax - xmin), (ymax - ymin)
