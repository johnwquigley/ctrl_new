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
    'plot_signal',
    'plot_truck_cost_design',
]


def plot_truck_xu(x, u):
    """
    Plot truck states and steering in stacked layout.

    Top: positions
    Middle: angles
    Bottom: steering
    """

    x_pos, y_pos, theta_cab, theta_truck = x.T
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

    with plt.style.context(['dark_background', 'bmh']):
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])

        # --- Positions ---
        ax0 = plt.subplot(gs[0])
        ax0.set_facecolor('black')
        plt.plot(x_pos, label='x')
        plt.plot(y_pos, '--', label='y')
        plt.ylabel('Position')
        plt.grid(True)
        ax0.legend(loc='upper right', fontsize=8, framealpha=0.6)

        # --- Angles ---
        ax1 = plt.subplot(gs[1])
        ax1.set_facecolor('black')
        plt.plot(theta_cab_deg, label='theta_cab (deg)')
        plt.plot(theta_truck_deg, '--', label='theta_truck (deg)')
        plt.plot(delta_theta_deg, ':', label='delta_theta (deg)')
        ylo, yhi = _fixed_or_dynamic_ylim([theta_cab_deg, theta_truck_deg, delta_theta_deg], fixed_abs=120.0)
        ax1.set_ylim(ylo, yhi)
        _shade_abs_limit(ax1, 90.0, ylo, yhi)
        plt.ylabel('Angle (deg)')
        plt.grid(True)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.6)

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
        _shade_abs_limit(ax2, 45.0, ylo, yhi)
        plt.ylabel('u (deg)')
        plt.xlabel('Time step')
        plt.grid(True)

    plt.tight_layout()
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


def plot_signal(signal, label='signal', labels=None):
    """
    Plot a 1D tensor over time.

    signal : tensor shape (N,)
    label  : string for title and legend
    """
    signals = list(signal) if isinstance(signal, (list, tuple)) else [signal]
    signals = [s.detach() for s in signals]
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
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
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
        ax.plot(pos_grid, pos_cost, color='C0', label='position cost')
        ax.set_title('Final Position Cost', fontsize=title_fs)
        ax.set_xlabel('position coordinate (m)', fontsize=label_fs)
        ax.set_ylabel('cost', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(pos_grid, pos_grad, color='C1', label='d(cost)/d(position)')
        ax.set_title('Final Position Gradient', fontsize=title_fs)
        ax.set_xlabel('position coordinate (m)', fontsize=label_fs)
        ax.set_ylabel('gradient', fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

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
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 2].plot(np.rad2deg(theta_grid), final_angle_grad, color='C1')
        gbot, gtop = _symmetric_ylim_from_visible(final_angle_grad, theta_mask)
        axes[1, 2].set_ylim(gbot, gtop)
        axes[1, 2].set_title('Final Cab/Trailer Angle Gradient', fontsize=title_fs)
        axes[1, 2].set_xlabel('theta (deg)', fontsize=label_fs)
        axes[1, 2].set_ylabel('gradient', fontsize=label_fs)
        axes[1, 2].tick_params(labelsize=tick_fs)
        axes[1, 2].set_xlim(delta_theta_xlim_deg)
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
        ax.grid(True, alpha=0.3)

        fig.suptitle('Truck Cost Design: Cost and Gradient Charts', fontsize=suptitle_fs)
        return fig, axes
