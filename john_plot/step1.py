import matplotlib.pyplot as plt
import torch
from IPython.display import display


# def plot_step_1_optimal_control_steering_angle_comparison_with_and_without_action_penalty(
#     steering_actions_without_penalty,
#     steering_actions_with_penalty,
#     initial_state,
# ):
#     """Plot the steering-angle comparison used in step 1."""


#     plt.figure(figsize=(12, 5))
#     plt.plot(
#         torch.rad2deg(steering_actions_without_penalty),
#         label="No action penalty",
#         linewidth=2,
#         alpha=0.8,
#     )
#     plt.plot(
#         torch.rad2deg(steering_actions_with_penalty),
#         label="With action penalty",
#         linewidth=2,
#         alpha=0.8,
#     )
#     plt.axhline(35, color="r", linestyle="--", alpha=0.5, label="+/-35 deg limit")
#     plt.axhline(-35, color="r", linestyle="--", alpha=0.5)
#     plt.xlabel("Time Step")
#     plt.ylabel("Steering Angle (degrees)")
#     plt.title(f"Steering Actions Comparison\nInitial: x0={initial_state.tolist()[:2]}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def plot_step_1_optimal_control_steering_angle_comparison_with_and_without_action_penalty(
    steering_actions_without_penalty,
    steering_actions_with_penalty,
    initial_state,
):
    """Plot the steering-angle comparison using dark styling."""

    # Set the global style
    plt.style.use(['dark_background', 'bmh'])

    # Initialize figure and axis with your specific requirements
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.set_facecolor('black')

    # Convert tensors to CPU/numpy for plotting if they aren't already
    actions_no_pen = torch.rad2deg(steering_actions_without_penalty).detach().cpu()
    actions_with_pen = torch.rad2deg(steering_actions_with_penalty).detach().cpu()

    # Plotting on the 'ax' object
    ax.plot(
        actions_no_pen,
        label="No action penalty",
        linewidth=2,
        alpha=0.9,  # Slightly higher alpha to pop against black
        color='#00d1ff' # Electric blue
    )
    ax.plot(
        actions_with_pen,
        label="With action penalty",
        linewidth=2,
        alpha=0.9,
        color='#00ff87' # Neon green
    )

    # Constraint lines - adjusted to a brighter red for dark mode visibility
    ax.axhline(35, color="#ff4d4d", linestyle="--", alpha=0.6, label="+/-35 deg limit")
    ax.axhline(-35, color="#ff4d4d", linestyle="--", alpha=0.6)

    # Labeling using the setter methods
    ax.set_xlabel("Time Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Steering Angle (degrees)", fontsize=10, fontweight='bold')
    ax.set_title(
        f"Steering Actions Comparison\nInitial: x0={initial_state.tolist()[:2]}",
        fontsize=12,
        pad=15
    )

    # Legend and Grid adjustments
    ax.legend(loc='upper right', frameon=True, facecolor='#111111', edgecolor='gray')
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.show()

def display_step_1_side_by_side_optimal_control_trajectory_animation_comparison(
    trajectories_to_compare,
    plotting_config,
    trajectory_plotter_function,
):
    """Create and display the step 1 trajectory animation comparison."""
    comparison_animation = trajectory_plotter_function(
        trajectories_to_compare,
        plotting_config,
        y_target=torch.zeros(4),
        labels=["No action penalty", "With action penalty"],
        colors=["C0", "C1"],
        save_path="vis/action_penalty_comparison.gif",
    )
    display(comparison_animation)
