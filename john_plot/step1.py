import matplotlib.pyplot as plt
import torch
from IPython.display import display


def plot_step_1_optimal_control_steering_angle_comparison_with_and_without_action_penalty(
    steering_actions_without_penalty,
    steering_actions_with_penalty,
    initial_state,
):
    """Plot the steering-angle comparison used in step 1."""
    plt.figure(figsize=(12, 5))
    plt.plot(
        torch.rad2deg(steering_actions_without_penalty),
        label="No action penalty",
        linewidth=2,
        alpha=0.8,
    )
    plt.plot(
        torch.rad2deg(steering_actions_with_penalty),
        label="With action penalty",
        linewidth=2,
        alpha=0.8,
    )
    plt.axhline(35, color="r", linestyle="--", alpha=0.5, label="+/-35 deg limit")
    plt.axhline(-35, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Time Step")
    plt.ylabel("Steering Angle (degrees)")
    plt.title(f"Steering Actions Comparison\nInitial: x0={initial_state.tolist()[:2]}")
    plt.legend()
    plt.grid(True, alpha=0.3)
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
