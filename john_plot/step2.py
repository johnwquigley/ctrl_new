import matplotlib.pyplot as plt
import torch
from IPython.display import display


def plot_step_2_single_trajectory_memorization_loss_and_closed_loop_trailer_drift(
    memorization_losses,
    trailer_position_drift_per_time_step,
):
    """Plot the two-panel memorization diagnostics used in step 2."""
    with plt.style.context(["dark_background", "bmh"]):
        fig, (loss_axis, drift_axis) = plt.subplots(1, 2, figsize=(16, 5), facecolor="black")

        loss_axis.set_facecolor("black")
        loss_axis.plot(memorization_losses, linewidth=2, color="#00d4ff")
        loss_axis.set_yscale("log")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("MSE Loss")
        loss_axis.set_title("Single-Trajectory Memorization Loss")
        loss_axis.grid(True, alpha=0.15)

        drift_axis.set_facecolor("black")
        drift_axis.plot(trailer_position_drift_per_time_step, linewidth=2, color="#ff2a6d")
        drift_axis.set_xlabel("Time Step")
        drift_axis.set_ylabel("Trailer Position Drift")
        drift_axis.set_title("Closed-Loop Drift Away From the Expert Trajectory")
        drift_axis.grid(True, alpha=0.15)

        plt.tight_layout()
        plt.show()


def display_step_2_supervised_controller_rollout_animation(
    controller_rollout_states,
    plotting_config,
    trajectory_plotter_function,
):
    """Create and display the step 2 validation rollout animation."""
    rollout_animation = trajectory_plotter_function(
        controller_rollout_states,
        plotting_config,
        y_target=torch.zeros(4),
        save_path="vis/rollout.gif",
    )
    display(rollout_animation)


def display_step_2_expert_trajectory_and_memorized_controller_rollout_animation(
    expert_trajectory_states,
    memorized_controller_rollout_states,
    plotting_config,
    target_state,
    trajectory_plotter_function,
):
    """Create and display the step 2 expert-vs-memorized rollout animation."""
    comparison_animation = trajectory_plotter_function(
        [expert_trajectory_states, memorized_controller_rollout_states],
        plotting_config,
        y_target=target_state,
        labels=["Expert trajectory", "Memorized controller rollout"],
        colors=["C0", "C1"],
        save_path="vis/memorized_vs_expert.gif",
    )
    display(comparison_animation)


def plot_step_2_supervised_learning_training_and_validation_loss_curves(
    training_losses,
    validation_losses,
):
    """Plot the supervised training curves used in step 2."""
    with plt.style.context(["dark_background", "bmh"]):
        plt.figure(figsize=(10, 4), facecolor="black")

        axis = plt.gca()
        axis.set_facecolor("black")

        plt.plot(training_losses, label="Train", linewidth=2, alpha=0.8, color="#1f77b4")
        plt.plot(validation_losses, label="Validation", linewidth=2, alpha=0.8, color="#ff7f0e")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Supervised Training Loss", pad=15)

        plt.legend(facecolor="black", edgecolor="none", framealpha=0.7)
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.show()

    print(f"Final training loss: {training_losses[-1]:.6f}")
    print(f"Final validation loss: {validation_losses[-1]:.6f}")
