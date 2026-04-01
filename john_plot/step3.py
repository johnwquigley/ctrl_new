import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_step_3_neural_emulator_training_curves(
    emulator_training_losses,
    emulator_validation_losses,
):
    """Plot the emulator learning curves used in step 3."""
    with plt.style.context(["dark_background", "bmh"]):
        plt.figure(figsize=(8, 4), facecolor="black")

        axis = plt.gca()
        axis.set_facecolor("black")

        epochs = list(range(1, len(emulator_training_losses) + 1))

        plt.plot(
            epochs,
            emulator_training_losses,
            "o-",
            label="Train",
            linewidth=2,
            markersize=6,
            color="#1f77b4",
        )
        plt.plot(
            epochs,
            emulator_validation_losses,
            "o-",
            label="Validation",
            linewidth=2,
            markersize=6,
            color="#ff7f0e",
        )

        plt.xlabel("Epoch", color="white")
        plt.ylabel("MSE Loss", color="white")
        plt.title("Emulator Training Curves", color="white", pad=10)

        plt.legend(facecolor="black", edgecolor="none", framealpha=0.7)
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.show()


def plot_step_3_controller_training_loss_across_all_batches(
    controller_batch_total_losses,
):
    """Plot the controller loss history used in step 3."""
    with plt.style.context(["dark_background", "bmh"]):
        plt.figure(figsize=(10, 4), facecolor="black")

        axis = plt.gca()
        axis.set_facecolor("black")

        plt.plot(
            controller_batch_total_losses,
            "o-",
            color="#00d4ff",
            linewidth=2,
            markersize=4,
            alpha=0.9,
        )

        plt.xlabel("Batch / Epoch", color="white")
        plt.ylabel("Total Loss", color="white")
        plt.title("Controller Training Loss (All Batches)", color="white", pad=15)

        plt.grid(True, alpha=0.15, linestyle="--")
        plt.tight_layout()
        plt.show()


def plot_step_3_emulator_vs_physics_rollout_comparison_dashboard(
    emulator_states,
    emulator_actions,
    emulator_final_distance,
    physics_states,
    physics_actions,
    physics_final_distance,
    initial_state,
    trailer_xy_function,
    plot_truck_state_and_action_function,
):
    """Plot the 2x2 rollout comparison dashboard used in step 3."""
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plot_truck_state_and_action_function(
        emulator_states,
        emulator_actions,
        title=f"Emulator Rollout\n(dist={emulator_final_distance:.3f})",
    )

    plt.subplot(2, 2, 2)
    plot_truck_state_and_action_function(
        physics_states,
        physics_actions,
        title=f"Physics Rollout\n(dist={physics_final_distance:.3f})",
    )

    plt.subplot(2, 2, 3)
    emulator_trailer_x, emulator_trailer_y = trailer_xy_function(emulator_states)
    physics_trailer_x, physics_trailer_y = trailer_xy_function(physics_states)

    plt.plot(emulator_trailer_x, emulator_trailer_y, label="Emulator", linewidth=2, alpha=0.8)
    plt.plot(physics_trailer_x, physics_trailer_y, label="Physics", linewidth=2, alpha=0.8)
    plt.scatter(0, 0, color="red", s=100, marker="*", label="Target", zorder=5)
    plt.scatter(
        initial_state[0].cpu(),
        initial_state[1].cpu(),
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.subplot(2, 2, 4)
    emulator_actions_to_plot = emulator_actions.squeeze() if len(emulator_actions) > 0 else torch.tensor([])
    physics_actions_to_plot = physics_actions.squeeze() if len(physics_actions) > 0 else torch.tensor([])

    if len(emulator_actions_to_plot) > 0:
        plt.plot(torch.rad2deg(emulator_actions_to_plot), label="Emulator", linewidth=2, alpha=0.8)
    if len(physics_actions_to_plot) > 0:
        plt.plot(torch.rad2deg(physics_actions_to_plot), label="Physics", linewidth=2, alpha=0.8)

    plt.axhline(45, color="r", linestyle="--", alpha=0.5, label="+/-45 deg limit")
    plt.axhline(-45, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Time Step")
    plt.ylabel("Steering Angle (degrees)")
    plt.title("Action Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_step_3_three_way_approach_comparison_summary(
    results_by_approach,
    controller_success_radius,
):
    """Plot the final three-panel comparison across all approaches."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    valid_results = {}
    for approach_name, error_values in results_by_approach.items():
        finite_errors = [error for error in error_values if error != float("inf")]
        if finite_errors:
            valid_results[approach_name] = finite_errors

    if valid_results:
        plt.boxplot(valid_results.values(), labels=valid_results.keys())
        plt.ylabel("Final Distance Error")
        plt.title("Error Distribution Comparison")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    success_rate_approach_labels = []
    success_rates = []
    for approach_name, error_values in results_by_approach.items():
        finite_errors = [error for error in error_values if error != float("inf")]
        if finite_errors:
            success_rate_approach_labels.append(approach_name.replace(" ", "\n"))
            success_rates.append(
                sum(1 for error in finite_errors if error < controller_success_radius) / len(finite_errors)
            )

    if success_rate_approach_labels:
        bars = plt.bar(success_rate_approach_labels, success_rates, alpha=0.7)
        plt.ylabel("Success Rate")
        plt.title("Success Rate Comparison")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        for bar, success_rate in zip(bars, success_rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{success_rate:.1%}",
                ha="center",
                va="bottom",
            )
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    mean_error_approach_labels = []
    mean_errors = []
    for approach_name, error_values in results_by_approach.items():
        finite_errors = [error for error in error_values if error != float("inf")]
        if finite_errors:
            mean_error_approach_labels.append(approach_name.replace(" ", "\n"))
            mean_errors.append(np.mean(finite_errors))

    if mean_error_approach_labels:
        bars = plt.bar(mean_error_approach_labels, mean_errors, alpha=0.7, color="orange")
        plt.ylabel("Mean Final Distance Error")
        plt.title("Mean Error Comparison")
        plt.xticks(rotation=45)
        for bar, mean_error in zip(bars, mean_errors):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mean_errors) * 0.02,
                f"{mean_error:.3f}",
                ha="center",
                va="bottom",
            )
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
