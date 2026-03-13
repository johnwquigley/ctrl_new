from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches


# Eval assumptions from eval.py
VALID_REGION_X = (0.0, 40.0)
VALID_REGION_Y = (-15.0, 15.0)

# Extracted from your reported eval output (xy_only, max_steps=600).
XY_ONLY_SUCCESS = {
    "final_dist<1.0": [0.863, 0.980, 1.000, 1.000, 0.992, 0.996, 0.973, 0.961, 0.910, 0.988],
    "final_dist<0.1": [0.316, 0.344, 0.414, 0.469, 0.434, 0.402, 0.332, 0.230, 0.199, 0.309],
}


def create_xy_only_curriculum_rectangles(
    num_lessons: int = 10,
    x_cab_range: tuple[float, float] = (10.0, 35.0),
    y_cab_range_abs: tuple[float, float] = (2.0, 7.0),
) -> list[dict]:
    """Mirror eval.py create_train_configs_tbu(..., persist_max_angles=True)."""
    n = num_lessons - 1
    x_first, x_final = x_cab_range
    y_first, y_final = y_cab_range_abs

    rects: list[dict] = []
    x_lower = x_first
    denom = max(1, n - 1)
    x_upper = x_first
    y_upper = y_first

    for stage in range(1, n + 1):
        t = (stage - 1) / float(denom)
        x_upper = x_first + (x_final - x_first) * t
        y_upper = y_first + (y_final - y_first) * t
        rects.append(
            {
                "stage": stage,
                "x_range": (x_lower, x_upper),
                "y_range": (-y_upper, y_upper),
            }
        )
        x_lower = x_upper

    rects.append(
        {
            "stage": n + 1,
            "x_range": (x_first, x_upper),
            "y_range": (-y_upper, y_upper),
        }
    )
    return rects


def _style_axes(ax):
    ax.set_facecolor("#0f141a")
    ax.grid(True, alpha=0.20, linestyle="-", linewidth=0.8)
    ax.tick_params(colors="#e6e6e6")
    for spine in ax.spines.values():
        spine.set_color("#bfc7cf")
    ax.xaxis.label.set_color("#f2f2f2")
    ax.yaxis.label.set_color("#f2f2f2")


def _draw_valid_region(ax):
    x0, x1 = VALID_REGION_X
    y0, y1 = VALID_REGION_Y
    valid = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor="#e5e5e5",
        linewidth=1.8,
        linestyle="-.",
        zorder=2,
        label="Valid Region",
    )
    ax.add_patch(valid)
    ax.scatter([0.0], [0.0], marker="x", s=80, c="#d9d9d9", linewidths=2.0, zorder=7, label="Dock (0,0)")


def _metric_title(metric_name: str) -> str:
    if metric_name == "final_dist<1.0":
        return "Success Rate by Curriculum Rectangle (final_dist < 1.0m)"
    return "Success Rate by Curriculum Rectangle (final_dist < 0.1m)"


def plot_curriculum_rectangles(metric_name: str, rates: list[float], rects: list[dict], out_path: Path) -> None:
    """Draw valid region + curriculum init rectangles colored by success rate."""
    with plt.style.context(["dark_background", "bmh"]):
        fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=220)
        fig.patch.set_facecolor("#0b0f14")
        _style_axes(ax)
        _draw_valid_region(ax)

        stage_to_rate = {i + 1: rates[i] for i in range(len(rates))}

        # Draw stage-10 full rectangle first so progressive strips remain visible.
        draw_order = [10] + list(range(1, 10))
        for stage in draw_order:
            rect = next(r for r in rects if r["stage"] == stage)
            rate = stage_to_rate[stage]
            # Keep visual simple: one hue family, alpha carries stage layering.
            col = "#2e7d32" if rate >= 0.95 else "#3f8f46" if rate >= 0.85 else "#66a86a"
            x0, x1 = rect["x_range"]
            y0, y1 = rect["y_range"]
            width = x1 - x0
            height = y1 - y0

            if abs(width) < 1e-8:
                ax.plot([x0, x0], [y0, y1], color=col, linewidth=3.0, alpha=0.95, zorder=5)
                label_x = x0 + 0.50
            else:
                alpha = 0.10 if stage == 10 else 0.18
                patch = patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor=col,
                    edgecolor=col,
                    linewidth=2.0,
                    alpha=alpha,
                    zorder=3 if stage == 10 else 5,
                )
                ax.add_patch(patch)
                label_x = x0 + 0.5 * width

            # Stage 1 has near-zero width in xy_only curriculum; move label out with connector.
            if stage == 1 or abs(width) < 1e-8:
                text_x = x0 - 1.4
                text_y = y1 + 0.8
                ax.annotate(
                    f"S{stage}: {100.0 * rate:.1f}%",
                    xy=(x0, y1 - 0.15),
                    xytext=(text_x, text_y),
                    textcoords="data",
                    ha="right",
                    va="bottom",
                    fontsize=9.2,
                    color="#f4f4f4",
                    bbox={"boxstyle": "round,pad=0.18", "fc": (0, 0, 0, 0.58), "ec": "#d0d0d0", "lw": 0.8},
                    arrowprops={"arrowstyle": "-", "color": "#d0d0d0", "lw": 1.0},
                    zorder=10,
                    clip_on=False,
                )
            else:
                label_y = y1 - 0.35
                ax.text(
                    label_x,
                    label_y,
                    f"S{stage}: {100.0 * rate:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#f4f4f4",
                    zorder=8,
                    bbox={"boxstyle": "round,pad=0.18", "fc": (0, 0, 0, 0.45), "ec": "none"},
                )

        ax.set_xlim(VALID_REGION_X[0] - 1.0, VALID_REGION_X[1] + 1.4)
        ax.set_ylim(VALID_REGION_Y[0] - 1.0, VALID_REGION_Y[1] + 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(_metric_title(metric_name), fontsize=13.5, pad=12, weight="semibold", color="#f2f2f2")
        handles, labels = ax.get_legend_handles_labels()
        curriculum_handle = patches.Patch(
            facecolor="#3f8f46",
            edgecolor="#3f8f46",
            alpha=0.20,
            label="Curriculum Init Regions",
        )
        handles.append(curriculum_handle)
        labels.append("Curriculum Init Regions")
        leg = ax.legend(
            handles,
            labels,
            loc="upper left",
            facecolor="#141b22",
            edgecolor="#cfcfcf",
            framealpha=0.95,
            fontsize=9,
        )
        for text in leg.get_texts():
            text.set_color("#f0f0f0")

        ax.text(
            0.985,
            0.035,
            "S1-S10: curriculum stage\nS#: xx.x% = stage success rate",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#f0f0f0",
            bbox={"boxstyle": "round,pad=0.24", "fc": (0, 0, 0, 0.45), "ec": "#cfcfcf", "lw": 0.7},
        )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    out_dir = Path("vis")
    curriculum_rects = create_xy_only_curriculum_rectangles()

    out_lt_1 = out_dir / "xy_only_curriculum_rectangles_lt_1p0.png"
    out_lt_01 = out_dir / "xy_only_curriculum_rectangles_lt_0p1.png"

    plot_curriculum_rectangles(
        "final_dist<1.0",
        XY_ONLY_SUCCESS["final_dist<1.0"],
        curriculum_rects,
        out_lt_1,
    )
    plot_curriculum_rectangles(
        "final_dist<0.1",
        XY_ONLY_SUCCESS["final_dist<0.1"],
        curriculum_rects,
        out_lt_01,
    )

    print("Wrote:")
    print(out_lt_1)
    print(out_lt_01)


if __name__ == "__main__":
    main()
