import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(r"E:\project\xiaozhi\Uni-Sign")
METRIC_DIR = PROJECT_ROOT / "out" / "paper_metrics"
FIGURE_DIR = PROJECT_ROOT / "docs" / "04_论文稿件" / "2026-04-25_论文定稿材料" / "figures"
OUT_PNG = FIGURE_DIR / "figure_low_resource_trend.png"
OUT_PDF = FIGURE_DIR / "figure_low_resource_trend.pdf"


def load_split(split: str) -> dict[str, dict[int, float]]:
    path = METRIC_DIR / f"low_resource_{split}100_summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, dict[int, float]] = {
        "pose-only": {},
        "naive RGB": {},
        "full model": {},
    }
    name_map = {
        "pose_only": "pose-only",
        "naive_rgb": "naive RGB",
        "full_model": "full model",
    }
    for row in rows:
        label = row["label"]
        ratio = int(label.split("_")[0].replace("lr", ""))
        for key, model_name in name_map.items():
            if key in label:
                parsed[model_name][ratio] = float(row["average_similarity"])
                break
    return parsed


def plot_panel(ax, title: str, data: dict[str, dict[int, float]]) -> None:
    ratios = [10, 25, 50, 100]
    colors = {
        "pose-only": "#1f4e79",
        "naive RGB": "#d97706",
        "full model": "#2f855a",
    }
    markers = {
        "pose-only": "o",
        "naive RGB": "s",
        "full model": "^",
    }

    for model_name in ["pose-only", "naive RGB", "full model"]:
        values = [data[model_name][ratio] for ratio in ratios]
        ax.plot(
            ratios,
            values,
            marker=markers[model_name],
            linewidth=2.3,
            markersize=7,
            color=colors[model_name],
            label=model_name,
        )
        for x, y in zip(ratios, values):
            ax.text(x, y + 0.0012, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color=colors[model_name])

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training ratio (%)", fontsize=10)
    ax.set_ylabel("AvgSim", fontsize=10)
    ax.set_xticks(ratios)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dev_data = load_split("dev")
    test_data = load_split("test")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.patch.set_facecolor("white")

    plot_panel(axes[0], "(a) Dev100", dev_data)
    plot_panel(axes[1], "(b) Test100", test_data)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Low-resource trend across CE-CSL training ratios", fontsize=14, fontweight="bold", y=1.08)
    fig.text(
        0.5,
        -0.02,
        "Pose-only remains the strongest anchor, and the full model becomes more competitive than naive RGB in the stronger settings without dominating every ratio.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
