import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reviewer gate and subgroup analysis outputs.")
    parser.add_argument("--analysis-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_gate(gate_dir):
    frames = read_csv(gate_dir / "gate_values_per_frame.csv")
    gate = np.asarray([float(row["gate_mean"]) for row in frames])
    confidence = np.asarray([float(row["pose_confidence"]) for row in frames])
    hands = np.asarray([row["hand"] for row in frames])

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    bins = np.linspace(0.05, 0.5, 31)
    axes[0].hist(gate[hands == "left"], bins=bins, density=True, alpha=0.65, label="Left hand")
    axes[0].hist(gate[hands == "right"], bins=bins, density=True, alpha=0.65, label="Right hand")
    axes[0].axvline(gate.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Overall mean={gate.mean():.3f}")
    axes[0].set_xlabel("Mean fusion gate per selected frame")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Gate distribution")

    edges = np.asarray([0.0, 0.3, 0.5, 0.7, 0.9, 1.000001])
    centers, means, errors, counts = [], [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = gate[(confidence >= lower) & (confidence < upper)]
        if selected.size:
            centers.append((lower + min(upper, 1.0)) / 2)
            means.append(selected.mean())
            errors.append(selected.std(ddof=1) if selected.size > 1 else 0.0)
            counts.append(selected.size)
    axes[1].errorbar(centers, means, yerr=errors, marker="o", capsize=3, color="#3b6fb6")
    for x_value, y_value, count in zip(centers, means, counts):
        axes[1].annotate(f"n={count}", (x_value, y_value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axes[1].set_xlabel("Pose-confidence metadata bin")
    axes[1].set_ylabel("Mean fusion gate (mean ± SD)")
    axes[1].set_ylim(0.08, 0.23)
    axes[1].set_title("Gate versus pose confidence")
    figure.savefig(gate_dir / "gate_distribution_figure.png", dpi=300)
    figure.savefig(gate_dir / "gate_distribution_figure.pdf")
    plt.close(figure)


def plot_subgroups(subgroup_dir):
    rows = read_csv(subgroup_dir / "subgroup_aggregate_metrics.csv")
    groups = ["short_le8", "long_ge11", "question", "negation", "numeral"]
    variants = ["pose_only", "rgb_gate", "rgb_metadata_gate"]
    labels = {"short_le8": "Short", "long_ge11": "Long", "question": "Question", "negation": "Negation", "numeral": "Numeral"}
    colors = {"pose_only": "#777777", "rgb_gate": "#3b6fb6", "rgb_metadata_gate": "#d05a4e"}
    names = {"pose_only": "Pose only", "rgb_gate": "RGB gate", "rgb_metadata_gate": "RGB+metadata gate"}
    lookup = {(row["subgroup"], row["variant"]): row for row in rows}
    metrics = [("bleu4", "BLEU-4"), ("average_similarity", "AvgSim"), ("standard_cer", "CER ↓")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    x_values = np.arange(len(groups))
    width = 0.24
    for axis, (field, ylabel) in zip(axes, metrics):
        for offset, variant in enumerate(variants):
            values = [float(lookup[(group, variant)][f"{field}_mean"]) for group in groups]
            errors = [float(lookup[(group, variant)][f"{field}_std_across_seeds"]) for group in groups]
            axis.bar(x_values + (offset - 1) * width, values, width, yerr=errors, capsize=2, color=colors[variant], alpha=0.9, label=names[variant])
        axis.set_xticks(x_values, [labels[group] for group in groups], rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(subgroup_dir / "subgroup_metrics_figure.png", dpi=300)
    figure.savefig(subgroup_dir / "subgroup_metrics_figure.pdf")
    plt.close(figure)


def main():
    args = parse_args()
    plot_gate(args.analysis_root / "gate_distribution_seed42_topk10")
    plot_subgroups(args.analysis_root / "subgroups_7seeds_test500")
    print(f"Figures written under: {args.analysis_root}")


if __name__ == "__main__":
    main()
