from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(
    r"E:\project\xiaozhi\Uni-Sign\docs\04_论文稿件\回复编辑论文内容\figures"
)


COLORS = {
    "pose": "#DCEEFF",
    "pose_edge": "#2774AE",
    "rgb": "#FFE4E8",
    "rgb_edge": "#C94C65",
    "meta": "#FFF1CC",
    "meta_edge": "#C58A16",
    "gate": "#E5F5E8",
    "gate_edge": "#358B55",
    "text": "#1F2933",
    "muted": "#52606D",
    "panel": "#F7F9FB",
    "panel_edge": "#A9B5C1",
    "output": "#EEE7FA",
    "output_edge": "#7654A7",
}


def rounded_box(ax, x, y, w, h, face, edge, title, body=None, title_size=14, body_size=11):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.69,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.31,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["text"],
            linespacing=1.35,
        )
    return box


def arrow(ax, start, end, color="#52606D", width=1.8, style="-|>", connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=width,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.3, 10.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Evaluated Sparse RGB Support Pipeline",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.945,
        "Pose remains the complete-sequence backbone; RGB supports selected hand frames only",
        ha="center",
        va="top",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    # 1. Inputs
    rounded_box(
        ax,
        0.08,
        0.81,
        0.36,
        0.105,
        COLORS["pose"],
        COLORS["pose_edge"],
        "Pose stream",
        "RTMLib whole-body keypoints\nFull retained timeline",
    )
    rounded_box(
        ax,
        0.56,
        0.81,
        0.36,
        0.105,
        COLORS["rgb"],
        COLORS["rgb_edge"],
        "RGB stream",
        "Left/right hand crops\nCandidate frames from the same video",
    )

    # 2. Selection and encoders
    rounded_box(
        ax,
        0.08,
        0.65,
        0.36,
        0.105,
        COLORS["pose"],
        COLORS["pose_edge"],
        "Per-part spatial pose encoding",
        "Body, face, and both hands\nShared weights for left/right hands",
    )
    rounded_box(
        ax,
        0.56,
        0.65,
        0.36,
        0.105,
        COLORS["rgb"],
        COLORS["rgb_edge"],
        "Per-hand frame selection",
        "Eligibility threshold + ranked sampling\nMain setting: top 10% per hand",
    )
    arrow(ax, (0.26, 0.81), (0.26, 0.755), COLORS["pose_edge"])
    arrow(ax, (0.74, 0.81), (0.74, 0.755), COLORS["rgb_edge"])
    arrow(ax, (0.44, 0.862), (0.56, 0.702), COLORS["meta_edge"], connectionstyle="arc3,rad=-0.18")
    ax.text(0.505, 0.79, "pose scores", ha="center", fontsize=9.5, color=COLORS["meta_edge"])

    rounded_box(
        ax,
        0.56,
        0.51,
        0.36,
        0.09,
        COLORS["rgb"],
        COLORS["rgb_edge"],
        "RGB appearance encoder",
        "EfficientNet-B0 on selected crops",
        body_size=11,
    )
    arrow(ax, (0.74, 0.65), (0.74, 0.60), COLORS["rgb_edge"])

    rounded_box(
        ax,
        0.08,
        0.51,
        0.36,
        0.09,
        COLORS["meta"],
        COLORS["meta_edge"],
        "Optional detector metadata",
        "Fingertip score, selection density,\nand hand-presence ratio",
        title_size=12.2,
        body_size=9.8,
    )
    arrow(ax, (0.68, 0.65), (0.34, 0.60), COLORS["meta_edge"], connectionstyle="arc3,rad=-0.28")
    ax.text(0.475, 0.625, "selection statistics", fontsize=9, color=COLORS["meta_edge"])

    # 3. Alignment and gate
    rounded_box(
        ax,
        0.14,
        0.36,
        0.72,
        0.09,
        COLORS["panel"],
        COLORS["panel_edge"],
        "Pose-guided RGB alignment",
        "Selected hand-node pose features + selected RGB appearance → fused RGB-pose features",
        title_size=13.2,
        body_size=9.8,
    )
    arrow(ax, (0.08, 0.70), (0.22, 0.43), COLORS["pose_edge"], connectionstyle="arc3,rad=0.28")
    arrow(ax, (0.74, 0.51), (0.62, 0.45), COLORS["rgb_edge"], connectionstyle="arc3,rad=0.08")

    rounded_box(
        ax,
        0.14,
        0.205,
        0.72,
        0.095,
        COLORS["gate"],
        COLORS["gate_edge"],
        "Node-wise pose-preserving gate",
        "Inputs: pose anchor + fused RGB-pose feature + optional metadata\nOnly selected hand nodes receive RGB-guided replacement",
        title_size=13.0,
        body_size=9.6,
    )
    arrow(ax, (0.5, 0.36), (0.5, 0.30), COLORS["gate_edge"])
    arrow(ax, (0.08, 0.68), (0.16, 0.25), COLORS["pose_edge"], connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.26, 0.51), (0.62, 0.30), COLORS["meta_edge"], connectionstyle="arc3,rad=-0.22")

    # 4. Hand-stream reinsertion and inherited temporal encoding
    rounded_box(
        ax,
        0.14,
        0.105,
        0.72,
        0.070,
        COLORS["pose"],
        COLORS["pose_edge"],
        "Hand-stream reinsertion and temporal encoding",
        "Restore hand-node order → add inherited body context → per-part temporal GCN",
        title_size=12.4,
        body_size=8.8,
    )
    arrow(ax, (0.5, 0.205), (0.5, 0.175), COLORS["pose_edge"])

    # 5. Sequence output
    rounded_box(
        ax,
        0.14,
        0.010,
        0.72,
        0.060,
        COLORS["output"],
        COLORS["output_edge"],
        "Part pooling and translation",
        "Mean over nodes (four parts) → concatenate + project → mT5-base encoder-decoder → Chinese sentence",
        title_size=12.4,
        body_size=8.3,
    )
    arrow(ax, (0.5, 0.105), (0.5, 0.070), COLORS["output_edge"])

    plt.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.015)
    fig.savefig(OUTPUT_DIR / "framework_core.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_DIR / "framework_core.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
