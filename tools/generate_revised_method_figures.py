from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


OUT = Path(r"E:\project\xiaozhi\Uni-Sign\docs\04_论文稿件\回复编辑论文内容\figures")

C = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "pose": "#E3F2FD",
    "pose_edge": "#1976B9",
    "rgb": "#FDE8EC",
    "rgb_edge": "#CF4F68",
    "meta": "#FFF2CF",
    "meta_edge": "#C98A12",
    "gate": "#E5F5E8",
    "gate_edge": "#2E8B57",
    "fusion": "#EEF1F5",
    "fusion_edge": "#8A99A8",
    "out": "#EEE7FA",
    "out_edge": "#7152A3",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, title, subtitle="", face="#FFFFFF", edge="#445566",
        title_size=13, subtitle_size=10.2, radius=0.018, lw=1.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=C["ink"])
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=C["ink"], linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#52606D", lw=1.9, rad=0.0, dashed=False, scale=14):
    style = "arc3" if rad == 0 else f"arc3,rad={rad}"
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, connectionstyle=style,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def gate_figure():
    fig, ax = plt.subplots(figsize=(14.2, 7.3), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "Node-Wise Pose-Preserving Gate", ha="center", va="top",
            fontsize=23, fontweight="bold", color=C["ink"])
    ax.text(0.5, 0.905,
            "The gate updates selected hand nodes while retaining an explicit pose path",
            ha="center", va="top", fontsize=12.5, color=C["muted"])

    ax.text(0.16, 0.84, "Inputs at one selected hand node", ha="center",
            fontsize=13.2, fontweight="bold", color=C["ink"])

    box(ax, 0.035, 0.64, 0.25, 0.14,
        "Aligned RGB-pose feature", "Output of pose-guided alignment\n(not raw RGB)",
        C["rgb"], C["rgb_edge"], 13.0, 10.1)
    box(ax, 0.035, 0.43, 0.25, 0.14,
        "Pose feature", "Feature from the inherited pose encoder",
        C["pose"], C["pose_edge"], 13.0, 10.1)
    box(ax, 0.035, 0.22, 0.25, 0.14,
        "Optional metadata embedding", "Detector-side summary after projection",
        C["meta"], C["meta_edge"], 12.2, 9.9)

    box(ax, 0.37, 0.52, 0.22, 0.20,
        "Gate predictor", "Concatenate available inputs\nTwo pointwise layers + sigmoid",
        C["gate"], C["gate_edge"], 14.2, 10.7)
    alpha = Circle((0.68, 0.62), 0.055, facecolor=C["white"], edgecolor=C["gate_edge"], linewidth=2.0)
    ax.add_patch(alpha)
    ax.text(0.68, 0.635, "α", ha="center", va="center", fontsize=22,
            fontweight="bold", color=C["gate_edge"])
    ax.text(0.68, 0.585, "node-wise weight", ha="center", va="center",
            fontsize=8.8, color=C["muted"])

    box(ax, 0.77, 0.46, 0.18, 0.23,
        "Pose-preserving blend", "Fused path weighted by α\nPose path weighted by 1 − α",
        C["fusion"], C["fusion_edge"], 13.0, 10.2)
    box(ax, 0.77, 0.18, 0.18, 0.14,
        "Updated hand-node feature", "Selected-node output",
        C["out"], C["out_edge"], 12.1, 10.0)

    # Predictor inputs
    arrow(ax, (0.285, 0.71), (0.37, 0.66), C["rgb_edge"])
    arrow(ax, (0.285, 0.50), (0.37, 0.62), C["pose_edge"])
    arrow(ax, (0.285, 0.29), (0.37, 0.57), C["meta_edge"])

    # Gate weight and blend inputs
    arrow(ax, (0.59, 0.62), (0.625, 0.62), C["gate_edge"])
    arrow(ax, (0.735, 0.62), (0.77, 0.60), C["gate_edge"])
    ax.text(0.752, 0.645, "controls", ha="center", fontsize=9.2, color=C["gate_edge"])
    arrow(ax, (0.285, 0.71), (0.77, 0.64), C["rgb_edge"], rad=-0.14)
    arrow(ax, (0.285, 0.50), (0.77, 0.51), C["pose_edge"], rad=0.04)
    arrow(ax, (0.86, 0.46), (0.86, 0.32), C["out_edge"])

    # Optionality and bypass note
    note = FancyBboxPatch((0.035, 0.055), 0.64, 0.095,
                          boxstyle="round,pad=0.012,rounding_size=0.014",
                          facecolor="#F8FAFC", edgecolor="#B8C2CC", linewidth=1.4)
    ax.add_patch(note)
    ax.text(0.055, 0.103,
            "RGB gate: metadata input omitted.   Pose-only: complete RGB-support block bypassed.\n"
            "Unselected pose nodes remain unchanged.",
            ha="left", va="center", fontsize=10.4, color=C["ink"], linespacing=1.35)

    ax.text(0.73, 0.105,
            "Color follows the three labeled inputs;\n"
            "green denotes the learned gate control.",
            fontsize=9.5, color=C["muted"], ha="left", va="center", linespacing=1.35)

    plt.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    save(fig, "figure4_revised_gate")


def framework_figure():
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "Sparse Hand-Crop RGB Support for Pose-Based SLT",
            ha="center", va="top", fontsize=17.2, fontweight="bold", color=C["ink"])
    ax.text(0.5, 0.945,
            "Core evaluated pipeline on CE-CSL",
            ha="center", va="top", fontsize=12.0, color=C["muted"])

    # Inputs
    box(ax, 0.10, 0.795, 0.35, 0.085,
        "Pose keypoints", "RTMLib whole-body sequence\nretained for all frames",
        C["pose"], C["pose_edge"], 13.2, 10.0)
    box(ax, 0.55, 0.795, 0.35, 0.085,
        "Hand RGB crops", "Left/right crop candidates\nfrom the same video",
        C["rgb"], C["rgb_edge"], 13.2, 10.0)

    # Selection and encoders
    box(ax, 0.10, 0.625, 0.35, 0.095,
        "Pose encoder", "Body, face, and both hands\nFull retained timeline",
        C["pose"], C["pose_edge"], 13.2, 10.0)
    box(ax, 0.55, 0.655, 0.35, 0.065,
        "Per-hand frame selection", "Eligibility check + ranked crop budget",
        C["rgb"], C["rgb_edge"], 12.6, 9.7)
    box(ax, 0.55, 0.555, 0.35, 0.065,
        "RGB appearance encoder", "Selected crops only",
        C["rgb"], C["rgb_edge"], 12.6, 9.7)
    box(ax, 0.10, 0.525, 0.35, 0.065,
        "Optional metadata", "Confidence and selection summaries",
        C["meta"], C["meta_edge"], 12.2, 9.5)

    arrow(ax, (0.275, 0.795), (0.275, 0.72), C["pose_edge"])
    arrow(ax, (0.725, 0.795), (0.725, 0.72), C["rgb_edge"])
    arrow(ax, (0.45, 0.835), (0.55, 0.688), C["meta_edge"], rad=-0.15)
    ax.text(0.505, 0.765, "pose scores", fontsize=8.8, color=C["meta_edge"], ha="center")
    arrow(ax, (0.725, 0.655), (0.725, 0.62), C["rgb_edge"])
    arrow(ax, (0.60, 0.655), (0.45, 0.558), C["meta_edge"], rad=-0.12)
    ax.text(0.515, 0.610, "selection summary", fontsize=8.5, color=C["meta_edge"], ha="center")

    # Alignment
    box(ax, 0.12, 0.410, 0.76, 0.070,
        "Pose-guided RGB alignment",
        "Selected pose features and selected RGB appearance are aligned",
        C["fusion"], C["fusion_edge"], 13.2, 9.8)
    arrow(ax, (0.275, 0.625), (0.34, 0.48), C["pose_edge"], rad=0.08)
    arrow(ax, (0.725, 0.555), (0.66, 0.48), C["rgb_edge"], rad=-0.08)

    # Gate
    box(ax, 0.12, 0.230, 0.76, 0.085,
        "Node-wise pose-preserving gate",
        "Inputs: pose feature, aligned RGB-pose feature, optional metadata\n"
        "Selected hand nodes are updated; unselected pose nodes pass through unchanged",
        C["gate"], C["gate_edge"], 13.0, 9.6)
    arrow(ax, (0.50, 0.410), (0.50, 0.315), C["gate_edge"])
    arrow(ax, (0.10, 0.67), (0.12, 0.265), C["pose_edge"],
          lw=1.5, rad=0.10, dashed=True)
    ax.text(0.065, 0.455, "pose skip path", fontsize=8.4,
            color=C["pose_edge"], rotation=90, ha="center", va="center")
    arrow(ax, (0.275, 0.525), (0.35, 0.315), C["meta_edge"], rad=-0.08)

    # Translation
    box(ax, 0.12, 0.055, 0.76, 0.085,
        "Sequence encoding and translation",
        "Merge body, face, and hand features\nProject to mT5-base → generate the Chinese sentence",
        C["out"], C["out_edge"], 13.0, 9.8)
    arrow(ax, (0.50, 0.230), (0.50, 0.14), C["out_edge"])

    # Variants footer
    ax.text(0.5, 0.028, "Evaluated configurations", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color=C["ink"])
    items = [(0.18, C["pose_edge"], "Pose-only"),
             (0.43, C["gate_edge"], "RGB gate"),
             (0.67, C["meta_edge"], "RGB + metadata gate")]
    for x, color, label in items:
        ax.plot([x, x + 0.035], [0.010, 0.010], lw=4, color=color, solid_capstyle="round")
        ax.text(x + 0.045, 0.010, label, ha="left", va="center", fontsize=8.8, color=C["ink"])

    plt.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.015)
    save(fig, "framework_revised_a4")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gate_figure()
    framework_figure()
