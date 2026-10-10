import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(r"E:\project\xiaozhi\Uni-Sign")
INPUT_JSON = PROJECT_ROOT / "out" / "paper_metrics" / "failure_case_analysis_test100.json"
FIGURE_DIR = PROJECT_ROOT / "docs" / "04_论文稿件" / "2026-04-25_论文定稿材料" / "figures"
OUT_PNG = FIGURE_DIR / "figure3_qualitative_cases.png"
OUT_PDF = FIGURE_DIR / "figure3_qualitative_cases.pdf"


def wrap_text(text: str, max_width: int = 34) -> str:
    text = str(text).replace("\n", " ").strip()
    if not text:
        return ""

    lines = []
    current = ""
    for char in text:
        if len(current) >= max_width:
            lines.append(current)
            current = char
        else:
            current += char

    if current:
        lines.append(current)

    return "\n".join(lines[:3])


def make_outcome_text(category: str, case_data: dict) -> str:
    if category == "rgb_helps":
        delta = case_data["full_sim"] - max(case_data["pose_sim"], case_data["rgb_sim"])
        return f"Outcome: Improvement (+{delta:.4f} vs best baseline)"
    if category == "rgb_hurts":
        delta = case_data["pose_sim"] - case_data["full_sim"]
        return f"Outcome: Degradation (-{delta:.4f} vs pose-only)"

    delta = case_data["full_sim"] - case_data["rgb_sim"]
    return f"Outcome: Fallback to pose (+{delta:.4f} vs naive RGB, still wrong)"


def create_publication_quality_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    english_text = {
        210: {
            "reference": "Please go to the reception desk and register first.",
            "pose_pred": "Please go to the front desk first.",
            "rgb_pred": "Please go to the waiting area and take a seat.",
            "full_pred": "Please go to the reception desk and register first.",
        },
        297: {
            "reference": "The medicine should be taken after meals.",
            "pose_pred": "The medicine should be taken after meals.",
            "rgb_pred": "The medicine should be taken before meals.",
            "full_pred": "The medicine should be taken before meals.",
        },
        209: {
            "reference": "You can transfer to the subway at the next station.",
            "pose_pred": "You can transfer to the subway at the next station.",
            "rgb_pred": "You need to get off at the current station.",
            "full_pred": "You can get off at the next station.",
        },
    }

    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig = plt.figure(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.975,
        "Figure 3. Representative Qualitative Cases of Reliability-Aware RGB-Pose Fusion",
        ha="center",
        fontsize=15,
        fontweight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.946,
        "Three representative cases showing when RGB helps, when RGB hurts, and when the gate partially recovers but still fails.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
        family="DejaVu Sans",
    )

    cases = [
        {
            "label": "(a) RGB Helps",
            "category": "rgb_helps",
            "y_top": 0.89,
            "color": "#1b5e20",
            "takeaway": "Takeaway: The full model recovers useful local semantics that both pose-only and naive RGB miss.",
        },
        {
            "label": "(b) RGB Hurts",
            "category": "rgb_hurts",
            "y_top": 0.63,
            "color": "#b71c1c",
            "takeaway": "Takeaway: RGB evidence pulls prediction away from a strong pose-only result, explaining why pose remains the more stable anchor.",
        },
        {
            "label": "(c) Partial Recovery",
            "category": "gate_recovers_but_still_fails",
            "y_top": 0.37,
            "color": "#0d47a1",
            "takeaway": "Takeaway: The gate falls back toward the pose branch, which is the intended behavior, but the final prediction remains incorrect.",
        },
    ]

    for case in cases:
        case_data = data[case["category"]][0]
        case_translation = english_text[case_data["dataset_index"]]
        y = case["y_top"]

        # Section border
        section = Rectangle(
            (0.03, y - 0.21),
            0.94,
            0.22,
            facecolor="white",
            edgecolor="#c9c9c9",
            linewidth=0.8,
        )
        fig.patches.append(section)

        # Header
        fig.text(
            0.05,
            y,
            case["label"],
            fontsize=12,
            fontweight="bold",
            color=case["color"],
            family="DejaVu Sans",
        )
        fig.text(
            0.31,
            y,
            f"Index: #{case_data['dataset_index']}",
            fontsize=11,
            fontweight="bold",
            family="DejaVu Sans",
        )
        fig.text(
            0.50,
            y,
            make_outcome_text(case["category"], case_data),
            fontsize=11,
            fontweight="bold",
            color=case["color"],
            family="DejaVu Sans",
        )

        # Table frame
        left = 0.03
        col_widths = [0.22, 0.245, 0.245, 0.245]
        x_positions = [left]
        for width in col_widths[:-1]:
            x_positions.append(x_positions[-1] + width)

        table_top = y - 0.02
        header_h = 0.055
        body_h = 0.105
        takeaway_h = 0.04

        for idx, width in enumerate(col_widths):
            fig.patches.append(
                Rectangle(
                    (x_positions[idx], table_top - header_h),
                    width,
                    header_h,
                    facecolor="#fafafa",
                    edgecolor="#222222",
                    linewidth=0.5,
                )
            )
            fig.patches.append(
                Rectangle(
                    (x_positions[idx], table_top - header_h - body_h),
                    width,
                    body_h,
                    facecolor="white",
                    edgecolor="#222222",
                    linewidth=0.5,
                )
            )

        headers = ["Ref.", "Pose-only", "Naive RGB", "Full model"]
        texts = [
            case_translation["reference"],
            case_translation["pose_pred"],
            case_translation["rgb_pred"],
            case_translation["full_pred"],
        ]
        scores = [None, case_data["pose_sim"], case_data["rgb_sim"], case_data["full_sim"]]
        colors = ["#111111", "#0d47a1", "#b71c1c", "#1b5e20"]

        for idx, header in enumerate(headers):
            x_center = x_positions[idx] + col_widths[idx] / 2
            fig.text(
                x_center,
                table_top - 0.028,
                header,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=colors[idx],
                family="DejaVu Sans",
            )

            if scores[idx] is not None:
                fig.text(
                    x_center,
                    table_top - 0.048,
                    f"AvgSim: {scores[idx]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=colors[idx],
                    family="DejaVu Sans",
                )

            fig.text(
                x_center,
                table_top - header_h - 0.055,
                wrap_text(texts[idx]),
                ha="center",
                va="center",
                fontsize=8.5,
                family="DejaVu Sans",
            )

        # Takeaway line
        fig.text(
            0.16,
            table_top - header_h - body_h - 0.028,
            case["takeaway"],
            fontsize=8.8,
            family="DejaVu Sans",
            color="#222222",
        )

    fig.text(
        0.5,
        0.03,
        "Cases are selected from Test100 failure-case analysis. Chinese references and predictions are translated into English for readability.",
        ha="center",
        fontsize=8,
        color="#555555",
        family="DejaVu Sans",
    )

    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none", pad_inches=0.15)
    plt.savefig(OUT_PDF, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.15)

    print(f"PNG: {OUT_PNG}")
    print(f"PDF: {OUT_PDF}")


if __name__ == "__main__":
    create_publication_quality_figure()
