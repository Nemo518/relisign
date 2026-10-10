from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _cover_page(pdf: PdfPages, title: str, controls: dict[str, Any], cards: dict[str, Any], quote_lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.06, 0.86, title, fontsize=26, fontweight="bold", color="#0B3954")
    ax.text(0.06, 0.79, f"评委视角：{controls['judge']}  |  展示层级：{controls['stage']}", fontsize=14, color="#3B4A5A")
    ax.text(0.06, 0.73, f"主展示模型：{cards.get('model_name', 'N/A')}", fontsize=16, color="#081E34")

    metric_lines = [
        f"BLEU-4：{cards.get('bleu4', 0.0):.2f}",
        f"ROUGE-L：{cards.get('rouge_l', 0.0):.2f}",
        f"WER：{cards.get('wer', 0.0):.2f}",
        f"Average Similarity：{cards.get('average_similarity', 0.0):.2f}",
    ]
    for idx, line in enumerate(metric_lines):
        ax.text(0.08, 0.63 - idx * 0.07, line, fontsize=15, color="#1F2D3D")

    ax.text(0.56, 0.70, "答辩核心金句", fontsize=18, fontweight="bold", color="#C08429")
    for idx, line in enumerate(quote_lines[:4]):
        ax.text(0.56, 0.63 - idx * 0.10, f"{idx + 1}. {line}", fontsize=13, color="#2D3A45", wrap=True)

    ax.text(0.06, 0.10, "Uni-Sign | view/ 独立演示模块 | FastAPI + Streamlit", fontsize=11, color="#6B7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    subtitle: str,
    column_order: list[str] | None = None,
    max_rows: int = 8,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.06, 0.90, title, fontsize=22, fontweight="bold", color="#0B3954")
    ax.text(0.06, 0.84, subtitle, fontsize=12, color="#4B5563")

    view = dataframe.copy()
    if column_order is not None:
        available = [col for col in column_order if col in view.columns]
        view = view[available]
    view = view.head(max_rows)

    if view.empty:
        ax.text(0.06, 0.70, "当前没有可导出的表格数据。", fontsize=14, color="#6B7280")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.14, 0.90, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#0B3954")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "#EEF2F7")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _script_page(pdf: PdfPages, scripts: list[str], section_rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.06, 0.90, "答辩脚本与演示节奏", fontsize=22, fontweight="bold", color="#0B3954")
    ax.text(0.06, 0.84, "建议按下面顺序讲述，并与界面自动轮播节奏同步。", fontsize=12, color="#4B5563")

    ax.text(0.06, 0.76, "讲解词", fontsize=16, fontweight="bold", color="#C08429")
    for idx, line in enumerate(scripts[:6]):
        ax.text(0.08, 0.71 - idx * 0.08, f"{idx + 1}. {line}", fontsize=13, color="#1F2D3D", wrap=True)

    section_df = pd.DataFrame(section_rows)
    if not section_df.empty:
        table = ax.table(
            cellText=section_df.values,
            colLabels=section_df.columns,
            loc="center",
            cellLoc="center",
            bbox=[0.56, 0.22, 0.37, 0.46],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D1D5DB")
            if row == 0:
                cell.set_facecolor("#087E8B")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#F8FAFC" if row % 2 else "#EEF2F7")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_judge_pdf(
    controls: dict[str, Any],
    cards: dict[str, Any],
    summary_df: pd.DataFrame,
    confusion_pairs: pd.DataFrame,
    quote_lines: list[str],
    script_lines: list[str],
    section_rows: list[dict[str, Any]],
) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _cover_page(pdf, "Uni-Sign 国奖答辩演示简报", controls, cards, quote_lines)
        _table_page(
            pdf,
            "模型对比摘要",
            summary_df,
            "建议现场优先讲 Top 模型与 Pose-only / Ablation 的差距。",
            column_order=["model_name", "bleu4", "rouge_l", "wer", "average_similarity"],
            max_rows=6,
        )
        _table_page(
            pdf,
            "高频混淆对",
            confusion_pairs,
            "用于回答“系统最容易错在哪些字/词”这类追问。",
            column_order=["reference_char", "prediction_char", "count", "operation"],
            max_rows=12,
        )
        _script_page(pdf, script_lines, section_rows)

    buffer.seek(0)
    return buffer.getvalue()
