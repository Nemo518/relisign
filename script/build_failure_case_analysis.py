import json
from pathlib import Path

import pandas as pd


BASE = Path(r"E:\project\xiaozhi\Uni-Sign-storage\out")
POSE = BASE / "ablation_pose_only" / "test100_eval.csv"
RGB = BASE / "ablation_rgb_support" / "test100_eval.csv"
FULL = BASE / "ablation_uncert_gate" / "test100_eval.csv"

OUT_JSON = Path(r"E:\project\xiaozhi\Uni-Sign\out\paper_metrics\failure_case_analysis_test100.json")
OUT_MD = Path(r"E:\project\xiaozhi\Uni-Sign\docs\04_论文稿件\2026-04-25_论文定稿材料\05_失败案例分析.md")


def load_df() -> pd.DataFrame:
    pose = pd.read_csv(POSE).rename(
        columns={"similarity": "pose_sim", "prediction": "pose_pred"}
    )[["dataset_index", "reference", "pose_sim", "pose_pred"]]
    rgb = pd.read_csv(RGB).rename(
        columns={"similarity": "rgb_sim", "prediction": "rgb_pred"}
    )[["dataset_index", "rgb_sim", "rgb_pred"]]
    full = pd.read_csv(FULL).rename(
        columns={"similarity": "full_sim", "prediction": "full_pred"}
    )[["dataset_index", "full_sim", "full_pred"]]
    return pose.merge(rgb, on="dataset_index").merge(full, on="dataset_index")


def pick_cases(df: pd.DataFrame) -> dict[str, list[dict]]:
    # Case 1: RGB helps — full model beats both pose and naive RGB
    case_a = df[(df["full_sim"] > df["pose_sim"]) & (df["full_sim"] > df["rgb_sim"])].copy()
    case_a["score"] = case_a["full_sim"] - case_a[["pose_sim", "rgb_sim"]].max(axis=1)
    case_a = case_a.sort_values(["score", "full_sim"], ascending=False).head(3)

    # Case 2: RGB hurts — pose is best and both multimodal variants lag
    case_b = df[(df["pose_sim"] > df["rgb_sim"]) & (df["pose_sim"] > df["full_sim"])].copy()
    case_b["score"] = case_b["pose_sim"] - case_b["rgb_sim"]
    case_b = case_b.sort_values(["score", "pose_sim"], ascending=False).head(3)

    # Case 3: gate suppresses RGB noise but still fails — ideal case:
    # full recovers over rgb but remains below pose.
    case_c = df[(df["full_sim"] > df["rgb_sim"]) & (df["full_sim"] < df["pose_sim"])].copy()
    case_c["score"] = case_c["full_sim"] - case_c["rgb_sim"]
    case_c = case_c.sort_values(["score", "pose_sim"], ascending=False).head(3)

    # Fallback for the current result set: if no strict recovery-below-pose samples exist,
    # keep cases where the gate pulls the prediction back to the pose branch and improves over rgb,
    # but the sample is still far from correct.
    if case_c.empty:
        case_c = df[
            (df["full_sim"] > df["rgb_sim"])
            & (df["full_pred"] == df["pose_pred"])
            & (df["full_sim"] < 0.5)
        ].copy()
        case_c["score"] = case_c["full_sim"] - case_c["rgb_sim"]
        case_c = case_c.sort_values(["score", "full_sim"], ascending=False).head(3)

    result = {
        "rgb_helps": case_a.to_dict(orient="records"),
        "rgb_hurts": case_b.to_dict(orient="records"),
        "gate_recovers_but_still_fails": case_c.to_dict(orient="records"),
    }
    return result


def write_outputs(cases: dict[str, list[dict]]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 失败案例分析",
        "",
        "汇总文件：",
        "",
        "1. [failure_case_analysis_test100.json](/abs/path/E:/project/xiaozhi/Uni-Sign/out/paper_metrics/failure_case_analysis_test100.json)",
        "",
        "固定对照组：",
        "",
        "1. `ablation_pose_only`",
        "2. `ablation_rgb_support`",
        "3. `ablation_uncert_gate`",
        "",
    ]

    sections = [
        ("RGB 帮忙", "rgb_helps", "full-model 明显优于其他两组，说明 RGB 在该样本上提供了有效补充。"),
        ("RGB 添乱", "rgb_hurts", "pose-only 更稳，说明 RGB 在该样本上引入了额外噪声。"),
        ("gate 抑制噪声但仍失败", "gate_recovers_but_still_fails", "full-model 相比 rgb-support 有恢复；在当前结果中，这类样本主要表现为门控把预测拉回 pose 分支，但样本仍未预测正确。"),
    ]

    for title, key, note in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(note)
        lines.append("")
        lines.append("| Case | dataset_index | Reference | pose-only | rgb-support | full-model |")
        lines.append("| --- | ---: | --- | --- | --- | --- |")
        for i, row in enumerate(cases[key], start=1):
            lines.append(
                f"| {i} | {int(row['dataset_index'])} | {row['reference']} | "
                f"{row['pose_pred']} ({row['pose_sim']:.4f}) | "
                f"{row['rgb_pred']} ({row['rgb_sim']:.4f}) | "
                f"{row['full_pred']} ({row['full_sim']:.4f}) |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    df = load_df()
    cases = pick_cases(df)
    write_outputs(cases)
    print(json.dumps(cases, ensure_ascii=False, indent=2))
