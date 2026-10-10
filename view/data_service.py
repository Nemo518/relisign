from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight environments
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import CE_CSL_ROOT, OUTPUT_ROOT, PROJECT_ROOT as CODE_PROJECT_ROOT, resolve_storage_path, DATA_ROOT as STORAGE_DATA_ROOT

PROJECT_ROOT = CODE_PROJECT_ROOT
OUT_ROOT = OUTPUT_ROOT
DATA_ROOT = CE_CSL_ROOT
VIDEO_ROOT = DATA_ROOT / "video"
POSE_ROOT = DATA_ROOT / "pose_format"
LABEL_ROOT = DATA_ROOT / "label_cleaned"
PAPER_METRICS_ROOT = OUT_ROOT / "paper_metrics"
DEFAULT_CHECKPOINT = OUT_ROOT / "stage3_finetuning" / "best_checkpoint.pth"
GAP_TOKEN = "∅"

SUMMARY_FILES = {
    "dev": PAPER_METRICS_ROOT / "dev100_metrics_summary.json",
    "test": PAPER_METRICS_ROOT / "test100_metrics_summary.json",
}

ERROR_ANALYSIS_FILE = PAPER_METRICS_ROOT / "test100_error_analysis.json"
DISPLAY_MODEL_HINT = {
    "round6_best_dev100": "Round6 RGB-support",
    "round6_best_test100": "Round6 RGB-support",
    "ablation_pose_only_dev100": "Pose-only",
    "ablation_pose_only_test100": "Pose-only",
    "ablation_uncert_gate_dev100": "Uncertainty Gate",
    "ablation_uncert_gate_test100": "Uncertainty Gate",
}


@dataclass(frozen=True)
class VideoRecord:
    name: str
    split: str
    relative_path: str
    video_path: Path
    pose_path: Path | None
    sample_id: str
    dataset_index: int | None
    reference: str | None
    gloss: str | None


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    duration_s: float
    width: int
    height: int


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_artifact_path(path_value: str | Path) -> Path:
    return Path(resolve_storage_path(str(path_value)))


def _empty_eval_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_index",
            "reference",
            "prediction",
            "gloss",
            "similarity",
        ]
    )


def _sanitize_sample_id(stem: str) -> str:
    return stem.split("(", 1)[0] if "(" in stem else stem


def _safe_mean(array: np.ndarray) -> float:
    return float(np.nanmean(array)) if array.size else 0.0


def _safe_frame_points(frame_points: Any, frame_scores: Any) -> np.ndarray:
    points = np.asarray(frame_points, dtype=np.float32)
    scores = np.asarray(frame_scores, dtype=np.float32)

    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if points.ndim == 3:
        points = points[0]
    if scores.ndim == 2:
        scores = scores[0]
    if points.shape[-1] != 2:
        return np.empty((0, 2), dtype=np.float32)
    if scores.size and len(scores) == len(points):
        points = points[scores > 0.1]
    return points


def load_label_catalog(split: str) -> pd.DataFrame:
    path = LABEL_ROOT / f"{split}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Number", "Chinese Sentences", "Gloss_Clean", "split", "dataset_index"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["split"] = split
    df["dataset_index"] = np.arange(len(df))
    return df


def load_all_labels() -> pd.DataFrame:
    return pd.concat([load_label_catalog("dev"), load_label_catalog("test")], ignore_index=True)


def video_to_pose_path(video_path: Path) -> Path:
    relative = video_path.relative_to(VIDEO_ROOT).with_suffix(".pkl")
    return POSE_ROOT / relative


def list_sample_videos(limit_per_split: int = 24) -> list[VideoRecord]:
    labels = load_all_labels()
    lookup = {row["Number"]: row for _, row in labels.iterrows()}
    records: list[VideoRecord] = []

    for split in ("dev", "test"):
        root = VIDEO_ROOT / split
        if not root.exists():
            continue
        count = 0
        for video_path in sorted(root.rglob("*.mp4")):
            sample_id = _sanitize_sample_id(video_path.stem)
            label_row = lookup.get(sample_id)
            pose_path = video_to_pose_path(video_path)
            relative_path = str(video_path.relative_to(VIDEO_ROOT)).replace("\\", "/")
            records.append(
                VideoRecord(
                    name=f"{sample_id} | {relative_path}",
                    split=split,
                    relative_path=relative_path,
                    video_path=video_path,
                    pose_path=pose_path if pose_path.exists() else None,
                    sample_id=sample_id,
                    dataset_index=int(label_row["dataset_index"]) if label_row is not None else None,
                    reference=str(label_row["Chinese Sentences"]) if label_row is not None else None,
                    gloss=str(label_row["Gloss_Clean"]) if label_row is not None else None,
                )
            )
            count += 1
            if count >= limit_per_split:
                break
    return records


def get_video_metadata(video_path: Path) -> VideoMetadata:
    if cv2 is None:
        raise ModuleNotFoundError("opencv-python is required for video metadata. Install it before launching the view module.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    duration_s = frame_count / fps if frame_count > 0 else 0.0
    return VideoMetadata(frame_count=frame_count, fps=fps, duration_s=duration_s, width=width, height=height)


def load_pose_payload(pose_path: Path) -> dict[str, Any]:
    """Load a pickle file with numpy 1.x/2.x compatibility.

    Pickle files created with numpy 2.x reference ``numpy._core.*`` modules
    which don't exist in numpy 1.x.  This custom unpickler remaps those
    references to the correct numpy 1.x location.
    """
    _NUMPY_MODULE_MAP = {
        "numpy._core": "numpy",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.defchararray": "numpy.core.defchararray",
        "numpy._core.records": "numpy.core.records",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.arrayprint": "numpy.core.arrayprint",
        "numpy._core.memmap": "numpy.core.memmap",
        "numpy._core.function_base": "numpy.core.function_base",
        "numpy._core.getlimits": "numpy.core.getlimits",
        "numpy._core.einsumfunc": "numpy.core.einsumfunc",
        "numpy._core._methods": "numpy.core._methods",
        "numpy._core._add_newdocs": "numpy.core._add_newdocs",
        "numpy._core._dtype": "numpy.core._dtype",
        "numpy._core._internal": "numpy.core._internal",
        "numpy._core._string_helpers": "numpy.core._string_helpers",
        "numpy._core._type_aliases": "numpy.core._type_aliases",
        "numpy._core._ufunc_config": "numpy.core._ufunc_config",
        "numpy._core.umath": "numpy.core.umath",
    }

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in _NUMPY_MODULE_MAP:
                module = _NUMPY_MODULE_MAP[module]
            elif module.startswith("numpy._core."):
                module = "numpy.core" + module[11:]
            return super().find_class(module, name)

    with pose_path.open("rb") as f:
        return _CompatUnpickler(f).load()


def scan_all_experiments(split: str = "test") -> list[dict[str, Any]]:
    """Scan all experiment directories under OUT_ROOT and compute avg similarity from eval CSVs."""
    import csv

    results: list[dict[str, Any]] = []
    if not OUT_ROOT.exists():
        return results

    for d in sorted(OUT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        eval_csv = d / f"{split}100_eval.csv"
        if not eval_csv.exists():
            continue

        similarities: list[float] = []
        sample_count = 0
        try:
            with eval_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_count += 1
                    try:
                        similarities.append(float(row.get("similarity", 0)))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue

        if sample_count == 0:
            continue

        avg_sim = sum(similarities) / len(similarities)
        results.append({
            "directory": d.name,
            "split": split,
            "sample_count": sample_count,
            "avg_similarity": round(avg_sim, 4),
            "max_similarity": round(max(similarities), 4),
            "min_similarity": round(min(similarities), 4),
        })

    results.sort(key=lambda r: r["avg_similarity"], reverse=True)
    return results


def build_frame_importance(pose_payload: dict[str, Any]) -> pd.DataFrame:
    keypoints = pose_payload.get("keypoints", [])
    scores = pose_payload.get("scores", [])
    rows: list[dict[str, Any]] = []
    previous_center: np.ndarray | None = None

    for index, (frame_points, frame_scores) in enumerate(zip(keypoints, scores)):
        points = _safe_frame_points(frame_points, frame_scores)
        score_array = np.asarray(frame_scores, dtype=np.float32)
        if score_array.ndim == 2:
            score_array = score_array[0]

        mean_score = _safe_mean(score_array)
        if points.size == 0:
            center = previous_center if previous_center is not None else np.zeros(2, dtype=np.float32)
            motion = 0.0
        else:
            center = points.mean(axis=0)
            motion = float(np.linalg.norm(center - previous_center)) if previous_center is not None else 0.0
            previous_center = center

        rows.append({"frame": index, "mean_score": mean_score, "motion": motion})

    frame_df = pd.DataFrame(rows)
    if frame_df.empty:
        return frame_df

    motion_scale = frame_df["motion"].max()
    frame_df["motion_norm"] = 0.0 if motion_scale <= 0 else frame_df["motion"] / motion_scale
    frame_df["importance"] = 0.7 * frame_df["mean_score"] + 0.3 * frame_df["motion_norm"]
    return frame_df


def build_pose_density(pose_payload: dict[str, Any], bins: int = 24) -> pd.DataFrame:
    keypoints = pose_payload.get("keypoints", [])
    scores = pose_payload.get("scores", [])
    all_points: list[np.ndarray] = []

    for frame_points, frame_scores in zip(keypoints, scores):
        points = _safe_frame_points(frame_points, frame_scores)
        if points.size:
            all_points.append(points)

    if not all_points:
        return pd.DataFrame(columns=["x_bin", "y_bin", "density"])

    stacked = np.vstack(all_points)
    clipped = np.clip(stacked, 0.0, 1.0)
    heatmap, x_edges, y_edges = np.histogram2d(
        clipped[:, 0],
        clipped[:, 1],
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )

    rows: list[dict[str, Any]] = []
    for x_index in range(len(x_edges) - 1):
        for y_index in range(len(y_edges) - 1):
            rows.append(
                {
                    "x_bin": float((x_edges[x_index] + x_edges[x_index + 1]) / 2),
                    "y_bin": float((y_edges[y_index] + y_edges[y_index + 1]) / 2),
                    "density": float(heatmap[x_index, y_index]),
                }
            )
    return pd.DataFrame(rows)


def load_metrics_summary(split: str) -> pd.DataFrame:
    summary_path = SUMMARY_FILES.get(split)
    if summary_path is None or not summary_path.exists():
        return pd.DataFrame(columns=["label", "model_name", "bleu4", "rouge_l", "wer", "average_similarity", "csv_path"])
    rows = _read_json(summary_path)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["label", "model_name", "bleu4", "rouge_l", "wer", "average_similarity", "csv_path"])
    for column, default in {
        "label": "",
        "bleu4": 0.0,
        "rouge_l": 0.0,
        "wer": 0.0,
        "average_similarity": 0.0,
        "csv_path": "",
    }.items():
        if column not in df.columns:
            df[column] = default
    df["model_name"] = df["label"].map(DISPLAY_MODEL_HINT).fillna(df["label"])
    return df


def load_error_analyses() -> list[dict[str, Any]]:
    if not ERROR_ANALYSIS_FILE.exists():
        return []
    return _read_json(ERROR_ANALYSIS_FILE)


def load_error_examples(label: str) -> dict[str, Any]:
    for row in load_error_analyses():
        if row["label"] == label:
            return row
    return {"best_examples": [], "worst_examples": [], "stats": {}}


def load_training_histories() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not OUT_ROOT.exists():
        return pd.DataFrame()
    for metrics_path in sorted(OUT_ROOT.glob("*/training_metrics.jsonl")):
        experiment = metrics_path.parent.name
        for item in _read_jsonl(metrics_path):
            payload = dict(item)
            payload["experiment"] = experiment
            rows.append(payload)
    return pd.DataFrame(rows)


def load_eval_rows(csv_path: Path) -> pd.DataFrame:
    resolved_path = _resolve_artifact_path(csv_path)
    if not resolved_path.exists():
        return _empty_eval_rows()
    df = pd.read_csv(resolved_path, encoding="utf-8-sig")
    for column, default in {
        "dataset_index": -1,
        "reference": "",
        "prediction": "",
        "gloss": "",
        "similarity": 0.0,
    }.items():
        if column not in df.columns:
            df[column] = default
    return df


def sample_id_to_index(sample_id: str) -> int | None:
    if not ("dev-" in sample_id or "test-" in sample_id):
        return None
    try:
        return int(sample_id.split("-")[-1]) - 1
    except ValueError:
        return None


def load_cached_prediction(sample_id: str, preferred_prefix: str = "round6_best") -> dict[str, Any] | None:
    split = "dev" if sample_id.startswith("dev-") else "test" if sample_id.startswith("test-") else None
    if split is None:
        return None

    dataset_index = sample_id_to_index(sample_id)
    if dataset_index is None:
        return None

    metrics_df = load_metrics_summary(split)
    target_row = None
    for _, row in metrics_df.iterrows():
        if str(row["label"]).startswith(preferred_prefix):
            target_row = row
            break
    if target_row is None and not metrics_df.empty:
        target_row = metrics_df.iloc[0]
    if target_row is None:
        return None

    csv_path = _resolve_artifact_path(str(target_row["csv_path"]))
    if not csv_path.exists():
        return None

    eval_df = load_eval_rows(csv_path)
    if eval_df.empty or "dataset_index" not in eval_df.columns:
        return None
    match = eval_df[eval_df["dataset_index"] == dataset_index]
    if match.empty:
        return None

    row = match.iloc[0]
    return {
        "model_label": str(target_row["label"]),
        "model_name": str(target_row["model_name"]),
        "prediction": str(row["prediction"]),
        "similarity": float(row["similarity"]),
        "reference": str(row["reference"]),
        "gloss": str(row["gloss"]),
    }


def build_metric_cards(split: str, preferred_prefix: str = "round6_best") -> dict[str, Any]:
    summary = load_metrics_summary(split)
    if summary.empty:
        return {}

    row = None
    for _, candidate in summary.iterrows():
        if str(candidate["label"]).startswith(preferred_prefix):
            row = candidate
            break
    if row is None:
        row = summary.iloc[0]

    return {
        "model_name": str(row["model_name"]),
        "bleu4": float(row["bleu4"]),
        "rouge_l": float(row["rouge_l"]),
        "wer": float(row["wer"]),
        "average_similarity": float(row["average_similarity"]),
    }


def build_model_radar(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["model_name", "metric", "value"])

    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        metrics = {
            "BLEU-4": float(row["bleu4"]),
            "ROUGE-L": float(row["rouge_l"]),
            "Similarity": float(row["average_similarity"]) * 100.0,
            "WER Advantage": max(0.0, 100.0 - float(row["wer"])),
        }
        for metric, value in metrics.items():
            rows.append({"model_name": row["model_name"], "metric": metric, "value": value})
    return pd.DataFrame(rows)


def build_error_category_frame(label: str) -> pd.DataFrame:
    payload = load_error_examples(label)
    stats = payload.get("stats", {})
    rows: list[dict[str, Any]] = []
    for category, values in stats.items():
        if category == "all":
            continue
        rows.append(
            {
                "category": category,
                "count": int(values.get("count", 0)),
                "avg_similarity": float(values.get("avg_similarity", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_latency_breakdown(total_latency_s: float) -> pd.DataFrame:
    total_ms = total_latency_s * 1000.0
    stages = [
        ("视频读取", 0.10),
        ("姿态提取", 0.36),
        ("时序编码", 0.24),
        ("文本生成", 0.22),
        ("可视化更新", 0.08),
    ]
    rows = []
    start = 0.0
    for name, ratio in stages:
        duration = total_ms * ratio
        rows.append({"stage": name, "start_ms": start, "duration_ms": duration})
        start += duration
    return pd.DataFrame(rows)


def generate_demo_prediction(sample_id: str, reference: str | None) -> str:
    if reference:
        return f"演示模式已载入样例 {sample_id}，可切换到真实推理获取模型输出。"
    return f"演示模式已载入样例 {sample_id}。"


def _align_texts(reference: str, prediction: str) -> list[tuple[str, str]]:
    ref_chars = list(reference)
    pred_chars = list(prediction)
    m, n = len(ref_chars), len(pred_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_chars[i - 1] == pred_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    aligned: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_chars[i - 1] == pred_chars[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((ref_chars[i - 1], pred_chars[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref_chars[i - 1], GAP_TOKEN))
            i -= 1
            continue
        aligned.append((GAP_TOKEN, pred_chars[j - 1]))
        j -= 1

    aligned.reverse()
    return aligned


def build_alignment_dataframe(reference: str, prediction: str) -> pd.DataFrame:
    aligned = _align_texts(reference, prediction)
    rows = []
    for index, (ref_char, pred_char) in enumerate(aligned):
        if ref_char == pred_char:
            operation = "match"
        elif ref_char == GAP_TOKEN:
            operation = "insert"
        elif pred_char == GAP_TOKEN:
            operation = "delete"
        else:
            operation = "replace"
        rows.append(
            {
                "position": index,
                "reference_char": ref_char,
                "prediction_char": pred_char,
                "operation": operation,
            }
        )
    return pd.DataFrame(rows)


def extract_video_frames(
    video_path: Path,
    frame_indices: list[int],
    resize_width: int = 360,
) -> list[dict[str, Any]]:
    if cv2 is None:
        raise ModuleNotFoundError("opencv-python is required for keyframe extraction.")

    unique_indices = sorted({int(idx) for idx in frame_indices if idx is not None and idx >= 0})
    if not unique_indices:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[dict[str, Any]] = []
    current_target_index = 0
    current_frame_index = 0
    success, frame = cap.read()
    while success and current_target_index < len(unique_indices):
        if current_frame_index == unique_indices[current_target_index]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and rgb.shape[1] > resize_width:
                scale = resize_width / rgb.shape[1]
                new_size = (resize_width, max(1, int(rgb.shape[0] * scale)))
                rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            frames.append({"frame": current_frame_index, "image": rgb})
            current_target_index += 1
        current_frame_index += 1
        success, frame = cap.read()

    cap.release()
    return frames


def build_confusion_artifacts(label: str, top_pairs: int = 20, top_chars: int = 12) -> dict[str, Any]:
    payload = load_error_examples(label)
    csv_path_value = payload.get("csv_path", "")
    if not csv_path_value:
        return {
            "pair_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count", "operation"]),
            "matrix_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count"]),
            "summary_df": pd.DataFrame(columns=["operation", "count"]),
        }

    csv_path = _resolve_artifact_path(str(csv_path_value))
    if not csv_path.exists():
        return {
            "pair_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count", "operation"]),
            "matrix_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count"]),
            "summary_df": pd.DataFrame(columns=["operation", "count"]),
        }
    eval_df = load_eval_rows(csv_path)
    if eval_df.empty:
        return {
            "pair_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count", "operation"]),
            "matrix_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count"]),
            "summary_df": pd.DataFrame(columns=["operation", "count"]),
        }
    pair_counter: Counter[tuple[str, str, str]] = Counter()
    op_counter: Counter[str] = Counter()

    for _, row in eval_df.iterrows():
        aligned = _align_texts(str(row["reference"]), str(row["prediction"]))
        for ref_char, pred_char in aligned:
            if ref_char == pred_char:
                continue
            if ref_char == GAP_TOKEN:
                operation = "插入"
            elif pred_char == GAP_TOKEN:
                operation = "删除"
            else:
                operation = "替换"
            pair_counter[(ref_char, pred_char, operation)] += 1
            op_counter[operation] += 1

    pair_rows = [
        {
            "reference_char": ref_char,
            "prediction_char": pred_char,
            "count": count,
            "operation": operation,
        }
        for (ref_char, pred_char, operation), count in pair_counter.most_common(top_pairs)
    ]
    pair_df = pd.DataFrame(pair_rows, columns=["reference_char", "prediction_char", "count", "operation"])

    substitute_rows = [row for row in pair_rows if row["operation"] == "替换" and row["reference_char"] != GAP_TOKEN and row["prediction_char"] != GAP_TOKEN]
    top_reference_chars = [row["reference_char"] for row in substitute_rows[:top_chars]]
    top_prediction_chars = [row["prediction_char"] for row in substitute_rows[:top_chars]]
    char_pool = list(dict.fromkeys(top_reference_chars + top_prediction_chars))[:top_chars]
    matrix_rows: list[dict[str, Any]] = []
    for ref_char in char_pool:
        for pred_char in char_pool:
            count = pair_counter.get((ref_char, pred_char, "替换"), 0)
            matrix_rows.append(
                {
                    "reference_char": ref_char,
                    "prediction_char": pred_char,
                    "count": count,
                }
            )

    summary_df = pd.DataFrame(
        [{"operation": operation, "count": count} for operation, count in op_counter.items()],
        columns=["operation", "count"],
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values("count", ascending=False)

    return {
        "pair_df": pair_df,
        "matrix_df": pd.DataFrame(matrix_rows),
        "summary_df": summary_df,
    }


def list_best_examples(
    split: str = "test",
    limit: int = 24,
    model_prefix: str = "round6_best",
) -> list[dict[str, Any]]:
    summary = load_metrics_summary(split)
    if summary.empty:
        return []

    csv_path_value = None
    for _, row in summary.iterrows():
        if str(row["label"]).startswith(model_prefix):
            csv_path_value = str(row["csv_path"])
            break
    if csv_path_value is None:
        csv_path_value = str(summary.iloc[0]["csv_path"])

    eval_df = load_eval_rows(Path(csv_path_value))
    if eval_df.empty:
        return []

    sorted_df = eval_df.sort_values("similarity", ascending=False)
    top = sorted_df.head(limit)

    labels = load_all_labels()
    label_lookup = {}
    for _, lr in labels.iterrows():
        label_lookup[int(lr["dataset_index"])] = lr

    # Pre-index video files by stem for fast lookup
    video_index: dict[str, Path] = {}
    for video_path in (VIDEO_ROOT / split).rglob("*.mp4"):
        video_index[_sanitize_sample_id(video_path.stem)] = video_path

    results: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        dataset_index = int(row["dataset_index"])
        label_row = label_lookup.get(dataset_index)
        sample_id = str(label_row["Number"]) if label_row is not None else ""
        sample_split = str(label_row.get("split", split)) if label_row is not None else split

        video_path = video_index.get(sample_id)
        if video_path is not None:
            relative_path = str(video_path.relative_to(VIDEO_ROOT)).replace("\\", "/")
        else:
            relative_path = f"{sample_split}/{sample_id}.mp4"

        results.append({
            "name": f"{sample_id} | {relative_path}",
            "sample_id": sample_id,
            "split": sample_split,
            "relative_path": relative_path,
            "dataset_index": dataset_index,
            "reference": str(row["reference"]),
            "prediction": str(row["prediction"]),
            "gloss": str(row["gloss"]),
            "similarity": float(row["similarity"]),
        })

    return results


def list_datasets() -> list[dict[str, Any]]:
    storage_data = Path(STORAGE_DATA_ROOT)
    if not storage_data.exists():
        return []

    dataset_dirs: list[Path] = []
    for entry in sorted(storage_data.iterdir()):
        if not entry.is_dir():
            continue
        # CE_CSL has nested structure (CE_CSL/CE-CSL)
        inner = entry / entry.name.replace("_", "-") if "_" in entry.name else None
        if inner and inner.exists():
            dataset_dirs.append(inner)
        else:
            dataset_dirs.append(entry)

    results: list[dict[str, Any]] = []
    for ds_path in dataset_dirs:
        name = ds_path.name
        video_count = 0
        video_root = ds_path / "video"
        if video_root.exists():
            video_count = len(list(video_root.rglob("*.mp4")))

        label_count = 0
        label_formats: list[str] = []
        for label_dir_name in ("label", "label_cleaned"):
            label_dir = ds_path / label_dir_name
            if label_dir.exists():
                for f in label_dir.iterdir():
                    if f.suffix in (".csv", ".json", ".txt"):
                        label_count += 1
                        if f.suffix not in label_formats:
                            label_formats.append(f.suffix)

        # Also count top-level label files
        for f in ds_path.glob("labels*"):
            if f.is_file():
                label_count += 1
                ext = f.suffix or ".pkl"
                if ext not in label_formats:
                    label_formats.append(ext)

        # Also check for gzip-style label files (no extension)
        for f in ds_path.glob("labels.*"):
            if f.is_file():
                if f not in ds_path.glob("labels*"):
                    label_count += 1

        pose_count = 0
        pose_root = ds_path / "pose_format"
        if pose_root.exists():
            pose_count = len(list(pose_root.rglob("*.pkl")))

        splits: list[str] = []
        if video_root.exists():
            for sub in sorted(video_root.iterdir()):
                if sub.is_dir() and any(sub.rglob("*.mp4")):
                    splits.append(sub.name)

        has_videos = video_count > 0
        has_labels = label_count > 0
        has_poses = pose_count > 0

        status = "完整" if (has_videos and has_labels) else ("仅标签" if (has_labels and not has_videos) else "空")

        results.append({
            "name": name,
            "path": str(ds_path),
            "video_count": video_count,
            "label_count": label_count,
            "label_formats": label_formats,
            "pose_count": pose_count,
            "splits": splits,
            "has_videos": has_videos,
            "has_labels": has_labels,
            "has_poses": has_poses,
            "status": status,
        })

    return results
