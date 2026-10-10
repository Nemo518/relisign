from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from paths import CE_CSL_ROOT, resolve_storage_path

try:
    from .bootstrap import add_vendor_path
    from .data_service import (
        load_pose_payload,
        list_sample_videos,
        get_video_metadata,
        load_metrics_summary,
        build_metric_cards,
        load_training_histories,
        load_error_examples,
        build_confusion_artifacts,
        build_error_category_frame,
        load_cached_prediction,
        build_model_radar,
        build_latency_breakdown,
        extract_video_frames,
        build_alignment_dataframe,
        generate_demo_prediction,
        list_best_examples,
        list_datasets,
    )
    from .inference_adapter import run_online_inference
    from .xai_service import analyze_video_xai
    from .report_service import build_judge_pdf
except ImportError:  # pragma: no cover - script mode fallback
    from bootstrap import add_vendor_path
    from data_service import (
        load_pose_payload,
        list_sample_videos,
        get_video_metadata,
        load_metrics_summary,
        build_metric_cards,
        load_training_histories,
        load_error_examples,
        build_confusion_artifacts,
        build_error_category_frame,
        load_cached_prediction,
        build_model_radar,
        build_latency_breakdown,
        extract_video_frames,
        build_alignment_dataframe,
        generate_demo_prediction,
        list_best_examples,
        list_datasets,
    )
    from inference_adapter import run_online_inference
    from xai_service import analyze_video_xai
    from report_service import build_judge_pdf

add_vendor_path()

import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parents[0]

app = FastAPI(title="Uni-Sign View API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    video_path: str
    checkpoint_path: str
    rgb_support: bool = False
    lightweight: bool = False
    ultra_lightweight: bool = False


class XAIRequest(BaseModel):
    video_path: str
    checkpoint_path: str
    rgb_support: bool = False
    reference_text: Optional[str] = None
    pose_path: Optional[str] = None
    device: Optional[str] = None


class ExtractFramesRequest(BaseModel):
    video_path: str
    frame_indices: list[int]
    resize_width: int = 360


class JudgePdfRequest(BaseModel):
    controls: dict[str, Any]
    cards: dict[str, Any]
    summary_df: list[dict[str, Any]]
    confusion_pairs: list[dict[str, Any]]
    quote_lines: list[str]
    script_lines: list[str]
    section_rows: list[dict[str, Any]]


def _resolve_existing_path(raw_path: str) -> Path:
    path = Path(resolve_storage_path(raw_path))
    if path.exists():
        return path
    video_path = Path(CE_CSL_ROOT) / "video" / raw_path
    if video_path.exists():
        return video_path
    raise HTTPException(status_code=404, detail=f"Path not found: {path}")


_MEDIA_TYPE_MAP = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _guess_media_type(suffix: str) -> str:
    return _MEDIA_TYPE_MAP.get(suffix.lower(), "application/octet-stream")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


UPLOAD_DIR = Path(tempfile.gettempdir()) / "uni-sign-uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/predict/upload-video")
async def predict_upload_video(file: UploadFile) -> dict[str, Any]:
    """Accept an uploaded video file, save it, and return the server-side path for prediction."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "uploaded_path": str(dest),
        "filename": safe_name,
        "size_bytes": dest.stat().st_size,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    video_path = _resolve_existing_path(req.video_path)
    checkpoint_path = _resolve_existing_path(req.checkpoint_path)
    result = run_online_inference(
        project_root=PROJECT_ROOT,
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        rgb_support=req.rgb_support,
        lightweight=req.lightweight,
        ultra_lightweight=req.ultra_lightweight,
    )
    # Build pose_sequence from inference result (for camera real-time display)
    pose_sequence: dict[str, Any] | None = None
    if result.pose_data and result.pose_data.get("keypoints"):
        import numpy as np
        raw_kp = result.pose_data["keypoints"]
        raw_sc = result.pose_data.get("scores", [])
        num_keypoints = len(raw_kp[0]) if raw_kp else 0
        frames_out: list[dict[str, Any]] = []
        for fi, (kp, sc) in enumerate(zip(raw_kp, raw_sc)):
            kp_arr = np.asarray(kp, dtype=np.float32)
            sc_arr = np.asarray(sc, dtype=np.float32)
            if kp_arr.ndim == 3:
                kp_arr = kp_arr[0]
            if sc_arr.ndim == 2:
                sc_arr = sc_arr[0]
            pts: list[dict[str, float]] = []
            n = min(len(kp_arr), len(sc_arr))
            for i in range(n):
                pts.append({
                    "x": max(0.0, min(1.0, float(kp_arr[i][0]))),
                    "y": max(0.0, min(1.0, float(kp_arr[i][1]))),
                    "confidence": max(0.0, min(1.0, float(sc_arr[i]))),
                })
            if pts:
                frames_out.append({"frame": fi, "points": pts})
        pose_sequence = {
            "num_frames": len(frames_out),
            "num_keypoints": num_keypoints,
            "fps": result.effective_fps if result.effective_fps > 0 else 25.0,
            "frames": frames_out,
        }

    return {
        "success": result.success,
        "prediction": result.prediction,
        "latency_s": result.latency_s,
        "effective_fps": result.effective_fps,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
        "command": result.command,
        "pose_sequence": pose_sequence,
    }


@app.post("/xai")
def xai(req: XAIRequest) -> dict[str, Any]:
    video_path = _resolve_existing_path(req.video_path)
    checkpoint_path = _resolve_existing_path(req.checkpoint_path)
    pose_payload = None
    if req.pose_path:
        pose_payload = load_pose_payload(_resolve_existing_path(req.pose_path))

    result = analyze_video_xai(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        rgb_support=req.rgb_support,
        reference_text=req.reference_text,
        pose_payload=pose_payload,
        device=req.device,
    )
    return {
        "prediction": result["prediction"],
        "target_sentence": result["target_sentence"],
        "loss": result["loss"],
        "xai_timeline": result["xai_timeline_df"].to_dict(orient="records"),
        "attention_matrix": result["attention_matrix_df"].to_dict(orient="records"),
        "rgb_cross": result["rgb_cross_df"].to_dict(orient="records"),
        "top_frames": result["top_frames_df"].to_dict(orient="records"),
        "part_summary": result["part_summary_df"].to_dict(orient="records"),
        "alignment": result["alignment_df"].to_dict(orient="records"),
    }


# ── Data API endpoints ────────────────────────────────────────────

@app.get("/data/sample-videos")
def data_sample_videos(limit_per_split: int = 24) -> list[dict[str, Any]]:
    videos = list_sample_videos(limit_per_split)
    return [
        {
            "name": v.name,
            "split": v.split,
            "relative_path": v.relative_path,
            "sample_id": v.sample_id,
            "dataset_index": v.dataset_index,
            "reference": v.reference,
            "gloss": v.gloss,
        }
        for v in videos
    ]


@app.get("/data/best-examples")
def data_best_examples(
    split: str = "test", limit: int = 24, model: str = "round6_best"
) -> list[dict[str, Any]]:
    return list_best_examples(split=split, limit=limit, model_prefix=model)


@app.get("/data/datasets")
def data_datasets() -> list[dict[str, Any]]:
    return list_datasets()


@app.get("/data/metric-cards")
def data_metric_cards(split: str = "test", preferred_prefix: str = "round6_best") -> dict[str, Any]:
    return build_metric_cards(split, preferred_prefix)


@app.get("/data/metrics-summary")
def data_metrics_summary(split: str = "test") -> list[dict[str, Any]]:
    df = load_metrics_summary(split)
    return df.to_dict(orient="records")


@app.get("/data/training-history")
def data_training_history() -> list[dict[str, Any]]:
    df = load_training_histories()
    return df.to_dict(orient="records")


@app.get("/data/error-analysis")
def data_error_analysis(label: str) -> dict[str, Any]:
    return load_error_examples(label)


@app.get("/data/confusion-artifacts")
def data_confusion_artifacts(
    label: str, top_pairs: int = 20, top_chars: int = 12
) -> dict[str, Any]:
    result = build_confusion_artifacts(label, top_pairs, top_chars)
    return {
        "pair_df": result["pair_df"].to_dict(orient="records"),
        "matrix_df": result["matrix_df"].to_dict(orient="records"),
        "summary_df": result["summary_df"].to_dict(orient="records"),
    }


@app.get("/data/video-metadata")
def data_video_metadata(path: str) -> dict[str, Any]:
    video_path = _resolve_existing_path(path)
    meta = get_video_metadata(video_path)
    return {
        "frame_count": meta.frame_count,
        "fps": meta.fps,
        "duration_s": meta.duration_s,
        "width": meta.width,
        "height": meta.height,
    }


@app.get("/data/cached-prediction")
def data_cached_prediction(
    sample_id: str, preferred_prefix: str = "round6_best"
) -> Optional[dict[str, Any]]:
    return load_cached_prediction(sample_id, preferred_prefix)


@app.get("/data/model-radar")
def data_model_radar(split: str = "test") -> list[dict[str, Any]]:
    summary_df = load_metrics_summary(split)
    radar_df = build_model_radar(summary_df)
    return radar_df.to_dict(orient="records")


@app.get("/data/all-experiments")
def data_all_experiments(split: str = "test") -> list[dict[str, Any]]:
    """Scan all experiment directories and return evaluation metrics."""
    try:
        from .data_service import scan_all_experiments
    except ImportError:
        from data_service import scan_all_experiments  # type: ignore[no-redef]
    return scan_all_experiments(split)


@app.get("/data/latency-breakdown")
def data_latency_breakdown(latency_s: float = 0.5) -> list[dict[str, Any]]:
    df = build_latency_breakdown(latency_s)
    return df.to_dict(orient="records")


@app.post("/data/extract-frames")
def data_extract_frames(req: ExtractFramesRequest) -> list[dict[str, Any]]:
    import base64
    import cv2

    video_path = _resolve_existing_path(req.video_path)
    frames = extract_video_frames(video_path, req.frame_indices, req.resize_width)
    result: list[dict[str, Any]] = []
    for item in frames:
        img: Any = item["image"]
        _, buffer = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buffer).decode("ascii")
        result.append({"frame": int(item["frame"]), "image_base64": b64})
    return result


@app.get("/data/error-category-frame")
def data_error_category_frame(label: str) -> list[dict[str, Any]]:
    df = build_error_category_frame(label)
    return df.to_dict(orient="records")


@app.post("/data/alignment")
def data_alignment(req: dict[str, Any]) -> list[dict[str, Any]]:
    reference = str(req.get("reference", ""))
    prediction = str(req.get("prediction", ""))
    df = build_alignment_dataframe(reference, prediction)
    return df.to_dict(orient="records")


@app.post("/data/judge-pdf")
def data_judge_pdf(req: JudgePdfRequest) -> Response:
    import io

    import pandas as pd

    pdf_bytes = build_judge_pdf(
        controls=req.controls,
        cards=req.cards,
        summary_df=pd.DataFrame(req.summary_df),
        confusion_pairs=pd.DataFrame(req.confusion_pairs),
        quote_lines=req.quote_lines,
        script_lines=req.script_lines,
        section_rows=req.section_rows,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=uni-sign-judge-brief.pdf"},
    )


@app.get("/data/pose-sequence")
def data_pose_sequence(video_path: str) -> dict[str, Any]:
    """Return real pre-extracted 133-keypoint pose data for a video."""
    import numpy as np
    try:
        from .data_service import video_to_pose_path, load_pose_payload, get_video_metadata
    except ImportError:
        from data_service import video_to_pose_path, load_pose_payload, get_video_metadata  # type: ignore[no-redef]

    resolved_video = _resolve_existing_path(video_path)
    pose_path = video_to_pose_path(resolved_video)
    if not pose_path.exists():
        raise HTTPException(status_code=404, detail=f"No pre-extracted pose found for: {video_path}")

    payload = load_pose_payload(pose_path)
    raw_keypoints = payload.get("keypoints", [])
    raw_scores = payload.get("scores", [])

    if not raw_keypoints:
        raise HTTPException(status_code=404, detail="Pose file contains no keypoints")

    # Get video dimensions for normalization
    meta = get_video_metadata(resolved_video)
    frame_width = meta.width or 1920
    frame_height = meta.height or 1080

    frames: list[dict[str, Any]] = []
    for frame_idx, (kp, sc) in enumerate(zip(raw_keypoints, raw_scores)):
        kp_arr = np.asarray(kp, dtype=np.float32)
        sc_arr = np.asarray(sc, dtype=np.float32)

        # Handle nested arrays from RTMPose
        if kp_arr.ndim == 3:
            kp_arr = kp_arr[0]
        if sc_arr.ndim == 2:
            sc_arr = sc_arr[0]

        points: list[dict[str, float]] = []
        num_pts = min(len(kp_arr), len(sc_arr)) if len(sc_arr) > 0 else len(kp_arr)

        for i in range(num_pts):
            x = float(kp_arr[i][0])
            y = float(kp_arr[i][1])
            conf = float(sc_arr[i]) if i < len(sc_arr) else 0.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            conf = max(0.0, min(1.0, conf))
            points.append({"x": x, "y": y, "confidence": conf})

        if points:
            frames.append({"frame": frame_idx, "points": points})

    return {
        "video_path": video_path,
        "num_frames": len(frames),
        "num_keypoints": len(frames[0]["points"]) if frames else 0,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": meta.fps,
        "frames": frames,
    }


@app.get("/data/media/{relative_path:path}")
def data_media_file(relative_path: str, request: Request) -> Response:
    from paths import CE_CSL_ROOT as _CE_CSL_ROOT, resolve_storage_path

    resolved = Path(resolve_storage_path(relative_path))
    if not resolved.exists():
        video_path = Path(_CE_CSL_ROOT) / "video" / relative_path
        if video_path.exists():
            resolved = video_path
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Media not found: {relative_path}")
    # Allow paths under storage root or CE-CSL video directory
    storage_root = Path(resolve_storage_path("")).resolve()
    cecsl_video = (Path(_CE_CSL_ROOT) / "video").resolve()
    resolved_abs = resolved.resolve()
    if not (str(resolved_abs).startswith(str(storage_root)) or
            str(resolved_abs).startswith(str(cecsl_video))):
        raise HTTPException(status_code=403, detail="Access denied")

    file_size = resolved.stat().st_size
    media_type = _guess_media_type(resolved.suffix)

    range_header = request.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        # Parse the requested byte range
        range_value = range_header[6:]
        try:
            start_str, end_str = range_value.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except ValueError:
            start, end = 0, file_size - 1

        # Clamp to valid range
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        chunk_size = end - start + 1

        # Read the requested byte range
        with resolved.open("rb") as f:
            f.seek(start)
            data = f.read(chunk_size)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(len(data)),
            "Accept-Ranges": "bytes",
            "Content-Type": media_type,
            "Cache-Control": "public, max-age=3600",
        }
        return Response(content=data, status_code=206, headers=headers, media_type=media_type)

    # Full file response with streaming for large files
    def file_iterator(path: Path, chunk_size: int = 256 * 1024):
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        file_iterator(resolved),
        status_code=200,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600",
        },
    )
