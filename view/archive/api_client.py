from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests


class ViewAPIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        response = requests.get(f"{self.base_url}/health", timeout=3)
        response.raise_for_status()
        payload = response.json()
        return payload.get("status") == "ok"

    def predict(
        self,
        video_path: Path,
        checkpoint_path: Path,
        rgb_support: bool,
        lightweight: bool = False,
        ultra_lightweight: bool = False,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict",
            json={
                "video_path": str(video_path),
                "checkpoint_path": str(checkpoint_path),
                "rgb_support": rgb_support,
                "lightweight": lightweight,
                "ultra_lightweight": ultra_lightweight,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def xai(
        self,
        video_path: Path,
        checkpoint_path: Path,
        rgb_support: bool,
        reference_text: str | None = None,
        pose_path: Path | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/xai",
            json={
                "video_path": str(video_path),
                "checkpoint_path": str(checkpoint_path),
                "rgb_support": rgb_support,
                "reference_text": reference_text,
                "pose_path": str(pose_path) if pose_path is not None else None,
                "device": device,
            },
            timeout=1800,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "prediction": payload["prediction"],
            "target_sentence": payload["target_sentence"],
            "loss": payload["loss"],
            "xai_timeline_df": pd.DataFrame(payload["xai_timeline"]),
            "attention_matrix_df": pd.DataFrame(payload["attention_matrix"]),
            "rgb_cross_df": pd.DataFrame(payload["rgb_cross"]),
            "top_frames_df": pd.DataFrame(payload["top_frames"]),
            "part_summary_df": pd.DataFrame(payload["part_summary"]),
            "alignment_df": pd.DataFrame(payload["alignment"]),
        }
