from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight environments
    cv2 = None


PREDICTION_PATTERN = re.compile(r"Prediction result is:\s*(.+)")


@dataclass(frozen=True)
class InferenceResult:
    success: bool
    prediction: str
    latency_s: float
    effective_fps: float
    stdout: str
    stderr: str
    return_code: int
    command: list[str]
    pose_data: dict | None = None


def _video_frame_count(video_path: Path) -> int:
    if cv2 is None:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count


def _parse_prediction(stdout: str, stderr: str) -> str:
    for text in (stdout, stderr):
        match = PREDICTION_PATTERN.search(text)
        if match:
            return match.group(1).strip()
    return ""


class _RealtimeInferenceSession:
    def __init__(
        self,
        project_root: Path,
        checkpoint_path: Path,
        rgb_support: bool,
        lightweight: bool = False,
        ultra_lightweight: bool = False,
    ) -> None:
        self.project_root = project_root
        self.checkpoint_path = checkpoint_path
        self.rgb_support = rgb_support
        self.lightweight = lightweight
        self.ultra_lightweight = ultra_lightweight
        self._lock = Lock()
        self._ready = False
        self._device = "cpu"
        self._wholebody = None
        self._fallback_pose_extraction = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader

        from datasets import S2T_Dataset_online
        from models import Uni_Sign
        import utils as utils
        from demo.online_inference import pose_extraction

        try:
            from rtmlib import Wholebody
        except Exception:  # pragma: no cover - optional acceleration path
            Wholebody = None

        parser = utils.get_args_parser()
        args = parser.parse_args([])
        args.finetune = str(self.checkpoint_path)
        args.online_video = ""
        args.rgb_support = self.rgb_support
        args.eval = True
        if self.ultra_lightweight:
            args.max_length = min(getattr(args, "max_length", 256), 64)
        elif self.lightweight:
            args.max_length = min(getattr(args, "max_length", 256), 96)
        utils.set_seed(args.seed)

        self._torch = torch
        self._pad_sequence = pad_sequence
        self._fallback_pose_extraction = pose_extraction
        self._target_dtype = torch.bfloat16
        self._dataset = S2T_Dataset_online(args=args)
        self._frame_stride = 3 if self.ultra_lightweight else 2 if self.lightweight else 1
        self._max_new_tokens = 40 if self.ultra_lightweight else 64 if self.lightweight else 100
        self._num_beams = 1 if (self.lightweight or self.ultra_lightweight) else 4
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=1,
            collate_fn=self._dataset.collate_fn,
            sampler=torch.utils.data.SequentialSampler(self._dataset),
        )

        model = Uni_Sign(args=args)
        if torch.cuda.is_available():
            model = model.cuda()
            self._device = "cuda"
        model.train()
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        if self._device == "cuda":
            model.to(torch.bfloat16)
        self._model = model
        self._tokenizer = model.mt5_tokenizer
        if Wholebody is not None:
            try:
                self._wholebody = Wholebody(
                    to_openpose=False,
                    mode="lightweight",
                    backend="onnxruntime",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception:
                self._wholebody = None
        self._ready = True

    def _extract_pose(self, video_path: Path) -> dict | None:
        if cv2 is None or self._wholebody is None:
            return self._fallback_pose_extraction(str(video_path))

        import numpy as np

        data = {"keypoints": [], "scores": []}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        try:
            source_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self._frame_stride > 1 and (source_index % self._frame_stride) != 0:
                    source_index += 1
                    continue
                if self.ultra_lightweight:
                    frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
                frame = np.uint8(frame)
                keypoints, scores = self._wholebody(frame)
                height, width = frame.shape[:2]
                data["keypoints"].append(keypoints / np.array([width, height])[None, None])
                data["scores"].append(scores)
                source_index += 1
        finally:
            cap.release()

        return data if data["scores"] else None

    def predict(self, video_path: Path) -> InferenceResult:
        with self._lock:
            start = time.perf_counter()
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            pose_data: dict | None = None
            try:
                self._ensure_ready()
                pose_data = self._extract_pose(video_path)
                if pose_data is None:
                    raise RuntimeError(f"pose extraction failed for {video_path}")

                self._dataset.rgb_data = str(video_path)
                self._dataset.pose_data = pose_data

                tgt_pres = []
                with self._torch.no_grad():
                    for src_input, tgt_input in self._dataloader:
                        if self._device == "cuda":
                            for key in src_input.keys():
                                if isinstance(src_input[key], self._torch.Tensor):
                                    src_input[key] = src_input[key].to(self._target_dtype).cuda()

                        stack_out = self._model(src_input, tgt_input)
                        output = self._model.generate(
                            stack_out,
                            max_new_tokens=self._max_new_tokens,
                            num_beams=self._num_beams,
                        )
                        for item in output:
                            tgt_pres.append(item)

                if not tgt_pres:
                    raise RuntimeError("model returned empty sequence")

                padding_value = self._tokenizer.eos_token_id
                if len(tgt_pres[0]) < 150:
                    pad_tensor = self._torch.ones(150 - len(tgt_pres[0]), device=tgt_pres[0].device) * padding_value
                    tgt_pres[0] = self._torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

                tgt_pres = self._pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
                decoded = self._tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
                prediction = decoded[0].strip() if decoded else ""
                stdout_parts.append(f"Prediction result is: {prediction}")
                return_code = 0 if prediction else 1
                success = bool(prediction)
            except Exception as exc:
                prediction = ""
                stderr_parts.append(str(exc))
                return_code = 1
                success = False

            latency_s = time.perf_counter() - start
            frame_count = _video_frame_count(video_path)
            effective_fps = frame_count / latency_s if latency_s > 0 and frame_count > 0 else 0.0
            return InferenceResult(
                success=success,
                prediction=prediction,
                latency_s=latency_s,
                effective_fps=effective_fps,
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                return_code=return_code,
                command=["inprocess", "demo.online_inference"],
                pose_data=pose_data,
            )


_SESSION_CACHE: dict[tuple[str, bool], _RealtimeInferenceSession] = {}
_SESSION_CACHE_LOCK = Lock()


def _run_online_inference_subprocess(
    project_root: Path,
    video_path: Path,
    checkpoint_path: Path,
    rgb_support: bool = False,
    python_executable: str | None = None,
) -> InferenceResult:
    python_executable = python_executable or sys.executable
    command = [
        python_executable,
        "demo/online_inference.py",
        "--online_video",
        str(video_path),
        "--finetune",
        str(checkpoint_path),
    ]
    if rgb_support:
        command.append("--rgb_support")

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    start = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    latency_s = time.perf_counter() - start
    prediction = _parse_prediction(process.stdout, process.stderr)
    frame_count = _video_frame_count(video_path)
    effective_fps = frame_count / latency_s if latency_s > 0 and frame_count > 0 else 0.0

    return InferenceResult(
        success=process.returncode == 0 and bool(prediction),
        prediction=prediction,
        latency_s=latency_s,
        effective_fps=effective_fps,
        stdout=process.stdout,
        stderr=process.stderr,
        return_code=process.returncode,
        command=command,
    )


def run_online_inference(
    project_root: Path,
    video_path: Path,
    checkpoint_path: Path,
    rgb_support: bool = False,
    python_executable: str | None = None,
    lightweight: bool = False,
    ultra_lightweight: bool = False,
) -> InferenceResult:
    cache_key = (
        str(checkpoint_path.resolve()),
        bool(rgb_support),
        bool(lightweight),
        bool(ultra_lightweight),
    )
    try:
        with _SESSION_CACHE_LOCK:
            session = _SESSION_CACHE.get(cache_key)
            if session is None:
                session = _RealtimeInferenceSession(
                    project_root=project_root,
                    checkpoint_path=checkpoint_path,
                    rgb_support=rgb_support,
                    lightweight=lightweight,
                    ultra_lightweight=ultra_lightweight,
                )
                _SESSION_CACHE[cache_key] = session
        return session.predict(video_path)
    except Exception as exc:
        fallback = _run_online_inference_subprocess(
            project_root=project_root,
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            rgb_support=rgb_support,
            python_executable=python_executable,
        )
        if not fallback.stderr:
            fallback = InferenceResult(
                success=fallback.success,
                prediction=fallback.prediction,
                latency_s=fallback.latency_s,
                effective_fps=fallback.effective_fps,
                stdout=fallback.stdout,
                stderr=f"in-process fallback error: {exc}",
                return_code=fallback.return_code,
                command=fallback.command,
            )
        return fallback
