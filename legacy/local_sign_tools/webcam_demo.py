import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import S2T_Dataset_online
from models import Uni_Sign
import utils as utils
from rtmlib import PoseTracker, Wholebody, draw_skeleton

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam demo for pose-only sign language translation.")
    parser.add_argument("--finetune", required=True, help="Path to Uni-Sign checkpoint.")
    parser.add_argument("--dataset", default="CE_CSL", choices=["CSL_News", "CSL_Daily", "CE_CSL", "WLASL", "How2Sign", "OpenASL"])
    parser.add_argument("--camera_id", type=int, default=0, help="OpenCV camera id.")
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--no_window", action="store_true", help="Disable cv2.imshow and run in terminal-only mode.")
    parser.add_argument("--save_video", default="", help="Optional path to save the rendered preview video.")
    parser.add_argument("--status_interval", type=float, default=2.0, help="Seconds between terminal status prints in no-window mode.")
    parser.add_argument("--max_runtime_seconds", type=float, default=0.0, help="Optional max runtime for no-window mode. 0 means unlimited.")
    parser.add_argument("--buffer_size", type=int, default=96, help="How many recent frames to keep.")
    parser.add_argument("--min_frames", type=int, default=24, help="Minimum buffered frames before inference.")
    parser.add_argument("--infer_every", type=int, default=12, help="Run rolling inference every N frames.")
    parser.add_argument("--cooldown_seconds", type=float, default=1.5, help="Minimum seconds between auto inferences.")
    parser.add_argument("--motion_window", type=int, default=12, help="How many recent motion scores to average.")
    parser.add_argument("--motion_threshold", type=float, default=0.012, help="Only auto-infer when recent motion is above this value.")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Minimum average pose confidence to keep a frame.")
    parser.add_argument("--kpt_thr", type=float, default=0.35, help="Skeleton drawing confidence threshold.")
    parser.add_argument("--det_frequency", type=int, default=7, help="PoseTracker detector frequency.")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--flip", action="store_true", help="Mirror the webcam preview.")
    parser.add_argument("--no_skeleton", action="store_true", help="Disable skeleton drawing.")
    parser.add_argument("--rgb_support", action="store_true", help="Not supported in webcam mode.")
    return parser.parse_args()


def select_primary_person(keypoints, scores):
    if keypoints is None or scores is None:
        return None, None

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.size == 0 or scores.size == 0:
        return None, None

    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]

    best_idx = int(np.nan_to_num(scores, nan=0.0).mean(axis=1).argmax())
    return keypoints[best_idx:best_idx + 1], scores[best_idx:best_idx + 1]


def normalize_keypoints(keypoints, width, height):
    scale = np.array([width, height], dtype=np.float32).reshape(1, 1, 2)
    return keypoints / scale


def compute_motion(previous_keypoints, current_keypoints, current_scores, score_threshold):
    if previous_keypoints is None or current_keypoints is None:
        return 0.0

    prev = previous_keypoints[0]
    curr = current_keypoints[0]
    scores = current_scores[0]
    valid = scores >= score_threshold
    if not np.any(valid):
        return 0.0

    motion = np.linalg.norm(curr[valid] - prev[valid], axis=-1)
    if motion.size == 0:
        return 0.0
    return float(motion.mean())


def move_src_input(src_input, device, dtype_mode="fp32"):
    tensor_dtype = torch.float32
    if dtype_mode == "bf16" and device.type == "cuda":
        tensor_dtype = torch.bfloat16

    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.dtype.is_floating_point:
                value = value.to(tensor_dtype)
            src_input[key] = value
    return src_input


def supports_bf16(device):
    return device.type == "cuda" and torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()


def load_model(args, device):
    model = Uni_Sign(args=args)
    checkpoint = torch.load(args.finetune, map_location="cpu")
    ret = model.load_state_dict(checkpoint["model"], strict=True)
    print(f"missing_keys: {len(ret.missing_keys)}")
    print(f"unexpected_keys: {len(ret.unexpected_keys)}")
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, device, use_autocast, args, pose_window):
    pose_data = {
        "keypoints": [np.asarray(item, dtype=np.float32) for item in pose_window["keypoints"]],
        "scores": [np.asarray(item, dtype=np.float32) for item in pose_window["scores"]],
    }

    dataset = S2T_Dataset_online(args=args)
    dataset.pose_data = pose_data
    sample = dataset[0]
    src_input, tgt_input = dataset.collate_fn([sample])
    src_input = move_src_input(src_input, device, args.dtype)

    start_time = time.time()
    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                stack_out = model(src_input, tgt_input)
                output = model.generate(
                    stack_out,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
        else:
            stack_out = model(src_input, tgt_input)
            output = model.generate(
                stack_out,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
    elapsed_ms = (time.time() - start_time) * 1000.0

    pred_text = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)
    return pred_text[0].strip(), elapsed_ms


def get_font(font_size):
    if ImageFont is None:
        return None

    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def wrap_text(text, width):
    if not text:
        return [""]
    return [text[idx:idx + width] for idx in range(0, len(text), width)]


def draw_text_panel(frame, lines):
    if Image is None or ImageDraw is None:
        y = 30
        for line in lines:
            cv2.putText(frame, line.encode("ascii", errors="ignore").decode("ascii"), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, line.encode("ascii", errors="ignore").decode("ascii"), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
            y += 30
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = get_font(24)

    x = 12
    y = 10
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y = bbox[3] + 12

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def resize_for_display(frame, display_width):
    if display_width <= 0 or frame.shape[1] <= display_width:
        return frame
    scale = display_width / frame.shape[1]
    display_height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (display_width, display_height))


def create_video_writer(save_path, frame_shape, fps=20.0):
    if not save_path:
        return None

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (frame_shape[1], frame_shape[0]))


def build_status_lines(prediction, frame_count, paused, motion_value, infer_ms, score_value):
    lines = [
        f"Prediction: {prediction or '[no prediction yet]'}",
        f"Buffered frames: {frame_count} | Status: {'paused' if paused else 'recording'}",
        f"Recent motion: {motion_value:.4f} | Pose score: {score_value:.3f} | Last infer: {infer_ms:.0f} ms",
        "Keys: [P] pause/resume  [Space] new phrase  [Enter] infer now  [C] clear text  [Q] quit",
    ]
    wrapped_lines = []
    for idx, line in enumerate(lines):
        wrap_width = 26 if idx == 0 else 80
        wrapped_lines.extend(wrap_text(line, wrap_width))
    return wrapped_lines


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    utils.set_seed(args.seed)

    if args.rgb_support:
        raise ValueError("webcam_demo.py currently supports pose-only inference only. Please omit --rgb_support.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    use_autocast = args.dtype == "bf16" and supports_bf16(device)
    if args.dtype == "bf16" and not use_autocast:
        print("bf16 is not available on this device, falling back to fp32 inference.")

    model = load_model(args, device)

    tracker = PoseTracker(
        Wholebody,
        det_frequency=args.det_frequency,
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    if os.name == "nt":
        cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}.")

    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    pose_buffer = deque(maxlen=args.buffer_size)
    score_buffer = deque(maxlen=args.buffer_size)
    motion_history = deque(maxlen=args.motion_window)

    paused = False
    last_prediction = ""
    last_infer_ms = 0.0
    last_infer_at = 0.0
    last_status_at = 0.0
    previous_keypoints = None
    frame_idx = 0
    start_at = time.time()
    window_enabled = not args.no_window
    video_writer = None

    window_name = "Uni-Sign Webcam Demo"
    print("Webcam demo started.")
    if window_enabled:
        print("Keys: P pause/resume | Space new phrase | Enter infer now | C clear text | Q quit")
    else:
        print("No-window mode enabled. Press Ctrl+C in the terminal to stop.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if args.flip:
                frame = cv2.flip(frame, 1)

            keypoints, scores = tracker(frame)
            primary_keypoints, primary_scores = select_primary_person(keypoints, scores)

            mean_score = 0.0
            motion_value = 0.0
            if primary_keypoints is not None and primary_scores is not None:
                mean_score = float(np.nan_to_num(primary_scores, nan=0.0).mean())
                normalized_keypoints = normalize_keypoints(primary_keypoints, frame.shape[1], frame.shape[0])
                motion_value = compute_motion(previous_keypoints, normalized_keypoints, primary_scores, args.score_threshold)
                previous_keypoints = normalized_keypoints
                motion_history.append(motion_value)

                if not paused and mean_score >= args.score_threshold:
                    pose_buffer.append(normalized_keypoints.astype(np.float32))
                    score_buffer.append(primary_scores.astype(np.float32))
            else:
                previous_keypoints = None
                motion_history.append(0.0)

            avg_motion = float(sum(motion_history) / len(motion_history)) if motion_history else 0.0

            should_infer = (
                not paused
                and len(pose_buffer) >= args.min_frames
                and frame_idx % args.infer_every == 0
                and avg_motion >= args.motion_threshold
                and (time.time() - last_infer_at) >= args.cooldown_seconds
            )

            if should_infer:
                pose_window = {"keypoints": list(pose_buffer), "scores": list(score_buffer)}
                try:
                    last_prediction, last_infer_ms = run_inference(model, device, use_autocast, args, pose_window)
                    last_infer_at = time.time()
                    print(f"[auto] prediction: {last_prediction}")
                except Exception as exc:  # pragma: no cover
                    print(f"[auto] inference failed: {exc}")

            display = frame.copy()
            if not args.no_skeleton and primary_keypoints is not None and primary_scores is not None:
                display = draw_skeleton(display, primary_keypoints, primary_scores, kpt_thr=args.kpt_thr)

            status_lines = build_status_lines(
                prediction=last_prediction,
                frame_count=len(pose_buffer),
                paused=paused,
                motion_value=avg_motion,
                infer_ms=last_infer_ms,
                score_value=mean_score,
            )
            display = draw_text_panel(display, status_lines)
            display = resize_for_display(display, args.display_width)

            if args.save_video:
                if video_writer is None:
                    video_writer = create_video_writer(args.save_video, display.shape)
                    print(f"Saving preview video to: {args.save_video}")
                video_writer.write(display)

            if window_enabled:
                try:
                    cv2.imshow(window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    window_enabled = False
                    print("OpenCV window is not available in this environment. Switching to no-window mode.")
                    print(f"OpenCV detail: {exc}")
                    print("You can keep the demo running and watch predictions in the terminal, or add --save_video to save the preview.")
                    continue

                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                    print("paused" if paused else "recording resumed")
                elif key == ord("c"):
                    last_prediction = ""
                    last_infer_ms = 0.0
                    print("prediction cleared")
                elif key == ord(" "):
                    pose_buffer.clear()
                    score_buffer.clear()
                    motion_history.clear()
                    previous_keypoints = None
                    print("started a new phrase buffer")
                elif key in (10, 13):
                    if len(pose_buffer) < args.min_frames:
                        print(f"not enough frames for inference: {len(pose_buffer)} / {args.min_frames}")
                    else:
                        pose_window = {"keypoints": list(pose_buffer), "scores": list(score_buffer)}
                        try:
                            last_prediction, last_infer_ms = run_inference(model, device, use_autocast, args, pose_window)
                            last_infer_at = time.time()
                            print(f"[manual] prediction: {last_prediction}")
                        except Exception as exc:  # pragma: no cover
                            print(f"[manual] inference failed: {exc}")
            else:
                now = time.time()
                if now - last_status_at >= args.status_interval:
                    print(
                        f"[status] buffered={len(pose_buffer)} paused={paused} "
                        f"motion={avg_motion:.4f} score={mean_score:.3f} "
                        f"prediction={last_prediction or '[no prediction yet]'}"
                    )
                    last_status_at = now

            if args.max_runtime_seconds > 0 and (time.time() - start_at) >= args.max_runtime_seconds:
                print(f"Reached max runtime: {args.max_runtime_seconds} seconds")
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if window_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
