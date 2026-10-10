import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from paths import CACHE_ROOT, CE_CSL_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "rtmlib"))

from rtmlib import Wholebody


DEFAULT_CECSL_ROOT = str(CE_CSL_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CE-CSL pose_format from videos.")
    parser.add_argument(
        "--cecsl_root",
        default=os.environ.get("CE_CSL_ROOT", DEFAULT_CECSL_ROOT),
        help="Root path of the CE-CSL dataset.",
    )
    parser.add_argument(
        "--src_subdir",
        default="video",
        help="Video subdirectory under CE-CSL root.",
    )
    parser.add_argument(
        "--tgt_subdir",
        default="pose_format",
        help="Target pose subdirectory under CE-CSL root.",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max_videos_per_split", type=int, default=0)
    return parser.parse_args()


def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    h, w, _ = frame.shape
    return keypoints, scores, [w, h]


def collect_video_files(split_dir):
    return sorted(split_dir.rglob("*.mp4"))


def process_video(video_path, output_path, wholebody, max_workers=16, overwrite=False):
    if output_path.exists() and not overwrite:
        return "skip"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"failed_open:{video_path}"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return f"empty_video:{video_path}"

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in frames]
        for future in futures:
            results.append(future.result())

    for keypoints, scores, w_h in results:
        data["keypoints"].append(keypoints / np.array(w_h)[None, None])
        data["scores"].append(scores)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return "ok"


def main():
    args = parse_args()
    cecsl_root = Path(args.cecsl_root)
    src_root = cecsl_root / args.src_subdir
    tgt_root = cecsl_root / args.tgt_subdir

    wholebody = Wholebody(
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    total = 0
    ok = 0
    skipped = 0
    failed = []

    for split in args.splits:
        split_src = src_root / split
        if not split_src.exists():
            print(f"missing split dir, skip: {split_src}")
            continue

        video_files = collect_video_files(split_src)
        if args.max_videos_per_split > 0:
            video_files = video_files[: args.max_videos_per_split]
        print(f"{split}: found {len(video_files)} videos")

        for video_path in tqdm(video_files, desc=f"Processing {split}"):
            rel_path = video_path.relative_to(src_root)
            output_path = (tgt_root / rel_path).with_suffix(".pkl")
            status = process_video(
                video_path=video_path,
                output_path=output_path,
                wholebody=wholebody,
                max_workers=args.max_workers,
                overwrite=args.overwrite,
            )
            total += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed.append(status)

    print(f"total: {total}")
    print(f"ok: {ok}")
    print(f"skipped: {skipped}")
    print(f"failed: {len(failed)}")
    if failed:
        print("failed samples:")
        for item in failed[:20]:
            print(item)


if __name__ == "__main__":
    main()
