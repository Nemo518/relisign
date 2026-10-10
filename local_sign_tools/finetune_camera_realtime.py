import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Base_Dataset, build_uniform_indices, load_part_kp, load_support_rgb_dict
from models import Uni_Sign
from paths import CACHE_ROOT, OUTPUT_ROOT, PRETRAINED_ROOT, resolve_output_dir, resolve_storage_path
from local_sign_tools.finetune_cecsl_rgb_support import (
    build_optimizer,
    build_stage_plan,
    configure_trainable_params,
    evaluate_decode,
    evaluate_loss,
    load_checkpoint_flexible,
    move_src_input,
    save_checkpoint,
    seed_worker_with_base,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a realtime-window Uni-Sign checkpoint on camera-domain manifests.")
    parser.add_argument("--train_manifest", required=True, help="JSON/JSONL/CSV manifest for realtime camera training samples.")
    parser.add_argument("--eval_manifest", default="", help="Optional manifest used for validation loss.")
    parser.add_argument("--selection_manifest", default="", help="Optional manifest used for decode-based checkpoint selection.")
    parser.add_argument("--checkpoint", default=str(OUTPUT_ROOT / "stage3_finetuning" / "best_checkpoint.pth"))
    parser.add_argument("--output_dir", default=str(OUTPUT_ROOT / "realtime_window_finetuning"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stage1_epochs", type=int, default=0)
    parser.add_argument("--stage2_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--stage1_lr", type=float, default=3e-5)
    parser.add_argument("--stage2_lr", type=float, default=0.0)
    parser.add_argument("--stage1_proj_lr_scale", type=float, default=0.3)
    parser.add_argument("--stage2_mt5_lr_scale", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_mt5", action="store_true")
    parser.add_argument("--window_seconds", type=float, default=2.0, help="Window length used to build realtime training clips.")
    parser.add_argument("--window_stride_seconds", type=float, default=1.0, help="Stride between consecutive realtime windows.")
    parser.add_argument("--min_window_seconds", type=float, default=1.0, help="Minimum allowed window duration before a clip is skipped.")
    parser.add_argument("--max_windows_per_sample", type=int, default=6, help="Cap windows per original sample. 0 means keep all.")
    parser.add_argument("--train_window_limit", type=int, default=0, help="Optional cap on train windows after expansion.")
    parser.add_argument("--eval_window_limit", type=int, default=0, help="Optional cap on eval windows after expansion.")
    parser.add_argument("--selection_window_limit", type=int, default=24, help="Optional cap on decode-selection windows after expansion.")
    parser.add_argument("--selection_sample_count", type=int, default=16)
    parser.add_argument("--selection_sample_mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--selection_offset", type=int, default=0)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--selection_batch_size", type=int, default=1)
    parser.add_argument("--selection_max_new_tokens", type=int, default=80)
    parser.add_argument("--selection_num_beams", type=int, default=2)
    parser.add_argument("--rgb_support", dest="rgb_support", action="store_true", help="Enable RGB-support in realtime fine-tuning.")
    parser.add_argument("--no_rgb_support", dest="rgb_support", action="store_false", help="Disable RGB-support for faster realtime adaptation.")
    parser.set_defaults(rgb_support=False)
    parser.add_argument(
        "--rgb_backbone_weights",
        default=str(PRETRAINED_ROOT / "efficientnet_b0_rwightman-7f5810bc.pth"),
    )
    parser.add_argument("--rgb_backbone_pretrained", action="store_true")
    parser.add_argument("--rgb_frame_sampling", default="confidence_topk", choices=["confidence_topk", "confidence_weighted"])
    parser.add_argument("--fusion_use_uncertainty_gate", action="store_true")
    parser.add_argument("--fusion_meta_dim", type=int, default=3)
    parser.add_argument("--pose_cache_dir", default=str(CACHE_ROOT / "realtime_camera_pose"))
    parser.add_argument("--save_epoch_checkpoints", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "samples" in payload:
            return payload["samples"]
        raise ValueError(f"Unsupported JSON manifest structure: {manifest_path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def choose_indices(total_size, sample_count, mode, offset, seed):
    if total_size <= 0:
        return []
    sample_count = max(1, min(sample_count, total_size))
    if mode == "random":
        rng = random.Random(seed)
        indices = list(range(total_size))
        rng.shuffle(indices)
        return sorted(indices[:sample_count])
    start = min(max(0, offset), total_size - 1)
    end = min(total_size, start + sample_count)
    return list(range(start, end))


def _resolve_optional_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    return Path(resolve_storage_path(str(raw_value)))


def _get_record_text(record: dict) -> str:
    for key in ("text", "reference", "label", "sentence"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _get_record_gloss(record: dict) -> str:
    value = record.get("gloss")
    return "" if value is None else str(value)


def _read_video_meta(video_path: Path) -> tuple[int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frame_count, fps


class CameraRealtimeWindowDataset(Base_Dataset):
    def __init__(self, manifest_path: Path, args, split_name: str):
        super().__init__()
        self.args = args
        self.split_name = split_name
        self.rgb_support = args.rgb_support
        self.max_length = args.max_length
        self.window_seconds = args.window_seconds
        self.window_stride_seconds = args.window_stride_seconds
        self.min_window_seconds = args.min_window_seconds
        self.pose_cache_dir = Path(resolve_storage_path(args.pose_cache_dir))
        self.pose_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.records = []
        self.window_records = []

        for idx, raw_record in enumerate(load_manifest(manifest_path)):
            video_raw = raw_record.get("video_path") or raw_record.get("video")
            if not video_raw:
                continue
            video_path = _resolve_optional_path(str(video_raw))
            if video_path is None or not video_path.exists():
                continue
            pose_path = _resolve_optional_path(raw_record.get("pose_path"))
            sample_id = str(raw_record.get("sample_id") or raw_record.get("id") or f"{split_name}_{idx:05d}")
            record = {
                "sample_id": sample_id,
                "video_path": video_path,
                "pose_path": pose_path,
                "text": _get_record_text(raw_record),
                "gloss": _get_record_gloss(raw_record),
            }
            self.records.append(record)

        self._build_windows()

    def _build_windows(self):
        for record_idx, record in enumerate(self.records):
            if record["pose_path"] is not None and record["pose_path"].exists():
                try:
                    pose = pickle.load(open(record["pose_path"], "rb"))
                    total_frames = len(pose.get("scores", []))
                    fps = float(record.get("fps") or 0.0)
                except Exception:
                    total_frames = 0
                    fps = 0.0
            else:
                total_frames, fps = _read_video_meta(record["video_path"])

            if total_frames <= 0:
                continue
            if fps <= 0:
                fps = 25.0

            window_frames = max(int(round(self.window_seconds * fps)), 1)
            stride_frames = max(int(round(self.window_stride_seconds * fps)), 1)
            min_frames = max(int(round(self.min_window_seconds * fps)), 1)
            if total_frames < min_frames:
                continue

            start_positions = list(range(0, max(total_frames - window_frames + 1, 1), stride_frames))
            if not start_positions:
                start_positions = [0]
            if start_positions[-1] + window_frames < total_frames:
                start_positions.append(max(total_frames - window_frames, 0))

            if self.args.max_windows_per_sample > 0 and len(start_positions) > self.args.max_windows_per_sample:
                keep_positions = build_uniform_indices(len(start_positions), self.args.max_windows_per_sample).tolist()
                start_positions = [start_positions[pos] for pos in keep_positions]

            for window_id, start_frame in enumerate(start_positions):
                end_frame = min(total_frames, start_frame + window_frames)
                if end_frame - start_frame < min_frames:
                    continue
                self.window_records.append(
                    {
                        "record_idx": record_idx,
                        "window_id": window_id,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "fps": fps,
                    }
                )

    def __len__(self):
        return len(self.window_records)

    def _pose_cache_path(self, record: dict) -> Path:
        stem = record["sample_id"].replace("/", "_").replace("\\", "_")
        return self.pose_cache_dir / f"{stem}.pkl"

    def _load_pose_payload(self, record: dict):
        if record["pose_path"] is not None and record["pose_path"].exists():
            return pickle.load(open(record["pose_path"], "rb"))

        cache_path = self._pose_cache_path(record)
        if cache_path.exists():
            return pickle.load(open(cache_path, "rb"))

        from demo.online_inference import pose_extraction

        pose_payload = pose_extraction(str(record["video_path"]))
        if pose_payload is None:
            raise RuntimeError(f"Failed to extract pose for {record['video_path']}")
        with cache_path.open("wb") as handle:
            pickle.dump(pose_payload, handle)
        return pose_payload

    def __getitem__(self, index):
        window_meta = self.window_records[index]
        record = self.records[window_meta["record_idx"]]
        pose = self._load_pose_payload(record)

        start_frame = window_meta["start_frame"]
        end_frame = window_meta["end_frame"]
        frame_indices = np.arange(start_frame, end_frame, dtype=np.int64)
        if len(frame_indices) == 0:
            raise RuntimeError(f"Empty realtime window for sample {record['sample_id']}")

        local_indices = build_uniform_indices(len(frame_indices), self.max_length)
        selected_indices = frame_indices[local_indices]

        skeletons = [pose["keypoints"][int(frame_idx)] for frame_idx in selected_indices]
        confs = [pose["scores"][int(frame_idx)] for frame_idx in selected_indices]
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_support_rgb_dict(
                selected_indices,
                skeletons,
                confs,
                str(record["video_path"]),
                self.data_transform,
                sample_mode=self.args.rgb_frame_sampling,
            )

        sample_name = f"{record['sample_id']}#w{window_meta['window_id']:03d}"
        return sample_name, kps_with_scores, record["text"], record["gloss"], support_rgb_dict

    def __str__(self):
        return f"{self.split_name}: windows={len(self.window_records)} samples={len(self.records)}"


def build_model_args(args):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=args.rgb_support,
        rgb_backbone_weights=args.rgb_backbone_weights,
        rgb_backbone_pretrained=args.rgb_backbone_pretrained,
        rgb_frame_sampling=args.rgb_frame_sampling,
        fusion_use_uncertainty_gate=args.fusion_use_uncertainty_gate,
        fusion_meta_dim=args.fusion_meta_dim,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=args.label_smoothing,
    )


def build_loader(args, manifest_path: str, split_name: str, window_limit: int, *, batch_size=None, num_workers=None, shuffle=None, subset_indices=None):
    if not manifest_path:
        return None, [], [], None
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if split_name == "train" else 1000))

    if batch_size is None:
        batch_size = args.batch_size
    if num_workers is None:
        num_workers = args.num_workers
    if shuffle is None:
        shuffle = split_name == "train"

    manifest = Path(resolve_storage_path(manifest_path))
    dataset = CameraRealtimeWindowDataset(manifest, args, split_name)
    available_indices = list(range(len(dataset)))
    if window_limit > 0:
        available_indices = available_indices[:window_limit]
    active_indices = available_indices if subset_indices is None else subset_indices
    subset = Subset(dataset, active_indices)
    worker_init_fn = None if num_workers == 0 else partial(seed_worker_with_base, args.seed)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return dataset, available_indices, active_indices, loader


def main():
    args = parse_args()
    args.train_manifest = resolve_storage_path(args.train_manifest)
    args.eval_manifest = resolve_storage_path(args.eval_manifest) if args.eval_manifest else ""
    args.selection_manifest = resolve_storage_path(args.selection_manifest) if args.selection_manifest else ""
    args.checkpoint = resolve_storage_path(args.checkpoint)
    args.rgb_backbone_weights = resolve_storage_path(args.rgb_backbone_weights)
    args.pose_cache_dir = resolve_storage_path(args.pose_cache_dir)
    args.output_dir = resolve_output_dir(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_available_indices, _, train_loader = build_loader(
        args,
        args.train_manifest,
        "train",
        args.train_window_limit,
        shuffle=True,
    )
    if train_dataset is None or not train_available_indices:
        raise RuntimeError("No realtime training windows were created from the provided train manifest.")
    print(train_dataset)
    print(f"train_windows: {len(train_available_indices)}")
    print(f"rgb_support: {args.rgb_support}")
    print(f"window_seconds: {args.window_seconds}")
    print(f"window_stride_seconds: {args.window_stride_seconds}")

    eval_loader = None
    if args.eval_manifest:
        eval_dataset, eval_available_indices, _, eval_loader = build_loader(
            args,
            args.eval_manifest,
            "eval",
            args.eval_window_limit,
            shuffle=False,
        )
        print(eval_dataset)
        print(f"eval_windows: {len(eval_available_indices)}")

    selection_loader = None
    selection_dataset_indices = []
    if args.selection_manifest and args.selection_sample_count > 0:
        _, selection_available_indices, _, _ = build_loader(
            args,
            args.selection_manifest,
            "selection",
            args.selection_window_limit,
            batch_size=args.selection_batch_size,
            num_workers=0,
            shuffle=False,
        )
        chosen_positions = choose_indices(
            len(selection_available_indices),
            args.selection_sample_count,
            args.selection_sample_mode,
            args.selection_offset,
            args.selection_seed,
        )
        selection_dataset_indices = [selection_available_indices[pos] for pos in chosen_positions]
        _, _, _, selection_loader = build_loader(
            args,
            args.selection_manifest,
            "selection",
            args.selection_window_limit,
            batch_size=args.selection_batch_size,
            num_workers=0,
            shuffle=False,
            subset_indices=selection_dataset_indices,
        )
        print(f"selection_windows: {selection_dataset_indices}")

    model_args = build_model_args(args)
    model = Uni_Sign(args=model_args)
    checkpoint, ret, skipped_keys = load_checkpoint_flexible(model, args.checkpoint)
    print(f"missing_keys: {len(ret.missing_keys)} | unexpected_keys: {len(ret.unexpected_keys)} | skipped_keys: {len(skipped_keys)}")

    model = model.to(device)
    model.train()

    use_autocast = args.dtype == "bf16" and device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not use_autocast:
        print("bf16 is not available on this device, falling back to fp32 training.")

    stage_plan = build_stage_plan(args)
    total_epochs = sum(stage["epochs"] for stage in stage_plan)
    history_path = output_dir / "training_metrics.jsonl"
    if history_path.exists():
        history_path.unlink()

    best_metrics = None
    global_step = 0
    global_epoch = 0

    for stage in stage_plan:
        configure_trainable_params(model, stage["name"], args.freeze_mt5)
        optimizer, optimizer_summaries = build_optimizer(model, args, stage["name"])
        print(f"stage_start: {stage['name']} | epochs={stage['epochs']}")
        for summary in optimizer_summaries:
            print(f"optimizer_group[{summary['name']}]: lr={summary['lr']:.2e} | tensors={summary['param_tensors']} | params={summary['param_count']}")

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            running_loss = 0.0
            batch_count = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (src_input, tgt_input) in enumerate(train_loader, start=1):
                src_input = move_src_input(src_input, device, args.dtype)
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        stack_out = model(src_input, tgt_input)
                        loss = stack_out["loss"]
                else:
                    stack_out = model(src_input, tgt_input)
                    loss = stack_out["loss"]

                (loss / args.gradient_accumulation_steps).backward()
                if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running_loss += float(loss.item())
                batch_count += 1

            avg_train_loss = running_loss / max(1, batch_count)
            eval_loss = evaluate_loss(model, eval_loader, device, args.dtype)
            decode_metrics = evaluate_decode(
                model,
                selection_loader,
                selection_dataset_indices,
                device,
                args.dtype,
                args.selection_max_new_tokens,
                args.selection_num_beams,
            )

            metrics = {
                "stage": stage["name"],
                "epoch": global_epoch,
                "stage_epoch": stage_epoch,
                "train_loss": avg_train_loss,
                "eval_loss": eval_loss,
                "selection_average_similarity": None if decode_metrics is None else decode_metrics["average_similarity"],
                "selection_exact_match_count": None if decode_metrics is None else decode_metrics["exact_match_count"],
                "selection_evaluated_samples": None if decode_metrics is None else decode_metrics["evaluated_samples"],
                "window_seconds": args.window_seconds,
                "window_stride_seconds": args.window_stride_seconds,
                "rgb_support": args.rgb_support,
            }

            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            checkpoint_extra = dict(metrics)
            if decode_metrics is not None:
                checkpoint_extra["selection_rows"] = decode_metrics["rows"]

            if args.save_epoch_checkpoints:
                save_checkpoint(output_dir / f"checkpoint_epoch_{global_epoch}.pth", model, args, extra=checkpoint_extra)

            if best_metrics is None or (
                (metrics["selection_average_similarity"] or -1.0) > (best_metrics.get("selection_average_similarity") or -1.0) + 1e-8
                or (
                    math.isclose(metrics["selection_average_similarity"] or -1.0, best_metrics.get("selection_average_similarity") or -1.0, abs_tol=1e-8)
                    and (metrics["eval_loss"] or math.inf) < (best_metrics.get("eval_loss") or math.inf)
                )
            ):
                best_metrics = dict(metrics)
                save_checkpoint(output_dir / "best_checkpoint.pth", model, args, extra=checkpoint_extra)
                save_checkpoint(output_dir / "best_realtime_window_checkpoint.pth", model, args, extra=checkpoint_extra)

    save_checkpoint(output_dir / "final_checkpoint.pth", model, args, extra={"epoch": global_epoch, "stage": stage_plan[-1]["name"]})
    print(f"realtime fine-tuning finished, checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
