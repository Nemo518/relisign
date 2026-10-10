import argparse
import csv
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import dev_label_paths, test_label_paths, train_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign
from paths import PRETRAINED_ROOT, resolve_storage_path


LABEL_PATHS = {
    "train": train_label_paths["CE_CSL"],
    "dev": dev_label_paths["CE_CSL"],
    "test": test_label_paths["CE_CSL"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch sample evaluation for CE-CSL checkpoints.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", required=True, help="Path to Uni-Sign checkpoint.")
    parser.add_argument("--output_csv", default="", help="Optional CSV path for detailed sample outputs.")
    parser.add_argument("--sample_count", type=int, default=20, help="Number of samples to evaluate.")
    parser.add_argument("--offset", type=int, default=0, help="Start index used in sequential mode.")
    parser.add_argument("--sample_mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--rgb_support", dest="rgb_support", action="store_true")
    parser.add_argument("--no_rgb_support", dest="rgb_support", action="store_false")
    parser.set_defaults(rgb_support=None)
    parser.add_argument(
        "--rgb_backbone_weights",
        default=str(PRETRAINED_ROOT / "efficientnet_b0_rwightman-7f5810bc.pth"),
        help="Local EfficientNet-B0 weight path used when --rgb_support is enabled.",
    )
    parser.add_argument(
        "--rgb_backbone_pretrained",
        action="store_true",
        help="Reserved compatibility flag for RGB backbone loading.",
    )
    parser.add_argument(
        "--rgb_frame_sampling",
        default=None,
        choices=["confidence_topk", "confidence_weighted", "uniform", "random"],
        help="How RGB-support selects hand frames before cropping.",
    )
    parser.add_argument("--fusion_use_uncertainty_gate", dest="fusion_use_uncertainty_gate", action="store_true")
    parser.add_argument("--no_fusion_use_uncertainty_gate", dest="fusion_use_uncertainty_gate", action="store_false")
    parser.set_defaults(fusion_use_uncertainty_gate=None)
    parser.add_argument("--fusion_meta_dim", type=int, default=3)
    parser.add_argument("--top_k_ratio", type=float, default=None)
    parser.add_argument(
        "--experiment_variant",
        default="auto",
        choices=["auto", "pose_only", "rgb_gate", "rgb_metadata_gate"],
        help="Infer the architecture from checkpoint contents by default.",
    )
    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        help="Diagnostic only. Reported Test500 runs must use strict checkpoint loading.",
    )
    parser.add_argument("--output_metrics_json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset_args(args):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=args.rgb_support,
        rgb_backbone_weights=args.rgb_backbone_weights,
        rgb_backbone_pretrained=args.rgb_backbone_pretrained,
        rgb_frame_sampling=args.rgb_frame_sampling,
        fusion_use_uncertainty_gate=args.fusion_use_uncertainty_gate,
        fusion_meta_dim=args.fusion_meta_dim,
        top_k_ratio=args.top_k_ratio,
        max_length=args.max_length,
        hidden_dim=256,
        label_smoothing=0.2,
    )


def move_src_input(src_input, device, dtype_mode):
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


def discover_available_indices(dataset, require_rgb):
    available_indices = []
    for idx, key in enumerate(dataset.list):
        sample = dataset.raw_data[key]
        pose_path = Path(dataset.pose_dir) / sample["video_path"].replace(".mp4", ".pkl")
        rgb_path = Path(dataset.rgb_dir) / sample["video_path"]
        if pose_path.exists() and (not require_rgb or rgb_path.exists()):
            available_indices.append(idx)
    return available_indices


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


def similarity_score(reference, prediction):
    return SequenceMatcher(None, reference, prediction).ratio()


def load_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def infer_checkpoint_variant(checkpoint):
    checkpoint_args = checkpoint.get("args", {}) or {}
    state_keys = checkpoint["model"].keys()
    has_rgb = any(key.startswith("rgb_support_backbone.") for key in state_keys)
    has_metadata = any(key.startswith("fusion_meta_proj.") for key in state_keys)
    inferred = "pose_only" if not has_rgb else ("rgb_metadata_gate" if has_metadata else "rgb_gate")

    declared = checkpoint_args.get("experiment_variant")
    if declared not in (None, "auto", inferred):
        raise RuntimeError(
            f"checkpoint declares experiment_variant={declared!r}, but state_dict implies {inferred!r}."
        )
    return inferred, checkpoint_args


def resolve_eval_protocol(args, checkpoint):
    inferred_variant, checkpoint_args = infer_checkpoint_variant(checkpoint)
    if args.experiment_variant != "auto" and args.experiment_variant != inferred_variant:
        raise ValueError(
            f"--experiment_variant {args.experiment_variant!r} conflicts with checkpoint variant {inferred_variant!r}."
        )

    expected_rgb = inferred_variant != "pose_only"
    expected_metadata = inferred_variant == "rgb_metadata_gate"
    if args.rgb_support is not None and args.rgb_support != expected_rgb:
        raise ValueError("Manual RGB flag conflicts with the checkpoint architecture.")
    if args.fusion_use_uncertainty_gate is not None and args.fusion_use_uncertainty_gate != expected_metadata:
        raise ValueError("Manual metadata-gate flag conflicts with the checkpoint architecture.")

    args.experiment_variant = inferred_variant
    args.rgb_support = expected_rgb
    args.fusion_use_uncertainty_gate = expected_metadata
    args.rgb_frame_sampling = args.rgb_frame_sampling or checkpoint_args.get(
        "eval_rgb_frame_sampling", checkpoint_args.get("rgb_frame_sampling", "confidence_topk")
    )
    if args.top_k_ratio is None:
        args.top_k_ratio = checkpoint_args.get("top_k_ratio", 0.1)
    if args.top_k_ratio is None:
        args.top_k_ratio = 0.1
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError(f"top_k_ratio must be in (0, 1], got {args.top_k_ratio}")
    return checkpoint_args


def load_checkpoint_into_model(model, checkpoint, allow_partial):
    model_state = model.state_dict()
    checkpoint_state = checkpoint["model"]

    if not allow_partial:
        ret = model.load_state_dict(checkpoint_state, strict=True)
        return ret, []

    filtered_state = {}
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    ret = model.load_state_dict(filtered_state, strict=False)
    return ret, skipped_keys


def main():
    args = parse_args()
    args.checkpoint = resolve_storage_path(args.checkpoint)
    args.rgb_backbone_weights = resolve_storage_path(args.rgb_backbone_weights)
    if args.output_csv:
        args.output_csv = resolve_storage_path(args.output_csv)
    if args.output_metrics_json:
        args.output_metrics_json = resolve_storage_path(args.output_metrics_json)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_args = resolve_eval_protocol(args, checkpoint)
    dataset_args = build_dataset_args(args)
    dataset = S2T_Dataset(path=LABEL_PATHS[args.split], args=dataset_args, phase=args.split)
    available_indices = discover_available_indices(dataset, args.rgb_support)
    subset_positions = choose_indices(len(available_indices), args.sample_count, args.sample_mode, args.offset, args.seed)
    indices = [available_indices[position] for position in subset_positions]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"experiment_variant: {args.experiment_variant}")
    print(f"rgb_support: {args.rgb_support}")
    if args.rgb_support:
        print(f"rgb_backbone_weights: {args.rgb_backbone_weights}")
        print(f"rgb_frame_sampling: {args.rgb_frame_sampling}")
        print(f"top_k_ratio: {args.top_k_ratio}")
    print(f"dataset_size: {len(dataset)}")
    print(f"available_samples: {len(available_indices)}")
    print(f"sample_indices: {indices}")

    model = Uni_Sign(args=dataset_args)
    ret, skipped_keys = load_checkpoint_into_model(model, checkpoint, args.allow_partial_checkpoint)
    del checkpoint
    print(f"missing_keys: {len(ret.missing_keys)}")
    print(f"unexpected_keys: {len(ret.unexpected_keys)}")
    print(f"skipped_shape_or_missing_keys: {len(skipped_keys)}")
    if args.allow_partial_checkpoint:
        print("warning: partial checkpoint loading enabled; do not use this run for reported results.")

    model = model.to(device)
    model.eval()
    if args.dtype == "bf16" and device.type == "cuda":
        model = model.to(torch.bfloat16)

    tokenizer = model.mt5_tokenizer
    rows = []

    with torch.no_grad():
        for batch_idx, (src_input, tgt_input) in enumerate(dataloader):
            src_input = move_src_input(src_input, device, args.dtype)
            stack_out = model(src_input, tgt_input)
            output = model.generate(
                stack_out,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

            for item_idx, prediction in enumerate(predictions):
                dataset_index = indices[batch_idx * args.batch_size + item_idx]
                reference = tgt_input["gt_sentence"][item_idx]
                gloss = tgt_input["gt_gloss"][item_idx]
                score = similarity_score(reference, prediction)
                rows.append(
                    {
                        "dataset_index": dataset_index,
                        "similarity": f"{score:.4f}",
                        "reference": reference,
                        "gloss": gloss,
                        "prediction": prediction,
                    }
                )

    average_similarity = sum(float(row["similarity"]) for row in rows) / max(1, len(rows))
    exact_match_count = sum(1 for row in rows if row["reference"] == row["prediction"])
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "experiment_variant": args.experiment_variant,
        "rgb_frame_sampling": args.rgb_frame_sampling if args.rgb_support else None,
        "top_k_ratio": args.top_k_ratio if args.rgb_support else None,
        "evaluated_samples": len(rows),
        "available_samples": len(available_indices),
        "average_similarity": average_similarity,
        "exact_match_count": exact_match_count,
        "strict_checkpoint_load": not args.allow_partial_checkpoint,
        "checkpoint_seed": checkpoint_args.get("seed"),
    }

    print(f"evaluated_samples: {len(rows)}")
    print(f"average_similarity: {average_similarity:.4f}")
    print(f"exact_match_count: {exact_match_count}")
    print("samples:")
    for row in rows:
        print(f"[{row['dataset_index']}] sim={row['similarity']}")
        print(f"  ref : {row['reference']}")
        print(f"  pred: {row['prediction']}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_index", "similarity", "reference", "gloss", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved_csv: {output_path}")

    if args.output_metrics_json:
        metrics_path = Path(args.output_metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_metrics_json: {metrics_path}")


if __name__ == "__main__":
    main()
