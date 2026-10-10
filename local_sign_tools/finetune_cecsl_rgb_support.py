import os
import argparse
import json
import hashlib
import math
import random
import sys
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import dev_label_paths, pose_dirs, rgb_dirs, test_label_paths, train_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign
from paths import OUTPUT_ROOT, PRETRAINED_ROOT, resolve_output_dir, resolve_storage_path


LABEL_PATHS = {
    "train": train_label_paths["CE_CSL"],
    "dev": dev_label_paths["CE_CSL"],
    "test": test_label_paths["CE_CSL"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU CE-CSL RGB-support fine-tuning without deepspeed.")
    parser.add_argument("--checkpoint", default=str(OUTPUT_ROOT / "stage3_finetuning" / "best_checkpoint.pth"))
    parser.add_argument("--output_dir", default=str(OUTPUT_ROOT / "cecsl_rgb_support_local"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--stage1_epochs", type=int, default=0, help="Freeze MT5 and train RGB-heavy modules first when > 0.")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="Joint fine-tuning epochs after stage 1 when > 0.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--stage1_lr", type=float, default=3e-5, help="Base LR for RGB-heavy stage 1.")
    parser.add_argument("--stage2_lr", type=float, default=0.0, help="Base LR for stage 2. 0 means reuse --lr.")
    parser.add_argument("--stage1_proj_lr_scale", type=float, default=0.3, help="Relative LR for small projection layers during stage 1.")
    parser.add_argument("--stage2_mt5_lr_scale", type=float, default=0.1, help="Relative LR for MT5 during stage 2.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_mt5", action="store_true")
    parser.add_argument(
        "--rgb_backbone_weights",
        default=str(PRETRAINED_ROOT / "efficientnet_b0_rwightman-7f5810bc.pth"),
        help="Local EfficientNet-B0 weight path. If missing, the RGB backbone falls back to random initialization.",
    )
    parser.add_argument(
        "--rgb_backbone_pretrained",
        action="store_true",
        help="Reserved compatibility flag. The current offline workflow still avoids online download and uses local weights when available.",
    )
    parser.add_argument(
        "--train_rgb_frame_sampling",
        default="confidence_topk",
        choices=["confidence_topk", "confidence_weighted", "uniform", "random"],
        help="How RGB-support selects hand frames during training.",
    )
    parser.add_argument(
        "--eval_rgb_frame_sampling",
        default="confidence_topk",
        choices=["confidence_topk", "confidence_weighted", "uniform", "random"],
        help="How RGB-support selects hand frames during eval and decode selection.",
    )
    parser.add_argument("--train_sample_limit", type=int, default=0, help="0 means use all available pose+video samples.")
    parser.add_argument("--eval_sample_limit", type=int, default=0, help="0 means use all available pose+video samples in eval split.")
    parser.add_argument("--eval_split", default="dev", choices=["none", "dev", "test"])
    parser.add_argument("--selection_split", default="dev", choices=["none", "train", "dev", "test"])
    parser.add_argument("--selection_sample_count", type=int, default=20)
    parser.add_argument("--selection_sample_mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--selection_offset", type=int, default=0)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--selection_batch_size", type=int, default=1)
    parser.add_argument("--selection_max_new_tokens", type=int, default=100)
    parser.add_argument("--selection_num_beams", type=int, default=4)
    parser.add_argument("--rgb_support", dest="rgb_support", action="store_true", help="Enable hand RGB support.")
    parser.add_argument("--no_rgb_support", dest="rgb_support", action="store_false", help="Disable hand RGB support.")
    parser.set_defaults(rgb_support=True)
    parser.add_argument("--fusion_use_uncertainty_gate", action="store_true", help="Enable reliability-aware metadata in RGB-pose fusion.")
    parser.add_argument("--fusion_meta_dim", type=int, default=3, help="Metadata dimension used by the uncertainty-aware fusion gate.")
    parser.add_argument("--top_k_ratio", type=float, default=0.1, help="Fraction of eligible hand frames retained by all RGB sampling modes.")
    parser.add_argument(
        "--experiment_variant",
        default="auto",
        choices=["auto", "pose_only", "rgb_gate", "rgb_metadata_gate"],
        help="Auditable variant label. 'rgb_gate' still uses the learned pose-preserving gate but excludes reliability metadata.",
    )
    parser.add_argument(
        "--allow_test_during_training",
        action="store_true",
        help="Explicit override for diagnostic runs only. Test data must not drive checkpoint selection in reported experiments.",
    )
    parser.add_argument(
        "--save_epoch_checkpoints",
        action="store_true",
        help="Save checkpoint_epoch_*.pth after every epoch. Disabled by default to reduce disk usage.",
    )
    return parser.parse_args()


def build_dataset_args(args, rgb_frame_sampling):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=args.rgb_support,
        rgb_backbone_weights=args.rgb_backbone_weights,
        rgb_backbone_pretrained=args.rgb_backbone_pretrained,
        rgb_frame_sampling=rgb_frame_sampling,
        fusion_use_uncertainty_gate=args.fusion_use_uncertainty_gate,
        fusion_meta_dim=args.fusion_meta_dim,
        top_k_ratio=args.top_k_ratio,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=args.label_smoothing,
    )


def discover_available_indices(dataset, sample_limit, require_rgb):
    available_indices = []
    for idx, key in enumerate(dataset.list):
        sample = dataset.raw_data[key]
        pose_path = Path(dataset.pose_dir) / sample["video_path"].replace(".mp4", ".pkl")
        rgb_path = Path(dataset.rgb_dir) / sample["video_path"]
        if pose_path.exists() and (not require_rgb or rgb_path.exists()):
            available_indices.append(idx)
        if sample_limit > 0 and len(available_indices) >= sample_limit:
            break
    return available_indices


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


def resolve_experiment_variant(args):
    inferred = "pose_only"
    if args.rgb_support:
        inferred = "rgb_metadata_gate" if args.fusion_use_uncertainty_gate else "rgb_gate"
    if args.experiment_variant != "auto" and args.experiment_variant != inferred:
        raise ValueError(
            f"experiment_variant={args.experiment_variant!r} conflicts with model flags; inferred {inferred!r}."
        )
    args.experiment_variant = inferred


def validate_protocol_args(args):
    if not 0 < args.top_k_ratio <= 1:
        raise ValueError(f"--top_k_ratio must be in (0, 1], got {args.top_k_ratio}")
    if args.eval_split == "test" and not args.allow_test_during_training:
        raise ValueError(
            "--eval_split test is blocked during training because eval_loss can affect checkpoint selection. "
            "Use --eval_split dev, then run script/eval_cecsl_samples.py once on the fixed test set."
        )
    if args.selection_split == "test" and not args.allow_test_during_training:
        raise ValueError(
            "--selection_split test is blocked because decoded similarity directly selects best_checkpoint.pth."
        )
    resolve_experiment_variant(args)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_protocol_manifest(output_dir, args, stage_plan, train_indices, eval_indices, selection_indices):
    label_files = {}
    for split, path in LABEL_PATHS.items():
        resolved = Path(path)
        label_files[split] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    manifest = {
        "schema_version": 1,
        "experiment_variant": args.experiment_variant,
        "arguments": vars(args),
        "stage_plan": stage_plan,
        "label_files": label_files,
        "train_indices": list(train_indices),
        "eval_split": args.eval_split,
        "eval_indices": list(eval_indices),
        "selection_split": args.selection_split,
        "selection_indices": list(selection_indices),
        "checkpoint_selection_rule": "selection_average_similarity_then_exact_match_then_eval_loss",
        "test_data_used_during_training": args.eval_split == "test" or args.selection_split == "test",
    }
    manifest_path = Path(output_dir) / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"protocol_manifest: {manifest_path}")


def seed_worker_with_base(base_seed, worker_id):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loader(args, split, sample_limit, *, batch_size=None, num_workers=None, shuffle=None, subset_indices=None, rgb_frame_sampling=None):
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if split == "train" else 1000))

    if batch_size is None:
        batch_size = args.batch_size
    if num_workers is None:
        num_workers = args.num_workers
    if shuffle is None:
        shuffle = split == "train"
    if rgb_frame_sampling is None:
        rgb_frame_sampling = args.train_rgb_frame_sampling if split == "train" else args.eval_rgb_frame_sampling

    dataset_args = build_dataset_args(args, rgb_frame_sampling=rgb_frame_sampling)
    dataset = S2T_Dataset(path=LABEL_PATHS[split], args=dataset_args, phase=split)
    available_indices = discover_available_indices(dataset, sample_limit, args.rgb_support)
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


def evaluate_loss(model, loader, device, dtype_mode):
    if loader is None:
        return None

    use_autocast = dtype_mode == "bf16" and device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for src_input, tgt_input in loader:
            src_input = move_src_input(src_input, device, dtype_mode)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stack_out = model(src_input, tgt_input)
                    loss = stack_out["loss"]
            else:
                stack_out = model(src_input, tgt_input)
                loss = stack_out["loss"]
            total_loss += float(loss.item())
            total_batches += 1
    if was_training:
        model.train()
    if total_batches == 0:
        return None
    return total_loss / total_batches


def evaluate_decode(model, loader, dataset_indices, device, dtype_mode, max_new_tokens, num_beams):
    if loader is None:
        return None

    use_autocast = dtype_mode == "bf16" and device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    was_training = model.training
    model.eval()
    tokenizer = model.mt5_tokenizer
    rows = []
    cursor = 0

    with torch.no_grad():
        for src_input, tgt_input in loader:
            src_input = move_src_input(src_input, device, dtype_mode)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stack_out = model(src_input, tgt_input)
                    output = model.generate(
                        stack_out,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
            else:
                stack_out = model(src_input, tgt_input)
                output = model.generate(
                    stack_out,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

            for item_idx, prediction in enumerate(predictions):
                dataset_index = dataset_indices[cursor]
                reference = tgt_input["gt_sentence"][item_idx]
                score = similarity_score(reference, prediction)
                rows.append(
                    {
                        "dataset_index": dataset_index,
                        "reference": reference,
                        "prediction": prediction,
                        "similarity": score,
                    }
                )
                cursor += 1

    if was_training:
        model.train()

    if not rows:
        return None

    average_similarity = sum(row["similarity"] for row in rows) / len(rows)
    exact_match_count = sum(1 for row in rows if row["reference"] == row["prediction"])
    return {
        "average_similarity": average_similarity,
        "exact_match_count": exact_match_count,
        "evaluated_samples": len(rows),
        "rows": rows,
    }


def save_checkpoint(path, model, args, extra=None):
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        try:
            torch.save(payload, tmp_path)
        except RuntimeError:
            # Some mounted filesystems are flaky with the newer zip-based format.
            torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_checkpoint_flexible(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    checkpoint_state = checkpoint["model"]

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
    return checkpoint, ret, skipped_keys


def build_stage_plan(args):
    if args.stage1_epochs > 0 or args.stage2_epochs > 0:
        stages = []
        if args.stage1_epochs > 0:
            stages.append({"name": "stage1_rgb_focus", "epochs": args.stage1_epochs})
        if args.stage2_epochs > 0:
            stages.append({"name": "stage2_joint", "epochs": args.stage2_epochs})
        return stages
    return [{"name": "single_stage", "epochs": args.epochs}]


def classify_param(name):
    if name.startswith("mt5_model."):
        return "mt5"
    if name.startswith(("rgb_support_backbone.", "rgb_proj.", "fusion_pose_rgb_DA.", "fusion_gate.", "fusion_pose_rgb_linear.", "fusion_meta_proj.")):
        return "rgb"
    if name == "part_para" or name.startswith("pose_proj."):
        return "projection"
    return "other"


def configure_trainable_params(model, stage_name, freeze_mt5):
    if stage_name == "stage1_rgb_focus":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if classify_param(name) in {"rgb", "projection"}:
                param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = True
    if freeze_mt5 and stage_name == "single_stage":
        for param in model.mt5_model.parameters():
            param.requires_grad = False


def build_optimizer(model, args, stage_name):
    if stage_name == "stage1_rgb_focus":
        lr_by_group = {
            "rgb": args.stage1_lr,
            "projection": args.stage1_lr * args.stage1_proj_lr_scale,
        }
    elif stage_name == "single_stage":
        lr_by_group = {
            "rgb": args.lr,
            "projection": args.lr,
            "other": args.lr,
            "mt5": 0.0 if args.freeze_mt5 else args.lr,
        }
    else:
        stage2_lr = args.stage2_lr if args.stage2_lr > 0 else args.lr
        lr_by_group = {
            "rgb": stage2_lr,
            "projection": stage2_lr,
            "other": stage2_lr,
            "mt5": 0.0 if (stage_name == "single_stage" and args.freeze_mt5) else stage2_lr * args.stage2_mt5_lr_scale,
        }

    grouped_params = {key: [] for key in lr_by_group}
    summaries = []
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = classify_param(name)
        if group_name not in lr_by_group:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        grouped_params[group_name].append(param)

    optimizer_groups = []
    for group_name, params in grouped_params.items():
        if not params:
            continue
        optimizer_groups.append(
            {
                "params": params,
                "lr": lr_by_group[group_name],
                "weight_decay": args.weight_decay,
            }
        )
        summaries.append(
            {
                "name": group_name,
                "lr": lr_by_group[group_name],
                "param_tensors": len(params),
                "param_count": sum(param.numel() for param in params),
            }
        )

    if not optimizer_groups:
        raise RuntimeError(f"No trainable parameters were assigned for stage: {stage_name}")

    return torch.optim.AdamW(optimizer_groups), summaries


def is_better_checkpoint(current_metrics, best_metrics):
    if best_metrics is None:
        return True

    current_decode = current_metrics.get("selection_average_similarity")
    best_decode = best_metrics.get("selection_average_similarity")
    if current_decode is not None or best_decode is not None:
        current_decode = -1.0 if current_decode is None else current_decode
        best_decode = -1.0 if best_decode is None else best_decode
        if current_decode > best_decode + 1e-8:
            return True
        if current_decode < best_decode - 1e-8:
            return False

        current_exact = current_metrics.get("selection_exact_match_count", -1)
        best_exact = best_metrics.get("selection_exact_match_count", -1)
        if current_exact > best_exact:
            return True
        if current_exact < best_exact:
            return False

    current_loss = math.inf if current_metrics.get("eval_loss") is None else current_metrics["eval_loss"]
    best_loss = math.inf if best_metrics.get("eval_loss") is None else best_metrics["eval_loss"]
    return current_loss < best_loss - 1e-8


def main():
    args = parse_args()
    validate_protocol_args(args)
    args.checkpoint = resolve_storage_path(args.checkpoint)
    args.rgb_backbone_weights = resolve_storage_path(args.rgb_backbone_weights)
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

    train_dataset, train_available_indices, train_indices, train_loader = build_loader(
        args,
        "train",
        args.train_sample_limit,
        shuffle=True,
        rgb_frame_sampling=args.train_rgb_frame_sampling,
    )
    print(f"train_dataset_size: {len(train_dataset)}")
    print(f"train_available_samples: {len(train_available_indices)}")
    print(f"train_pose_dir: {pose_dirs['CE_CSL']}")
    print(f"train_rgb_dir: {rgb_dirs['CE_CSL']}")
    print(f"rgb_support: {args.rgb_support}")
    print(f"experiment_variant: {args.experiment_variant}")
    print(f"train_rgb_frame_sampling: {args.train_rgb_frame_sampling}")
    print(f"top_k_ratio: {args.top_k_ratio}")

    if not train_available_indices:
        missing_reason = "pose+video" if args.rgb_support else "pose"
        raise RuntimeError(f"No CE-CSL train samples with {missing_reason} files are available.")

    eval_loader = None
    eval_active_indices = []
    if args.eval_split != "none":
        _, eval_available_indices, eval_active_indices, eval_loader = build_loader(
            args,
            args.eval_split,
            args.eval_sample_limit,
            shuffle=False,
            rgb_frame_sampling=args.eval_rgb_frame_sampling,
        )
        print(f"{args.eval_split}_available_samples: {len(eval_available_indices)}")
        if not eval_available_indices:
            missing_reason = "pose+video" if args.rgb_support else "pose"
            print(f"warning: {args.eval_split} split has no {missing_reason} samples yet, evaluation will be skipped.")
            eval_loader = None

    selection_loader = None
    selection_dataset_indices = []
    if args.selection_split != "none" and args.selection_sample_count > 0:
        _, selection_available_indices, _, _ = build_loader(
            args,
            args.selection_split,
            0,
            shuffle=False,
            batch_size=args.selection_batch_size,
            num_workers=0,
            rgb_frame_sampling=args.eval_rgb_frame_sampling,
        )
        chosen_positions = choose_indices(
            len(selection_available_indices),
            args.selection_sample_count,
            args.selection_sample_mode,
            args.selection_offset,
            args.selection_seed,
        )
        selection_dataset_indices = [selection_available_indices[position] for position in chosen_positions]
        _, _, _, selection_loader = build_loader(
            args,
            args.selection_split,
            0,
            batch_size=args.selection_batch_size,
            num_workers=0,
            shuffle=False,
            subset_indices=selection_dataset_indices,
            rgb_frame_sampling=args.eval_rgb_frame_sampling,
        )
        print(f"selection_split: {args.selection_split}")
        print(f"selection_rgb_frame_sampling: {args.eval_rgb_frame_sampling}")
        print(f"selection_indices: {selection_dataset_indices}")

    model_args = build_dataset_args(args, rgb_frame_sampling=args.eval_rgb_frame_sampling)
    model = Uni_Sign(args=model_args)
    checkpoint, ret, skipped_keys = load_checkpoint_flexible(model, args.checkpoint)
    print(f"missing_keys: {len(ret.missing_keys)}")
    print(f"unexpected_keys: {len(ret.unexpected_keys)}")
    print(f"skipped_shape_or_missing_keys: {len(skipped_keys)}")
    if ret.missing_keys:
        print("missing_keys_sample:")
        for item in ret.missing_keys[:10]:
            print(f"  {item}")
    if skipped_keys:
        print("skipped_keys_sample:")
        for item in skipped_keys[:10]:
            print(f"  {item}")
    if ret.unexpected_keys:
        print("unexpected_keys_sample:")
        for item in ret.unexpected_keys[:10]:
            print(f"  {item}")

    model = model.to(device)
    model.train()

    use_autocast = args.dtype == "bf16" and device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not use_autocast:
        print("bf16 is not available on this device, falling back to fp32 training.")

    stage_plan = build_stage_plan(args)
    write_protocol_manifest(
        output_dir,
        args,
        stage_plan,
        train_indices,
        eval_active_indices,
        selection_dataset_indices,
    )
    total_epochs = sum(stage["epochs"] for stage in stage_plan)
    history_path = output_dir / "training_metrics.jsonl"
    if history_path.exists():
        history_path.unlink()

    best_eval = math.inf
    best_metrics = None
    global_step = 0
    global_epoch = 0

    for stage in stage_plan:
        configure_trainable_params(model, stage["name"], args.freeze_mt5)
        optimizer, optimizer_summaries = build_optimizer(model, args, stage["name"])
        print(f"stage_start: {stage['name']} | epochs={stage['epochs']}")
        for summary in optimizer_summaries:
            print(
                f"optimizer_group[{summary['name']}]: lr={summary['lr']:.2e} "
                f"| tensors={summary['param_tensors']} | params={summary['param_count']}"
            )

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

                loss_to_backprop = loss / args.gradient_accumulation_steps
                loss_to_backprop.backward()

                if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running_loss += float(loss.item())
                batch_count += 1

                if batch_idx % max(1, args.gradient_accumulation_steps) == 0 or batch_idx == len(train_loader):
                    print(
                        f"stage={stage['name']} | epoch {global_epoch}/{total_epochs} | "
                        f"stage_epoch {stage_epoch}/{stage['epochs']} | batch {batch_idx}/{len(train_loader)} | "
                        f"loss={loss.item():.4f} | step={global_step}"
                    )

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
            }

            print(
                f"epoch {global_epoch} finished | stage={stage['name']} | train_loss={avg_train_loss:.4f}"
                + (f" | {args.eval_split}_loss={eval_loss:.4f}" if eval_loss is not None else "")
                + (
                    f" | {args.selection_split}_decode_avg={decode_metrics['average_similarity']:.4f}"
                    f" | {args.selection_split}_decode_exact={decode_metrics['exact_match_count']}"
                    if decode_metrics is not None
                    else ""
                )
            )

            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            checkpoint_extra = dict(metrics)
            if decode_metrics is not None:
                checkpoint_extra["selection_rows"] = decode_metrics["rows"]

            if args.save_epoch_checkpoints:
                checkpoint_path = output_dir / f"checkpoint_epoch_{global_epoch}.pth"
                save_checkpoint(checkpoint_path, model, args, extra=checkpoint_extra)

            if eval_loss is not None and eval_loss < best_eval:
                best_eval = eval_loss
                save_checkpoint(output_dir / "best_loss_checkpoint.pth", model, args, extra=checkpoint_extra)

            if is_better_checkpoint(metrics, best_metrics):
                best_metrics = dict(metrics)
                save_checkpoint(output_dir / "best_checkpoint.pth", model, args, extra=checkpoint_extra)

    save_checkpoint(output_dir / "final_checkpoint.pth", model, args, extra={"epoch": global_epoch, "stage": stage_plan[-1]["name"]})
    print(f"fine-tuning finished, checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
