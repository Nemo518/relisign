import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import pose_dirs, train_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CE-CSL pose-only training smoke test.")
    parser.add_argument("--sample_limit", type=int, default=4, help="How many train samples with existing pose files to use.")
    parser.add_argument("--max_steps", type=int, default=2, help="How many optimizer steps to run.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--checkpoint", default="./out/stage3_finetuning/best_checkpoint.pth", help="Path to Uni-Sign checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_mt5", action="store_true", help="Freeze the MT5 backbone to lower memory usage.")
    return parser.parse_args()


def build_dataset_args(args):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=False,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=0.2,
    )


def discover_available_indices(dataset, sample_limit):
    available_indices = []
    for idx, key in enumerate(dataset.list):
        sample = dataset.raw_data[key]
        pose_path = Path(dataset.pose_dir) / sample["video_path"].replace(".mp4", ".pkl")
        if pose_path.exists():
            available_indices.append(idx)
        if len(available_indices) >= sample_limit:
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    dataset_args = build_dataset_args(args)
    dataset = S2T_Dataset(path=train_label_paths["CE_CSL"], args=dataset_args, phase="train")
    available_indices = discover_available_indices(dataset, args.sample_limit)

    print(f"train_label_path: {train_label_paths['CE_CSL']}")
    print(f"pose_dir: {pose_dirs['CE_CSL']}")
    print(f"dataset_size: {len(dataset)}")
    print(f"available_subset_size: {len(available_indices)}")

    if not available_indices:
        raise RuntimeError("No CE-CSL train samples with existing pose files were found. Generate pose first.")

    subset = Subset(dataset, available_indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    )

    model = Uni_Sign(args=dataset_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ret = model.load_state_dict(checkpoint["model"], strict=True)
    print(f"missing_keys: {len(ret.missing_keys)}")
    print(f"unexpected_keys: {len(ret.unexpected_keys)}")

    if args.freeze_mt5:
        for param in model.mt5_model.parameters():
            param.requires_grad = False
        print("mt5 backbone frozen for lower memory usage")

    model = model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    use_autocast = args.dtype == "bf16" and device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not use_autocast:
        print("bf16 is not available on this device, falling back to fp32 training.")

    steps_ran = 0
    for src_input, tgt_input in dataloader:
        src_input = move_src_input(src_input, device, args.dtype)
        optimizer.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                stack_out = model(src_input, tgt_input)
                loss = stack_out["loss"]
        else:
            stack_out = model(src_input, tgt_input)
            loss = stack_out["loss"]

        loss.backward()
        optimizer.step()

        steps_ran += 1
        sample_names = src_input["name_batch"]
        print(f"step {steps_ran}/{args.max_steps} | loss={loss.item():.4f} | samples={sample_names}")

        if steps_ran >= args.max_steps:
            break

    print("smoke training finished successfully")


if __name__ == "__main__":
    main()
