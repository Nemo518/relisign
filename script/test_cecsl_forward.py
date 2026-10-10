import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import test_label_paths, train_label_paths, dev_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign
from paths import OUTPUT_ROOT, resolve_storage_path


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CE-CSL forward/generation test for Uni-Sign.")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--checkpoint",
        default=str(OUTPUT_ROOT / "stage3_finetuning" / "best_checkpoint.pth"),
        help="Path to Uni-Sign checkpoint.",
    )
    parser.add_argument("--rgb_support", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    return parser.parse_args()


def build_dataset_args(args):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=args.rgb_support,
        max_length=args.max_length,
        hidden_dim=256,
        label_smoothing=0.2,
    )


def get_label_path(split):
    if split == "train":
        return train_label_paths["CE_CSL"]
    if split == "dev":
        return dev_label_paths["CE_CSL"]
    return test_label_paths["CE_CSL"]


def move_src_input(src_input, device, dtype_mode):
    tensor_dtype = torch.float32
    if dtype_mode == "bf16" and device.type == "cuda":
        tensor_dtype = torch.bfloat16

    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if tensor_dtype is not None and value.dtype.is_floating_point:
                value = value.to(tensor_dtype)
            src_input[key] = value
    return src_input


def main():
    args = parse_args()
    args.checkpoint = resolve_storage_path(args.checkpoint)
    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"rgb_support: {args.rgb_support}")

    dataset_args = build_dataset_args(args)
    label_path = get_label_path(args.split)
    dataset = S2T_Dataset(path=label_path, args=dataset_args, phase=args.split)
    print(f"dataset_size: {len(dataset)}")

    subset = [dataset[args.sample_index]]
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )
    src_input, tgt_input = next(iter(dataloader))

    model = Uni_Sign(args=dataset_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ret = model.load_state_dict(checkpoint["model"], strict=True)
    print(f"missing_keys: {len(ret.missing_keys)}")
    print(f"unexpected_keys: {len(ret.unexpected_keys)}")

    model = model.to(device)
    model.eval()
    if args.dtype == "bf16" and device.type == "cuda":
        model = model.to(torch.bfloat16)

    src_input = move_src_input(src_input, device, args.dtype)

    with torch.no_grad():
        stack_out = model(src_input, tgt_input)
        output = model.generate(
            stack_out,
            max_new_tokens=100,
            num_beams=4,
        )

    tokenizer = model.mt5_tokenizer
    pred_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    print("ground_truth_text:")
    for item in tgt_input["gt_sentence"]:
        print(item)

    print("ground_truth_gloss:")
    for item in tgt_input["gt_gloss"]:
        print(item)

    print("prediction:")
    for item in pred_text:
        print(item)


if __name__ == "__main__":
    main()
