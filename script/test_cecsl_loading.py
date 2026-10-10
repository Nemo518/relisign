import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CE-CSL dataset loading test for Uni-Sign.")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--rgb_support", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--print_batch_only", action="store_true")
    return parser.parse_args()


def build_args(args):
    return SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=args.rgb_support,
        max_length=args.max_length,
    )


def get_label_path(split):
    if split == "train":
        return train_label_paths["CE_CSL"]
    if split == "dev":
        return dev_label_paths["CE_CSL"]
    return test_label_paths["CE_CSL"]


def print_pose_info(pose_sample):
    print("pose parts:")
    for key, value in pose_sample.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")


def print_support_info(support_rgb_dict):
    if not support_rgb_dict:
        print("rgb support: disabled or empty")
        return
    print("rgb support:")
    for key, value in support_rgb_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


def main():
    args = parse_args()
    dataset_args = build_args(args)
    label_path = get_label_path(args.split)

    print(f"split: {args.split}")
    print(f"label_path: {label_path}")
    print(f"rgb_support: {args.rgb_support}")

    dataset = S2T_Dataset(path=label_path, args=dataset_args, phase=args.split)
    print(f"dataset_size: {len(dataset)}")

    name_sample, pose_sample, text, gloss, support_rgb_dict = dataset[args.sample_index]
    print("single sample:")
    print(f"  name: {name_sample}")
    print(f"  text: {text}")
    print(f"  gloss: {gloss}")
    print_pose_info(pose_sample)
    print_support_info(support_rgb_dict)

    if args.print_batch_only:
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    src_input, tgt_input = next(iter(dataloader))
    print("batch summary:")
    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            print(f"  src[{key}]: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  src[{key}]: {value}")

    print(f"  tgt[gt_sentence]: {tgt_input['gt_sentence'][:2]}")
    print(f"  tgt[gt_gloss]: {tgt_input['gt_gloss'][:2]}")


if __name__ == "__main__":
    main()
