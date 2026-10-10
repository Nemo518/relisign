import argparse
import gzip
import pickle
from pathlib import Path

import torch

from paths import CE_CSL_ROOT, DATA_ROOT, MT5_ROOT, OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_PROJECT_ROOT = PROJECT_ROOT
DEFAULT_CECSL_ROOT = CE_CSL_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check mt5-base, Uni-Sign checkpoint, CE-CSL labels, and CE-CSL pose resources."
    )
    parser.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--cecsl_root", default=str(DEFAULT_CECSL_ROOT))
    parser.add_argument("--check_checkpoint_load", action="store_true")
    parser.add_argument("--check_pose_content", action="store_true")
    parser.add_argument("--sample_limit", type=int, default=5)
    return parser.parse_args()


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_mt5(project_root):
    mt5_dir = MT5_ROOT
    required = [
        "config.json",
        "pytorch_model.bin",
        "spiece.model",
        "tokenizer_config.json",
    ]
    print_header("MT5 CHECK")
    print(f"path: {mt5_dir}")
    missing = [name for name in required if not (mt5_dir / name).exists()]
    if missing:
        print("status: missing files")
        for item in missing:
            print(f"missing: {item}")
        return False
    print("status: ok")
    for item in required:
        file_path = mt5_dir / item
        print(f"{item}: {file_path.stat().st_size}")
    return True


def check_checkpoint(project_root, check_load):
    ckpt_path = OUTPUT_ROOT / "stage3_finetuning" / "best_checkpoint.pth"
    print_header("CHECKPOINT CHECK")
    print(f"path: {ckpt_path}")
    if not ckpt_path.exists():
        print("status: missing")
        return False
    print(f"status: exists ({ckpt_path.stat().st_size} bytes)")
    if check_load:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"type: {type(ckpt)}")
        print(f"keys: {list(ckpt.keys())[:10]}")
        print(f"has_model_key: {'model' in ckpt}")
    return True


def load_label_file(label_path):
    with gzip.open(label_path, "rb") as f:
        return pickle.load(f)


def check_cecsl_labels(project_root, sample_limit):
    label_root = DATA_ROOT / "CE_CSL"
    print_header("CE-CSL LABEL CHECK")
    print(f"path: {label_root}")
    all_ok = True
    for split in ["train", "dev", "test"]:
        label_path = label_root / f"labels.{split}"
        if not label_path.exists():
            print(f"{split}: missing")
            all_ok = False
            continue
        labels = load_label_file(label_path)
        print(f"{split}: ok ({len(labels)} samples)")
        for idx, (name, sample) in enumerate(labels.items()):
            if idx >= sample_limit:
                break
            print(
                f"  sample {idx+1}: name={name}, text={sample.get('text', '')}, "
                f"gloss_len={len(sample.get('gloss', []))}, video_path={sample.get('video_path', '')}"
            )
    return all_ok


def validate_pose_file(pose_path):
    with open(pose_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return False, "not_dict"
    if "keypoints" not in obj or "scores" not in obj:
        return False, "missing_keys"
    if len(obj["keypoints"]) == 0 or len(obj["scores"]) == 0:
        return False, "empty"
    return True, "ok"


def check_cecsl_pose(cecsl_root, check_content, sample_limit):
    pose_root = cecsl_root / "pose_format"
    print_header("CE-CSL POSE CHECK")
    print(f"path: {pose_root}")
    if not pose_root.exists():
        print("status: pose_format directory missing")
        return False

    all_ok = True
    for split in ["train", "dev", "test"]:
        split_root = pose_root / split
        if not split_root.exists():
            print(f"{split}: missing split directory")
            all_ok = False
            continue
        files = list(split_root.rglob("*.pkl"))
        print(f"{split}: {len(files)} pose files")
        for pose_path in files[:sample_limit]:
            print(f"  sample: {pose_path}")
            if check_content:
                is_ok, reason = validate_pose_file(pose_path)
                print(f"    content_check: {reason}")
                if not is_ok:
                    all_ok = False
    return all_ok


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    cecsl_root = Path(args.cecsl_root)

    ok_mt5 = check_mt5(project_root)
    ok_ckpt = check_checkpoint(project_root, args.check_checkpoint_load)
    ok_labels = check_cecsl_labels(project_root, args.sample_limit)
    ok_pose = check_cecsl_pose(cecsl_root, args.check_pose_content, args.sample_limit)

    print_header("SUMMARY")
    print(f"mt5: {'ok' if ok_mt5 else 'missing_or_incomplete'}")
    print(f"checkpoint: {'ok' if ok_ckpt else 'missing_or_incomplete'}")
    print(f"cecsl_labels: {'ok' if ok_labels else 'missing_or_incomplete'}")
    print(f"cecsl_pose: {'ok' if ok_pose else 'missing_or_incomplete'}")


if __name__ == "__main__":
    main()
