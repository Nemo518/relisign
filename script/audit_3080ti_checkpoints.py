# -*- coding: utf-8 -*-
r"""盘点 3080 Ti 上的 checkpoint，并输出可复用性证据。

用法（在 3080 Ti 机器上）:
    python script/audit_3080ti_checkpoints.py "E:\project\xiaozhi\Uni-Sign-storage\out"

不传参数则使用下方 DEFAULT_OUT；请按 3080 Ti 上的实际路径改。
"""
import argparse
import json
from pathlib import Path

import torch

DEFAULT_OUT = r"E:\project\xiaozhi\Uni-Sign-storage\out"


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_checkpoint(path, root):
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("model", {})
    checkpoint_args = checkpoint.get("args", {}) or {}
    has_rgb = any(key.startswith("rgb_support_backbone.") for key in state)
    has_metadata = any(key.startswith("fusion_meta_proj.") for key in state)
    variant = "pose_only" if not has_rgb else ("rgb_metadata_gate" if has_metadata else "rgb_gate")
    protocol_fields = [
        "seed", "stage1_epochs", "stage2_epochs", "eval_split", "selection_split",
        "top_k_ratio", "experiment_variant", "train_sample_limit",
    ]
    protocol = {field: checkpoint_args.get(field) for field in protocol_fields}
    manifest_path = path.parent / "protocol_manifest.json"
    return {
        "path": str(path.relative_to(root)),
        "size_gb": round(path.stat().st_size / (1024 ** 3), 3),
        "inferred_variant": variant,
        "protocol": protocol,
        "has_protocol_manifest": manifest_path.is_file(),
        "has_args": "args" in checkpoint,
        "epoch": checkpoint.get("epoch"),
        "stage": checkpoint.get("stage"),
        "eligible_for_new_protocol_main_result": manifest_path.is_file(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=DEFAULT_OUT)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"[错误] 目录不存在: {root}")
        print("请把 DEFAULT_OUT 改成 3080Ti 上的实际 out 目录，或作为第一个参数传入。")
        raise SystemExit(1)

    print(f"扫描: {root}\n")
    checkpoint_paths = sorted(root.rglob("*.pth"))
    records = []
    for path in checkpoint_paths:
        try:
            record = inspect_checkpoint(path, root)
        except Exception as exc:
            record = {"path": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    summary = {
        "root": str(root.resolve()),
        "checkpoint_count": len(checkpoint_paths),
        "eligible_for_new_protocol_main_result": sum(
            bool(record.get("eligible_for_new_protocol_main_result")) for record in records
        ),
        "records": records,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
