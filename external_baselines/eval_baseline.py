"""
Unified evaluation script for external baseline models on CE-CSL.

Outputs CSV files compatible with compute_cecsl_text_metrics.py
for consistent comparison with Uni-Sign results.
"""

import argparse
import csv
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

from config import dev_label_paths, test_label_paths
from datasets import S2T_Dataset

LABEL_PATHS = {
    "dev": dev_label_paths["CE_CSL"],
    "test": test_label_paths["CE_CSL"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate external baselines on CE-CSL.")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--model_type", required=True, choices=["gfslt_vlp", "unislow"])
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_csv", default="", help="Output CSV path.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--sample_count", type=int, default=0, help="0 = all samples")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def similarity_score(reference, prediction):
    return SequenceMatcher(None, reference, prediction).ratio()


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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    dataset_args = SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=False,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=0.2,
    )

    dataset = S2T_Dataset(
        path=LABEL_PATHS[args.split], args=dataset_args, phase=args.split
    )

    if args.sample_count > 0:
        indices = list(range(min(args.sample_count, len(dataset))))
        subset = Subset(dataset, indices)
    else:
        subset = dataset
        indices = list(range(len(dataset)))

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    print(f"model_type: {args.model_type}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"samples: {len(subset)}")

    # Load model based on type
    if args.model_type == "gfslt_vlp":
        from external_baselines.gfslt_vlp_model import GFSLT_VLP_Simplified

        model = GFSLT_VLP_Simplified(args=dataset_args)
    elif args.model_type == "unislow":
        from models import Uni_Sign

        model = Uni_Sign(args=dataset_args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state = model.state_dict()
    checkpoint_state = checkpoint.get("model", checkpoint)

    filtered_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered_state[key] = value

    ret = model.load_state_dict(filtered_state, strict=False)
    print(f"missing_keys: {len(ret.missing_keys)}, unexpected_keys: {len(ret.unexpected_keys)}, "
          f"skipped: {len(skipped)}")

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
            output = model.generate(stack_out, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

            for item_idx, prediction in enumerate(predictions):
                global_idx = batch_idx * args.batch_size + item_idx
                if global_idx >= len(indices):
                    continue
                dataset_index = indices[global_idx]
                reference = tgt_input["gt_sentence"][item_idx]
                gloss = tgt_input.get("gt_gloss", [""] * len(tgt_input["gt_sentence"]))[item_idx]
                if isinstance(gloss, list):
                    gloss = " ".join(gloss)
                score = similarity_score(reference, prediction)
                rows.append({
                    "dataset_index": dataset_index,
                    "similarity": f"{score:.4f}",
                    "reference": reference,
                    "gloss": gloss,
                    "prediction": prediction,
                })

            if (batch_idx + 1) % 10 == 0:
                print(f"progress: {batch_idx + 1}/{len(dataloader)}")

    avg_sim = sum(float(r["similarity"]) for r in rows) / max(1, len(rows))
    exact = sum(1 for r in rows if r["reference"] == r["prediction"])
    print(f"evaluated: {len(rows)}, avg_similarity: {avg_sim:.4f}, exact_match: {exact}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_index", "similarity", "reference", "gloss", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved_csv: {output_path}")


if __name__ == "__main__":
    main()
