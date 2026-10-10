"""
Gloss-based baseline pipelines for CE-CSL.

Two variants:
1. Oracle Gloss-to-Text: Train mT5 on (ground-truth gloss → Chinese text)
   Provides the upper bound for text generation quality.

2. Two-Stage Pipeline (S2G2T):
   Stage 1: CSLR model predicts glosses from pose
   Stage 2: mT5 translates predicted glosses to Chinese text
"""

import argparse
import csv
import gzip
import json
import pickle
import random
import sys
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    get_scheduler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import mt5_path, train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset
from paths import resolve_output_dir, resolve_storage_path


def load_label_dict(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def extract_text_and_gloss(label_path: str):
    raw = load_label_dict(label_path)
    pairs = []
    for key, sample in raw.items():
        gloss = sample.get("gloss", [])
        text = sample.get("text", "")
        if isinstance(gloss, list):
            gloss_str = " ".join(gloss)
        else:
            gloss_str = str(gloss)
        if gloss_str.strip() and text.strip():
            pairs.append((gloss_str, text))
    return pairs


# ---------------------------------------------------------------------------
# Oracle Gloss-to-Text Training
# ---------------------------------------------------------------------------

def train_gloss_to_text(args):
    import time as time_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"Loading mT5 from {mt5_path}...", flush=True)

    tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
    model = model.to(device)
    print(f"Model loaded on {device}.", flush=True)

    train_pairs = extract_text_and_gloss(resolve_storage_path(train_label_paths["CE_CSL"]))
    dev_pairs = extract_text_and_gloss(resolve_storage_path(dev_label_paths["CE_CSL"]))
    print(f"train pairs: {len(train_pairs)}, dev: {len(dev_pairs)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * (len(train_pairs) // args.batch_size)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_dev_loss = float("inf")
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_pairs)
        total_loss = 0.0
        n_batches = 0
        skipped = 0
        epoch_start = time_mod.time()

        for i in range(0, len(train_pairs), args.batch_size):
            try:
                batch_pairs = train_pairs[i : i + args.batch_size]
                glosses = [p[0] for p in batch_pairs]
                texts = [p[1] for p in batch_pairs]

                if not any(g.strip() for g in glosses) or not any(t.strip() for t in texts):
                    skipped += 1
                    continue

                src_tokens = tokenizer(
                    glosses, return_tensors="pt", padding=True, truncation=True, max_length=128
                ).to(device)
                tgt_tokens = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=128
                ).to(device)

                labels = tgt_tokens["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100

                out = model(**src_tokens, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()
                n_batches += 1

                if n_batches % 200 == 0:
                    mem_gb = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0
                    elapsed = time_mod.time() - epoch_start
                    progress = i / len(train_pairs) * 100
                    print(f"  epoch {epoch} [{i}/{len(train_pairs)} ({progress:.0f}%)] "
                          f"loss={loss.item():.2f} mem={mem_gb:.2f}GB elapsed={elapsed:.0f}s",
                          flush=True)

            except RuntimeError as e:
                print(f"  WARNING batch {i} RuntimeError: {e}", flush=True)
                torch.cuda.empty_cache()
                skipped += 1
                optimizer.zero_grad()
                continue

        avg_loss = total_loss / max(n_batches, 1)
        epoch_time = time_mod.time() - epoch_start
        print(f"Epoch {epoch} DONE: loss={avg_loss:.4f} batches={n_batches} "
              f"skipped={skipped} time={epoch_time:.0f}s", flush=True)

        # Save per-epoch checkpoint
        if output_dir:
            ckpt_dir = output_dir / f"checkpoint_epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"  Saved: {ckpt_dir}", flush=True)

        # Dev evaluation
        model.eval()
        dev_loss = 0.0
        dev_batches = 0
        with torch.no_grad():
            for i in range(0, min(len(dev_pairs), 200), args.batch_size):
                try:
                    batch_pairs = dev_pairs[i : i + args.batch_size]
                    glosses = [p[0] for p in batch_pairs]
                    texts = [p[1] for p in batch_pairs]
                    if not any(g.strip() for g in glosses):
                        continue
                    src_tokens = tokenizer(
                        glosses, return_tensors="pt", padding=True, truncation=True, max_length=128
                    ).to(device)
                    tgt_tokens = tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
                    ).to(device)
                    labels = tgt_tokens["input_ids"].clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    out = model(**src_tokens, labels=labels)
                    dev_loss += out.loss.item()
                    dev_batches += 1
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue

        dev_loss_avg = dev_loss / max(dev_batches, 1) if dev_batches > 0 else float("inf")
        print(f"Epoch {epoch}: dev_loss={dev_loss_avg:.4f} (on {dev_batches} batches)", flush=True)

        if dev_loss_avg < best_dev_loss and output_dir:
            best_dev_loss = dev_loss_avg
            best_dir = output_dir / "best_model"
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            print(f"  >>> Saved best model: dev_loss={best_dev_loss:.4f}", flush=True)

    # Final model
    if output_dir:
        final_dir = output_dir / "final_model"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Training complete. Final model saved to {final_dir}", flush=True)


# ---------------------------------------------------------------------------
# Generate: gloss → text (shared by oracle and two-stage)
# ---------------------------------------------------------------------------

def generate_gloss_to_text(model_dir: str, input_csv: str, output_csv: str, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading gloss→text model from {model_dir}...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    with open(input_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            glosses_list = [row.get("gloss_prediction", row.get("gloss", "")) for row in batch_rows]

            if not any(g.strip() for g in glosses_list):
                for j, row in enumerate(batch_rows):
                    results.append({**row, "prediction": "", "similarity": "0.0000"})
                continue

            try:
                src_tokens = tokenizer(
                    glosses_list, return_tensors="pt", padding=True, truncation=True, max_length=128
                ).to(device)
                outputs = model.generate(**src_tokens, max_new_tokens=100, num_beams=4)
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for j, (row, pred) in enumerate(zip(batch_rows, predictions)):
                    ref = row.get("reference", "")
                    sim = SequenceMatcher(None, ref, pred).ratio()
                    results.append({**row, "prediction": pred, "similarity": f"{sim:.4f}"})
            except RuntimeError as e:
                print(f"  WARNING batch {i} error: {e}", flush=True)
                torch.cuda.empty_cache()
                for j, row in enumerate(batch_rows):
                    results.append({**row, "prediction": "", "similarity": "0.0000"})

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset_index", "similarity", "reference", "gloss", "prediction"]
        writer = csv.DictWriter(f, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    avg_sim = sum(float(r.get("similarity", 0)) for r in results) / max(len(results), 1)
    print(f"Generated {len(results)} predictions, avg_sim={avg_sim:.4f}", flush=True)
    print(f"Saved: {output_csv}", flush=True)


# ---------------------------------------------------------------------------
# CSLR gloss predictions (Stage 1 of two-stage pipeline)
# ---------------------------------------------------------------------------

def generate_cslr_predictions(checkpoint_path: str, split: str, output_csv: str):
    from types import SimpleNamespace
    from models import Uni_Sign

    label_paths_map = {"dev": dev_label_paths["CE_CSL"], "test": test_label_paths["CE_CSL"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_args = SimpleNamespace(
        dataset="CE_CSL", rgb_support=False, max_length=256,
        hidden_dim=256, label_smoothing=0.2,
    )
    dataset = S2T_Dataset(
        path=resolve_storage_path(label_paths_map[split]),
        args=dataset_args, phase=split,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0,
        collate_fn=dataset.collate_fn, shuffle=False,
    )

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model = Uni_Sign(args=dataset_args)
    ckpt = torch.load(resolve_storage_path(checkpoint_path), map_location="cpu")
    ckpt_state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for idx, (src_input, tgt_input) in enumerate(dataloader):
            for key in list(src_input.keys()):
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(device, dtype=torch.float32)

            cslr_tgt = {"gt_sentence": tgt_input["gt_gloss"]}
            stack_out = model(src_input, cslr_tgt)
            output = model.generate(stack_out, max_new_tokens=100, num_beams=4)
            tokenizer = model.mt5_tokenizer
            pred_gloss = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

            ref_text = tgt_input["gt_sentence"][0] if tgt_input["gt_sentence"] else ""
            ref_gloss = tgt_input["gt_gloss"][0] if tgt_input["gt_gloss"] else ""
            if isinstance(ref_gloss, list):
                ref_gloss = " ".join(ref_gloss)

            rows.append({
                "dataset_index": idx,
                "reference": ref_text,
                "gloss": ref_gloss,
                "gloss_prediction": pred_gloss,
                "prediction": "",
                "similarity": "0.0000",
            })

            if (idx + 1) % 20 == 0:
                print(f"  CSLR progress: {idx + 1}/{len(dataset)}", flush=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset_index", "reference", "gloss", "gloss_prediction", "prediction", "similarity"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSLR predictions saved: {output_csv}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Gloss-based baselines for CE-CSL")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train_oracle")
    p_train.add_argument("--output_dir", type=str, default="out/gloss_oracle_model")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=5e-5)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--model_dir", required=True)
    p_gen.add_argument("--input_csv", required=True)
    p_gen.add_argument("--output_csv", required=True)

    p_cslr = sub.add_parser("generate_cslr")
    p_cslr.add_argument("--checkpoint", required=True)
    p_cslr.add_argument("--split", default="test", choices=["dev", "test"])
    p_cslr.add_argument("--output_csv", required=True)

    args = parser.parse_args()

    if args.cmd == "train_oracle":
        if args.output_dir:
            args.output_dir = resolve_output_dir(args.output_dir)
        train_gloss_to_text(args)
    elif args.cmd == "generate":
        generate_gloss_to_text(args.model_dir, args.input_csv, args.output_csv)
    elif args.cmd == "generate_cslr":
        generate_cslr_predictions(args.checkpoint, args.split, args.output_csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
