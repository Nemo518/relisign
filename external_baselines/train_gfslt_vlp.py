"""
Training script for GFSLT-VLP Simplified baseline on CE-CSL.

Adapts the fine_tuning.py pipeline to train the GFSLT-VLP model.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset
from gfslt_vlp_model import GFSLT_VLP_Simplified
from SLRT_metrics import translation_performance
from transformers import get_scheduler
from paths import resolve_output_dir

import utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser("GFSLT-VLP Simplified training on CE-CSL")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16")
    return parser


def build_args():
    parser = get_args_parser()
    args = parser.parse_args()
    args.dataset = "CE_CSL"
    args.task = "SLT"
    args.rgb_support = False
    args.distributed = False
    args.gpu = 0
    args.pin_mem = False
    if args.output_dir:
        args.output_dir = resolve_output_dir(args.output_dir)
    return args


class SafeDataLoader:
    """Wraps a DataLoader to skip items with data-loading errors, preserving __len__."""
    def __init__(self, dataloader, label="data", max_data_errors=20):
        self.dataloader = dataloader
        self.label = label
        self.max_data_errors = max_data_errors

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        skipped = 0
        iterator = iter(self.dataloader)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  SKIP {self.label}: {e}", flush=True)
                if skipped > self.max_data_errors:
                    raise RuntimeError(
                        f"Too many data errors ({skipped}) for {self.label}"
                    )
                continue


def train_one_epoch(model, data_loader, optimizer, epoch, args, device):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}/{args.epochs}]"

    for step, (src_input, tgt_input) in enumerate(
        metric_logger.log_every(SafeDataLoader(data_loader, "train"), 10, header)
    ):
        for key in list(src_input.keys()):
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device)

        try:
            stack_out = model(src_input, tgt_input)
            loss = stack_out["loss"]

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_value = loss.item() * args.gradient_accumulation_steps
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        except RuntimeError as e:
            print(f"  WARNING batch error: {e}", flush=True)
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, args, device, phase="dev"):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    tgt_pres = []
    tgt_refs = []

    for src_input, tgt_input in metric_logger.log_every(
        SafeDataLoader(data_loader, phase), 10, f"Eval {phase}"
    ):
        for key in list(src_input.keys()):
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device)

        try:
            stack_out = model(src_input, tgt_input)
            metric_logger.update(loss=stack_out["loss"].item())

            output = model.generate(stack_out, max_new_tokens=100, num_beams=4)
            tokenizer = model.mt5_tokenizer

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input["gt_sentence"][i])
        except RuntimeError as e:
            print(f"  WARNING eval error: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

    if len(tgt_pres) == 0:
        return {"bleu4": 0.0, "rouge": 0.0}

    # Pad and decode
    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    if len(tgt_pres[0]) < 150:
        pad_tensor = torch.ones(150 - len(tgt_pres[0]), device=device) * padding_value
        tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # Compute metrics
    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)

    for k, v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters["rouge"].update(rouge_score)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(model, path):
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.clone()
    torch.save({"model": state_dict}, path)


def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(args)

    utils.set_seed(args.seed)

    # Build datasets
    dataset_args = SimpleNamespace(
        dataset="CE_CSL",
        rgb_support=False,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=args.label_smoothing,
    )

    train_data = S2T_Dataset(
        path=train_label_paths["CE_CSL"], args=dataset_args, phase="train"
    )
    dev_data = S2T_Dataset(
        path=dev_label_paths["CE_CSL"], args=dataset_args, phase="dev"
    )
    test_data = S2T_Dataset(
        path=test_label_paths["CE_CSL"], args=dataset_args, phase="test"
    )

    print(f"train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        shuffle=False,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        shuffle=False,
        pin_memory=False,
    )

    # Build model
    model = GFSLT_VLP_Simplified(args=dataset_args)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f}M")

    # Evaluation only mode
    if args.eval:
        if args.checkpoint:
            state_dict = torch.load(args.checkpoint, map_location="cpu")["model"]
            model.load_state_dict(state_dict, strict=False)
        print("dev result")
        dev_stats = evaluate(model, dev_loader, args, device, "dev")
        print(dev_stats)
        print("test result")
        test_stats = evaluate(model, test_loader, args, device, "test")
        print(test_stats)
        return

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_epochs * len(train_loader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_bleu = 0.0

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, epoch, args, device)
        lr_scheduler.step()

        dev_stats = evaluate(model, dev_loader, args, device, "dev")
        test_stats = evaluate(model, test_loader, args, device, "test")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"dev_{k}": v for k, v in dev_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        print(f"[Epoch {epoch}] dev BLEU-4: {dev_stats.get('bleu4', 0):.2f}, "
              f"test BLEU-4: {test_stats.get('bleu4', 0):.2f}")

        if dev_stats.get("bleu4", 0) > max_bleu:
            max_bleu = dev_stats["bleu4"]
            if output_dir:
                save_checkpoint(model, output_dir / "best_checkpoint.pth")

        if output_dir:
            save_checkpoint(model, output_dir / f"checkpoint_{epoch}.pth")
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        print(f"Max dev BLEU-4: {max_bleu:.2f}")

    total_time = time.time() - start_time
    print(f"Training time: {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main()
