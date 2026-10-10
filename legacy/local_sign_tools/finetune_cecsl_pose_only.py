import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import dev_label_paths, pose_dirs, test_label_paths, train_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign


LABEL_PATHS = {
    'train': train_label_paths['CE_CSL'],
    'dev': dev_label_paths['CE_CSL'],
    'test': test_label_paths['CE_CSL'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Single-GPU CE-CSL pose-only fine-tuning without deepspeed.')
    parser.add_argument('--checkpoint', default='./out/stage3_finetuning/best_checkpoint.pth')
    parser.add_argument('--output_dir', default='./out/cecsl_pose_only_local')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', default='bf16', choices=['fp32', 'bf16'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_mt5', action='store_true')
    parser.add_argument('--train_sample_limit', type=int, default=0, help='0 means use all available pose samples.')
    parser.add_argument('--eval_sample_limit', type=int, default=0, help='0 means use all available pose samples in eval split.')
    parser.add_argument('--eval_split', default='dev', choices=['none', 'dev', 'test'])
    parser.add_argument(
        '--save_epoch_checkpoints',
        action='store_true',
        help='Save checkpoint_epoch_*.pth after every epoch. Disabled by default to reduce disk usage.',
    )
    return parser.parse_args()


def build_dataset_args(args):
    return SimpleNamespace(
        dataset='CE_CSL',
        rgb_support=False,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        label_smoothing=args.label_smoothing,
    )


def discover_available_indices(dataset, sample_limit):
    available_indices = []
    for idx, key in enumerate(dataset.list):
        sample = dataset.raw_data[key]
        pose_path = Path(dataset.pose_dir) / sample['video_path'].replace('.mp4', '.pkl')
        if pose_path.exists():
            available_indices.append(idx)
        if sample_limit > 0 and len(available_indices) >= sample_limit:
            break
    return available_indices


def move_src_input(src_input, device, dtype_mode='fp32'):
    tensor_dtype = torch.float32
    if dtype_mode == 'bf16' and device.type == 'cuda':
        tensor_dtype = torch.bfloat16

    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.dtype.is_floating_point:
                value = value.to(tensor_dtype)
            src_input[key] = value
    return src_input


def build_loader(args, split, sample_limit):
    dataset_args = build_dataset_args(args)
    dataset = S2T_Dataset(path=LABEL_PATHS[split], args=dataset_args, phase=split)
    available_indices = discover_available_indices(dataset, sample_limit)
    subset = Subset(dataset, available_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=(split == 'train'),
    )
    return dataset, available_indices, loader


def evaluate(model, loader, device, dtype_mode):
    if loader is None:
        return None

    use_autocast = dtype_mode == 'bf16' and device.type == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for src_input, tgt_input in loader:
            src_input = move_src_input(src_input, device, dtype_mode)
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    stack_out = model(src_input, tgt_input)
                    loss = stack_out['loss']
            else:
                stack_out = model(src_input, tgt_input)
                loss = stack_out['loss']
            total_loss += float(loss.item())
            total_batches += 1
    model.train()
    if total_batches == 0:
        return None
    return total_loss / total_batches


def save_checkpoint(path, model, args, extra=None):
    payload = {
        'model': model.state_dict(),
        'args': vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_indices, train_loader = build_loader(args, 'train', args.train_sample_limit)
    print(f'train_dataset_size: {len(train_dataset)}')
    print(f'train_available_pose: {len(train_indices)}')
    print(f'train_pose_dir: {pose_dirs["CE_CSL"]}')

    if not train_indices:
        raise RuntimeError('No CE-CSL train samples with pose files are available.')

    eval_loader = None
    eval_indices = []
    if args.eval_split != 'none':
        _, eval_indices, eval_loader = build_loader(args, args.eval_split, args.eval_sample_limit)
        print(f'{args.eval_split}_available_pose: {len(eval_indices)}')
        if not eval_indices:
            print(f'warning: {args.eval_split} split has no pose files yet, evaluation will be skipped.')
            eval_loader = None

    model_args = build_dataset_args(args)
    model = Uni_Sign(args=model_args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ret = model.load_state_dict(checkpoint['model'], strict=True)
    print(f'missing_keys: {len(ret.missing_keys)}')
    print(f'unexpected_keys: {len(ret.unexpected_keys)}')

    if args.freeze_mt5:
        for param in model.mt5_model.parameters():
            param.requires_grad = False
        print('mt5 backbone frozen for lower memory usage')

    model = model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    use_autocast = args.dtype == 'bf16' and device.type == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
    if args.dtype == 'bf16' and not use_autocast:
        print('bf16 is not available on this device, falling back to fp32 training.')

    best_eval = math.inf
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (src_input, tgt_input) in enumerate(train_loader, start=1):
            src_input = move_src_input(src_input, device, args.dtype)

            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    stack_out = model(src_input, tgt_input)
                    loss = stack_out['loss']
            else:
                stack_out = model(src_input, tgt_input)
                loss = stack_out['loss']

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
                    f'epoch {epoch}/{args.epochs} | batch {batch_idx}/{len(train_loader)} | '
                    f'loss={loss.item():.4f} | step={global_step}'
                )

        avg_train_loss = running_loss / max(1, batch_count)
        eval_loss = evaluate(model, eval_loader, device, args.dtype)
        print(f'epoch {epoch} finished | train_loss={avg_train_loss:.4f}' + (f' | {args.eval_split}_loss={eval_loss:.4f}' if eval_loss is not None else ''))
        if args.save_epoch_checkpoints:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(checkpoint_path, model, args, extra={'epoch': epoch, 'train_loss': avg_train_loss, 'eval_loss': eval_loss})

        if eval_loss is not None and eval_loss < best_eval:
            best_eval = eval_loss
            save_checkpoint(output_dir / 'best_checkpoint.pth', model, args, extra={'epoch': epoch, 'train_loss': avg_train_loss, 'eval_loss': eval_loss})

    save_checkpoint(output_dir / 'final_checkpoint.pth', model, args, extra={'epoch': args.epochs})
    print(f'fine-tuning finished, checkpoints saved to: {output_dir}')


if __name__ == '__main__':
    main()
