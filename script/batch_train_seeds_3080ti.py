"""
Batch training for RTX 3080 Ti 20GB — optimized config.
Runs: N seeds × 3 models (Full, Pose-only, Naive RGB)
GPU: RTX 3080 Ti 20GB — much faster than 5060 8GB.

Usage: python script/batch_train_seeds_3080ti.py --seeds 789,101,202,303,404,505 --dry-run

=== BEFORE RUNNING ON 3080Ti MACHINE ===
1. Copy entire Uni-Sign project to the 3080Ti machine
2. Create conda env: conda create -n torch13 python=3.10
3. Install deps: pip install torch numpy transformers timm
4. Update paths below (PYTHON_EXE, PROJECT_ROOT, STORAGE_OUT)
5. Run this script
"""
import os, subprocess, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════
# PATHS — adjust these for your 3080Ti machine
# ═══════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CHANGE THIS to your conda python path on 3080Ti machine
PYTHON_EXE = Path("C:/Users/Nemo518/.conda/envs/torch13/python.exe")

TRAIN_SCRIPT = PROJECT_ROOT / "local_sign_tools" / "finetune_cecsl_rgb_support.py"

# CHANGE THIS to your storage path on 3080Ti machine
STORAGE_OUT = Path("E:/project/xiaozhi/Uni-Sign-storage/out")

# ═══════════════════════════════════════════════════
# 3080 Ti OPTIMIZED CONFIG (20GB VRAM)
# ═══════════════════════════════════════════════════
BASE_ARGS = [
    # Both values are explicit. Previously --epochs 2 was ignored whenever
    # --stage2_epochs was non-zero, so the batch script ran only two epochs.
    "--stage1_epochs", "2", "--stage2_epochs", "2",
    "--stage2_lr", "1e-7", "--stage2_mt5_lr_scale", "0.1",
    # 3080Ti 20GB: use batch_size=8, no gradient accumulation needed
    "--batch_size", "8",
    "--gradient_accumulation_steps", "1",
    "--weight_decay", "1e-4", "--max_length", "256", "--hidden_dim", "256",
    # Development data may select checkpoints; test data is evaluated once
    # after the checkpoint and protocol are fixed.
    "--eval_sample_limit", "0", "--eval_split", "dev",
    "--selection_split", "dev", "--selection_sample_count", "100",
    "--selection_sample_mode", "random", "--selection_seed", "42",
    "--dtype", "bf16",
    # 3080Ti has enough CPU RAM for parallel data loading
    "--num_workers", "4",
]

# Pass-through environment with PYTHONUNBUFFERED for live output
_ENV = os.environ.copy()
_ENV["PYTHONUNBUFFERED"] = "1"

# ── experiment definitions ─────────────────────────────
VARIANTS = {
    "rgb_metadata_gate": {
        "label": "RGB + learned gate + reliability metadata",
        "extra_args": [
            "--experiment_variant", "rgb_metadata_gate",
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
            "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
        ],
    },
    "pose_only": {
        "label": "Pose-only",
        "extra_args": ["--experiment_variant", "pose_only", "--no_rgb_support"],
    },
    "rgb_gate": {
        "label": "RGB + learned gate without reliability metadata",
        "extra_args": [
            "--experiment_variant", "rgb_gate",
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
        ],
    },
}


def run_train(output_dir, extra_args, label, dry_run=False):
    """Run a single training. Returns True on success."""
    cmd = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--output_dir", output_dir] + BASE_ARGS + extra_args
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START: {label}")
    print(f"    dir: {output_dir}")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    if dry_run:
        return True, 0.0
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, env=_ENV)
        ok = result.returncode == 0
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        ok = False
    elapsed = time.time() - t0
    status = "OK" if ok else "FAILED"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE: {label} -> {status} ({elapsed/3600:.1f} h)")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="789,101,202,303,404,505",
                        help="Comma-separated seed list")
    parser.add_argument("--variants", default="pose_only,rgb_gate,rgb_metadata_gate",
                        help="Which variants to run")
    parser.add_argument("--start-from", default="",
                        help="Skip until this label, e.g. 'seed789_pose_only'")
    parser.add_argument("--top-k-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Print and validate the run plan without training.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Allow reuse of a non-empty output directory.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    variant_keys = [v.strip() for v in args.variants.split(",")]
    unknown_variants = sorted(set(variant_keys) - set(VARIANTS))
    if unknown_variants:
        parser.error(f"unknown variants: {', '.join(unknown_variants)}")
    if not 0 < args.top_k_ratio <= 1:
        parser.error("--top-k-ratio must be in (0, 1]")

    results_log = []
    skip = bool(args.start_from)
    total = len(seeds) * len(variant_keys)
    done_count = 0

    print(f"\n{'#'*70}")
    print(f"# BATCH SEED TRAINING — 3080Ti 20GB")
    print(f"# Seeds: {seeds}")
    print(f"# Variants: {variant_keys}")
    print(f"# Total runs: {total}")
    print(f"# batch_size=8 | num_workers=4 | bf16")
    print(f"# Start from: '{args.start_from}' (skip={skip})")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    for seed in seeds:
        for vk in variant_keys:
            label = f"seed{seed}_{vk}"

            if skip:
                if label != args.start_from:
                    print(f"[SKIP] {label} (before start_from)")
                    continue
                else:
                    skip = False
                    print(f"[RESUME] Starting from {label}")

            vdef = VARIANTS[vk]
            extra = vdef["extra_args"] + ["--seed", str(seed), "--top_k_ratio", str(args.top_k_ratio)]
            out_dir = str(STORAGE_OUT / f"seed{seed}_{vk}")

            output_path = Path(out_dir)
            if output_path.exists() and any(output_path.iterdir()) and not args.overwrite_existing:
                print(f"[BLOCKED] {label}: non-empty output directory: {output_path}")
                print("          Use a new directory or pass --overwrite-existing after auditing its contents.")
                results_log.append({"label": label, "seed": seed, "variant": vk, "ok": False, "elapsed_h": 0.0, "output_dir": out_dir, "reason": "non_empty_output_dir"})
                continue

            ok, elapsed = run_train(out_dir, extra, f"{label}  [{vdef['label']}]", dry_run=args.dry_run)
            results_log.append({
                "label": label,
                "seed": seed,
                "variant": vk,
                "ok": ok,
                "elapsed_h": round(elapsed / 3600, 2),
                "output_dir": out_dir,
            })
            done_count += 1
            print(f"\n[PROGRESS] {done_count}/{total} completed\n")

    # ── Summary ──────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# ALL DONE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    print(f"\n{'Seed':<8} {'Variant':<12} {'Status':<10} {'Time (h)':<10}")
    print("-" * 45)
    for r in results_log:
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['seed']:<8} {r['variant']:<12} {status:<10} {r['elapsed_h']:<10.1f}")

    failed = [r for r in results_log if not r["ok"]]
    if failed:
        print(f"\n⚠ {len(failed)} failures:")
        for r in failed:
            print(f"  - {r['label']}: {r['output_dir']}")

    # Save log
    if not args.dry_run:
        log_path = STORAGE_OUT / f"batch_log_3080ti_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
