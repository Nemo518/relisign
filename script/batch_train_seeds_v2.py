"""
Batch training v2: Multi-seed expansion for reviewer revision.
Runs: N seeds × 3 models (Full, Pose-only, Naive RGB)
GPU: RTX 5060 8GB — sequential execution only.

Usage: python script/batch_train_seeds_v2.py --seeds 456,789,101,202 --start-from seed456_full
"""
import os, subprocess, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path("C:/Users/Nemo518/.conda/envs/torch13/python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "local_sign_tools" / "finetune_cecsl_rgb_support.py"
STORAGE_OUT = Path("E:/project/xiaozhi/Uni-Sign-storage/out")

# Pass-through environment with PYTHONUNBUFFERED for live output
_ENV = os.environ.copy()
_ENV["PYTHONUNBUFFERED"] = "1"

BASE_ARGS = [
    "--epochs", "2", "--stage2_epochs", "2",
    "--stage2_lr", "1e-7", "--stage2_mt5_lr_scale", "0.1",
    "--batch_size", "1", "--gradient_accumulation_steps", "4",
    "--weight_decay", "1e-4", "--max_length", "256", "--hidden_dim", "256",
    "--eval_sample_limit", "100", "--eval_split", "test",
    "--dtype", "bf16", "--num_workers", "0",
]

# ── experiment definitions ─────────────────────────────
# Three core variants per seed
VARIANTS = {
    "full_model": {
        "label": "Full model (gate+metadata)",
        "suffix": "full_model",  # matches existing naming seed42_full_model
        "extra_args": [
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
            "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
        ],
    },
    "pose_only": {
        "label": "Pose-only",
        "suffix": "pose_only",  # matches existing naming seed42_pose_only
        "extra_args": ["--no_rgb_support"],
    },
    "naive_rgb": {
        "label": "Naive RGB",
        "suffix": "naive_rgb",  # matches existing naming seed42_naive_rgb
        "extra_args": [
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
        ],
    },
}

def run_train(output_dir, extra_args, label):
    """Run a single training. Returns True on success."""
    cmd = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--output_dir", output_dir] + BASE_ARGS + extra_args
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START: {label}")
    print(f"    dir: {output_dir}")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False, env=_ENV)
        ok = result.returncode == 0
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        ok = False
    elapsed = time.time() - t0
    status = "OK" if ok else "FAILED"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE: {label} → {status} ({elapsed/3600:.1f} h)")
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="456,789,101,202", help="Comma-separated seed list")
    parser.add_argument("--variants", default="pose_only,naive_rgb,full_model", help="Which variants to run (pose_only,naive_rgb,full_model)")
    parser.add_argument("--start-from", default="", help="Skip until this label, e.g. 'seed456_full'")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    variant_keys = [v.strip() for v in args.variants.split(",")]

    results_log = []
    skip = bool(args.start_from)
    total = len(seeds) * len(variant_keys)
    done_count = 0

    print(f"\n{'#'*70}")
    print(f"# BATCH SEED TRAINING v2")
    print(f"# Seeds: {seeds}")
    print(f"# Variants: {variant_keys}")
    print(f"# Total runs: {total}")
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
                    skip = False  # Found the start point, stop skipping
                    print(f"[RESUME] Starting from {label}")

            vdef = VARIANTS[vk]
            extra = vdef["extra_args"] + ["--seed", str(seed)]
            out_dir = str(STORAGE_OUT / f"seed{seed}_{vk}")  # e.g. seed456_pose_only

            ok, elapsed = run_train(out_dir, extra, f"{label}  [{vdef['label']}]")
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
    print(f"\n{'Seed':<8} {'Variant':<8} {'Status':<10} {'Time (h)':<10}")
    print("-" * 40)
    for r in results_log:
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['seed']:<8} {r['variant']:<8} {status:<10} {r['elapsed_h']:<10.1f}")

    failed = [r for r in results_log if not r["ok"]]
    if failed:
        print(f"\n⚠ {len(failed)} failures:")
        for r in failed:
            print(f"  - {r['label']}: {r['output_dir']}")

    # Save log
    log_path = STORAGE_OUT / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
