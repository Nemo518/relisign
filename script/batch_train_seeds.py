"""
Batch training: 7 more seeds + bias ablation for reviewer revision.
Runs: 7 seeds × 3 models = 21 runs + 3 bias ablation = 24 total.
"""
import subprocess, sys, json, time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path("C:/Users/Nemo518/.conda/envs/torch13/python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "local_sign_tools" / "finetune_cecsl_rgb_support.py"
STORAGE_OUT = Path("E:/project/xiaozhi/Uni-Sign-storage/out")

BASE_ARGS = [
    "--epochs", "2", "--stage2_epochs", "2",
    "--stage2_lr", "1e-7", "--stage2_mt5_lr_scale", "0.1",
    "--batch_size", "1", "--gradient_accumulation_steps", "4",
    "--weight_decay", "1e-4", "--max_length", "256", "--hidden_dim", "256",
    "--eval_sample_limit", "100", "--eval_split", "test",
    "--dtype", "bf16",
]

def run_train(output_dir, extra_args, label):
    """Run a single training, return True on success."""
    cmd = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--output_dir", output_dir] + BASE_ARGS + extra_args
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}")
    print(f"    dir: {output_dir}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: {'OK' if ok else 'FAILED'} ({elapsed/60:.1f} min)")
    return ok

def main():
    results_log = []

    # ========================================
    # 7 seeds (456, 789 re-run + 101, 202, 303, 404, 505 new)
    # ========================================
    seeds = [456, 789, 101, 202, 303, 404, 505]

    for seed in seeds:
        # Full model
        run_train(
            str(STORAGE_OUT / f"seed{seed}_full_model"),
            [
                "--seed", str(seed),
                "--train_rgb_frame_sampling", "confidence_topk",
                "--eval_rgb_frame_sampling", "confidence_topk",
                "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
            ],
            f"seed{seed}_full"
        )

        # Pose-only
        run_train(
            str(STORAGE_OUT / f"seed{seed}_pose_only"),
            ["--seed", str(seed), "--train_sample_limit", "0"],
            f"seed{seed}_pose"
        )

        # Naive RGB
        run_train(
            str(STORAGE_OUT / f"seed{seed}_naive_rgb"),
            [
                "--seed", str(seed),
                "--train_rgb_frame_sampling", "confidence_topk",
                "--eval_rgb_frame_sampling", "confidence_topk",
            ],
            f"seed{seed}_naive"
        )

    # ========================================
    # Bias ablation: bias=0, bias=-1 (bias=-2 is default full model)
    # ========================================
    # These require code support for --fusion_gate_bias flag
    # For now, skip (needs code change in models.py)
    print("\n[SKIP] Bias ablation requires --fusion_gate_bias flag in training script.")
    print("[TODO] Add --fusion_gate_bias argument to finetune_cecsl_rgb_support.py and models.py")

    print(f"\n{'='*60}")
    print(f"ALL DONE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
