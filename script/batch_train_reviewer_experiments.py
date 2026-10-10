"""
Batch training script for reviewer-requested experiments.
Runs: multi-seed (MIN-1), top-K sensitivity (MIN-2), comparison baselines (MAJ-3).

Usage: python script/batch_train_reviewer_experiments.py
"""
import subprocess
import sys
from pathlib import Path
import shutil
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_PY = PROJECT_ROOT / "datasets.py"
BACKUP_PY = PROJECT_ROOT / "datasets.py.backup"

PYTHON_EXE = Path("C:/Users/Nemo518/.conda/envs/torch13/python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "local_sign_tools" / "finetune_cecsl_rgb_support.py"
EVAL_SCRIPT = PROJECT_ROOT / "script" / "eval_cecsl_samples.py"
METRICS_SCRIPT = PROJECT_ROOT / "script" / "compute_cecsl_text_metrics.py"
STORAGE_OUT = Path("E:/project/xiaozhi/Uni-Sign-storage/out")

# Base config (same as paper's main experiments)
BASE_ARGS = [
    "--epochs", "2",
    "--stage2_epochs", "2",
    "--stage2_lr", "1e-7",
    "--stage2_mt5_lr_scale", "0.1",
    "--batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--weight_decay", "1e-4",
    "--max_length", "256",
    "--hidden_dim", "256",
    "--eval_sample_limit", "100",
    "--eval_split", "test",
    "--dtype", "bf16",
]


def patch_topk_ratio(ratio: float):
    """Patch the hardcoded 0.1 in datasets.py to a new ratio."""
    if not BACKUP_PY.exists():
        shutil.copy(DATASETS_PY, BACKUP_PY)

    content = DATASETS_PY.read_text(encoding="utf-8")
    original_line = "    sample_size = max(1, int(np.ceil(0.1 * len(candidate_indices))))"
    new_line = f"    sample_size = max(1, int(np.ceil({ratio} * len(candidate_indices))))"

    if new_line not in content:
        content = content.replace(original_line, new_line)
        DATASETS_PY.write_text(content, encoding="utf-8")
        print(f"  [PATCH] top-K ratio: 0.1 -> {ratio}")
    else:
        print(f"  [PATCH] top-K ratio already set to {ratio}")


def restore_datasets():
    """Restore the original datasets.py."""
    if BACKUP_PY.exists():
        shutil.copy(BACKUP_PY, DATASETS_PY)
        BACKUP_PY.unlink()
        print("  [RESTORE] datasets.py restored")


def run_train(output_dir: str, extra_args: list[str]):
    """Run a single training experiment."""
    cmd = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--output_dir", output_dir] + BASE_ARGS + extra_args
    print(f"\n{'='*60}")
    print(f">>> Training: {output_dir}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: Training failed for {output_dir}")
        return False
    return True


def run_eval(csv_path: str, eval_extra_args: list[str]):
    """Run evaluation and compute metrics."""
    cmd = [str(PYTHON_EXE), str(EVAL_SCRIPT), "--output_csv", csv_path] + eval_extra_args
    print(f">>> Eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: Eval failed for {csv_path}")
        return None

    # Compute metrics
    label = Path(csv_path).stem
    cmd2 = [str(PYTHON_EXE), str(METRICS_SCRIPT), "--input_csv", csv_path, "--label", label]
    result2 = subprocess.run(cmd2, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result2.returncode == 0:
        try:
            metrics = json.loads(result2.stdout)
            if metrics:
                m = metrics[0]
                print(f"  AvgSim={m.get('average_similarity', '?'):.4f}, BLEU-4={m.get('bleu4', '?'):.2f}")
                return metrics[0]
        except:
            pass
    return None


def main():
    results = {}

    # ============================================
    # MIN-1: Multi-seed experiments (2 more seeds)
    # ============================================
    print("\n" + "="*60)
    print("MIN-1: Multi-Seed Experiments (seeds 456, 789)")
    print("="*60)

    extra_seeds = [456, 789]
    for seed in extra_seeds:
        # Full model
        output_dir = str(STORAGE_OUT / f"seed{seed}_full_model")
        ok = run_train(output_dir, [
            "--seed", str(seed),
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
            "--fusion_use_uncertainty_gate",
            "--fusion_meta_dim", "3",
        ])
        if ok:
            csv = str(STORAGE_OUT / f"seed{seed}_full_model" / "test100_eval.csv")
            results[f"seed{seed}_full"] = run_eval(csv, [
                "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
                "--rgb_frame_sampling", "confidence_topk",
            ])

        # Pose-only
        output_dir = str(STORAGE_OUT / f"seed{seed}_pose_only")
        ok = run_train(output_dir, [
            "--seed", str(seed),
            "--train_sample_limit", "0",
        ])
        # Pose-only doesn't use rgb_frame_sampling
        if ok:
            csv = str(STORAGE_OUT / f"seed{seed}_pose_only" / "test100_eval.csv")
            results[f"seed{seed}_pose"] = run_eval(csv, [])

        # Naive RGB
        output_dir = str(STORAGE_OUT / f"seed{seed}_naive_rgb")
        ok = run_train(output_dir, [
            "--seed", str(seed),
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
        ])
        if ok:
            csv = str(STORAGE_OUT / f"seed{seed}_naive_rgb" / "test100_eval.csv")
            results[f"seed{seed}_naive"] = run_eval(csv, [
                "--rgb_frame_sampling", "confidence_topk",
            ])

    # ============================================
    # MIN-2: top-K sensitivity analysis
    # ============================================
    print("\n" + "="*60)
    print("MIN-2: Top-K Sensitivity (5%, 15%, 20%)")
    print("="*60)

    for ratio in [0.05, 0.15, 0.20]:
        patch_topk_ratio(ratio)
        pct = int(ratio * 100)

        output_dir = str(STORAGE_OUT / f"topk{pct}_full_model")
        ok = run_train(output_dir, [
            "--seed", "42",
            "--train_rgb_frame_sampling", "confidence_topk",
            "--eval_rgb_frame_sampling", "confidence_topk",
            "--fusion_use_uncertainty_gate",
            "--fusion_meta_dim", "3",
        ])
        if ok:
            csv = str(STORAGE_OUT / f"topk{pct}_full_model" / "test100_eval.csv")
            results[f"topk{pct}_full"] = run_eval(csv, [
                "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
                "--rgb_frame_sampling", "confidence_topk",
            ])

    restore_datasets()
    patch_topk_ratio(0.10)  # restore default 10%

    # ============================================
    # MAJ-3: Comparison baselines
    # ============================================
    print("\n" + "="*60)
    print("MAJ-3: Comparison Baselines")
    print("="*60)

    # Confidence-weighted RGB (no gate)
    output_dir = str(STORAGE_OUT / "baseline_conf_weighted")
    ok = run_train(output_dir, [
        "--seed", "42",
        "--train_rgb_frame_sampling", "confidence_weighted",
        "--eval_rgb_frame_sampling", "confidence_weighted",
    ])
    if ok:
        csv = str(STORAGE_OUT / "baseline_conf_weighted" / "test100_eval.csv")
        results["conf_weighted"] = run_eval(csv, [
            "--rgb_frame_sampling", "confidence_weighted",
        ])

    # Uniform sampling + gate (no explicit confidence)
    output_dir = str(STORAGE_OUT / "baseline_uniform_gate")
    ok = run_train(output_dir, [
        "--seed", "42",
        "--train_rgb_frame_sampling", "uniform",
        "--eval_rgb_frame_sampling", "uniform",
        "--fusion_use_uncertainty_gate",
        "--fusion_meta_dim", "3",
    ])
    if ok:
        csv = str(STORAGE_OUT / "baseline_uniform_gate" / "test100_eval.csv")
        results["uniform_gate"] = run_eval(csv, [
            "--fusion_use_uncertainty_gate", "--fusion_meta_dim", "3",
            "--rgb_frame_sampling", "uniform",
        ])

    # ============================================
    # Summary
    # ============================================
    restore_datasets()

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)

    # Save results
    results_path = PROJECT_ROOT / "out" / "paper_metrics" / "reviewer_experiments_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if v is not None}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print(f"\n{'Experiment':<30} {'AvgSim':<10} {'BLEU-4':<10} {'WER':<10}")
    print("-"*60)
    for name, m in sorted(results.items()):
        if m:
            print(f"{name:<30} {m.get('average_similarity', 0):.4f}     {m.get('bleu4', 0):.2f}      {m.get('wer', 0):.2f}")


if __name__ == "__main__":
    main()
