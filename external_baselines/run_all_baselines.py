"""
Master script to run all external baselines and collect comparison results.

Pipeline:
  1. Oracle Gloss-to-Text: train mT5 on gloss→text, evaluate on dev100/test100
  2. Two-Stage Gloss Pipeline: CSLR gloss predictions → gloss→text translation
  3. GFSLT-VLP Simplified: train + evaluate

After all baselines complete, prints a unified comparison table with
existing Uni-Sign results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "external_baselines"

PYTHON_EXE = Path("C:/Users/Nemo518/.conda/envs/torch13/python.exe")
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(str(PYTHON_EXE))

sys.path.insert(0, str(PROJECT_ROOT))
from paths import STORAGE_ROOT
STORAGE_OUT = Path(STORAGE_ROOT) / "out"


def run_cmd(cmd, desc=""):
    cmd = [str(c) for c in cmd]
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        return False
    return True


def compute_metrics(csv_path, label):
    """Compute full metrics from eval CSV."""
    script = PROJECT_ROOT / "script" / "compute_cecsl_text_metrics.py"
    if not script.exists():
        print(f"WARNING: metrics script not found at {script}")
        return None

    result = subprocess.run(
        [
            str(PYTHON_EXE), str(script),
            "--input_csv", str(csv_path),
            "--label", label,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR computing metrics for {csv_path}: {result.stderr}")
        return None

    try:
        return json.loads(result.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        print(f"ERROR parsing metrics output: {result.stdout[:500]}")
        return None


def step1_oracle_gloss(args):
    """Step 1: Train oracle gloss→text model and evaluate."""
    print("\n" + "=" * 60)
    print("STEP 1: Oracle Gloss-to-Text Baseline")
    print("=" * 60)

    model_dir = args.output_dir / "oracle_gloss_model"
    script = BASELINE_DIR / "gloss_pipeline.py"

    # Train
    ok = run_cmd(
        [str(PYTHON_EXE), str(script), "train_oracle",
         "--output_dir", str(model_dir),
         "--epochs", str(args.oracle_epochs),
         "--batch_size", str(args.oracle_batch_size),
         "--lr", str(args.oracle_lr)],
        "Training oracle gloss-to-text model"
    )
    if not ok:
        return None

    # Evaluate on dev100
    dev_input = STORAGE_OUT / "ablation_pose_only" / "dev100_eval.csv"
    dev_output = args.output_dir / "oracle_gloss_dev100.csv"
    if dev_input.exists():
        ok = run_cmd(
            [str(PYTHON_EXE), str(script), "generate",
             "--model_dir", str(model_dir / "best_model"),
             "--input_csv", str(dev_input),
             "--output_csv", str(dev_output)],
            "Oracle gloss→text on dev100"
        )
        if ok:
            metrics = compute_metrics(dev_output, "oracle_gloss_dev100")
            if metrics:
                print(f"Oracle Dev100: AvgSim={metrics.get('average_similarity', 0):.4f}, "
                      f"BLEU-4={metrics.get('bleu4', 0):.2f}, ROUGE-L={metrics.get('rouge_l', 0):.2f}")

    # Evaluate on test100
    test_input = STORAGE_OUT / "ablation_pose_only" / "test100_eval.csv"
    test_output = args.output_dir / "oracle_gloss_test100.csv"
    if test_input.exists():
        ok = run_cmd(
            [str(PYTHON_EXE), str(script), "generate",
             "--model_dir", str(model_dir / "best_model"),
             "--input_csv", str(test_input),
             "--output_csv", str(test_output)],
            "Oracle gloss→text on test100"
        )
        if ok:
            metrics = compute_metrics(test_output, "oracle_gloss_test100")
            if metrics:
                print(f"Oracle Test100: AvgSim={metrics.get('average_similarity', 0):.4f}, "
                      f"BLEU-4={metrics.get('bleu4', 0):.2f}, ROUGE-L={metrics.get('rouge_l', 0):.2f}")
    return {"dev": dev_output, "test": test_output}


def step2_two_stage(args):
    """Step 2: Two-Stage Gloss Pipeline (CSLR → gloss→text)."""
    print("\n" + "=" * 60)
    print("STEP 2: Two-Stage Gloss Pipeline")
    print("=" * 60)

    script = BASELINE_DIR / "gloss_pipeline.py"

    # Stage 1: Generate CSLR gloss predictions
    for split in ["dev", "test"]:
        cslr_output = args.output_dir / f"cslr_gloss_{split}100.csv"
        ok = run_cmd(
            [str(PYTHON_EXE), str(script), "generate_cslr",
             "--checkpoint", str(args.cslr_checkpoint),
             "--split", split,
             "--output_csv", str(cslr_output)],
            f"CSLR gloss prediction on {split}100"
        )
        if not ok:
            print(f"WARNING: CSLR failed for {split}, skipping two-stage for this split")

    # Stage 2: Translate glosses → text using oracle model
    oracle_model = args.output_dir / "oracle_gloss_model" / "best_model"
    if not oracle_model.exists():
        print("ERROR: Oracle model not found. Run step 1 first.")
        return None

    results = {}
    for split in ["dev", "test"]:
        cslr_csv = args.output_dir / f"cslr_gloss_{split}100.csv"
        ts_output = args.output_dir / f"twostage_{split}100.csv"
        if cslr_csv.exists():
            ok = run_cmd(
                [str(PYTHON_EXE), str(script), "generate",
                 "--model_dir", str(oracle_model),
                 "--input_csv", str(cslr_csv),
                 "--output_csv", str(ts_output)],
                f"Two-stage gloss→text on {split}100"
            )
            if ok:
                metrics = compute_metrics(ts_output, f"twostage_{split}100")
                if metrics:
                    print(f"Two-Stage {split}100: AvgSim={metrics.get('average_similarity', 0):.4f}, "
                          f"BLEU-4={metrics.get('bleu4', 0):.2f}")
                results[split] = ts_output
    return results


def step3_gfslt_vlp(args):
    """Step 3: Train and evaluate GFSLT-VLP Simplified model."""
    print("\n" + "=" * 60)
    print("STEP 3: GFSLT-VLP Simplified Baseline")
    print("=" * 60)

    gfslt_dir = args.output_dir / "gfslt_vlp"
    train_script = BASELINE_DIR / "train_gfslt_vlp.py"

    # Train
    ok = run_cmd(
        [str(PYTHON_EXE), str(train_script),
         "--output_dir", str(gfslt_dir),
         "--epochs", str(args.gfslt_epochs),
         "--batch_size", "1",
         "--lr", str(args.gfslt_lr),
         "--hidden_dim", "256",
         "--max_length", "256"],
        "Training GFSLT-VLP Simplified"
    )
    if not ok:
        return None

    # Evaluate best checkpoint on dev100 and test100
    eval_script = BASELINE_DIR / "eval_baseline.py"
    results = {}
    best_ckpt = gfslt_dir / "best_checkpoint.pth"

    if not best_ckpt.exists():
        print(f"WARNING: best_checkpoint not found at {best_ckpt}")
        # Try latest epoch checkpoint
        checkpoints = sorted(gfslt_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            best_ckpt = checkpoints[-1]
            print(f"Using {best_ckpt} instead")

    for split in ["dev", "test"]:
        output_csv = args.output_dir / f"gfslt_vlp_{split}100.csv"
        ok = run_cmd(
            [str(PYTHON_EXE), str(eval_script),
             "--model_type", "gfslt_vlp",
             "--checkpoint", str(best_ckpt),
             "--split", split,
             "--output_csv", str(output_csv),
             "--sample_count", "0",
             "--batch_size", "1"],
            f"Evaluating GFSLT-VLP on {split}100"
        )
        if ok:
            metrics = compute_metrics(output_csv, f"gfslt_vlp_{split}100")
            if metrics:
                print(f"GFSLT-VLP {split}100: AvgSim={metrics.get('average_similarity', 0):.4f}, "
                      f"BLEU-4={metrics.get('bleu4', 0):.2f}")
            results[split] = output_csv
        else:
            print(f"WARNING: GFSLT-VLP eval failed for {split}")

    return results


def print_comparison_table(args):
    """Print a unified comparison table from existing Uni-Sign + new baseline results."""
    print("\n" + "=" * 60)
    print("UNIFIED COMPARISON TABLE")
    print("=" * 60)

    # Load existing Uni-Sign results
    existing_csvs = {
        "Pose-only (Uni-Sign)": STORAGE_OUT / "ablation_pose_only",
        "Naive RGB (Uni-Sign)": STORAGE_OUT / "ablation_rgb_support",
        "Full model (Uni-Sign)": STORAGE_OUT / "ablation_uncert_gate",
    }

    # New baseline CSVs
    new_csvs = {
        "GFSLT-VLP (Simplified)": args.output_dir / "gfslt_vlp",
        "Two-Stage Gloss2Text": args.output_dir / "twostage",
        "Oracle Gloss→Text": args.output_dir / "oracle_gloss",
    }

    script = PROJECT_ROOT / "script" / "compute_cecsl_text_metrics.py"

    all_results = []

    for label, csv_dir in {**existing_csvs, **new_csvs}.items():
        for split in ["test", "dev"]:
            csv_path = csv_dir / f"{split}100_eval.csv" if "Uni-Sign" in label else Path(str(csv_dir) + f"_{split}100.csv")
            if not csv_path.exists():
                # Try alternative naming
                alt_path = Path(str(csv_dir).replace("_eval", "")) if isinstance(csv_dir, Path) else csv_dir
                continue
            try:
                result = subprocess.run(
                    [str(PYTHON_EXE), str(script), "--input_csv", str(csv_path), "--label", label],
                    cwd=str(PROJECT_ROOT), capture_output=True, text=True
                )
                if result.returncode == 0:
                    metrics = json.loads(result.stdout)[0]
                    metrics["split"] = split
                    all_results.append(metrics)
            except Exception:
                pass

    if not all_results:
        print("No results found yet. Run the baselines first.")
        return

    # Print table
    print(f"\n{'Model':<30} {'Split':<6} {'AvgSim':<8} {'BLEU-4':<8} {'ROUGE-L':<8} {'chrF':<8} {'WER':<8}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: (x.get("split", ""), -x.get("average_similarity", 0))):
        print(f"{r.get('label', '?'):<30} {r.get('split', '?'):<6} "
              f"{r.get('average_similarity', 0):.4f}   "
              f"{r.get('bleu4', 0):.2f}     "
              f"{r.get('rouge_l', 0):.2f}     "
              f"{r.get('chrf', 0):.2f}     "
              f"{r.get('wer', 0):.2f}")


def main():
    parser = argparse.ArgumentParser("Run all external baselines for CE-CSL")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "out" / "external_baselines"))
    parser.add_argument("--steps", type=str, default="1,2,3",
                        help="Comma-separated steps to run: 1=Oracle, 2=TwoStage, 3=GFSLT-VLP")
    parser.add_argument("--cslr_checkpoint", type=str,
                        default=str(STORAGE_OUT / "stage3_finetuning" / "best_checkpoint.pth"))

    # Training hyperparams
    parser.add_argument("--oracle_epochs", type=int, default=20)
    parser.add_argument("--oracle_batch_size", type=int, default=4)
    parser.add_argument("--oracle_lr", type=float, default=5e-5)
    parser.add_argument("--gfslt_epochs", type=int, default=30)
    parser.add_argument("--gfslt_lr", type=float, default=1e-5)

    # Final report only
    parser.add_argument("--report_only", action="store_true",
                        help="Only print comparison table without running training")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        print_comparison_table(args)
        return

    steps = set(int(s.strip()) for s in args.steps.split(","))

    if 1 in steps:
        step1_oracle_gloss(args)

    if 2 in steps:
        step2_two_stage(args)

    if 3 in steps:
        step3_gfslt_vlp(args)

    print_comparison_table(args)

    print("\n" + "=" * 60)
    print("ALL BASELINES COMPLETE")
    print(f"Results in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
