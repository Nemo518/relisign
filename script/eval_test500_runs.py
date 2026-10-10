import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "script" / "eval_cecsl_samples.py"
METRICS_SCRIPT = PROJECT_ROOT / "script" / "compute_cecsl_text_metrics.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Strict, unified Test500 evaluation for completed CE-CSL runs.")
    parser.add_argument("--run", action="append", default=[], help="Run directory or checkpoint path; repeatable.")
    parser.add_argument("--run_root", default="", help="Discover protocol-manifest runs below this directory.")
    parser.add_argument("--checkpoint_name", default="best_checkpoint.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_json", default="out/test500_summary.json")
    parser.add_argument("--summary_csv", default="out/test500_summary.csv")
    return parser.parse_args()


def discover_runs(args):
    candidates = [Path(item) for item in args.run]
    if args.run_root:
        candidates.extend(path.parent for path in Path(args.run_root).rglob("protocol_manifest.json"))

    checkpoints = []
    for candidate in candidates:
        checkpoint = candidate if candidate.is_file() else candidate / args.checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        checkpoints.append(checkpoint.resolve())
    return list(dict.fromkeys(checkpoints))


def validate_manifest(run_dir):
    manifest_path = run_dir / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"reported Test500 evaluation requires protocol_manifest.json: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_data_used_during_training") is not False:
        raise RuntimeError(f"test data leakage flag is not false: {manifest_path}")
    if manifest.get("eval_split") != "dev" or manifest.get("selection_split") != "dev":
        raise RuntimeError(f"training selection was not dev/dev: {manifest_path}")
    return manifest


def run_command(command, dry_run):
    print(" ".join(str(item) for item in command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    checkpoints = discover_runs(args)
    if not checkpoints:
        raise SystemExit("No runs found. Pass --run or --run_root.")

    summary = []
    for checkpoint in checkpoints:
        run_dir = checkpoint.parent
        manifest = validate_manifest(run_dir)
        output_csv = run_dir / "test500_eval.csv"
        eval_metrics = run_dir / "test500_eval_manifest.json"
        text_metrics = run_dir / "test500_metrics.json"

        if args.skip_existing and output_csv.is_file() and text_metrics.is_file():
            print(f"skip existing: {run_dir}")
        else:
            eval_command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--split", "test",
                "--checkpoint", str(checkpoint),
                "--sample_count", "500",
                "--sample_mode", "sequential",
                "--offset", "0",
                "--batch_size", str(args.batch_size),
                "--num_beams", str(args.num_beams),
                "--max_new_tokens", str(args.max_new_tokens),
                "--experiment_variant", "auto",
                "--device", args.device,
                "--dtype", args.dtype,
                "--output_csv", str(output_csv),
                "--output_metrics_json", str(eval_metrics),
            ]
            run_command(eval_command, args.dry_run)
            metrics_command = [
                sys.executable,
                str(METRICS_SCRIPT),
                "--input_csv", str(output_csv),
                "--label", run_dir.name,
                "--output_json", str(text_metrics),
            ]
            run_command(metrics_command, args.dry_run)

        if args.dry_run:
            continue
        eval_info = json.loads(eval_metrics.read_text(encoding="utf-8"))
        metric_rows = json.loads(text_metrics.read_text(encoding="utf-8"))
        if eval_info["evaluated_samples"] != 500 or eval_info["available_samples"] != 500:
            raise RuntimeError(f"Test500 was incomplete: {run_dir}")
        metric_info = metric_rows[0]
        summary.append(
            {
                "run": run_dir.name,
                "checkpoint": str(checkpoint),
                "variant": eval_info["experiment_variant"],
                "seed": eval_info.get("checkpoint_seed"),
                "top_k_ratio": eval_info.get("top_k_ratio"),
                "average_similarity": metric_info["average_similarity"],
                "bleu4": metric_info["bleu4"],
                "rouge_l": metric_info["rouge_l"],
                "chrf": metric_info["chrf"],
                "wer": metric_info["wer"],
                "standard_cer": metric_info["standard_cer"],
                "strict_checkpoint_load": eval_info["strict_checkpoint_load"],
                "protocol_variant": manifest.get("experiment_variant"),
            }
        )

    if args.dry_run:
        print(f"dry-run complete: {len(checkpoints)} checkpoint(s)")
        return

    summary_json = Path(args.summary_json)
    summary_csv = Path(args.summary_csv)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"saved summary: {summary_json}")
    print(f"saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
