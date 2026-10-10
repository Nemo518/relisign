import argparse
import csv
import json
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import portalocker  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    # sacrebleu only uses portalocker when touching cache files. Our local
    # metric workflow does not rely on that path, so a tiny no-op shim is
    # sufficient and avoids adding a new dependency just for offline scoring.
    portalocker_stub = ModuleType("portalocker")

    class _DummyLock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    portalocker_stub.Lock = _DummyLock
    sys.modules["portalocker"] = portalocker_stub

from SLRT_metrics import bleu, chrf, rouge, wer_list


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unit_edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + (ref_char != pred_char),
                )
            )
        previous = current
    return previous[-1]


def corpus_character_error_rate(references, predictions):
    total_reference_characters = sum(len(reference) for reference in references)
    total_errors = sum(
        unit_edit_distance(reference, prediction)
        for reference, prediction in zip(references, predictions)
    )
    return 100.0 * total_errors / max(1, total_reference_characters)


def compute_metrics(rows):
    references = [row["reference"].strip() for row in rows]
    predictions = [row["prediction"].strip() for row in rows]
    references_char = [" ".join(list(text)) for text in references]
    predictions_char = [" ".join(list(text)) for text in predictions]

    bleu_scores = bleu(references, predictions, level="char")
    rouge_l = rouge(references, predictions, level="char")
    chrf_score = chrf(references, predictions)
    wer_scores = wer_list(references_char, predictions_char)
    standard_cer = corpus_character_error_rate(references, predictions)

    exact_match = sum(1 for ref, pred in zip(references, predictions) if ref == pred)
    avg_similarity = (
        sum(float(row.get("similarity", 0.0)) for row in rows) / len(rows) if rows else 0.0
    )

    return {
        "rows": len(rows),
        "average_similarity": avg_similarity,
        "exact_match_count": exact_match,
        "bleu1": bleu_scores["bleu1"],
        "bleu2": bleu_scores["bleu2"],
        "bleu3": bleu_scores["bleu3"],
        "bleu4": bleu_scores["bleu4"],
        "rouge_l": rouge_l,
        "chrf": chrf_score,
        "wer": wer_scores["wer"],
        "standard_cer": standard_cer,
        "wer_del_rate": wer_scores["del_rate"],
        "wer_ins_rate": wer_scores["ins_rate"],
        "wer_sub_rate": wer_scores["sub_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute text generation metrics from CE-CSL eval CSV files.")
    parser.add_argument("--input_csv", action="append", required=True, help="Input eval CSV path. Can be repeated.")
    parser.add_argument("--label", action="append", default=[], help="Optional display label matching --input_csv order.")
    parser.add_argument("--output_json", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.input_csv]
    labels = args.label if args.label else [p.stem for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("--label count must match --input_csv count when provided.")

    results = []
    for label, csv_path in zip(labels, csv_paths):
        rows = load_rows(csv_path)
        metrics = compute_metrics(rows)
        metrics["label"] = label
        metrics["csv_path"] = str(csv_path)
        results.append(metrics)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
