import argparse
import csv
import json
import re
from pathlib import Path


TIME_PAT = re.compile(r"(今天|明天|昨天|早上|晚上|周末|这周|下周|之前|以后|刚才|上午|下午|一会|分钟|小时)")
NEG_PAT = re.compile(r"(不|没|不能|不要|别|未|无)")
NUM_PAT = re.compile(r"([0-9０-９一二三四五六七八九十百千万两几多])")
QUESTION_PAT = re.compile(r"[？?]$")


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify(reference: str):
    text = reference.strip()
    return {
        "all": True,
        "question": bool(QUESTION_PAT.search(text)),
        "negation": bool(NEG_PAT.search(text)),
        "temporal": bool(TIME_PAT.search(text)),
        "number": bool(NUM_PAT.search(text)),
        "long": len(text) >= 12,
        "short": len(text) < 12,
    }


def summarize(rows):
    categories = ["all", "question", "negation", "temporal", "number", "long", "short"]
    stats = {name: {"count": 0, "avg_similarity": 0.0} for name in categories}

    sortable_rows = []
    for row in rows:
        sim = float(row.get("similarity", 0.0))
        reference = row["reference"]
        prediction = row["prediction"]
        flags = classify(reference)
        for name, enabled in flags.items():
            if enabled:
                stats[name]["count"] += 1
                stats[name]["avg_similarity"] += sim
        sortable_rows.append(
            {
                "similarity": sim,
                "reference": reference,
                "prediction": prediction,
            }
        )

    for name in categories:
        count = stats[name]["count"]
        if count:
            stats[name]["avg_similarity"] /= count

    sortable_rows.sort(key=lambda x: x["similarity"], reverse=True)
    best_examples = sortable_rows[:5]
    worst_examples = list(reversed(sortable_rows[-5:]))
    return stats, best_examples, worst_examples


def build_markdown(results):
    lines = ["# CE-CSL Eval Error Analysis", ""]
    for result in results:
        lines.append(f"## {result['label']}")
        lines.append("")
        lines.append("| Category | Count | Avg Similarity |")
        lines.append("| --- | ---: | ---: |")
        for name, values in result["stats"].items():
            lines.append(f"| {name} | {values['count']} | {values['avg_similarity']:.4f} |")
        lines.append("")
        lines.append("### Best Examples")
        for item in result["best_examples"]:
            lines.append(f"- sim={item['similarity']:.4f} | ref={item['reference']} | pred={item['prediction']}")
        lines.append("")
        lines.append("### Worst Examples")
        for item in result["worst_examples"]:
            lines.append(f"- sim={item['similarity']:.4f} | ref={item['reference']} | pred={item['prediction']}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze CE-CSL eval CSV files by sentence category.")
    parser.add_argument("--input_csv", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.input_csv]
    labels = args.label if args.label else [p.stem for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("--label count must match --input_csv count when provided.")

    results = []
    for label, csv_path in zip(labels, csv_paths):
        rows = load_rows(csv_path)
        stats, best_examples, worst_examples = summarize(rows)
        results.append(
            {
                "label": label,
                "csv_path": str(csv_path),
                "stats": stats,
                "best_examples": best_examples,
                "worst_examples": worst_examples,
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
