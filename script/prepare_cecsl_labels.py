import argparse
import csv
import gzip
import os
import pickle
from pathlib import Path

from paths import CE_CSL_ROOT, DATA_ROOT
DEFAULT_CECSL_ROOT = str(CE_CSL_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CE-CSL CSV annotations into Uni-Sign label files.")
    parser.add_argument(
        "--cecsl_root",
        default=os.environ.get("CE_CSL_ROOT", DEFAULT_CECSL_ROOT),
        help="Root path of the CE-CSL dataset.",
    )
    parser.add_argument(
        "--label_dir",
        default="label_cleaned",
        choices=["label", "label_cleaned"],
        help="Which CE-CSL label directory to convert.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATA_ROOT / "CE_CSL"),
        help="Output directory for converted Uni-Sign labels.",
    )
    return parser.parse_args()


def split_gloss(gloss_text):
    if gloss_text is None:
        return []
    gloss_text = str(gloss_text).strip()
    if not gloss_text or gloss_text.lower() == "nan":
        return []
    return [token.strip() for token in gloss_text.split("/") if token.strip()]


def build_sample(row, split):
    number = row["Number"].strip()
    translator = row["Translator"].strip()
    text = row["Chinese Sentences"].strip()

    gloss_source = row.get("Gloss_Clean") or row.get("Gloss") or ""
    gloss = split_gloss(gloss_source)

    return {
        "name": number,
        "text": text,
        "gloss": gloss,
        "video_path": f"{split}/{translator}/{number}.mp4",
    }


def convert_split(csv_path, split):
    samples = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = build_sample(row, split)
            samples[sample["name"]] = sample
    return samples


def dump_labels(obj, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(obj, f)


def main():
    args = parse_args()
    cecsl_root = Path(args.cecsl_root)
    label_root = cecsl_root / args.label_dir
    output_root = Path(args.output_dir)

    for split, output_name in [("train", "labels.train"), ("dev", "labels.dev"), ("test", "labels.test")]:
        csv_path = label_root / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CE-CSL annotation file: {csv_path}")

        converted = convert_split(csv_path, split)
        output_path = output_root / output_name
        dump_labels(converted, output_path)
        print(f"{split}: converted {len(converted)} samples -> {output_path}")


if __name__ == "__main__":
    main()
