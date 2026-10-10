from __future__ import annotations

import difflib
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


def normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^0-9a-z%+\-.]+", "", text)
    return text


def extract_words(path: Path):
    tokens = []
    locations = []
    page_sizes = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_sizes.append((float(page.width), float(page.height)))
            words = page.extract_words(
                x_tolerance=1.5,
                y_tolerance=2.0,
                keep_blank_chars=False,
                use_text_flow=True,
            )
            for word in words:
                token = normalize(word["text"])
                if not token:
                    continue
                tokens.append(token)
                locations.append(
                    {
                        "page": page_index,
                        "x0": float(word["x0"]),
                        "x1": float(word["x1"]),
                        "top": float(word["top"]),
                        "bottom": float(word["bottom"]),
                    }
                )
    return tokens, locations, page_sizes


def changed_indices(old_tokens, new_tokens):
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    unchanged = set()
    for block in matcher.get_matching_blocks():
        unchanged.update(range(block.b, block.b + block.size))
    return set(range(len(new_tokens))) - unchanged


def group_rectangles(locations, changed):
    by_page_line = defaultdict(list)
    for index in sorted(changed):
        loc = locations[index]
        line_key = round(loc["top"] / 2.5) * 2.5
        by_page_line[(loc["page"], line_key)].append(loc)

    page_rectangles = defaultdict(list)
    for (page_index, _), words in by_page_line.items():
        words.sort(key=lambda item: item["x0"])
        current = None
        for word in words:
            if current is None:
                current = dict(word)
                continue
            same_band = abs(word["top"] - current["top"]) <= 3.0
            close = word["x0"] - current["x1"] <= 10.0
            if same_band and close:
                current["x1"] = max(current["x1"], word["x1"])
                current["top"] = min(current["top"], word["top"])
                current["bottom"] = max(current["bottom"], word["bottom"])
            else:
                page_rectangles[page_index].append(current)
                current = dict(word)
        if current is not None:
            page_rectangles[page_index].append(current)
    return page_rectangles


def build_overlay(width, height, rectangles):
    stream = io.BytesIO()
    layer = canvas.Canvas(stream, pagesize=(width, height))
    layer.setFillColorRGB(1.0, 0.92, 0.0)
    layer.setFillAlpha(0.42)
    for rect in rectangles:
        x = max(0.0, rect["x0"] - 0.8)
        y = max(0.0, height - rect["bottom"] - 0.6)
        w = min(width - x, rect["x1"] - rect["x0"] + 1.6)
        h = min(height - y, rect["bottom"] - rect["top"] + 1.2)
        layer.rect(x, y, w, h, stroke=0, fill=1)
    layer.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def main():
    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    old_tokens, _, _ = extract_words(old_path)
    new_tokens, locations, page_sizes = extract_words(new_path)
    changed = changed_indices(old_tokens, new_tokens)
    rectangles = group_rectangles(locations, changed)

    reader = PdfReader(new_path)
    writer = PdfWriter()
    for page_index, page in enumerate(reader.pages):
        width, height = page_sizes[page_index]
        page.merge_page(build_overlay(width, height, rectangles.get(page_index, [])), over=True)
        writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        writer.write(stream)

    print(f"baseline tokens: {len(old_tokens)}")
    print(f"revised tokens: {len(new_tokens)}")
    print(f"highlighted tokens: {len(changed)}")
    print(f"highlight rectangles: {sum(len(v) for v in rectangles.values())}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
