from pathlib import Path


OUT = Path(r"E:\project\xiaozhi\Uni-Sign\docs\04_论文稿件\回复编辑论文内容\figures")

W, H = 1240, 1754

COLORS = {
    "ink": "#17212B",
    "muted": "#52606D",
    "line": "#66788A",
    "panel": "#F8FAFC",
    "pose": "#E8F4FF",
    "pose_edge": "#2286D2",
    "rgb": "#FFF0F3",
    "rgb_edge": "#D94C70",
    "meta": "#FFF7DF",
    "meta_edge": "#D9950B",
    "gate": "#EAF8ED",
    "gate_edge": "#29945A",
    "align": "#F0F3F6",
    "align_edge": "#8191A2",
    "out": "#F1EBFF",
    "out_edge": "#7654B5",
}


def esc(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def rect(x, y, w, h, fill, stroke, radius=16, width=2.5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>' )


def text(x, y, lines, size=24, weight=400, fill=None, anchor="middle", gap=1.18,
         family="Arial, Helvetica, sans-serif", italic=False):
    if isinstance(lines, str):
        lines = [lines]
    fill = fill or COLORS["ink"]
    style = "font-style:italic;" if italic else ""
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else int(size * gap)
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'style="{style}">' + "".join(spans) + "</text>")


def arrow(points, color, width=4, dash=None, marker="arrow"):
    path = " ".join(f"{x},{y}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" '
            f'marker-end="url(#{marker})"{dash_attr}/>' )


def title_box(x, y, w, h, title, subtitle, fill, edge, title_size=25, subtitle_size=18,
              dash=None):
    parts = [rect(x, y, w, h, fill, edge, 16, 2.5, dash)]
    title_lines = title if isinstance(title, list) else [title]
    parts.append(text(x + w / 2, y + 40, title_lines, title_size, 700, gap=1.05))
    if subtitle:
        lines = subtitle if isinstance(subtitle, list) else [subtitle]
        subtitle_y = y + 75 + max(0, len(title_lines) - 1) * int(title_size * 1.05)
        parts.append(text(x + w / 2, subtitle_y, lines, subtitle_size, 400, COLORS["muted"]))
    return "\n".join(parts)


def skeleton_icon(x, y, scale=1.0):
    c = COLORS["pose_edge"]
    sw = 3 * scale
    parts = [f'<circle cx="{x}" cy="{y}" r="7" fill="none" stroke="{c}" stroke-width="{sw}"/>']
    segs = [
        (x, y + 8, x, y + 37),
        (x, y + 15, x - 22, y + 30),
        (x, y + 15, x + 22, y + 30),
        (x, y + 37, x - 17, y + 68),
        (x, y + 37, x + 17, y + 68),
    ]
    for x1, y1, x2, y2 in segs:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}" stroke-linecap="round"/>')
    return "\n".join(parts)


def crop_cards(x, y):
    parts = []
    for i, label in enumerate(["L", "R", "L", "R"]):
        px = x + i * 47
        parts.append(rect(px, y + (i % 2) * 7, 38, 54, "#FFFFFF", COLORS["rgb_edge"], 5, 2))
        parts.append(text(px + 19, y + 35 + (i % 2) * 7, label, 18, 700, COLORS["rgb_edge"]))
    return "\n".join(parts)


def build_svg():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append('<defs>')
    for name, color in [("arrow", COLORS["line"]), ("poseArrow", COLORS["pose_edge"]),
                        ("rgbArrow", COLORS["rgb_edge"]), ("metaArrow", COLORS["meta_edge"]),
                        ("gateArrow", COLORS["gate_edge"]), ("outArrow", COLORS["out_edge"])]:
        s.append(f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{color}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    s.append(text(W / 2, 54, "Reliability-Aware Sparse RGB Support for Pose-Based SLT", 35, 700))
    s.append(text(W / 2, 91, "Evaluated Uni-Sign adaptation on CE-CSL", 20, 400, COLORS["muted"]))

    # Panel A
    s.append(rect(24, 120, 1192, 1060, COLORS["panel"], "#4B84B4", 22, 3))
    s.append(text(50, 163, "(a) Overall architecture and data flow", 27, 700, anchor="start"))

    # Column headers
    headers = [
        (48, 190, 342, "1  Input and sparse selection", COLORS["pose"]),
        (444, 190, 342, "2  Dual-stream representation", COLORS["rgb"]),
        (840, 190, 352, "3  Gate and translation", COLORS["gate"]),
    ]
    for x, y, w, label, fill in headers:
        s.append(rect(x, y, w, 52, fill, COLORS["line"], 12, 1.8))
        s.append(text(x + w / 2, y + 34, label, 21, 700))

    # Stage 1
    s.append(title_box(60, 270, 318, 150, ["Full-timeline", "pose keypoints"],
                       ["RTMLib body, face, and both hands", "Pose scores retained for every frame"],
                       COLORS["pose"], COLORS["pose_edge"], 22, 16))

    s.append(title_box(60, 452, 318, 135, "Candidate hand crops",
                       "Left and right hands from the same video",
                       COLORS["rgb"], COLORS["rgb_edge"], 22, 16))

    s.append(title_box(60, 625, 318, 150, ["Pose-guided", "hand-frame selection"],
                       ["Rank eligible frames separately per hand", "Keep sparse high-confidence crops"],
                       "#FFF8FB", COLORS["rgb_edge"], 21, 16))
    s.append(arrow([(378, 345), (420, 345), (420, 665), (378, 665)], COLORS["pose_edge"], 4, marker="poseArrow"))
    s.append(arrow([(219, 587), (219, 625)], COLORS["rgb_edge"], 4, marker="rgbArrow"))

    s.append(title_box(60, 815, 318, 120, ["Selection metadata", "(optional)"],
                       ["Fingertip confidence", "selection density and hand presence"],
                       COLORS["meta"], COLORS["meta_edge"], 19, 14, "8 6"))
    s.append(arrow([(219, 775), (219, 815)], COLORS["meta_edge"], 4, "8 6", "metaArrow"))

    s.append(rect(60, 978, 318, 130, "#FFFFFF", "#B8C4CE", 14, 2))
    s.append(text(219, 1018, ["Key protocol fact", "Pose keeps the complete timeline;", "RGB processes selected hand frames only."],
                  18, 700, COLORS["ink"]))

    # Stage 2
    s.append(title_box(456, 270, 318, 150, "Pose encoder",
                       ["Inherited Uni-Sign pose stream", "Body, face, and both hands"],
                       COLORS["pose"], COLORS["pose_edge"], 25, 18))
    s.append(text(615, 397, "Pose features", 18, 700, COLORS["pose_edge"]))
    s.append(arrow([(378, 345), (456, 345)], COLORS["pose_edge"], 4, marker="poseArrow"))

    s.append(title_box(456, 486, 318, 145, "RGB appearance encoder",
                       ["EfficientNet-B0", "Selected hand crops only"],
                       COLORS["rgb"], COLORS["rgb_edge"], 24, 18))
    s.append(text(615, 608, "Appearance features", 18, 700, COLORS["rgb_edge"]))
    s.append(arrow([(378, 700), (420, 700), (420, 558), (456, 558)], COLORS["rgb_edge"], 4, marker="rgbArrow"))

    s.append(title_box(456, 690, 318, 170, "Pose-guided RGB alignment",
                       ["Align sparse appearance cues", "to selected hand nodes"],
                       COLORS["align"], COLORS["align_edge"], 24, 18))
    s.append(text(615, 833, "Aligned RGB–pose features", 18, 700, COLORS["rgb_edge"]))
    s.append(arrow([(570, 420), (570, 690)], COLORS["pose_edge"], 4, marker="poseArrow"))
    s.append(arrow([(660, 631), (660, 690)], COLORS["rgb_edge"], 4, marker="rgbArrow"))

    s.append(rect(456, 915, 318, 193, "#FFFFFF", "#B8C4CE", 14, 2))
    s.append(text(615, 954, "Inherited and evaluated components", 19, 700))
    s.append(text(480, 992,
                  ["• Pose encoding: inherited Uni-Sign", "• RGB support: sparse hand crops", "• Alignment: selected hand nodes", "• No raw RGB enters the gate"],
                  17, 400, COLORS["ink"], anchor="start", gap=1.55))

    # Stage 3
    s.append(title_box(852, 300, 328, 245, ["Node-wise pose-preserving", "reliability gate"],
                       ["Inputs:", "pose feature", "aligned RGB–pose feature", "optional selection metadata"],
                       COLORS["gate"], COLORS["gate_edge"], 20, 16))
    s.append(text(1016, 515, "Update selected hand nodes", 18, 700, COLORS["gate_edge"]))
    # Pose direct path into gate
    s.append(arrow([(774, 335), (820, 335), (820, 370), (852, 370)], COLORS["pose_edge"], 4, marker="poseArrow"))
    # Aligned feature path
    s.append(arrow([(774, 780), (806, 780), (806, 445), (852, 445)], COLORS["rgb_edge"], 4, marker="rgbArrow"))
    # Metadata path
    s.append(arrow([(378, 875), (810, 875), (810, 500), (852, 500)],
                   COLORS["meta_edge"], 3.5, "9 7", "metaArrow"))

    s.append(rect(852, 585, 328, 105, COLORS["pose"], COLORS["pose_edge"], 14, 2.2, "8 6"))
    s.append(text(1016, 619, ["Pose-preserving bypass", "Unselected pose nodes", "remain unchanged"], 18, 700, COLORS["pose_edge"], gap=1.05))

    s.append(title_box(852, 730, 328, 105, "Updated pose sequence",
                       "Selected nodes updated; original order restored",
                       "#F4FBF6", COLORS["gate_edge"], 21, 14))
    s.append(arrow([(1016, 545), (1016, 730)], COLORS["gate_edge"], 4, marker="gateArrow"))
    s.append(arrow([(1016, 690), (1016, 730)], COLORS["pose_edge"], 4, "8 6", "poseArrow"))

    s.append(title_box(852, 875, 328, 185, ["Sequence encoding", "and translation"],
                       ["Part-wise pooling and visual projection", "mT5-base encoder–decoder"],
                       COLORS["out"], COLORS["out_edge"], 21, 16))
    s.append(arrow([(1016, 835), (1016, 875)], COLORS["out_edge"], 4, marker="outArrow"))
    s.append(rect(890, 1080, 252, 66, "#FFFFFF", COLORS["out_edge"], 12, 2.2))
    s.append(text(1016, 1107, ["Generated Chinese sentence", "我想去学校学习"], 18, 700, COLORS["ink"]))
    s.append(arrow([(1016, 1060), (1016, 1080)], COLORS["out_edge"], 4, marker="outArrow"))

    # Panel B
    s.append(rect(24, 1205, 1192, 510, "#FBFCFD", "#4B84B4", 22, 3))
    s.append(text(50, 1248, "(b) Core pose-preserving reliability gate", 27, 700, anchor="start"))

    # Inputs
    s.append(title_box(58, 1290, 260, 82, "Pose feature", "from pose encoder", COLORS["pose"], COLORS["pose_edge"], 21, 15))
    s.append(title_box(58, 1390, 260, 82, ["Aligned RGB–pose", "feature"], "", COLORS["rgb"], COLORS["rgb_edge"], 18, 13))
    s.append(title_box(58, 1490, 260, 92, ["Optional metadata", "embedding"], "", COLORS["meta"], COLORS["meta_edge"], 18, 13, "8 6"))

    # Predictor
    s.append(title_box(385, 1325, 265, 220, "Gate predictor",
                       ["Concatenate available inputs", "Pointwise layers", "Sigmoid"],
                       COLORS["gate"], COLORS["gate_edge"], 23, 18))
    s.append(arrow([(318, 1331), (385, 1370)], COLORS["pose_edge"], 4, marker="poseArrow"))
    s.append(arrow([(318, 1431), (385, 1431)], COLORS["rgb_edge"], 4, marker="rgbArrow"))
    s.append(arrow([(318, 1536), (385, 1492)], COLORS["meta_edge"], 3.5, "8 6", "metaArrow"))

    s.append(rect(692, 1375, 145, 78, "#F3FBF5", COLORS["gate_edge"], 39, 2.5))
    s.append(text(764, 1406, ["Learned", "node-wise weight"], 17, 700, COLORS["gate_edge"]))
    s.append(arrow([(650, 1435), (692, 1414)], COLORS["gate_edge"], 4, marker="gateArrow"))

    # Blend and outputs
    s.append(title_box(880, 1295, 286, 215, "Pose-preserving blend",
                       ["Weight aligned RGB–pose path", "Complement with pose path"],
                       "#F3FBF5", COLORS["gate_edge"], 22, 17))
    s.append(arrow([(837, 1414), (880, 1414)], COLORS["gate_edge"], 4, marker="gateArrow"))
    s.append(rect(880, 1535, 286, 82, COLORS["out"], COLORS["out_edge"], 14, 2.2))
    s.append(text(1023, 1567, ["Updated selected", "hand-node feature"], 19, 700))
    s.append(arrow([(1023, 1510), (1023, 1535)], COLORS["out_edge"], 4, marker="outArrow"))

    s.append(rect(385, 1592, 452, 76, COLORS["pose"], COLORS["pose_edge"], 14, 2.2, "8 6"))
    s.append(text(611, 1622, ["Unselected nodes bypass the RGB-support path", "and retain their original pose features"], 17, 700, COLORS["pose_edge"]))
    s.append(arrow([(58, 1331), (40, 1331), (40, 1630), (385, 1630)], COLORS["pose_edge"], 3.5, "8 6", "poseArrow"))

    s.append(text(W / 2, 1702,
                  "Evaluated variants: Pose-only   |   RGB gate   |   RGB + metadata gate",
                  18, 700, COLORS["muted"]))
    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / "framework_a4_ieee_core_v5.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: A4 portrait; margin: 0; }}
    html, body {{ margin:0; padding:0; width:210mm; height:297mm; background:white; }}
    svg {{ display:block; width:210mm; height:297mm; }}
    </style></head><body>{build_svg()}</body></html>'''
    (OUT / "framework_a4_ieee_core_v5.html").write_text(html, encoding="utf-8")
