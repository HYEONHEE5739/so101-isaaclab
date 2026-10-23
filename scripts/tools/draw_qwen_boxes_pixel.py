#!/usr/bin/env python3
"""Overlay ABSOLUTE PIXEL bounding boxes predicted by compare_qwen_q1_q2_pixel.py.

The inference script must ask Qwen for image pixel coordinates, not 0..1000
normalized coordinates. This script intentionally REJECTS old normalized JSON.

Usage (from repository root):
    python scripts/tools/draw_qwen_boxes_pixel.py \
      --results outputs/vlm_diagnostics/qwen_q1q2_pixels/results.json \
      --images-root outputs/vlm_diagnostics/first_frames_test \
      --output outputs/vlm_diagnostics/qwen_q1q2_boxes_pixels
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BBOX_RE = re.compile(
    r"\bbbox\s*[:=]\s*(\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]|null)",
    re.IGNORECASE,
)


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_pixel_bbox(answer: str, width: int, height: int) -> tuple[list[int] | None, str]:
    """Return image pixel corners if and only if valid; NO scaling/clamping."""
    found = BBOX_RE.search(answer or "")
    if not found:
        return None, "MISSING_BBOX"
    if found.group(1).lower() == "null":
        return None, "NULL_BBOX"
    values = [float(found.group(i)) for i in range(2, 6)]
    x1, y1, x2, y2 = [int(round(v)) for v in values]
    # Coordinates use the actual raster pixel indices (0 to width-1/height-1).
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        return None, f"OUT_OF_IMAGE_PIXEL_RANGE ({width}x{height}): {values}"
    return [x1, y1, x2, y2], "BOX_DRAWN"


def annotate(src: Path, output: Path, answer: str, label: str) -> tuple[list[int] | None, str, list[int]]:
    with Image.open(src) as original:
        image = original.convert("RGB")
    box, status = parse_pixel_bbox(answer, image.width, image.height)
    painter = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if box is not None:
        painter.rectangle(box, outline=(255, 25, 25), width=3)
        title = f"{label}: {box} px"
        anchor = (min(box[0], max(0, image.width - 210)), max(0, box[1] - 17))
        painter.text(anchor, title, fill=(255, 255, 255), stroke_width=2,
                     stroke_fill=(0, 0, 0), font=font)
    else:
        painter.text((6, 6), f"{label}: {status}", fill=(255, 255, 255),
                     stroke_width=2, stroke_fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return box, status, [image.height, image.width]


def report_html(out: Path, rows: list[dict]) -> None:
    fragments = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Qwen Q1/Q2 pixel bbox evaluation</title>
<style>
body{font:15px system-ui,sans-serif;margin:24px auto;max-width:1250px;padding:0 16px;background:#f8f9fa}
article{background:white;border:1px solid #dedede;border-radius:12px;margin:17px 0;padding:13px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
img{width:100%;max-height:400px;object-fit:contain;background:#eee}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:10px;border-radius:6px}
.info{color:#656b75}
@media(max-width:700px){.pair{grid-template-columns:1fr}}
</style><h1>Qwen Q1/Q2 — pixel Bounding Boxes</h1>
<p>빨간 박스 = Qwen이 각 이미지에 대해 직접 예측한 픽셀 좌표. Q1=원본 해상도, Q2=256×256.
좌표가 픽셀 범위를 벗어나면 임의 보정하지 않고 오류로 표시합니다.</p>"""]
    for row in rows:
        fragments.append(f'<article><h2>Episode {row["episode_index"]} · {html.escape(str(row["target"]))}</h2><div class="pair">')
        for key, title in [("q1", "Q1: original"), ("q2", "Q2: MiniVLA 256")]:
            r = row[key]
            rel = html.escape(row[f"{key}_annotated"], quote=True)
            fragments.append(f'<div><h3>{title}</h3><img src="{rel}">'
                             f'<p class="info">size={r["image_hw"]} · bbox_px={r["pixel_bbox"]} · '
                             f'{html.escape(r["box_status"])} · visual_tokens={r.get("visual_tokens")}</p>'
                             f'<pre>{html.escape(r.get("answer", ""))}</pre></div>')
        fragments.append('</div></article>')
    fragments.append('</html>')
    (out / "report_boxes.html").write_text("\n".join(fragments), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results_file = args.results.expanduser().resolve()
    source = args.images_root.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not results_file.is_file():
        parser.error(f"Missing results.json: {results_file}")
    if not source.is_dir():
        parser.error(f"Missing image folder: {source}")
    if out.exists():
        parser.error(f"Output exists; use a NEW directory: {out}")
    if out == source or out in source.parents or source in out.parents:
        parser.error("Output must be separate from source images")

    obj = json.loads(results_file.read_text(encoding="utf-8"))
    if obj.get("bbox_coordinate_system") != "pixel" or obj.get("bbox_coordinate_version") != 1:
        parser.error("OLD NORMALIZED RESULTS: regenerate with updated compare_qwen_q1_q2_pixel.py. "
                     "Do not interpret old 0-1000 coordinates as pixels.")
    if not obj.get("cases"):
        parser.error("No cases in results.json")

    # Validate all source file references BEFORE writing output.
    for case in obj["cases"]:
        for condition in ("q1", "q2"):
            src = (source / case[f"{condition}_path"]).resolve()
            if not src.is_relative_to(source) or not src.is_file():
                raise FileNotFoundError(src)

    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, case in enumerate(obj["cases"]):
        row = {"episode_index": case["episode_index"], "target": case["target"]}
        for condition in ("q1", "q2"):
            pred = case[condition]
            img = (source / case[f"{condition}_path"]).resolve()
            relative = Path(condition) / f'episode_{int(case["episode_index"]):04d}_{case["target"]}_{i:03d}.png'
            bbox, status, hw = annotate(img, out / relative, pred.get("answer", ""), str(case["target"]))
            row[condition] = {**pred, "image_hw": hw, "pixel_bbox": bbox, "box_status": status}
            row[f"{condition}_annotated"] = relative.as_posix()
            print(f"episode={case['episode_index']} {condition}: {status} px={bbox}")
        rows.append(row)
    atomic_json(out / "boxed_results.json", {"bbox_coordinate_system": "pixel", "cases": rows})
    report_html(out, rows)
    print(f"Saved: {out / 'report_boxes.html'}")


if __name__ == "__main__":
    main()
