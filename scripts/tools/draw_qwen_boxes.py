#!/usr/bin/env python3
"""Draw Qwen-predicted bounding boxes on Q1/Q2 images from compare_qwen_q1_q2.py output.

Input:
  - results.json produced by compare_qwen_q1_q2.py
  - original extracted image folder used in that run

Output:
  - annotated q1/q2 images
  - report_boxes.html for side-by-side visual inspection

Example:
PYTHONPATH=source/soarm101_lab python scripts/tools/draw_qwen_boxes.py \
  --results outputs/vlm_diagnostics/qwen_q1q2_test/results.json \
  --images-root outputs/vlm_diagnostics/first_frames_all \
  --output outputs/vlm_diagnostics/qwen_q1q2_boxes
"""
from __future__ import annotations
import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


BBOX_RE = re.compile(
    r'bbox\s*[:=]\s*(?:\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]|null)',
    re.IGNORECASE
)


def parse_bbox(answer: str) -> Optional[list[float]]:
    if not answer:
        return None
    m = BBOX_RE.search(answer)
    if not m:
        # fallback: first [a,b,c,d] in text
        generic = re.search(
            r'\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]',
            answer
        )
        if not generic:
            return None
        vals = [float(generic.group(i)) for i in range(1, 5)]
    else:
        if m.group(1) is None:
            return None
        vals = [float(m.group(i)) for i in range(1, 5)]

    x1, y1, x2, y2 = vals
    if max(vals) > 1000 or min(vals) < 0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return vals


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm1000_to_pixels(bbox, width, height):
    x1, y1, x2, y2 = bbox
    px = [
        int(round(clamp(x1 / 1000.0, 0, 1) * width)),
        int(round(clamp(y1 / 1000.0, 0, 1) * height)),
        int(round(clamp(x2 / 1000.0, 0, 1) * width)),
        int(round(clamp(y2 / 1000.0, 0, 1) * height)),
    ]
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return px


def draw_box(image_path: Path, bbox_norm, label: str, output_path: Path):
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if bbox_norm is not None:
        px = norm1000_to_pixels(bbox_norm, image.width, image.height)
    else:
        px = None

    if px is not None:
        x1, y1, x2, y2 = px
        color = (255, 0, 0)
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color, width=1)
        text = f"{label} {bbox_norm}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x1
        ty = max(0, y1 - th - 4)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(255, 255, 255))
        draw.text((tx + 3, ty + 2), text, fill=(0, 0, 0), font=font)
        status = "BOX_DRAWN"
    else:
        text = f"{label} bbox: none/parse-failed"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([5, 5, 5 + tw + 6, 5 + th + 4], fill=(255, 245, 200))
        draw.text((8, 7), text, fill=(0, 0, 0), font=font)
        status = "NO_BOX"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return status


def build_html(output_dir: Path, cases: list[dict]):
    lines = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Qwen bounding box overlay</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 14px;background:#fafafa}
article{background:white;border:1px solid #ddd;border-radius:12px;padding:14px;margin:18px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
img{max-width:100%;max-height:420px;object-fit:contain;border:1px solid #ddd}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;border-radius:6px;padding:10px}
small{color:#666}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><h1>Qwen predicted bbox overlay</h1>
<p>빨간 박스는 Qwen 답변에서 파싱한 bbox(0~1000 정규화 좌표)입니다. 정확도는 직접 이미지와 비교해 판단하세요.</p>"""]
    for c in cases:
        lines.append(f'<article><h2>Episode {c["episode_index"]} · target={html.escape(c["target"])}</h2><div class="grid">')
        for key, title in (("q1", "Q1"), ("q2", "Q2")):
            img_rel = Path(os.path.relpath(output_dir / c[f"{key}_annotated_rel"], output_dir)).as_posix()
            lines.append(
                f'<div><h3>{title}</h3><img src="{html.escape(img_rel, quote=True)}">'
                f'<p><small>{html.escape(c[key].get("box_status", ""))} · parsed_bbox={html.escape(str(c[key].get("parsed_bbox")))}'
                f' · processed_hw={html.escape(str(c[key].get("processed_hw")))}'
                f' · visual_tokens={html.escape(str(c[key].get("visual_tokens")))}'
                f'</small></p><pre>{html.escape(c[key].get("answer",""))}</pre></div>'
            )
        lines.append("</div></article>")
    (output_dir / "report_boxes.html").write_text("\n".join(lines) + "</html>", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path, help="results.json from compare_qwen_q1_q2.py")
    ap.add_argument("--images-root", required=True, type=Path, help="original extracted image root with q1/q2 images")
    ap.add_argument("--output", required=True, type=Path, help="new output directory")
    args = ap.parse_args()

    results_path = args.results.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not results_path.is_file():
        ap.error(f"results.json not found: {results_path}")
    if not images_root.is_dir():
        ap.error(f"images root not found: {images_root}")
    if output.exists():
        ap.error("output already exists; choose a new directory")
    if output == images_root or output in images_root.parents or images_root in output.parents:
        ap.error("output must be separate from images-root")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        ap.error("No cases found in results.json")

    output.mkdir(parents=True, exist_ok=False)
    rendered = []

    for i, case in enumerate(cases):
        row = {
            "episode_index": case["episode_index"],
            "target": case["target"],
            "q1": dict(case["q1"]),
            "q2": dict(case["q2"]),
        }
        for key in ("q1", "q2"):
            answer = case.get(key, {}).get("answer", "")
            bbox = parse_bbox(answer)
            image_path = (images_root / case[f"{key}_path"]).resolve()
            if not image_path.is_relative_to(images_root) or not image_path.is_file():
                raise FileNotFoundError(f"Missing source image: {image_path}")
            out_rel = Path(key) / f'episode_{int(case["episode_index"]):04d}_{case["target"]}_{i:03d}.png'
            out_path = output / out_rel
            status = draw_box(image_path, bbox, case["target"], out_path)
            row[key]["parsed_bbox"] = bbox
            row[key]["box_status"] = status
            row[f"{key}_annotated_rel"] = str(out_rel.as_posix())
        rendered.append(row)

    atomic_json(output / "boxed_results.json", {"cases": rendered})
    build_html(output, rendered)
    print(f"Saved overlay report: {output / 'report_boxes.html'}")
    print(f"Saved structured results: {output / 'boxed_results.json'}")


if __name__ == "__main__":
    main()
