#!/usr/bin/env python3
"""Qwen3-VL first-frame comparison: Q1 original vs Q2 MiniVLA 256 letterbox.

Uses ONE pretrained NF4 Qwen3-VL model and two image processors with
different pixel budgets. No policy/action inference and no training.
Run from repo root with PYTHONPATH=source/soarm101_lab.
"""
from __future__ import annotations
import argparse
import html
import json
import os
import time
from pathlib import Path


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def make_html(folder, source, cases):
    lines = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Qwen Q1-Q2 comparison</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1250px;margin:25px auto;padding:0 15px;background:#fafafa}
article{background:#fff;border:1px solid #ddd;border-radius:12px;padding:14px;margin:18px 0}
section{display:grid;grid-template-columns:1fr 1fr;gap:14px}
img{max-width:100%;max-height:380px;object-fit:contain;border:1px solid #ddd}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;border-radius:6px;padding:10px}
small{color:#656565}
@media(max-width:650px){section{grid-template-columns:1fr}}
</style><h1>Qwen3-VL Q1 vs Q2</h1>
<p>동일 모델·동일 대상 물체의 Q1/Q2 해상도별 출력. 좌표는 각 이미지의 실제 픽셀 단위입니다.
Q1/Q2의 픽셀 치수가 달라 질문 속 크기 수치는 서로 다릅니다. 답변의 정확도를 직접 확인하세요.</p>"""]
    for case in cases:
        lines.append(f'<article><h2>Episode {case["episode_index"]} / {html.escape(case["target"])}</h2><section>')
        for key, title in (("q1", "Q1: original"), ("q2", "Q2: MiniVLA 256")):
            result = case.get(key, {})
            image_path = source / case[f"{key}_path"]
            relative = Path(os.path.relpath(image_path, folder)).as_posix()
            lines.append(f'<div><h3>{title}</h3>'
                         f'<img src="{html.escape(relative, quote=True)}">'
                         f'<p><small>PNG {html.escape(str(result.get("png_hw")))} · '
                         f'processor {html.escape(str(result.get("processed_hw")))} · '
                         f'visual tokens {result.get("visual_tokens", "?")}</small></p>'
                         f'<pre>{html.escape(result.get("answer", "NOT RUN"))}</pre></div>')
        lines.append("</section></article>")
    (folder / "report.html").write_text("\n".join(lines)+"</html>", encoding="utf-8")


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--images-root", required=True, type=Path,
                   help="Folder containing manifest.json, q1_original/, q2_minivla/")
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--camera", default="side_cam")
    a.add_argument("--targets", default="red", help="red or red,blue,green")
    a.add_argument("--max-episodes", type=int, default=4)
    a.add_argument("--q1-max-pixels", type=int, default=640*480)
    a.add_argument("--max-new-tokens", type=int, default=100)
    args = a.parse_args()
    source = args.images_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "manifest.json").is_file():
        a.error(f"manifest.json not found in: {source}")
    if args.max_episodes < 1 or args.q1_max_pixels <= 65536 or args.max_new_tokens < 1:
        a.error("Require max-episodes>=1, q1-max-pixels>65536, max-new-tokens>=1")
    if source == output or source in output.parents or output in source.parents:
        a.error("Choose an output folder separate from the extracted images")
    if output.exists():
        a.error("Output directory already exists. Choose a new folder name.")

    data = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    records = [r for r in data["records"] if r["camera"] == args.camera]
    unique = {}
    for r in records:
        unique.setdefault(int(r["episode_index"]), r)
    rows = list(unique.values())[:args.max_episodes]
    if not rows:
        a.error(f"No images found for camera '{args.camera}'.")
    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    if not targets or any(not t.isalpha() for t in targets):
        a.error("Use --targets red or --targets red,blue,green")
    for r in rows:
        for key in ("q1_path","q2_path"):
            file = (source / r[key]).resolve()
            if not file.is_relative_to(source) or not file.is_file():
                raise FileNotFoundError(file)

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from soarm101_lab.learning.minivla.config import MiniVLAConfig
    from soarm101_lab.learning.minivla.backbone import QwenBackbone

    cfg = MiniVLAConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for project Qwen NF4 model")
    print("Loading Qwen3-VL once in NF4 ...", flush=True)
    p2, model = QwenBackbone.load_pretrained(cfg, "cuda:0")
    p1 = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision,
                                       min_pixels=32*32, max_pixels=args.q1_max_pixels)
    model.eval()

    result = {
        "model_id": cfg.model_id, "model_revision": cfg.revision,
        "q1_max_pixels": args.q1_max_pixels,
        "q2_fixed_pixels": cfg.image_width*cfg.image_height,
        "bbox_coordinate_system": "pixel",
        "bbox_coordinate_version": 1,
        "warning": "Pixel coordinates are Qwen predictions, not ground truth. Manually inspect the overlays.",
        "cases": [],
    }
    output.mkdir(parents=True, exist_ok=False)

    def ask(img_path, processor, target):
        with Image.open(img_path) as file:
            image = file.convert("RGB")
        width, height = image.size
        question = (
            f"Find the {target} cube or block in this robot camera image. "
            f"The image is exactly {width} pixels wide and {height} pixels high. "
            "Return the bounding box in ABSOLUTE IMAGE PIXEL COORDINATES, not normalized coordinates. "
            f"The top-left image corner is (0,0), x ranges from 0 to {width-1}, "
            f"and y ranges from 0 to {height-1}. "
            "The box must tightly enclose ONLY the requested object. "
            "Answer exactly as: visible: yes/no/unsure; "
            "bbox: [x1,y1,x2,y2] with pixel integer coordinates, "
            "or bbox: null when not visible or uncertain. "
            "Do not invent an object. Do not use a 0-to-1000 coordinate system."
        )
        prompt = processor.apply_chat_template(
            [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}],
            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        grid = inputs["image_grid_thw"].tolist()
        patch = int(processor.image_processor.patch_size)
        merge = int(model.config.vision_config.spatial_merge_size)
        processed = [[int(g[1])*patch,int(g[2])*patch] for g in grid]
        ntokens = sum(int(g[0])*(int(g[1])//merge)*(int(g[2])//merge) for g in grid)
        inputs = inputs.to("cuda:0")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=args.max_new_tokens)
        torch.cuda.synchronize()
        answer = processor.batch_decode(ids[:,inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip()
        return {"answer": answer, "question": question, "bbox_coordinate_system": "pixel",
                "png_hw": [image.height, image.width],
                "processed_hw": processed, "image_grid_thw": grid,
                "visual_tokens": ntokens, "elapsed_s": round(time.perf_counter()-start,2)}

    for idx, r in enumerate(rows,1):
        for target in targets:
            entry = {"episode_index":r["episode_index"],"frame_index":r["frame_index"],
                     "target":target,"camera":r["camera"],
                     "q1_path":r["q1_path"],"q2_path":r["q2_path"]}
            for key, processor in (("q1",p1),("q2",p2)):
                print(f"[{idx}/{len(rows)}] ep {r['episode_index']} {target} {key} ...", flush=True)
                entry[key] = ask(source / r[f"{key}_path"], processor, target)
                print(f"  {entry[key]['answer'][:170]} | tokens={entry[key]['visual_tokens']}",
                      flush=True)
            entry["resolution_is_different"] = (
                entry["q1"]["processed_hw"] != entry["q2"]["processed_hw"])
            if not entry["resolution_is_different"]:
                print("WARNING: Q1 and Q2 entered Qwen at the SAME resolution!", flush=True)
            result["cases"].append(entry)
            atomic_json(output/"results.json",result)
            make_html(output,source,result["cases"])
    print(f"\nFinished. Open {output/'report.html'} and compare visual accuracy manually.")


if __name__ == "__main__":
    main()
