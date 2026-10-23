#!/usr/bin/env python3
"""Export the first frame of each episode from a LOCAL LeRobot dataset.

Q1: decoded original-resolution RGB PNG.
Q2: the SAME MiniVLA letterbox() as the current project (256x256 by default).

Read-only with respect to the input dataset. No Hugging Face download/upload.
Run from the so101-isaaclab repository root with PYTHONPATH=source/soarm101_lab.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rgb_uint8(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "convert") and not isinstance(value, np.ndarray):
        value = np.asarray(value.convert("RGB"))
    a = np.asarray(value)

    if a.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got {a.shape}")
    if a.shape[-1] != 3 and a.shape[0] == 3:
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC or CHW, got {a.shape}")

    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError("Image has nonnumeric or nonfinite values")
    lo, hi = float(a.min()), float(a.max())
    if lo < -1e-5 or hi > 255 + 1e-5:
        raise ValueError(f"Unexpected image pixel range [{lo}, {hi}]")
    # LeRobot decodes video frames to float32 CHW in [0,1] by default.
    if hi <= 1 + 1e-5:
        a = a * 255.0
    return np.ascontiguousarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))


def _episode_starts(hf_dataset):
    """Locate first actual frame per episode using native episode/frame columns."""
    cols = set(hf_dataset.column_names)
    if not {"episode_index", "frame_index"} <= cols:
        raise ValueError("Missing episode_index/frame_index in LeRobot Parquet data")
    episodes = hf_dataset["episode_index"]
    frames = hf_dataset["frame_index"]
    if len(episodes) != len(frames):
        raise ValueError("Episode/frame columns have different lengths")

    first = {}
    for row, (ep, frame) in enumerate(zip(episodes, frames)):
        ep, frame = int(ep), int(frame)
        if ep not in first or frame < first[ep][0]:
            first[ep] = (frame, row)
    return [(ep, fr, row) for ep, (fr, row) in sorted(first.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Local LeRobot dataset root (contains meta/info.json)")
    parser.add_argument("--output", required=True, type=Path,
                        help="NEW output directory, separate from the dataset")
    parser.add_argument("--camera", default="side",
                        help="Camera name such as side/wrist; 'all' for every RGB camera")
    parser.add_argument("--repo-id", default="local/first-frame-export",
                        help="LeRobotDataset identifier (local-only when --dataset is populated)")
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav"], default="torchcodec")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="0 = all episodes; positive integer for a smoke test")
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        parser.error(f"Not a LeRobot dataset root: {root} (meta/info.json missing)")
    if output == root or root in output.parents or output in root.parents:
        parser.error("Output must be separate from input dataset, not its parent/child")
    if output.exists():
        parser.error(f"Output already exists; choose a NEW directory: {output}")
    if args.max_episodes < 0:
        parser.error("--max-episodes must be >= 0")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    cameras = sorted(
        k.removeprefix("observation.images.")
        for k, spec in features.items()
        if k.startswith("observation.images.")
        and spec.get("dtype") in ("video", "image")
    )
    if not cameras:
        parser.error("No observation.images.* RGB camera features in meta/info.json")
    chosen = cameras if args.camera == "all" else [args.camera]
    unknown = sorted(set(chosen) - set(cameras))
    if unknown:
        parser.error(f"Camera(s) {unknown} not available. Detected: {cameras}")

    # Import the existing project implementation; never copy its preprocessing math.
    from soarm101_lab.learning.minivla.backbone import letterbox
    from soarm101_lab.learning.minivla.config import MiniVLAConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    cfg = MiniVLAConfig()
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        download_videos=False,
        video_backend=args.video_backend,
    )
    starts = _episode_starts(ds.hf_dataset)
    if not starts:
        raise ValueError("No episode frames found")
    if args.max_episodes:
        starts = starts[:args.max_episodes]

    # Do not create output until dataset/camera/index validation has succeeded.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    total = len(starts)
    for n, (episode_index, frame_index, row_index) in enumerate(starts, start=1):
        item = ds[row_index]
        for camera in chosen:
            key = f"observation.images.{camera}"
            if key not in item:
                raise KeyError(f"{key} not returned by LeRobotDataset[{row_index}]")
            rgb = _rgb_uint8(item[key])
            original_hw = [int(rgb.shape[0]), int(rgb.shape[1])]
            processed = letterbox(rgb, cfg.image_width, cfg.image_height)

            name = f"episode_{episode_index:06d}.png"
            q1_rel = Path("q1_original") / camera / name
            q2_rel = Path("q2_minivla") / camera / name
            (output / q1_rel).parent.mkdir(parents=True, exist_ok=True)
            (output / q2_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb, "RGB").save(output / q1_rel)
            processed.save(output / q2_rel)
            records.append({
                "episode_index": episode_index,
                "frame_index": frame_index,
                "dataset_row": row_index,
                "camera": camera,
                "original_hw": original_hw,
                "q2_hw": [cfg.image_height, cfg.image_width],
                "q1_path": q1_rel.as_posix(),
                "q2_path": q2_rel.as_posix(),
            })
        print(f"[{n:>3}/{total}] episode={episode_index} frame={frame_index} OK", flush=True)

    report = {
        "input_dataset": str(root),
        "read_only_source": True,
        "camera_features": chosen,
        "episode_count": len(starts),
        "image_count_per_condition": len(records),
        "preprocessing": "MiniVLA backbone.letterbox (LANCZOS, black padding)",
        "minivla_config": {
            "image_height": cfg.image_height,
            "image_width": cfg.image_width,
            "cameras": list(cfg.cameras),
        },
        "note": "Q1 is decoded dataset resolution, not lossless pre-encoding camera data.",
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone: {len(starts)} episodes, {len(records)} camera images per condition")
    print(f"Q1: {output / 'q1_original'}")
    print(f"Q2: {output / 'q2_minivla'}")
    print(f"Manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
