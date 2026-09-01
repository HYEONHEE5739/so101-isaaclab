import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

"""
python scripts/tools/inspect_annotate_mimic.py \
    datasets/annotated_1_episode.hdf5 \
    --radius 7
   #--demo demo_0

"""
# ============================================================
# Annotation dataset search
# ============================================================

def find_annotation_datasets(group, prefix=""):
    """Find datasets related to Mimic annotation."""

    results = []

    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key

        if isinstance(value, h5py.Group):
            results.extend(
                find_annotation_datasets(
                    value,
                    path,
                )
            )

        elif isinstance(value, h5py.Dataset):
            lower = path.lower()

            if any(
                keyword in lower
                for keyword in (
                    "subtask",
                    "term",
                    "grasp",
                    "annotation",
                )
            ):
                results.append(path)

    return results


def extract_annotation_steps(dataset):
    """Extract candidate annotation frame indices."""

    data = np.asarray(dataset)

    print(
        f"    shape={data.shape}, "
        f"dtype={data.dtype}"
    )

    # --------------------------------------------------------
    # Scalar step
    # --------------------------------------------------------
    if data.ndim == 0:
        return [int(data)]

    flat = data.reshape(-1)

    # --------------------------------------------------------
    # bool:
    # False False True True ...
    # -> first rising edge
    # --------------------------------------------------------
    if data.dtype == np.bool_:

        if len(flat) < 2:
            return []

        transitions = np.where(
            (~flat[:-1])
            & flat[1:]
        )[0] + 1

        return transitions.tolist()

    # --------------------------------------------------------
    # binary integer signal
    # --------------------------------------------------------
    if np.issubdtype(
        data.dtype,
        np.integer,
    ):
        unique_values = set(
            np.unique(flat).tolist()
        )

        if (
            unique_values.issubset({0, 1})
            and len(flat) > 2
        ):
            transitions = np.where(
                (flat[:-1] == 0)
                & (flat[1:] == 1)
            )[0] + 1

            return transitions.tolist()

        # Otherwise assume integer values are step indices.
        return [
            int(x)
            for x in flat
        ]

    return []


# ============================================================
# Image utility
# ============================================================

def make_annotated_image(
    frame,
    step,
    center_step,
    camera_name,
):
    """Add camera/step information to one image."""

    image = Image.fromarray(
        np.asarray(frame)
    )

    draw = ImageDraw.Draw(image)

    if step == center_step:
        text = (
            f"{camera_name.upper()} | "
            f"STEP {step} | ANNOTATION"
        )
    else:
        text = (
            f"{camera_name.upper()} | "
            f"STEP {step}"
        )

    draw.rectangle(
        (5, 5, 360, 35),
        fill=(0, 0, 0),
    )

    draw.text(
        (10, 10),
        text,
        fill=(255, 255, 255),
    )

    return image


def save_camera_frames(
    camera_dataset,
    camera_name,
    center_step,
    output_dir,
    radius,
):
    """Save one camera's frames around annotation."""

    num_frames = camera_dataset.shape[0]

    start = max(
        0,
        center_step - radius,
    )

    end = min(
        num_frames - 1,
        center_step + radius,
    )

    camera_dir = (
        output_dir / camera_name
    )

    camera_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for step in range(
        start,
        end + 1,
    ):

        image = make_annotated_image(
            frame=camera_dataset[step],
            step=step,
            center_step=center_step,
            camera_name=camera_name,
        )

        suffix = (
            "_ANNOTATION"
            if step == center_step
            else ""
        )

        filename = (
            f"step_{step:04d}"
            f"{suffix}.png"
        )

        image.save(
            camera_dir / filename
        )


# ============================================================
# Demo processing
# ============================================================

def inspect_demo(
    f,
    demo_name,
    output_root,
    cameras,
    radius,
):
    """Inspect one demo."""

    demo_path = f"data/{demo_name}"

    print()
    print("=" * 80)
    print(f"DEMO: {demo_name}")
    print("=" * 80)

    demo = f[demo_path]

    # --------------------------------------------------------
    # Find annotation datasets
    # --------------------------------------------------------
    candidates = find_annotation_datasets(
        demo,
        demo_path,
    )

    if not candidates:
        print("No annotation-related datasets found.")
        return

    annotation_steps = []

    for path in candidates:

        print(f"\n[FOUND] {path}")

        steps = extract_annotation_steps(
            f[path]
        )

        if steps:
            print(
                "    candidate steps:",
                steps,
            )

            annotation_steps.extend(
                steps
            )

    annotation_steps = sorted(
        set(annotation_steps)
    )

    if not annotation_steps:
        print(
            "No annotation step could be extracted."
        )
        return

    print()
    print(
        "Final candidate annotation steps:",
        annotation_steps,
    )

    # --------------------------------------------------------
    # Check requested cameras
    # --------------------------------------------------------
    available_cameras = {}

    for camera_name in cameras:

        camera_path = (
            f"{demo_path}/obs/camera/"
            f"{camera_name}"
        )

        if camera_path not in f:
            print(
                f"[WARN] Camera not found: "
                f"{camera_path}"
            )
            continue

        available_cameras[
            camera_name
        ] = f[camera_path]

        print(
            f"[CAMERA] {camera_name}: "
            f"{f[camera_path].shape}"
        )

    if not available_cameras:
        print(
            "No requested camera datasets found."
        )
        return

    # --------------------------------------------------------
    # Save frames for every annotation step
    # --------------------------------------------------------
    for annotation_idx, step in enumerate(
        annotation_steps
    ):

        print()
        print(
            f"Saving annotation #{annotation_idx}: "
            f"step={step}"
        )

        step_dir = (
            output_root
            / demo_name
            / f"annotation_{annotation_idx:02d}"
            / f"step_{step:04d}"
        )

        for (
            camera_name,
            camera_dataset,
        ) in available_cameras.items():

            if (
                step < 0
                or step >= camera_dataset.shape[0]
            ):
                print(
                    f"[SKIP] {camera_name}: "
                    f"step {step} out of range"
                )
                continue

            save_camera_frames(
                camera_dataset=(
                    camera_dataset
                ),
                camera_name=camera_name,
                center_step=step,
                output_dir=step_dir,
                radius=radius,
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "hdf5",
        type=str,
        help="Annotated Mimic HDF5 file",
    )

    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help=(
            "Specific demo name, e.g. demo_0. "
            "If omitted, inspect ALL demos."
        ),
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=7,
        help=(
            "Frames before/after annotation "
            "to save."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="annotation_frames",
    )

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side", "wrist"],
        help="Camera names to export.",
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)

    output_root = Path(
        args.output
    )

    with h5py.File(
        hdf5_path,
        "r",
    ) as f:

        if "data" not in f:
            raise KeyError(
                "HDF5 has no 'data' group."
            )

        # ----------------------------------------------------
        # Determine demos
        # ----------------------------------------------------
        if args.demo is not None:

            demo_names = [
                args.demo
            ]

        else:

            demo_names = sorted(
                [
                    name
                    for name in f["data"].keys()
                    if name.startswith("demo_")
                ],
                key=lambda x: int(
                    x.split("_")[-1]
                ),
            )

        print(
            f"Found {len(demo_names)} demos:"
        )

        print(demo_names)

        # ----------------------------------------------------
        # Process all demos
        # ----------------------------------------------------
        for demo_name in demo_names:

            if (
                f"data/{demo_name}"
                not in f
            ):
                print(
                    f"[WARN] Missing demo: "
                    f"{demo_name}"
                )
                continue

            inspect_demo(
                f=f,
                demo_name=demo_name,
                output_root=output_root,
                cameras=args.cameras,
                radius=args.radius,
            )

    print()
    print("Done.")
    print(
        f"Saved under: "
        f"{output_root.resolve()}"
    )


if __name__ == "__main__":
    main()