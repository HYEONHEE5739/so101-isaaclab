#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Convert Isaac Lab HDF5 demonstrations to LeRobotDataset.

Supported state schemas:
- teleop source:    obs/policy/joint_pos
- Mimic generated:  obs/joint_pos

Action sources:
- joint_targets:    use recorded joint targets
- next_joint_pos:   use q_(t+1) as action for q_t
- mimic_actions:    use raw Mimic Cartesian action (normally NOT for SO101 policy)
- auto:             joint_targets if present, otherwise next_joint_pos

Example:

python scripts/tools/convert_isaac2lerobot.py \
    --input_file ./datasets/generated_demo_20ep_50ep.hdf5 \
    --repo_id ilikirobot/generated_demo_20ep_50ep \
    --root ./datasets/LerobotDataset/generated_demo_20ep_50ep \
    --task "pick red cube and place it in to the bin a" \
    --action_source joint_targets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", required=True)
parser.add_argument("--repo_id", required=True)
parser.add_argument("--root", required=True)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--task", required=True)
parser.add_argument("--robot_type", default="so101_follower_sim")

parser.add_argument(
    "--action_source",
    choices=["auto", "joint_targets", "next_joint_pos", "mimic_actions"],
    default="auto",
)

args = parser.parse_args()


# ============================================================
# HDF5 helpers
# ============================================================

def get_node(group, path):
    node = group
    for part in path.split("/"):
        node = node[part]
    return node


def exists(group, path):
    try:
        get_node(group, path)
        return True
    except KeyError:
        return False


def get_joint_pos_path(demo):
    """Support both teleop and Mimic-generated HDF5 schemas."""

    candidates = [
        "obs/policy/joint_pos",  # teleop_record_hdf5
        "obs/joint_pos",         # Mimic generated dataset
    ]

    for path in candidates:
        if exists(demo, path):
            return path

    raise KeyError(
        "Could not find joint position observation. "
        f"Tried: {candidates}"
    )


def image_to_uint8(image):
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.5:
            image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)


# ============================================================
# Action selection
# ============================================================

def choose_action_source(demo):
    if args.action_source != "auto":
        return args.action_source

    if exists(demo, "joint_targets"):
        return "joint_targets"

    return "next_joint_pos"


def load_actions(demo, states, action_source):
    if action_source == "joint_targets":
        if not exists(demo, "joint_targets"):
            raise KeyError(
                "joint_targets was requested but does not exist. "
                "Use --action_source next_joint_pos."
            )

        return np.asarray(
            get_node(demo, "joint_targets"),
            dtype=np.float32,
        )

    if action_source == "mimic_actions":
        if not exists(demo, "actions"):
            raise KeyError("Mimic actions were requested but 'actions' does not exist.")

        return np.asarray(
            get_node(demo, "actions"),
            dtype=np.float32,
        )

    # state_t -> state_(t+1)
    return np.concatenate(
        [states[1:], states[-1:]],
        axis=0,
    ).astype(np.float32)


# ============================================================
# Conversion
# ============================================================

input_file = Path(args.input_file).expanduser().resolve()
dataset_root = Path(args.root).expanduser().resolve()

if not input_file.exists():
    raise FileNotFoundError(f"Input HDF5 not found: {input_file}")


with h5py.File(input_file, "r") as h5:
    if "data" not in h5:
        raise KeyError("Input HDF5 has no 'data' group.")

    demos = h5["data"]

    demo_names = sorted(
        demos.keys(),
        key=lambda name: int(name.split("_")[-1]),
    )

    if not demo_names:
        raise RuntimeError("No demo_* groups found in HDF5.")

    # --------------------------------------------------------
    # Inspect first demo and create LeRobot features
    # --------------------------------------------------------

    first = demos[demo_names[0]]

    joint_pos_path = get_joint_pos_path(first)

    first_states = np.asarray(
        get_node(first, joint_pos_path),
        dtype=np.float32,
    )

    if len(first_states) == 0:
        raise RuntimeError(f"{demo_names[0]} has no joint states.")

    action_source = choose_action_source(first)
    first_actions = load_actions(first, first_states, action_source)

    if action_source != "mimic_actions" and first_states.shape[-1] != first_actions.shape[-1]:
        raise ValueError(
            f"State/action dimension mismatch: "
            f"state={first_states.shape}, action={first_actions.shape}"
        )

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(first_states[0].shape),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(first_actions[0].shape),
            "names": None,
        },
    }

    # --------------------------------------------------------
    # Cameras
    # --------------------------------------------------------

    camera_paths = {}

    for camera_name in ("side_cam", "wrist_cam"):
        candidates = [
            f"obs/policy/{camera_name}",
            f"obs/{camera_name}",
        ]

        path = next(
            (candidate for candidate in candidates if exists(first, candidate)),
            None,
        )

        if path is None:
            continue

        image0 = image_to_uint8(get_node(first, path)[0])

        if image0.shape[-1] == 4:
            image0 = image0[..., :3]

        camera_paths[camera_name] = path

        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": tuple(image0.shape),
            "names": ["height", "width", "channel"],
        }

    print("=" * 70)
    print("CONVERSION CONFIG")
    print("=" * 70)
    print(f"Input        : {input_file}")
    print(f"Output       : {dataset_root}")
    print(f"Demos        : {len(demo_names)}")
    print(f"State path   : {joint_pos_path}")
    print(f"Action source: {action_source}")
    print(f"State shape  : {first_states.shape}")
    print(f"Action shape : {first_actions.shape}")
    print(f"Cameras      : {list(camera_paths.keys())}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create LeRobot dataset
    # --------------------------------------------------------

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=bool(camera_paths),
        image_writer_processes=0,
        image_writer_threads=4,
    )

    # --------------------------------------------------------
    # Convert demos
    # --------------------------------------------------------

    for demo_name in demo_names:
        demo = demos[demo_name]

        joint_pos_path = get_joint_pos_path(demo)

        states = np.asarray(
            get_node(demo, joint_pos_path),
            dtype=np.float32,
        )

        action_source = choose_action_source(demo)
        actions = load_actions(demo, states, action_source)

        if action_source != "mimic_actions" and states.shape[-1] != actions.shape[-1]:
            raise ValueError(
                f"{demo_name}: state/action dimension mismatch: "
                f"{states.shape} vs {actions.shape}"
            )

        length = min(len(states), len(actions))

        # ----------------------------------------------------
        # Cameras for this demo
        # ----------------------------------------------------

        camera_arrays = {}

        for camera_name, path in camera_paths.items():
            if not exists(demo, path):
                raise KeyError(
                    f"{demo_name} is missing camera path: {path}"
                )

            camera_arrays[camera_name] = np.asarray(get_node(demo, path))
            length = min(length, len(camera_arrays[camera_name]))

        # ----------------------------------------------------
        # Frames
        # ----------------------------------------------------

        for t in range(length):
            frame = {
                "observation.state": states[t],
                "action": actions[t],
                "task": args.task,
            }

            for camera_name, images in camera_arrays.items():
                image = image_to_uint8(images[t])

                if image.shape[-1] == 4:
                    image = image[..., :3]

                frame[f"observation.images.{camera_name}"] = image

            dataset.add_frame(frame)

        dataset.save_episode()

        print(
            f"{demo_name}: {length} frames "
            f"(state={joint_pos_path}, action={action_source})"
        )


dataset.stop_image_writer()
dataset.finalize()

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"LeRobotDataset: {dataset_root}")