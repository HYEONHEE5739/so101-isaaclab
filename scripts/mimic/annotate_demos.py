# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Add Mimic annotations to demonstrations.

Examples:

# All demos
python scripts/mimic/annotate_demos.py \
    --task SO101-PickPlace-Mimic-v1 \
    --input_file datasets/demo_20ep.hdf5 \
    --output_file datasets/annotated_demo_20ep.hdf5 \
    --device cuda:0 \
    --enable_cameras \
    --auto

# Selected demos: demo_0, demo_1, demo_3~4, demo_6~19
python scripts/mimic/annotate_demos.py \
    --task SO101-PickPlace-Mimic-v1 \
    --input_file datasets/pick_red_block_into_bin_a.hdf5 \
    --output_file datasets/annotated_pick_red_block_into_bin_a.hdf5 \
    --device cuda:0 \
    --enable_cameras \
    --auto \
    --include 0 1 3-4 6-19
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--input_file", type=str, default="./datasets/dataset.hdf5")
parser.add_argument("--output_file", type=str, default="./datasets/dataset_annotated.hdf5")
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--annotate_subtask_start_signals", action="store_true", default=False)
parser.add_argument(
    "--include",
    nargs="+",
    default=None,
    help="Only annotate selected demos. Example: --include 0 1 3-4 6-19",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# Imports after AppLauncher
# ============================================================

import os

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import soarm101_lab.tasks  # noqa: F401
import soarm101_lab.tasks.manager_based.soarm101_lab
if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ============================================================
# Global state
# ============================================================

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


# ============================================================
# Demo selection
# ============================================================

def parse_demo_selection(tokens):
    if tokens is None:
        return None

    indices = set()

    for token in tokens:
        if "-" in token:
            start, end = map(int, token.split("-", 1))

            if start > end:
                raise ValueError(f"Invalid range '{token}': start must be <= end")

            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))

    if not indices:
        raise ValueError("--include was given but no demo index was parsed")

    return [f"demo_{i}" for i in sorted(indices)]


# ============================================================
# Keyboard callbacks
# ============================================================

def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices

    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


# ============================================================
# Mimic recorders
# ============================================================

class PreStepDatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        eef_pose_dict = {}

        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }

        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


# ============================================================
# Replay
# ============================================================

def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    global current_action_index, skip_episode, is_paused

    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]

    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)

    first_action = True

    for action_index, action in enumerate(actions):
        current_action_index = action_index

        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()

                if skip_episode:
                    return False

        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1)
        env.step(action_tensor)

    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False

    return True


# ============================================================
# Automatic annotation
# ============================================================

def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    global skip_episode

    skip_episode = False
    success = replay_episode(env, episode, success_term)

    if skip_episode:
        print("\tSkipping the episode.")
        return False

    if not success:
        print("\tThe final task was not completed.")
        return False

    annotated_episode = env.recorder_manager.get_episode(0)

    subtask_term_signals = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]

    for signal_name, signal_flags in subtask_term_signals.items():
        signal_flags = torch.as_tensor(signal_flags, device=env.device)

        if not torch.any(signal_flags):
            success = False
            print(f'\tDid not detect completion for subtask "{signal_name}".')

    if args_cli.annotate_subtask_start_signals:
        subtask_start_signals = annotated_episode.data["obs"]["datagen_info"]["subtask_start_signals"]

        for signal_name, signal_flags in subtask_start_signals.items():
            signal_flags = torch.as_tensor(signal_flags, device=env.device)

            if not torch.any(signal_flags):
                success = False
                print(f'\tDid not detect start for subtask "{signal_name}".')

    return success


# ============================================================
# Manual annotation
# ============================================================

def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    subtask_start_signal_names: dict[str, list[str]] = {},
) -> bool:
    global is_paused, marked_subtask_action_indices, skip_episode

    subtask_term_signal_action_indices = {}
    subtask_start_signal_action_indices = {}

    for eef_name, term_names in subtask_term_signal_names.items():
        start_names = subtask_start_signal_names[eef_name]

        if len(term_names) == 0 and len(start_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False

            print(f'\tPlaying episode for eef "{eef_name}".')
            print(f"\tTermination signals: {term_names}")

            if start_names:
                print(f"\tStart signals: {start_names}")

            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate.')
            print('\tPress "Q" to skip.\n')

            marked_subtask_action_indices = []
            task_success = replay_episode(env, episode, success_term)

            if skip_episode:
                return False

            expected_count = len(term_names) + len(start_names)

            if task_success and expected_count == len(marked_subtask_action_indices):
                for marked_index in range(expected_count):
                    action_index = marked_subtask_action_indices[marked_index]

                    if args_cli.annotate_subtask_start_signals and marked_index % 2 == 0:
                        start_names_index = int(marked_index / 2)
                        subtask_start_signal_action_indices[start_names[start_names_index]] = action_index

                    elif args_cli.annotate_subtask_start_signals:
                        term_names_index = math.floor(marked_index / 2)
                        subtask_term_signal_action_indices[term_names[term_names_index]] = action_index

                    else:
                        subtask_term_signal_action_indices[term_names[marked_index]] = action_index

                break

            if not task_success:
                print("\tThe final task was not completed.")
                return False

            print(
                f"\tExpected {expected_count} signals, "
                f"but marked {len(marked_subtask_action_indices)}. Replaying."
            )

    annotated_episode = env.recorder_manager.get_episode(0)
    num_actions = len(episode.data["actions"])

    for signal_name, action_index in subtask_term_signal_action_indices.items():
        signals = torch.ones(num_actions, dtype=torch.bool)
        signals[:action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{signal_name}", signals)

    for signal_name, action_index in subtask_start_signal_action_indices.items():
        signals = torch.ones(num_actions, dtype=torch.bool)
        signals[:action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_start_signals/{signal_name}", signals)

    return True


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"Input HDF5 not found: {args_cli.input_file}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)

    env_name = dataset_file_handler.get_env_name()
    all_episode_names = list(dataset_file_handler.get_episode_names())

    if not all_episode_names:
        print("No episodes found in the dataset.")
        return 0

    selected_demos = parse_demo_selection(args_cli.include)

    if selected_demos is None:
        episode_names = all_episode_names
    else:
        missing = [name for name in selected_demos if name not in all_episode_names]

        if missing:
            raise ValueError(f"Requested demos do not exist: {missing}")

        episode_names = selected_demos

    print("\n" + "=" * 70)
    print("ANNOTATION DATASET")
    print("=" * 70)
    print(f"Input          : {args_cli.input_file}")
    print(f"Total demos    : {len(all_episode_names)}")
    print(f"Selected demos : {len(episode_names)}")
    print("Demos          : " + ", ".join(episode_names))
    print("=" * 70)

    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]

    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    if env_cfg.terminations is not None and hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None
    env_cfg.recorders = MimicRecorderManagerCfg()

    if not args_cli.auto:
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    if not args_cli.auto or not args_cli.annotate_subtask_start_signals:
        env_cfg.recorders.record_pre_step_subtask_start_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("Environment must derive from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError("Environment does not implement get_subtask_term_signals().")

        if (
            args_cli.annotate_subtask_start_signals
            and env.get_subtask_start_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_start_signals
        ):
            raise NotImplementedError("Environment does not implement get_subtask_start_signals().")

    else:
        subtask_term_signal_names = {}
        subtask_start_signal_names = {}

        for eef_name, subtask_configs in env.cfg.subtask_configs.items():
            if args_cli.annotate_subtask_start_signals:
                subtask_start_signal_names[eef_name] = [
                    cfg.subtask_term_signal for cfg in subtask_configs
                ]
            else:
                subtask_start_signal_names[eef_name] = []

            subtask_term_signal_names[eef_name] = [
                cfg.subtask_term_signal for cfg in subtask_configs
            ]

            if args_cli.annotate_subtask_start_signals:
                if any(name in (None, "") for name in subtask_start_signal_names[eef_name]):
                    raise ValueError(f"Missing subtask_term_signal for eef '{eef_name}'.")

            subtask_term_signal_names[eef_name].pop()

    env.reset()

    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)

        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)

        keyboard_interface.reset()

    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0

    try:
        with torch.inference_mode():
            for episode_name in episode_names:
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break

                processed_episode_count += 1

                print(
                    f"\nAnnotating {processed_episode_count}/{len(episode_names)} "
                    f"({episode_name})"
                )

                episode = dataset_file_handler.load_episode(episode_name, env.device)

                if args_cli.auto:
                    success = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    success = annotate_episode_in_manual_mode(
                        env,
                        episode,
                        success_term,
                        subtask_term_signal_names,
                        subtask_start_signal_names,
                    )

                if success and not skip_episode:
                    env.recorder_manager.set_success_to_episodes(
                        None,
                        torch.tensor([[True]], dtype=torch.bool, device=env.device),
                    )
                    env.recorder_manager.export_episodes()

                    exported_episode_count += 1
                    successful_task_count += 1

                    print("\t✅ Exported annotated episode.")
                else:
                    print("\t❌ Skipped episode.")

    except KeyboardInterrupt:
        print("\n🛑 Annotation interrupted.")
        print("Already exported episodes remain saved.")

    print("\n" + "=" * 70)
    print("ANNOTATION RESULT")
    print("=" * 70)
    print(f"Processed : {processed_episode_count}")
    print(f"Exported  : {exported_episode_count}")
    print(f"Output    : {args_cli.output_file}")
    print("=" * 70)

    env.close()

    return successful_task_count


if __name__ == "__main__":
    main()
    simulation_app.close()