# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.

python scripts/mimic/generate_dataset.py \
    --task SO101-PickPlace-Mimic-v1 \
    --input_file datasets/annotated_demo_20ep.hdf5 \
    --output_file datasets/generated_demo_20ep_30ep.hdf5 \
    --generation_num_trials 30 \
    --num_envs 1 \
    --device cuda:0 \
    --enable_cameras
"""
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--generation_seed", type=int, default=None, help="Workspace Datagen seed; default: fresh seed per run")
parser.add_argument("--workspace", help="Published workspace ID/path")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.workspace:
    args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401
import soarm101_lab.tasks.manager_based.soarm101_lab
from isaaclab.managers import RecorderTermCfg
from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import (
    PreStepPolicyObservationsCpuRecorder, PostStepJointTargetsRecorder
)
def main():
    from soarm101_lab.so101_dataset_contract import inherit_contract, MIMIC_SPACE
    inherit_contract(args_cli.input_file, args_cli.output_file, MIMIC_SPACE)
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.mimic import make_mimic_config, validate_input
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers import DatasetExportMode
        workspace, _ = validate_input(args_cli.input_file, args_cli.workspace, annotated=True)
        from soarm101_lab.real2sim.workspaces.reset import datagen_seed
        from soarm101_lab.real2sim.storage import atomic
        from pathlib import Path
        seed = datagen_seed(workspace['task']['reset']['seed'], args_cli.generation_seed)
        generation_provenance = dict(generation_seed=seed, workspace_reset_seed=workspace['task']['reset']['seed'],
                                     seed_mode='automatic' if args_cli.generation_seed is None else 'explicit')
        atomic(Path(args_cli.output_file).with_suffix('.generation.json'), generation_provenance)
        print(f'[WORKFLOW] Datagen seed={seed} (workspace seed={workspace["task"]["reset"]["seed"]})', flush=True)
        env_cfg = make_mimic_config(args_cli.workspace, args_cli.device, num_envs, randomization_seed=seed)
        env_name = env_cfg.env_name
        success_term = env_cfg.terminations.success
        env_cfg.terminations = None
        target = 10 if args_cli.generation_num_trials is None else args_cli.generation_num_trials
        if target < 1:
            raise ValueError('Successful demo target must be positive')
        env_cfg.datagen_config.generation_num_trials = target
        print(f'[WORKFLOW] Datagen 목표: 성공 데이터 {target}개 (실패 시도 제외)', flush=True)
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    else:
        env_cfg, success_term = setup_env_config(
            env_name=env_name,
            output_dir=output_dir,
            output_file_name=output_file_name,
            num_envs=num_envs,
            device=args_cli.device,
            generation_num_trials=args_cli.generation_num_trials,
        )

    # override
    env_cfg.recorders.record_pre_step_flat_policy_observations = RecorderTermCfg(
        class_type=PreStepPolicyObservationsCpuRecorder
    )

    # add new recorder term
    env_cfg.recorders.record_post_step_joint_targets = RecorderTermCfg(
        class_type=PostStepJointTargetsRecorder
    )

    # Create environment
    if args_cli.workspace:
        from soarm101_lab.tasks.manager_based.soarm101_lab.so101_mimic_env import SO101PickPlaceMimicEnv
        env = SO101PickPlaceMimicEnv(cfg=env_cfg)
    else:
        env = gym.make(env_name, cfg=env_cfg).unwrapped

    from soarm101_lab.workflow.presentation import attach
    bridge = attach(env)
    if bridge:
        from soarm101_lab.workflow.progress import GenerationProgress
        import isaaclab_mimic.datagen.generation as generation
        bridge.progress_provider = GenerationProgress(generation, env.cfg.datagen_config.generation_num_trials)

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # Create one motion planner per environment
        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            # Create a config instance from the task name
            planner_config = CuroboPlannerCfg.from_task_name(env_name)

            # Ensure visualization is only enabled for the first environment
            # If not, sphere and plan visualization will be too slow in isaac lab
            # It is efficient to visualize the spheres and plan for the first environment in rerun
            if env_id != 0:
                planner_config.visualize_spheres = False
                planner_config.visualize_plan = False

            motion_planners[env_id] = CuroboPlanner(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,  # Pass the config object
                env_id=env_id,  # Pass environment ID
            )

        env.cfg.datagen_config.use_skillgen = True

    from soarm101_lab.workflow.data_resume import deferred_interrupt
    interrupt_boundary = deferred_interrupt()
    original_step = env.step
    def interruptible_step(*args, **kwargs):
        interrupt_boundary()
        return original_step(*args, **kwargs)
    env.step = interruptible_step

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("[WORKFLOW] Datagen 중지 요청 · 완료 episode 저장 후 종료", flush=True)
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()

    env.close()  # close/flush recorder before opening the completed dataset
    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.mimic import preserve_metadata
        preserve_metadata(args_cli.input_file, args_cli.output_file, args_cli.workspace)
        import h5py
        import json
        with h5py.File(args_cli.output_file, 'a') as dataset:
            dataset['data'].attrs['generation_provenance'] = json.dumps(generation_provenance)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
