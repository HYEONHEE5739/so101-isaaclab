#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Record SO101 sim teleoperation to Isaac Lab HDF5 for Mimic.

Recording keyboard controls:
- RIGHT : save current episode early, then reset with NEW random state
- LEFT  : discard current episode, then reset with SAME random state
- Q     : quit and discard unfinished episode
- Timeout:
    wait reset_time_s;
    LEFT during waiting -> discard/retry same random state
    otherwise -> auto-save and start next episode

python scripts/teleoperation/record_mimic_dataset.py \
    --dataset_file ./datasets/demo_20ep.hdf5 \
    --mimic_task SO101-PickPlace-Mimic-v1 \
    --grasp_object cube_red \
    --place_bin bin_a \
    --num_episodes 20 \
    --episode_time_s 20   
    
    
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import time

from isaaclab.app import AppLauncher


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", help="Published workspace ID or package path (opt-in)")
    parser.add_argument("--source_seed", type=int, help="Explicit reproducible seed; default is random per run")
    parser.add_argument('--identity_sim_for_validation',action='store_true',help='Explicitly exclude Scene model residual; Precision mapping is unchanged')
    parser.add_argument('--resume_data',help='Read-only original Source contract for a new continuation segment')
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--calibration_file_name", default="so101_leader.json")
    parser.add_argument("--dataset_file", default="./datasets/pick_red_place_into_bin_a.hdf5")
    parser.add_argument("--mimic_task", default="SO101-PickPlace-Mimic-v0")
    parser.add_argument("--grasp_object", default="cube_red")
    parser.add_argument("--place_bin", default="bin_a")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--episode_time_s", type=float, default=20.0)
    parser.add_argument("--reset_time_s", type=float, default=5.0)
    parser.add_argument("--num_envs", type=int, default=1)

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app


def reset_env(env, episode_random_state):
    cfg = env.event_manager.get_term_cfg("reset_episode")
    cfg.params["random_state"] = episode_random_state
    env.event_manager.set_term_cfg("reset_episode", cfg)
    obs, _ = env.reset()
    return obs


def to_cpu(value):
    return value.detach().cpu()


def pose_delta_action(prev_ee, next_ee, gripper):
    import torch
    import isaaclab.utils.math as math_utils

    prev_pos, prev_quat = prev_ee[:3], prev_ee[3:7]
    next_pos, next_quat = next_ee[:3], next_ee[3:7]

    delta_pos = next_pos - prev_pos

    prev_rot = math_utils.matrix_from_quat(prev_quat.unsqueeze(0))[0]
    next_rot = math_utils.matrix_from_quat(next_quat.unsqueeze(0))[0]

    delta_rot_matrix = next_rot @ prev_rot.transpose(-1, -2)
    delta_quat = math_utils.quat_from_matrix(delta_rot_matrix.unsqueeze(0))[0]
    delta_rot = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]

    gripper = gripper.to(device=delta_pos.device, dtype=delta_pos.dtype)

    return torch.cat(
        [delta_pos, delta_rot, gripper.reshape(1)],
        dim=0,
    )


def root_pose_for_hdf5(pos_w, quat_w):
    import torch
    import isaaclab.utils.math as math_utils

    try:
        from isaaclab.utils.datasets.hdf5_dataset_file_handler import DATASET_FORMAT_VERSION
    except ImportError:
        DATASET_FORMAT_VERSION = 0

    quat = quat_w
    if DATASET_FORMAT_VERSION >= 1:
        quat = math_utils.convert_quat(quat_w, to="xyzw")

    return torch.cat([pos_w, quat], dim=-1)


def add_initial_state(episode, env):
    import torch

    robot = env.scene["robot"]

    episode.add(
        "initial_state/articulation/robot/joint_position",
        to_cpu(robot.data.joint_pos[0]),
    )
    episode.add(
        "initial_state/articulation/robot/joint_velocity",
        to_cpu(robot.data.joint_vel[0]),
    )
    episode.add(
        "initial_state/articulation/robot/root_pose",
        to_cpu(
            root_pose_for_hdf5(
                robot.data.root_pos_w[0],
                robot.data.root_quat_w[0],
            )
        ),
    )
    episode.add(
        "initial_state/articulation/robot/root_velocity",
        to_cpu(
            torch.cat(
                [
                    robot.data.root_lin_vel_w[0],
                    robot.data.root_ang_vel_w[0],
                ],
                dim=-1,
            )
        ),
    )

    for name, obj in env.scene.rigid_objects.items():

            episode.add(
                f"initial_state/rigid_object/{name}/root_pose",
                to_cpu(
                    root_pose_for_hdf5(
                        obj.data.root_pos_w[0],
                        obj.data.root_quat_w[0],
                    )
                ),
            )

            velocity = torch.cat(
                [
                    obj.data.root_lin_vel_w[0],
                    obj.data.root_ang_vel_w[0],
                ],
                dim=-1,
            )

            episode.add(
                f"initial_state/rigid_object/{name}/root_velocity",
                to_cpu(velocity),
            )

def add_observation(episode, obs, env=None):
    import torch

    episode.add(
        "obs/policy/joint_pos",
        to_cpu(obs["policy"]["joint_pos"][0]),
    )
    episode.add(
        "obs/policy/joint_vel",
        to_cpu(obs["policy"]["joint_vel"][0]),
    )
    episode.add(
        "obs/policy/ee_state",
        to_cpu(obs["policy"]["ee_state"][0]),
    )

    stream_rgb=env is not None and hasattr(env,'_workflow_video_parent')
    if stream_rgb:
        from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import stream_observation_rgb
        stream_observation_rgb(episode,{k:v[0] for k,v in obs['policy'].items()},env._workflow_video_parent,env.step_dt**-1)
    for camera_name in ([] if stream_rgb else (name for name in obs["policy"] if name.endswith("_cam"))):
        if camera_name not in obs["policy"]:
            continue

        image = obs["policy"][camera_name][0]

        if image.dtype.is_floating_point:
            image = image.detach()
            if image.numel() > 0 and image.max().item() <= 1.0:
                image = image * 255.0
            image = image.clamp(0, 255).to(torch.uint8)

        episode.add(
            f"obs/policy/{camera_name}",
            image.detach().cpu(),
        )
    if env is not None:
        from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import observation_telemetry
        telemetry=observation_telemetry(env)
        episode.add('obs/timestamp',telemetry['timestamp'][0])
        for name,value in telemetry['object_poses'].items():episode.add('obs/object_poses/'+name,value[0])
        for name,value in telemetry.get('events',{}).items():episode.add('obs/events/'+name,value[0])


def add_post_state(episode, env):
    robot = env.scene["robot"]

    episode.add(
        "states/joint_pos",
        to_cpu(robot.data.joint_pos[0]),
    )
    episode.add(
        "states/joint_vel",
        to_cpu(robot.data.joint_vel[0]),
    )


def save_episode(writer, episode, episode_id):
    episode.success = True
    episode.pre_export()
    writer.write_episode(episode)
    if hasattr(writer, 'workflow_task_definition'):
        from soarm101_lab.workflow.tasks import stamp_group
        stamp_group(writer._hdf5_data_group[f'demo_{episode_id}'], writer.workflow_task_definition,
                    json.loads(writer._hdf5_data_group.attrs['env_args']).get('scene_configuration'))
    writer._hdf5_data_group[f'demo_{episode_id}'].attrs['success_source']='operator_saved_unverified'
    group=writer._hdf5_data_group[f'demo_{episode_id}']
    group.attrs['source_group']=str(Path(group.file.filename).resolve())+f'/demo_{episode_id}'
    writer.flush()

    print(
        f"✅ Saved demo_{episode_id}",
        flush=True,
    )


def main():
    from pathlib import Path
    if Path(args_cli.dataset_file).exists():raise FileExistsError('Choose a new dataset output; original data is preserved')
    import torch

    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

    from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
    from soarm101_lab.devices import SO101Leader, SO101LeaderCfg
    from soarm101_lab.utils.episode_randomizer import EpisodeRandomizer
    from soarm101_lab.utils.keyboard import KeyboardControl
    from soarm101_lab.utils.voice import log_say

    from soarm101_lab.workflow.presentation import current, attach, RecorderControls
    bridge = current()
    if bridge:
        keyboard_control = RecorderControls(bridge, None if args_cli.headless else KeyboardControl())
    else:
        keyboard_control = KeyboardControl()

    controller_cfg = SO101LeaderCfg(
        port=args_cli.port,
        recalibrate=args_cli.recalibrate,
        calibration_file_name=args_cli.calibration_file_name,
        urdf_path=str(Path(args_cli.workspace)/'assets/SO101/urdf/so101_isaaclab.urdf') if args_cli.workspace else None,
    )

    simulation=None
    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.package import load as load_workspace
        from soarm101_lab.workflow.tasks import configured
        from soarm101_lab.real2sim.precision import approved_mappings,simulation_configuration
        from soarm101_lab.real2sim.workspaces.reset import source_workspace
        workspace=source_workspace(configured(load_workspace(args_cli.workspace)),args_cli.resume_data)
        mappings=approved_mappings(workspace)
        resume_coordinates=None
        if args_cli.resume_data:
            from soarm101_lab.so101_dataset_contract import read_contract,MIMIC_SPACE
            from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
            from soarm101_lab.representation import require_compatible
            resume_coordinates=read_contract(args_cli.resume_data)
            require_compatible(resume_coordinates,coordinate_contract(workspace),MIMIC_SPACE)
            if resume_coordinates.get('hardware_mapping')!=mappings.get('leader'):
                raise ValueError('Source resume hardware calibration differs; use the original mapping or a new dataset')
        simulation=simulation_configuration(workspace,mappings=mappings,hardware_role='leader',coordinates=resume_coordinates,
            identity=args_cli.identity_sim_for_validation,backend='isaac',scene_configuration=workspace['scene_configuration'])
    controller = SO101Leader(controller_cfg)
    controller.connect()
    if args_cli.workspace:
        actual=getattr(controller,'precision_mapper',None)
        if (actual.metadata()['sha256'] if actual else None)!=mappings.get('leader',{}).get('sha256'):
            raise ValueError('Leader mapping changed during Source preparation')

    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.environment import make_config
        from soarm101_lab.real2sim.workspaces.package import load as load_workspace
        from soarm101_lab.workflow.tasks import configured
        workspace = source_workspace(configured(load_workspace(args_cli.workspace)),args_cli.resume_data)
        env_cfg = make_config(workspace, args_cli.device, args_cli.num_envs,
                              simulation=simulation)
        workspace = env_cfg.workflow_workspace
        from soarm101_lab.real2sim.workspaces.reset import source_seed
        env_cfg.events.reset_episode.params['randomization_seed'] = source_seed(args_cli.source_seed)
        print('SOURCE_RANDOMIZATION_SEED', env_cfg.events.reset_episode.params['randomization_seed'], flush=True)
        args_cli.grasp_object = workspace['task']['pick']
        args_cli.place_bin = workspace['task']['place']
        args_cli.mimic_task = 'SO101-Workspace-Source-v1'
    else:
        env_cfg = SO101TeleopEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    attach(env)

    if env.num_envs != 1:
        raise ValueError("This recorder currently supports num_envs=1.")

    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
        randomizer = WorkspaceRandomizer(workspace)
    else:
        randomizer = EpisodeRandomizer()

    control_dt = env.step_dt
    control_hz = 1.0 / control_dt
    max_step = round(args_cli.episode_time_s / control_dt)

    print(f"✅ max_step   : {max_step}")
    print(f"✅ control_dt : {control_dt}")
    print(f"✅ control_hz : {control_hz}")
    print(f"✅ sim_dt     : {env_cfg.sim.dt}")

    from soarm101_lab.so101_dataset_contract import current_contract, write_contract, MIMIC_SPACE
    coordinates = current_contract(MIMIC_SPACE)
    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
        coordinates = coordinate_contract(workspace)
    from soarm101_lab.real2sim.precision import collection_coordinates
    coordinates = collection_coordinates(coordinates, getattr(controller, 'precision_mapper', None),
        getattr(env_cfg,'simulation_configuration',None))
    if args_cli.workspace and args_cli.resume_data:coordinates=resume_coordinates
    # Persist before recording, including interrupted runs. Never mix mapping revisions.
    write_contract(args_cli.dataset_file, coordinates)
    from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import VideoHDF5DatasetFileHandler
    writer = VideoHDF5DatasetFileHandler()
    env._workflow_video_parent=Path(args_cli.dataset_file).resolve().parent
    writer.create(
        args_cli.dataset_file,
        env_name=args_cli.mimic_task,
    )
    writer.add_env_args(
        {
            "grasp_object": args_cli.grasp_object,
            "place_bin": args_cli.place_bin,
            "control_dt": float(control_dt),
            "so101_coordinates": coordinates,
        }
    )

    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.package import metadata
        try:
            from isaaclab.utils.datasets.hdf5_dataset_file_handler import DATASET_FORMAT_VERSION
        except ImportError:
            DATASET_FORMAT_VERSION = 0
        from soarm101_lab.representation import describe
        writer.add_env_args({'representation': describe(coordinates), 'workspace': metadata(workspace), 'workspace_path': str(workspace['root']),
                             'simulation_configuration':env_cfg.simulation_configuration,
                             'root_quaternion_order': 'xyzw' if DATASET_FORMAT_VERSION >= 1 else 'wxyz',
                             'task_definitions': [workspace['task_definition']], 'scene_configuration':workspace['scene_configuration']})
        writer.workflow_task_definition = workspace['task_definition']
    from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import telemetry_metadata
    writer.add_env_args(telemetry_metadata(workspace['task']['dynamic'] if args_cli.workspace else list(env.scene.rigid_objects)))

    episode_count = 0
    step_count = 0

    recording = True
    timeout = False
    waiting_start_time = None

    episode_random_state = randomizer.sample()
    obs = reset_env(env, episode_random_state)

    episode = EpisodeData()
    add_initial_state(episode, env)

    prev_ee = obs["policy"]["ee_state"][0].clone()

    log_say(
        f"Episode {episode_count}",
        blocking=True,
    )
    print(
        f"✅ Episode {episode_count}",
        flush=True,
    )

    episode_start_time = time.perf_counter()
    print('[WORKFLOW] Episode 0 시작 · 오른쪽: 저장/다음 · 왼쪽: 폐기/같은 배치 리셋', flush=True)

    try:
        from soarm101_lab.workflow.data_resume import deferred_interrupt
        interrupt_boundary = deferred_interrupt()
        while simulation_app.is_running():
            interrupt_boundary()

            # ====================================================
            # Quit
            # ====================================================

            if bridge:
                bridge.details.update(episode_index=episode_count + 1, episode_total=args_cli.num_episodes, recording=recording)
            if keyboard_control.should_quit():
                log_say("stop recording", blocking = True)
                print(
                    "✅ Quit requested. "
                    "Current unfinished episode will be discarded.",
                    flush=True,
                )
                break

            # ====================================================
            # RECORDING
            # ====================================================

            if recording:
                # Store obs_t first.
                add_observation(
                    episode,
                    obs,
                    env,
                )

                # Joint-space teleop input.
                joint_action = (
                    controller.advance()
                    .to(
                        device=env.device,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                )

                # Move the normal teleop environment.
                next_obs, _ = env.step(
                    joint_action
                )

                # Convert real sim EE movement to Mimic task-space action.
                next_ee = next_obs["policy"]["ee_state"][0]

                mimic_action = pose_delta_action(
                    prev_ee=prev_ee,
                    next_ee=next_ee,
                    gripper=joint_action[0, -1],
                )

                episode.add(
                    "actions",
                    to_cpu(mimic_action),
                )

                # Keep original joint target for later LeRobot conversion.
                episode.add(
                    "joint_targets",
                    to_cpu(joint_action[0]),
                )

                add_post_state(
                    episode,
                    env,
                )

                obs = next_obs
                prev_ee = next_ee.clone()
                step_count += 1

                # ------------------------------------------------
                # Save early (RIGHT)
                # ------------------------------------------------

                if keyboard_control.consume_save():
                    log_say(f"save episode {episode_count}", blocking=False)

                    save_episode(
                        writer,
                        episode,
                        episode_count,
                    )

                    episode_count += 1

                    if episode_count >= args_cli.num_episodes:
                        break

                    # Next episode uses NEW random state.
                    episode_random_state = randomizer.sample()

                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                    if getattr(episode,'_rgb_video',None):episode._rgb_video.abort()
                    episode = EpisodeData()
                    add_initial_state(
                        episode,
                        env,
                    )

                    prev_ee = obs["policy"]["ee_state"][0].clone()

                    step_count = 0
                    recording = False
                    timeout = False
                    waiting_start_time = time.perf_counter()

                    continue

                # ------------------------------------------------
                # Discard (LEFT)
                # ------------------------------------------------

                if keyboard_control.consume_discard():
                    log_say(f"resetting", blocking=False)
                    print(
                        "✅ Discarding current demo.",
                        flush=True,
                    )

                    # SAME random state.
                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                    if getattr(episode,'_rgb_video',None):episode._rgb_video.abort()
                    episode = EpisodeData()
                    add_initial_state(
                        episode,
                        env,
                    )

                    prev_ee = obs["policy"]["ee_state"][0].clone()

                    step_count = 0
                    recording = False
                    timeout = False
                    waiting_start_time = time.perf_counter()

                    continue

                # ------------------------------------------------
                # Timeout
                # ------------------------------------------------

                if step_count >= max_step:
                    episode_take_time = (
                        time.perf_counter()
                        - episode_start_time
                    )

                    print(
                        f"episode_take_time: "
                        f"{episode_take_time:.2f} seconds",
                        flush=True,
                    )

                    recording = False
                    timeout = True
                    waiting_start_time = time.perf_counter()

                    continue

            # ====================================================
            # WAITING
            # ====================================================

            else:
                assert waiting_start_time is not None

                elapsed = (
                    time.perf_counter()
                    - waiting_start_time
                )

                # Timeout 상태에서 LEFT:
                # current timed-out demo discard + SAME random state retry.
                if (
                    timeout
                    and keyboard_control.consume_discard()
                ):
                    log_say(f"resetting", blocking=False)
                    print(
                        "✅ Discarding timed-out demo.",
                        flush=True,
                    )

                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                    if getattr(episode,'_rgb_video',None):episode._rgb_video.abort()
                    episode = EpisodeData()
                    add_initial_state(
                        episode,
                        env,
                    )

                    prev_ee = obs["policy"]["ee_state"][0].clone()

                    step_count = 0
                    timeout = False
                    recording = False
                    waiting_start_time = time.perf_counter()

                    continue

                if elapsed < args_cli.reset_time_s:
                    continue

                # Timeout and no LEFT -> auto-save.
                if timeout:
                    log_say(f"save episode {episode_count}", blocking=False)
                    save_episode(
                        writer,
                        episode,
                        episode_count,
                    )

                    episode_count += 1

                    if episode_count >= args_cli.num_episodes:
                        break

                    episode_random_state = randomizer.sample()

                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                    if getattr(episode,'_rgb_video',None):episode._rgb_video.abort()
                    episode = EpisodeData()
                    add_initial_state(
                        episode,
                        env,
                    )

                    prev_ee = obs["policy"]["ee_state"][0].clone()

                step_count = 0
                timeout = False
                recording = True
                waiting_start_time = None

                episode_start_time = time.perf_counter()

                log_say(
                    f"Episode {episode_count}",
                    blocking=True,
                )

                print(
                    f"✅ Episode {episode_count}",
                    flush=True,
                )

    finally:
        print(
            "✅ Closing HDF5 recorder...",
            flush=True,
        )

        if getattr(episode,'_rgb_video',None):episode._rgb_video.abort()

        # Current EpisodeData is not written here:
        # Q/exception means unfinished demo is discarded.

        try:
            writer.close()
        except Exception as exc:
            print(
                f"Failed to close HDF5 writer: {exc}",
                flush=True,
            )

        keyboard_control.destroy()
        controller.disconnect()

        try:
            env.close()
        except Exception as exc:
            print(
                f"Failed to close environment: {exc}",
                flush=True,
            )
        if args_cli.workspace:
            writer.finalize_common()


if __name__ == "__main__":
    main()
