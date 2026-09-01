# SPDX-License-Identifier: BSD-3-Clause
"""SO101 teleoperation + direct LeRobotDataset recording."""

import argparse
import time

from isaaclab.app import AppLauncher
"""
python scripts/teleoperation/record_lerobot_dataset.py \
    --repo_id ilikirobot/so101_pick_place \
    --single_task "pick red cube and place in bin a" \
    --repo_root ./datasets \
    --num_episodes 50 \
    --episode_time_s 20 \
    --reset_time_s 5 \
    --port /dev/ttyACM0
    #--resume
    #--push_to_hub
"""

parser = argparse.ArgumentParser(
    description="SO101 Teleoperation with direct LeRobotDataset recording"
)

parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--recalibrate", action="store_true")
parser.add_argument(
    "--calibration_file_name",
    type=str,
    default="so101_leader.json",
)
parser.add_argument("--num_envs", type=int, default=1)

parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--repo_root", type=str, default="./datasets")
parser.add_argument("--single_task", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--episode_time_s", type=float, default=20.0)
parser.add_argument("--reset_time_s", type=float, default=5.0)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push finalized dataset to Hugging Face Hub when recording exits.",
)
parser.add_argument(
    "--private",
    action="store_true",
    help="Create/push the Hugging Face dataset repository as private.",
)

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


def main():
    from isaaclab.envs import ManagerBasedEnv

    from soarm101_lab.datasets import LeRobotRecorder
    from soarm101_lab.devices import SO101Leader, SO101LeaderCfg
    from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
    from soarm101_lab.utils.episode_randomizer import EpisodeRandomizer
    from soarm101_lab.utils.keyboard import KeyboardControl
    from soarm101_lab.utils.voice import log_say


    keyboard_control = KeyboardControl()

    controller_cfg = SO101LeaderCfg(
        port=args_cli.port,
        recalibrate=args_cli.recalibrate,
        calibration_file_name=args_cli.calibration_file_name,
    )

    controller = SO101Leader(controller_cfg)
    controller.connect()

    env_cfg = SO101TeleopEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    randomizer = EpisodeRandomizer()

    control_dt = env.step_dt
    control_hz = 1.0 / control_dt
    max_step = round(args_cli.episode_time_s / control_dt)

    print(f"✅ max_step   : {max_step}")
    print(f"✅ control_dt : {control_dt}")
    print(f"✅ control_hz : {control_hz}")
    print(f"✅ sim_dt     : {env_cfg.sim.dt}")

    recorder = LeRobotRecorder(
        repo_id=args_cli.repo_id,
        root=args_cli.repo_root,
        fps=round(control_hz),
        task=args_cli.single_task,
        camera_names=("side_cam", "wrist_came"),
        resume=args_cli.resume,
    )

    step_count = 0
    episode_count = 0

    recording = True
    timeout = False
    waiting_start_time = None

    episode_random_state = randomizer.sample()
    obs = reset_env(env, episode_random_state)

    # With --resume, this now announces the real next episode index.
    log_say(
        f"Episode {recorder.episode_id}",
        blocking=True,
    )
    print(f"✅ Episode {recorder.episode_id}", flush=True)

    episode_start_time = time.perf_counter()

    try:
        while simulation_app.is_running():

            # ====================================================
            # Quit
            # ====================================================

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
                action = controller.advance()
                action = action.unsqueeze(0)

                obs, _ = env.step(action)

                recorder.add_frame(
                    obs=obs,
                    action=action,
                )

                step_count += 1

                # ------------------------------------------------
                # Save early
                # ------------------------------------------------

                if keyboard_control.consume_save():
                    recorder.save_episode()
                    episode_count += 1

                    log_say(f"save episode {recorder.episode_id - 1}", blocking=False)
                    print(
                        f"✅ Saved episode {recorder.episode_id - 1}",
                        flush=True,
                    )

                    if episode_count >= args_cli.num_episodes:
                        break

                    episode_random_state = randomizer.sample()
                    obs = reset_env(env, episode_random_state)

                    step_count = 0
                    recording = False
                    timeout = False
                    waiting_start_time = time.perf_counter()

                    continue

                # ------------------------------------------------
                # Discard
                # ------------------------------------------------

                if keyboard_control.consume_discard():
                    log_say(f"resetting", blocking=False)
                    recorder.discard_episode()

                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

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
                elapsed = (
                    time.perf_counter() - waiting_start_time
                )

                # Timeout 상태에서 discard -> 같은 랜덤 배치 재시도
                if (
                    timeout
                    and keyboard_control.consume_discard()
                ):
                    log_say(f"resetting", blocking=False)
                    recorder.discard_episode()

                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                    step_count = 0
                    timeout = False
                    recording = False
                    waiting_start_time = time.perf_counter()

                    continue

                if elapsed < args_cli.reset_time_s:
                    continue

                # Timeout 후 discard가 없으면 자동 저장
                if timeout:
                    recorder.save_episode()
                    episode_count += 1
                    log_say(f"save episode {recorder.episode_id - 1}", blocking=False)
                    print(
                        f"✅ Saved episode {recorder.episode_id - 1}",
                        flush=True,
                    )

                    if episode_count >= args_cli.num_episodes:
                        break
                    
                    episode_random_state = randomizer.sample()
                    obs = reset_env(
                        env,
                        episode_random_state,
                    )

                step_count = 0
                timeout = False
                recording = True
                waiting_start_time = None

                episode_start_time = time.perf_counter()

                log_say(
                    f"Episode {recorder.episode_id}",
                    blocking=True,
                )
                print(
                    f"✅ Episode {recorder.episode_id}",
                    flush=True,
                )

    finally:
        print("✅ Finalizing dataset...", flush=True)

        # Q/exception during an episode should not silently create a short,
        # incomplete demonstration.
        if recorder.has_pending_frames():
            print(
                "✅ Discarding unfinished current episode.",
                flush=True,
            )
            recorder.discard_episode()

        recorder.finalize()

        if args_cli.push_to_hub:
            print(
                f"✅ Pushing dataset to Hub: {args_cli.repo_id}",
                flush=True,
            )
            recorder.push_to_hub(
                private=args_cli.private,
            )
            print("✅ Hub push complete.", flush=True)

        keyboard_control.destroy()
        controller.disconnect()

        try:
            env.close()
        except Exception as exc:
            print(
                f"Failed to close environment: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()