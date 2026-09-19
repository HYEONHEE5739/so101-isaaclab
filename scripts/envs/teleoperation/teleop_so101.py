#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Teleoperate the simulated SO-101 robot with a physical SO-101 leader arm."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Teleoperate SO-101 in Isaac Lab using a physical SO-101 leader arm."
)

parser.add_argument("--workspace", help="Published workspace ID or package path (opt-in)")
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Serial port of the SO-101 leader arm.",
)

parser.add_argument(
    "--recalibrate",
    action="store_true",
    help="Recalibrate the SO-101 leader arm.",
)

parser.add_argument(
    "--calibration_file_name",
    type=str,
    default="so101_leader.json",
    help="Calibration file used by the SO-101 leader arm.",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of simulation environments.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Screenshot
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

# IMPORTANT:
# Replace these names with the actual camera names in SO101TeleopEnvCfg.
CAMERA_NAMES = (
    "camera_sideview",
    "camera_wristview",
)


def save_camera_screenshots(env):
    """Save RGB images from the configured Isaac Lab cameras."""

    from PIL import Image

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for camera_name in CAMERA_NAMES:
        try:
            camera = env.scene[camera_name]

            # Shape is normally [num_envs, height, width, channels].
            rgb = camera.data.output["rgb"][0]

            # Keep RGB only if the sensor returns RGBA.
            rgb = rgb[..., :3]

            image = rgb.detach().cpu().numpy()

            # Convert to uint8 only when required.
            if image.dtype != "uint8":
                image = image.clip(0, 255).astype("uint8")

            save_path = (
                SCREENSHOT_DIR
                / f"{camera_name}_{timestamp}.png"
            )

            Image.fromarray(image).save(save_path)

            print(f"[INFO]: Screenshot saved: {save_path}")

        except Exception as exc:
            print(
                f"[WARNING]: Failed to save camera "
                f"'{camera_name}': {exc}",
                flush=True,
            )




# -----------------------------------------------------------------------------
# Isaac Sim
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    from isaaclab.envs import ManagerBasedEnv

    from soarm101_lab.devices import SO101Leader, SO101LeaderCfg
    from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
    from soarm101_lab.utils.keyboard import KeyboardControl

    # -------------------------------------------------------------------------
    # Keyboard
    # -------------------------------------------------------------------------

    keyboard = KeyboardControl()

    # -------------------------------------------------------------------------
    # Physical leader arm
    # -------------------------------------------------------------------------

    leader_cfg = SO101LeaderCfg(
        port=args_cli.port,
        recalibrate=args_cli.recalibrate,
        calibration_file_name=args_cli.calibration_file_name,
    )

    leader = SO101Leader(leader_cfg)
    leader.connect()

    # -------------------------------------------------------------------------
    # Simulation environment
    # -------------------------------------------------------------------------

    if args_cli.workspace:
        from soarm101_lab.real2sim.workspaces.environment import make_config
        env_cfg = make_config(args_cli.workspace, args_cli.device, args_cli.num_envs)
    else:
        env_cfg = SO101TeleopEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)

    print("SO-101 teleoperation started.")
    print("Q : Quit")
    print(f"Control frequency : {1.0 / env.step_dt:.1f} Hz")

    try:
        while simulation_app.is_running():

            if keyboard.should_quit():
                print("Quit requested.")
                break

            # Read joint targets from the physical SO-101 leader.
            action = leader.advance()

            # Isaac Lab environments expect [num_envs, action_dim].
            action = action.unsqueeze(0)

            # Apply the leader joint targets to the simulated follower.
            env.step(action)

            if keyboard.consume_screenshot():
                save_camera_screenshots(env)

    finally:
        keyboard.destroy()
        leader.disconnect()

        try:
            env.close()
        except Exception as exc:
            print(
                f"Failed to close environment: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()