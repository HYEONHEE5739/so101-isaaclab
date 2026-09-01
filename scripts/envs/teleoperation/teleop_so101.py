#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Teleoperate the simulated SO-101 robot with a physical SO-101 leader arm."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Teleoperate SO-101 in Isaac Lab using a physical SO-101 leader arm."
)

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