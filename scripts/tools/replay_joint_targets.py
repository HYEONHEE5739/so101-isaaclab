# SPDX-License-Identifier: BSD-3-Clause
"""Replay generated Mimic joint targets in SO101 joint-space env."""

import argparse
import h5py
import numpy as np
import torch

from isaaclab.app import AppLauncher


# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
)

parser.add_argument(
    "--episode",
    type=str,
    default="demo_0",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# Main
# ============================================================

def main():

    from isaaclab.envs import ManagerBasedEnv

    from soarm101_lab.tasks.manager_based.soarm101_lab import (
        SO101TeleopEnvCfg,
    )

    # --------------------------------------------------------
    # Load joint targets
    # --------------------------------------------------------

    with h5py.File(args_cli.dataset, "r") as f:

        demo = f["data"][args_cli.episode]

        joint_targets = demo["joint_targets"][:]

    joint_targets = np.asarray(
        joint_targets,
        dtype=np.float32,
    )

    print("====================================")
    print("dataset :", args_cli.dataset)
    print("episode :", args_cli.episode)
    print("targets :", joint_targets.shape)
    print("====================================")

    assert joint_targets.ndim == 2
    assert joint_targets.shape[1] == 6

    # --------------------------------------------------------
    # Create JOINT-SPACE environment
    # --------------------------------------------------------

    env_cfg = SO101TeleopEnvCfg()

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    # IMPORTANT:
    # For simple trajectory verification, randomization can
    # interfere with the original generated episode.
    #
    # We'll deal with restoring the exact initial scene later.

    env = ManagerBasedEnv(cfg=env_cfg)

    print(f"control_dt : {env.step_dt}")
    print(f"control_hz : {1.0 / env.step_dt}")

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    obs, _ = env.reset()

    # --------------------------------------------------------
    # Replay
    # --------------------------------------------------------

    for t in range(len(joint_targets)):

        action = torch.tensor(
            joint_targets[t],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)

        obs, _ = env.step(action)

        # Debug
        if t % 30 == 0:

            robot = env.scene["robot"]

            q = (
                robot.data.joint_pos[0]
                .detach()
                .cpu()
                .numpy()
            )

            print(
                f"\n[{t:04d}/{len(joint_targets)}]"
            )

            print(
                "target:",
                np.round(joint_targets[t], 4),
            )

            print(
                "actual:",
                np.round(q, 4),
            )

    print("\nReplay finished.")

    env.close()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()