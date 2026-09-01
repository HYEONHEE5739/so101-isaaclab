# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

gym.register(
    id="SO101-PickPlace-Mimic-v0",

    entry_point=
        "soarm101_lab.tasks.manager_based.soarm101_lab:SO101PickPlaceMimicEnv",

    disable_env_checker=True,

    kwargs={
        "env_cfg_entry_point":
            "soarm101_lab.tasks.manager_based.soarm101_lab:SO101PickPlaceMimicEnvCfg",
    },
)