# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .base_pick_place_teleop_env_cfg import SO101TeleopEnvCfg
from .so101_mimic_env import SO101PickPlaceMimicEnv
from .so101_mimic_env_cfg import SO101PickPlaceMimicEnvCfg
__all__ = [
    "SO101PickPlaceMimicEnv",
    "SO101PickPlaceMimicEnvCfg",
]

##
# Register Gym environments.
##

import gymnasium as gym


gym.register(
    id="SO101-PickPlace-Mimic-v1",
    entry_point=(
        "soarm101_lab.tasks.manager_based.soarm101_lab:"
        "SO101PickPlaceMimicEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "soarm101_lab.tasks.manager_based.soarm101_lab:"
            "SO101PickPlaceMimicEnvCfg"
        ),
    },
)
