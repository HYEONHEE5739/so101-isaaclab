# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import random
import numpy as np

from soarm101_lab.assets import SO101_FOLLOWER_INITIAL_JOINT_POS


@dataclass
class EpisodeRandomState:
    """한 에피소드에서 사용할 랜덤 환경"""

    cube_positions: dict[str, tuple]
    light_exposure: float
    joint_positions: dict[str, float]


class EpisodeRandomizer:

    def __init__(
        self,
        pick_zone_center=(0.0, 0.265, 0.61),
        offset=0.0375,
        exposure_range=(-2.0, 1.0),
        joint_error_range=(-0.05, 0.05),
    ):
        self.pick_zone_center = pick_zone_center
        self.offset = offset

        self.exposure_range = exposure_range
        self.joint_error_range = joint_error_range

    def sample(self) -> EpisodeRandomState:

        ##########################
        # Cube Position
        ##########################

        # 4개의 기준 위치
        P1 = (-self.offset, +self.offset)  # 왼쪽 위
        P2 = (+self.offset, +self.offset)  # 오른쪽 위
        P3 = (+self.offset, -self.offset)  # 오른쪽 아래
        P4 = (-self.offset, -self.offset)  # 왼쪽 아래

        configs = {
            "C1": {
                "cube_red":   P1,
                "cube_blue":  P2,
                "cube_green": P3,
            },
            "C2": {
                "cube_red":   P4,   
                "cube_blue":  P1,
                "cube_green": P2,
            },
            "C3": {
                "cube_red":   P3,
                "cube_blue":  P4,
                "cube_green": P1,
            },
            "C4": {
                "cube_red":   P2,
                "cube_blue":  P3,
                "cube_green": P4,
            },
        }

        # C1 ~ C4 중 하나 랜덤 선택
        config_name = random.choice(list(configs.keys()))
        selected_config = configs[config_name]

        # 기준 위치 주변 jitter
        position_jitter = 0.01  # ±1cm

        cube_positions = {}

        for cube_name, (dx, dy) in selected_config.items():
            jitter_x = random.uniform(-position_jitter, position_jitter)
            jitter_y = random.uniform(-position_jitter, position_jitter)

            cube_positions[cube_name] = (
                self.pick_zone_center[0] + dx + jitter_x,
                self.pick_zone_center[1] + dy + jitter_y,
                self.pick_zone_center[2] + 0.02,
            )

        ##########################
        # Light
        ##########################

        exposure = random.uniform(*self.exposure_range)

        ##########################
        # Joint Noise
        ##########################

        joint_positions = {}

        for joint_name, default_pos in SO101_FOLLOWER_INITIAL_JOINT_POS.items():
            err = random.uniform(*self.joint_error_range)
            joint_positions[joint_name] = default_pos + err

        return EpisodeRandomState(
            cube_positions=cube_positions,
            light_exposure=exposure,
            joint_positions=joint_positions,
        )