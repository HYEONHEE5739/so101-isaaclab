# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soarm101_lab.assets.scenes.pick_place import PickPlaceSceneCfg
from .mdp import ee_frame_state, image_raw, apply_episode_randomization
from isaaclab.sensors import CameraCfg

@configclass
class PickPlaceSceneCfg(PickPlaceSceneCfg):
    camera_wristview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/Camera_WristView",  # 실제 USD 프림 경로
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,  # 이미 USD에 존재하는 카메라
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the SO-ARM 101."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        scale=1.0,
        use_default_offset=False,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for teleoperation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        ee_state = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        side_cam = ObsTerm(
            func=image_raw,
            params={
                "sensor_cfg": SceneEntityCfg("camera_sideview"),
                "data_type": "rgb",
            },
        )

        wrist_cam = ObsTerm(
            func=image_raw,
            params={
                "sensor_cfg": SceneEntityCfg("camera_wristview"),
                "data_type": "rgb",
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (resets)."""
    
    # reset episode
    reset_episode = EventTerm(
        func=apply_episode_randomization,
        mode="reset",
        params={
            "random_state": None,
            "cube_red_cfg": SceneEntityCfg("cube_red"),
            "cube_green_cfg": SceneEntityCfg("cube_green"),
            "cube_blue_cfg": SceneEntityCfg("cube_blue"),
            "asset_cfg": SceneEntityCfg("dome_light"),
            "robot_asset": "robot",
        },
    )


@configclass
class SO101TeleopEnvCfg(ManagerBasedEnvCfg):
    """Base teleoperation environment configuration."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards = None  # No rewards for teleoperation
    terminations = None  # No terminations for teleoperation

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 30

        self.sim.dt = 1 / 60    # decimation -> 30fps
        self.sim.render_interval = self.decimation

        self.scene.env_spacing = 2.0
