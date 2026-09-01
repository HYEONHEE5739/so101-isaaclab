# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations


from isaaclab.envs.mdp.actions import JointPositionActionCfg

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Adjust these two imports to your actual module path.
from .base_pick_place_teleop_env_cfg import (
    ObservationsCfg as BaseObservationsCfg,
    SO101TeleopEnvCfg,
)
from .mdp import so101_mimic_mdp as mimic_mdp
from .mdp import SO101PinocchioIKActionCfg

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


@configclass
class MimicActionsCfg:
    """Mimic action space: 6D relative TCP command + 1 gripper joint command."""

    arm = SO101PinocchioIKActionCfg(
        asset_name="robot",

        joint_names=ARM_JOINT_NAMES,

        body_name="tool0",

        urdf_path=(
            "/home/hyeonhee/soarm_isaaclab/"
            "soarm101_lab/assets/SO101/urdf/"
            "so101_isaaclab.urdf"
        ),

        scale=(
            1.0,  # dx
            1.0,  # dy
            1.0,  # dz
            1.0,  # rx 
            1.0,  # ry 
            1.0,  # rz 
        ),

        position_weight=1.0,
        orientation_weight=0.01,

        max_iterations=200,
        tolerance=1e-3,
        damping=1e-3,
        ik_step_size=0.3,

        debug=False,
    )

    gripper = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class MimicObservationsCfg(BaseObservationsCfg):
    """Your existing observations + Mimic annotation/debug observations."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Heuristic signals used for automatic Mimic annotation."""

        # cube_red is only a DEFAULT.
        # run_so101_mimic.py replaces object_cfg before env creation.
        grasp = ObsTerm(
            func=mimic_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_red"),
                "distance_threshold": 0.045,
                "gripper_closed_threshold": 0.3,
                "gripper_open_threshold": 0.45
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    @configclass
    class TaskStateCfg(ObsGroup):
        """Debug/success signals. Not used to split the two Mimic subtasks."""

        # Defaults are replaced from main before environment creation.
        place_success = ObsTerm(
            func=mimic_mdp.object_in_bin,
            params={
                "object_cfg": SceneEntityCfg("cube_red"),
                "bin_cfg": SceneEntityCfg("bin_a"),
                # Tune these to your bin's actual interior dimensions.
                "x_bounds": (-0.0255, 0.0255),
                "y_bounds": (-0.0255, 0.0255),
                "z_bounds": (0.0, 0.02),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    task_state: TaskStateCfg = TaskStateCfg()

@configclass
class TerminationsCfg:
    """Termination terms for Mimic."""

    success = DoneTerm(
        func=mimic_mdp.object_in_bin,
        params={
            "object_cfg": SceneEntityCfg("cube_red"),
            "bin_cfg": SceneEntityCfg("bin_a"),

            "x_bounds": (-0.0255, 0.0255),
            "y_bounds": (-0.0255, 0.0255),
            "z_bounds": (0.01, 0.06),
        },
    )


@configclass
class SO101PickPlaceMimicEnvCfg(SO101TeleopEnvCfg, MimicEnvCfg):
    """SO101 pick-place config for Isaac Lab Mimic."""
    """
    [MimicObservationsCfg]

    grasp = ObsTerm(func=object_grasped)
            │
            │ 매 step ObservationManager가 실행
            ↓
    object_grasped()
            │
            │ True / False 반환
            ↓
    self.obs_buf
    ┌──────────────────────────────┐
    │ "subtask_terms"              │
    │    └─ "grasp": True/False    │
    └──────────────────────────────┘
            │
            │ env.py
            ↓
    get_subtask_term_signals()
            │
            ↓
    { "grasp": True/False }  
            │
            │ Mimic이 읽음
            ↓
    SubTaskConfig(subtask_term_signal="grasp") -- 즉 이 singal 사용하려면 get_subtask_term_signals에 {"grasp": value}가 있어야 한다.
            │
            ↓
    "아! grasp=True가 됐네"
            ↓
    Subtask 1 종료
    """
    # Completely replace the old joint_positions action group.
    actions: MimicActionsCfg = MimicActionsCfg()
    observations: MimicObservationsCfg = MimicObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()

        # Exact geometry for data generation.
        self.observations.policy.enable_corruption = False

        # Start minimal; tune Mimic generation parameters later.
        self.datagen_config.name = "so101_pick_place_mimic"

        # Defaults only. run_so101_mimic.py replaces these object_ref values
        # from --grasp_object and --place_bin BEFORE the env is created.
        self.subtask_configs["tool0"] = [
            # Subtask 1: approach + grasp.
            SubTaskConfig(
                object_ref="cube_red",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(0, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),

            # Subtask 2: transport + place in bin.
            # Final subtask does not need a termination annotation signal.
            SubTaskConfig(
                object_ref="bin_a",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
