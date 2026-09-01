from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from soarm101_lab.utils.so101_kinematics import SO101Kinematics


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .so101_ik_actions_cfg import SO101PinocchioIKActionCfg


class SO101PinocchioIKAction(ActionTerm):
    """SO101 task-space action using custom Pinocchio IK.

    Raw action:
        [dx, dy, dz, dRx, dRy, dRz]

    The action is a relative Cartesian delta pose in the robot
    base frame.

    IMPORTANT:
    Recorded Mimic deltas are integrated from the previous
    commanded pose instead of being re-anchored to the actual
    replay pose every control step.
    """

    cfg: "SO101PinocchioIKActionCfg"

    _asset: Articulation

    def __init__(
        self,
        cfg: "SO101PinocchioIKActionCfg",
        env: "ManagerBasedEnv",
    ):
        # ----------------------------------------------------------
        # Isaac Lab ActionTerm initialization
        # ----------------------------------------------------------
        super().__init__(cfg, env)

        # ----------------------------------------------------------
        # Resolve arm joints
        # ----------------------------------------------------------
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names,
            preserve_order=True,
        )

        self._num_joints = len(self._joint_ids)

        if self._num_joints != 5:
            raise ValueError(
                "SO101PinocchioIKAction expects exactly 5 arm joints. "
                f"Found {self._num_joints}: "
                f"{self._joint_names} [{self._joint_ids}]"
            )

        # ----------------------------------------------------------
        # Resolve tool0 body
        # ----------------------------------------------------------
        body_ids, body_names = self._asset.find_bodies(
            self.cfg.body_name
        )

        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching "
                f"'{self.cfg.body_name}'. "
                f"Found {len(body_ids)}: {body_names}"
            )

        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        omni.log.info(
            f"Resolved joint names for action term "
            f"{self.__class__.__name__}: "
            f"{self._joint_names} [{self._joint_ids}]"
        )

        omni.log.info(
            f"Resolved body name for action term "
            f"{self.__class__.__name__}: "
            f"{self._body_name} [{self._body_idx}]"
        )

        # ----------------------------------------------------------
        # SO101 Pinocchio kinematics
        # ----------------------------------------------------------
        self._kinematics = SO101Kinematics(
            urdf_path=self.cfg.urdf_path,
            target_frame_name=self.cfg.body_name,
            joint_names=self._joint_names,
        )

        # ----------------------------------------------------------
        # Action buffers
        #
        # [dx, dy, dz, dRx, dRy, dRz]
        # ----------------------------------------------------------
        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        self._processed_actions = torch.zeros_like(
            self._raw_actions
        )

        self._joint_position_targets = torch.zeros(
            self.num_envs,
            self._num_joints,
            device=self.device,
        )

        # ----------------------------------------------------------
        # Scale
        # ----------------------------------------------------------
        self._scale = torch.zeros(
            (self.num_envs, self.action_dim),
            device=self.device,
        )

        self._scale[:] = torch.tensor(
            self.cfg.scale,
            device=self.device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------------
        # Final IK target pose
        # robot base frame
        # ----------------------------------------------------------
        self._target_pos_b = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )

        self._target_quat_b = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
        )

        # ----------------------------------------------------------
        # Integrated command pose
        #
        # Demo:
        #
        # delta[t] = pose[t+1] relative to pose[t]
        #
        # We therefore reconstruct the command trajectory by
        # accumulating these deltas.
        # ----------------------------------------------------------
        self._command_pos_b = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )

        self._command_quat_b = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
        )

        self._command_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # ----------------------------------------------------------
        # Debug information
        # ----------------------------------------------------------
        self._last_ik_success = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    # ==============================================================
    # Properties
    # ==============================================================

    @property
    def action_dim(self) -> int:
        """Dimension of the arm task-space action."""
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_position_targets(self) -> torch.Tensor:
        return self._joint_position_targets

    # ==============================================================
    # Action processing
    # ==============================================================

    def process_actions(self, actions: torch.Tensor):
        """Process relative 6D task-space action.

        action:
            [dx, dy, dz, dRx, dRy, dRz]

        Translation:
            command_pos[t+1] =
                command_pos[t] + delta_pos[t]

        Rotation:
            R_command[t+1] =
                R_delta[t] @ R_command[t]
        """

        # ----------------------------------------------------------
        # Store / scale
        # ----------------------------------------------------------
        self._raw_actions[:] = actions

        self._processed_actions[:] = (
            self._raw_actions * self._scale
        )

        # ----------------------------------------------------------
        # Current ACTUAL tool0 pose
        #
        # This is used only to initialize the command trajectory.
        # ----------------------------------------------------------
        ee_pos_b, ee_quat_b = self._compute_frame_pose()

        # ----------------------------------------------------------
        # Initialize command pose once after reset
        # ----------------------------------------------------------
        not_initialized = ~self._command_initialized

        if not_initialized.any():

            self._command_pos_b[not_initialized] = (
                ee_pos_b[not_initialized]
            )

            self._command_quat_b[not_initialized] = (
                ee_quat_b[not_initialized]
            )

            self._command_initialized[not_initialized] = True

        # ==========================================================
        # Translation
        # ==========================================================

        delta_pos = self._processed_actions[:, 0:3]

        # IMPORTANT:
        #
        # Do NOT use:
        #
        #   actual_ee_pos + delta
        #
        # Instead integrate from previous commanded pose.
        self._command_pos_b[:] = (
            self._command_pos_b
            + delta_pos
        )
        """"""
        expected_target_from_action = ee_pos_b + delta_pos

        command_error = (
            self._command_pos_b
            - expected_target_from_action
        )

        if self.cfg.debug:
            print(
                "\n[COMMAND SEMANTICS DEBUG]"
                "\n actual_pos      =",
                ee_pos_b[0].detach().cpu().numpy(),
                "\n delta_pos       =",
                delta_pos[0].detach().cpu().numpy(),
                "\n expected_target =",
                expected_target_from_action[0].detach().cpu().numpy(),
                "\n command_pos     =",
                self._command_pos_b[0].detach().cpu().numpy(),
                "\n command_error   =",
                command_error[0].detach().cpu().numpy(),
                flush=True,
            )
        """"""
        # ==========================================================
        # Rotation
        # ==========================================================

        delta_rot_vec = self._processed_actions[:, 3:6]

        # Axis-angle vector:
        #
        # vector direction = axis
        # vector norm      = angle
        delta_angle = torch.linalg.norm(
            delta_rot_vec,
            dim=-1,
        )

        eps = 1.0e-8

        safe_angle = torch.clamp(
            delta_angle,
            min=eps,
        )

        delta_axis = (
            delta_rot_vec
            / safe_angle.unsqueeze(-1)
        )

        # Axis does not matter when angle == 0.
        default_axis = torch.zeros_like(
            delta_axis
        )

        default_axis[:, 0] = 1.0

        zero_mask = delta_angle < eps

        delta_axis = torch.where(
            zero_mask.unsqueeze(-1),
            default_axis,
            delta_axis,
        )

        delta_quat = math_utils.quat_from_angle_axis(
            delta_angle,
            delta_axis,
        )

        # ----------------------------------------------------------
        # Recorder convention:
        #
        # delta_R = R_next @ R_prev.T
        #
        # Therefore:
        #
        # R_next = delta_R @ R_prev
        #
        # Quaternion:
        #
        # q_next = q_delta * q_prev
        # ----------------------------------------------------------
        self._command_quat_b[:] = math_utils.quat_mul(
            delta_quat,
            self._command_quat_b,
        )

        # ----------------------------------------------------------
        # Final IK target
        # ----------------------------------------------------------
        self._target_pos_b[:] = self._command_pos_b
        self._target_quat_b[:] = self._command_quat_b

        # ----------------------------------------------------------
        # Debug
        # ----------------------------------------------------------
        if self.cfg.debug:

            tracking_error = (
                self._target_pos_b[0]
                - ee_pos_b[0]
            )

            print(
                "[ACTION DEBUG]",
                "\n raw          =",
                self._raw_actions[0]
                .detach()
                .cpu()
                .numpy(),

                "\n delta_pos    =",
                delta_pos[0]
                .detach()
                .cpu()
                .numpy(),

                "\n actual_pos   =",
                ee_pos_b[0]
                .detach()
                .cpu()
                .numpy(),

                "\n command_pos  =",
                self._command_pos_b[0]
                .detach()
                .cpu()
                .numpy(),

                "\n tracking_err =",
                tracking_error
                .detach()
                .cpu()
                .numpy(),

                flush=True,
            )

    # ==============================================================
    # Apply actions
    # ==============================================================

    def apply_actions(self):
        """Solve IK and apply resulting joint-position targets."""

        # ----------------------------------------------------------
        # Current arm joint positions
        # ----------------------------------------------------------
        joint_pos = self._asset.data.joint_pos[
            :, self._joint_ids
        ]

        # Hold current configuration if IK fails.
        joint_pos_des = joint_pos.clone()

        # ----------------------------------------------------------
        # Pinocchio solver is CPU / numpy based.
        # Solve each environment independently.
        # ----------------------------------------------------------
        for env_id in range(self.num_envs):

            q_current = (
                joint_pos[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            # ------------------------------------------------------
            # Current FK
            # ------------------------------------------------------
            T_current = (
                self._kinematics.forward_kinematics(
                    q_current
                )
            )

            # ------------------------------------------------------
            # Build target transform
            # ------------------------------------------------------
            T_target = T_current.copy()

            target_pos_b = (
                self._target_pos_b[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            target_quat_b = (
                self._target_quat_b[env_id]
                .unsqueeze(0)
            )

            target_rot_b = (
                math_utils.matrix_from_quat(
                    target_quat_b
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            T_target[:3, 3] = target_pos_b
            T_target[:3, :3] = target_rot_b

            # ------------------------------------------------------
            # Solve IK
            # ------------------------------------------------------
            q_target, success, info = (
                self._kinematics.inverse_kinematics(
                    current_joint_pos_rad=q_current,
                    desired_ee_pose=T_target,

                    position_weight=(
                        self.cfg.position_weight
                    ),

                    orientation_weight=(
                        self.cfg.orientation_weight
                    ),

                    max_iterations=(
                        self.cfg.max_iterations
                    ),

                    tolerance=(
                        self.cfg.tolerance
                    ),

                    damping=(
                        self.cfg.damping
                    ),

                    step_size=(
                        self.cfg.ik_step_size
                    ),
                )
            )

            self._last_ik_success[env_id] = success

            # ------------------------------------------------------
            # IK success
            # ------------------------------------------------------
            if success:

                q_target_tensor = torch.as_tensor(
                    q_target,
                    device=self.device,
                    dtype=joint_pos.dtype,
                )

                joint_pos_des[env_id] = (
                    q_target_tensor
                )
            
            # ------------------------------------------------------
            # IK failure -> hold current q
            # ------------------------------------------------------
            else:

                joint_pos_des[env_id] = (
                    joint_pos[env_id]
                )

                if self.cfg.debug:
                    print(
                        "[SO101PinocchioIKAction] "
                        f"env={env_id} IK failed | "
                        f"info={info}",
                        flush=True,
                    )

        # ----------------------------------------------------------
        # Store joint position targets for recorder
        # ----------------------------------------------------------
        self._joint_position_targets[:] = joint_pos_des
        
        # ----------------------------------------------------------
        # Apply joint position targets
        # ----------------------------------------------------------
        self._asset.set_joint_position_target(
            joint_pos_des,
            joint_ids=self._joint_ids,
        )

    # ==============================================================
    # Reset
    # ==============================================================

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:

        if env_ids is None:
            env_ids = slice(None)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        self._target_pos_b[env_ids] = 0.0
        self._target_quat_b[env_ids] = 0.0

        self._command_pos_b[env_ids] = 0.0
        self._command_quat_b[env_ids] = 0.0

        # Next process_actions() initializes from actual tool0.
        self._command_initialized[env_ids] = False

    # ==============================================================
    # Helpers
    # ==============================================================

    def _compute_frame_pose(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute physical tool0 pose in robot base frame."""

        # ----------------------------------------------------------
        # tool0 in world
        # ----------------------------------------------------------
        ee_pos_w = self._asset.data.body_pos_w[
            :, self._body_idx
        ]

        ee_quat_w = self._asset.data.body_quat_w[
            :, self._body_idx
        ]

        # ----------------------------------------------------------
        # robot base/root in world
        # ----------------------------------------------------------
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w

        # ----------------------------------------------------------
        # world -> robot base
        #
        # T_base_tool0
        # ----------------------------------------------------------
        ee_pos_b, ee_quat_b = (
            math_utils.subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                ee_pos_w,
                ee_quat_w,
            )
        )

        return ee_pos_b, ee_quat_b