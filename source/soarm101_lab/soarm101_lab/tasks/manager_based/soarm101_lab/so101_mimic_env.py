# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SO101PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """SO101 interface required by Isaac Lab Mimic."""

    EEF_NAME = "tool0"

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(
            cfg=cfg,
            render_mode=render_mode,
            **kwargs,
        )

        # env별로 gripper가 한 번이라도 열렸는지 기억
        self._gripper_opened_once = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self._grasp_completed = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if hasattr(self, "_gripper_opened_once"):
            self._gripper_opened_once[env_ids] = False

        if hasattr(self, "_grasp_completed"):
            self._grasp_completed[env_ids] = False


    def get_robot_eef_pose(
        self,
        eef_name: str,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return Robot/base -> tool0 pose as homogeneous matrices."""
        if eef_name != self.EEF_NAME:
            raise ValueError(f"Unknown EEF '{eef_name}'. Expected '{self.EEF_NAME}'.")

        if env_ids is None:
            env_ids = slice(None)

        ee_state = self.obs_buf["policy"]["ee_state"][env_ids]
        pos = ee_state[..., :3]
        quat = ee_state[..., 3:7]  # (w, x, y, z)

        return math_utils.make_pose(
            pos,
            math_utils.matrix_from_quat(quat),
        )

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Target tool0 pose -> command-relative [dpos(3), drot(3), gripper]."""

        # ---------------------------------------------------------
        # 1. Mimic이 원하는 absolute target pose
        # ---------------------------------------------------------
        target_pose = target_eef_pose_dict[self.EEF_NAME]
        target_pos, target_rot = math_utils.unmake_pose(target_pose)

        # ---------------------------------------------------------
        # 2. Arm ActionTerm 가져오기
        #    SO101PinocchioIKAction이 직전에 명령했던 command pose 사용
        # ---------------------------------------------------------
        arm_term = self.action_manager.get_term("arm")

        # reset 직후 아직 command pose가 초기화되지 않았을 수도 있으므로
        # 그 경우에만 실제 EEF pose를 기준으로 사용
        if not arm_term._command_initialized[env_id].item():
            current_pose = self.get_robot_eef_pose(
                self.EEF_NAME,
                env_ids=[env_id],
            )[0]

            command_pos, command_rot = math_utils.unmake_pose(current_pose)

        else:
            command_pos = arm_term._command_pos_b[env_id]
            command_quat = arm_term._command_quat_b[env_id]

            command_rot = math_utils.matrix_from_quat(
                command_quat.unsqueeze(0)
            )[0]

        # ---------------------------------------------------------
        # 3. Previous command -> Mimic target delta
        #
        # ActionTerm 내부에서:
        #
        #   new_command = previous_command + delta
        #
        # 를 하기 때문에 여기서는 반드시
        #
        #   delta = target - previous_command
        #
        # 이어야 함.
        # ---------------------------------------------------------
        delta_pos = target_pos - command_pos

        delta_rot_matrix = (
            target_rot
            @ command_rot.transpose(-1, -2)
        )

        delta_quat = math_utils.quat_from_matrix(
            delta_rot_matrix
        )

        delta_rot = math_utils.axis_angle_from_quat(
            delta_quat
        )

        pose_action = torch.cat(
            [delta_pos, delta_rot],
            dim=-1,
        )

        # ---------------------------------------------------------
        # 4. DEBUG
        # ---------------------------------------------------------
        # if not hasattr(self, "_mimic_action_debug_count"):
        #     self._mimic_action_debug_count = 0

        # if self._mimic_action_debug_count % 20 == 0:
        #     print(
        #         "\n[MIMIC ACTION DEBUG]"
        #         f"\n  command_pos = {command_pos.detach().cpu().numpy()}"
        #         f"\n  target_pos  = {target_pos.detach().cpu().numpy()}"
        #         f"\n  delta_pos   = {delta_pos.detach().cpu().numpy()}"
        #         f"\n  delta_rot   = {delta_rot.detach().cpu().numpy()}",
        #         flush=True,
        #     )

        # self._mimic_action_debug_count += 1

        # ---------------------------------------------------------
        # 5. Action noise
        # ---------------------------------------------------------
        if action_noise_dict is not None:
            noise = action_noise_dict.get(
                self.EEF_NAME,
                None,
            )

            if noise is not None:

                if isinstance(noise, dict):
                    pose_action = pose_action.clone()

                    pose_action[:3] += (
                        torch.randn_like(
                            pose_action[:3]
                        )
                        * float(
                            noise.get(
                                "position",
                                0.0,
                            )
                        )
                    )

                    pose_action[3:] += (
                        torch.randn_like(
                            pose_action[3:]
                        )
                        * float(
                            noise.get(
                                "orientation",
                                0.0,
                            )
                        )
                    )

                else:
                    pose_action += (
                        torch.randn_like(pose_action)
                        * float(noise)
                    )

        # ---------------------------------------------------------
        # 6. Gripper
        # ---------------------------------------------------------
        gripper_action = (
            gripper_action_dict[self.EEF_NAME]
            .reshape(-1)
        )

        return torch.cat(
            [pose_action, gripper_action],
            dim=-1,
        )

    def action_to_target_eef_pose(
        self,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Relative Cartesian action -> absolute target tool0 pose."""
        delta_pos = action[..., :3]
        delta_rot = action[..., 3:6]

        current_pose = self.get_robot_eef_pose(self.EEF_NAME)
        current_pos, current_rot = math_utils.unmake_pose(current_pose)

        target_pos = current_pos + delta_pos

        angle = torch.linalg.norm(delta_rot, dim=-1)
        safe_angle = torch.clamp(angle, min=1.0e-8)
        axis = delta_rot / safe_angle.unsqueeze(-1)

        delta_quat = math_utils.quat_from_angle_axis(angle, axis)

        zero_mask = angle < 1.0e-8
        if zero_mask.any():
            delta_quat = delta_quat.clone()
            delta_quat[zero_mask] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=delta_quat.device,
                dtype=delta_quat.dtype,
            )

        target_rot = math_utils.matrix_from_quat(delta_quat) @ current_rot
        target_pose = math_utils.make_pose(target_pos, target_rot)

        return {self.EEF_NAME: target_pose}

    def actions_to_gripper_actions(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract gripper command from full action sequence."""
        return {self.EEF_NAME: actions[..., -1:]}

    def get_object_poses(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return poses for every object_ref currently used by subtask_configs."""
        if env_ids is None:
            env_ids = slice(None)

        # No extra hard-coded mimic_object_names list:
        # derive the needed references directly from SubTaskConfig.
        object_names = {
            subtask.object_ref
            for subtask in self.cfg.subtask_configs[self.EEF_NAME]
            if subtask.object_ref is not None
        }

        robot = self.scene["robot"]
        base_pos_w = robot.data.root_pos_w[env_ids]
        base_quat_w = robot.data.root_quat_w[env_ids]

        result = {}

        for object_name in object_names:
            obj = self.scene[object_name]

            pos_b, quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                obj.data.root_pos_w[env_ids],
                obj.data.root_quat_w[env_ids],
            )

            result[object_name] = math_utils.make_pose(
                pos_b,
                math_utils.matrix_from_quat(quat_b),
            )

        return result

    def get_subtask_term_signals(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Automatic annotation signal for the only non-final boundary."""
        if env_ids is None:
            env_ids = slice(None)

        terms = self.obs_buf["subtask_terms"]

        grasp = terms["grasp"][env_ids]

        if torch.any(grasp):
            print(
                "🔥 GRASP SIGNAL TRUE:",
                grasp.detach().cpu().numpy(),
                flush=True,
            )

        return {
            "grasp": grasp,
        }
