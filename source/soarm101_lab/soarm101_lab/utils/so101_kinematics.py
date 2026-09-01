from __future__ import annotations

import numpy as np
import pinocchio as pin


class SO101Kinematics:
    """
    SO101 FK / IK using Pinocchio.

    - Input/output joint positions: radians
    - End-effector pose: 4x4 homogeneous transform
    - Default EE frame: tool0
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "tool0",
        joint_names: list[str] | None = None,
    ):
        # ---------------------------------------------------------
        # Load URDF
        # ---------------------------------------------------------
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.target_frame_name = target_frame_name

        if joint_names is None:
            raise ValueError("joint_names must be provided.")

        self.joint_names = joint_names

        # ---------------------------------------------------------
        # Find target frame
        # ---------------------------------------------------------
        if not self.model.existFrame(target_frame_name):
            frame_names = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"Frame '{target_frame_name}' not found in URDF.\n"
                f"Available frames:\n{frame_names}"
            )

        self.frame_id = self.model.getFrameId(target_frame_name)

        # ---------------------------------------------------------
        # Map our arm joints -> Pinocchio q indices
        #
        # SO101 revolute joints are 1-DoF, so idx_q points to the
        # corresponding position inside Pinocchio's q vector.
        # ---------------------------------------------------------
        self.q_indices = []

        for joint_name in self.joint_names:
            joint_id = self.model.getJointId(joint_name)

            if joint_id == 0:
                raise ValueError(
                    f"Joint '{joint_name}' not found in URDF."
                )

            joint_model = self.model.joints[joint_id]

            if joint_model.nq != 1 or joint_model.nv != 1:
                raise ValueError(
                    f"Joint '{joint_name}' is not 1-DoF: "
                    f"nq={joint_model.nq}, nv={joint_model.nv}"
                )

            self.q_indices.append(joint_model.idx_q)

        self.q_indices = np.asarray(self.q_indices, dtype=int)

        print("[SO101Kinematics]")
        print("  nq:", self.model.nq)
        print("  nv:", self.model.nv)
        print("  target frame:", target_frame_name)
        print("  frame id:", self.frame_id)
        print("  arm joints:", self.joint_names)
        print("  q indices:", self.q_indices.tolist())

    # =============================================================
    # Internal helpers
    # =============================================================

    def _make_full_q(self, arm_joint_pos: np.ndarray) -> np.ndarray:
        """
        Construct Pinocchio's full q vector from the 5 arm joints.

        Other movable joints (for example the gripper) are initialized
        to Pinocchio's neutral configuration.
        """
        arm_joint_pos = np.asarray(
            arm_joint_pos,
            dtype=np.float64,
        ).reshape(-1)

        if len(arm_joint_pos) != len(self.joint_names):
            raise ValueError(
                f"Expected {len(self.joint_names)} arm joints, "
                f"got {len(arm_joint_pos)}"
            )

        q = pin.neutral(self.model)
        q[self.q_indices] = arm_joint_pos

        return q

    def _extract_arm_q(self, q: np.ndarray) -> np.ndarray:
        return q[self.q_indices].copy()

    # =============================================================
    # Forward kinematics
    # =============================================================

    def forward_kinematics(
        self,
        joint_pos_rad: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tool0 pose from arm joint positions.

        Args:
            joint_pos_rad:
                shape (5,), radians

        Returns:
            T_base_tool0:
                shape (4, 4)
        """
        q = self._make_full_q(joint_pos_rad)

        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        placement = self.data.oMf[self.frame_id]

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = placement.rotation
        T[:3, 3] = placement.translation

        return T

    # =============================================================
    # Inverse kinematics
    # =============================================================

    def inverse_kinematics(
        self,
        current_joint_pos_rad: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        damping: float = 1e-4,
        step_size: float = 0.5,
    ) -> tuple[np.ndarray, bool, dict]:
        """
        Iterative weighted DLS IK.

        For the first SO101 test, use orientation_weight=0.0.
        This gives position-only IK, which is useful for a 5-DoF arm.

        Returns:
            q_arm:
                solved arm joint positions, radians

            success:
                True if weighted task error converged

            info:
                debugging information
        """

        current_joint_pos_rad = np.asarray(
            current_joint_pos_rad,
            dtype=np.float64,
        ).reshape(-1)

        desired_ee_pose = np.asarray(
            desired_ee_pose,
            dtype=np.float64,
        )

        if desired_ee_pose.shape != (4, 4):
            raise ValueError(
                f"desired_ee_pose must be (4,4), "
                f"got {desired_ee_pose.shape}"
            )

        q = self._make_full_q(current_joint_pos_rad)

        target = pin.SE3(
            desired_ee_pose[:3, :3],
            desired_ee_pose[:3, 3],
        )

        # Task weighting:
        # Pinocchio spatial vectors are [linear, angular] for the
        # frame Jacobian returned here.
        weights = np.array(
            [
                position_weight,
                position_weight,
                position_weight,
                orientation_weight,
                orientation_weight,
                orientation_weight,
            ],
            dtype=np.float64,
        )

        active_rows = weights > 0.0

        if not np.any(active_rows):
            raise ValueError(
                "At least one IK weight must be > 0."
            )

        success = False
        pos_error_norm = np.inf
        rot_error_norm = np.inf
        weighted_error_norm = np.inf

        for iteration in range(max_iterations):

            # -----------------------------------------------------
            # FK
            # -----------------------------------------------------
            pin.forwardKinematics(
                self.model,
                self.data,
                q,
            )

            pin.updateFramePlacements(
                self.model,
                self.data,
            )

            current = self.data.oMf[self.frame_id]

            # -----------------------------------------------------
            # Pose error
            #
            # Translation error is expressed in base/world coordinates.
            # Orientation error uses SO(3) logarithm.
            # -----------------------------------------------------
            pos_error = (
                target.translation
                - current.translation
            )

            R_error = (
                target.rotation
                @ current.rotation.T
            )

            rot_error = pin.log3(R_error)

            error = np.concatenate(
                [pos_error, rot_error]
            )

            pos_error_norm = np.linalg.norm(pos_error)
            rot_error_norm = np.linalg.norm(rot_error)

            weighted_error = (
                weights[active_rows]
                * error[active_rows]
            )

            weighted_error_norm = np.linalg.norm(
                weighted_error
            )

            if weighted_error_norm < tolerance:
                success = True
                break

            # -----------------------------------------------------
            # tool0 Jacobian
            #
            # WORLD_ALIGNED gives linear/angular components aligned
            # with the base/world axes, matching the error above.
            # -----------------------------------------------------
            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            # Only use our five arm velocity columns.
            #
            # For 1-DoF revolute joints idx_q == idx_v for this
            # fixed-base SO101 model in the normal URDF layout.
            # Build velocity indices explicitly for safety.
            v_indices = []

            for joint_name in self.joint_names:
                joint_id = self.model.getJointId(joint_name)
                v_indices.append(
                    self.model.joints[joint_id].idx_v
                )

            J_arm = J[:, v_indices]

            # Keep only enabled task rows.
            J_task = J_arm[active_rows, :]
            e_task = error[active_rows]
            w_task = weights[active_rows]

            J_weighted = (
                w_task[:, None] * J_task
            )

            e_weighted = (
                w_task * e_task
            )

            # -----------------------------------------------------
            # Damped Least Squares
            #
            # dq = J^T (J J^T + lambda I)^-1 e
            # -----------------------------------------------------
            A = (
                J_weighted @ J_weighted.T
                + damping
                * np.eye(J_weighted.shape[0])
            )

            dq_arm = (
                J_weighted.T
                @ np.linalg.solve(
                    A,
                    e_weighted,
                )
            )

            dq_arm *= step_size

            # -----------------------------------------------------
            # Update only the five arm joints
            # -----------------------------------------------------
            q[self.q_indices] += dq_arm

            # -----------------------------------------------------
            # Joint limits
            # -----------------------------------------------------
            lower = self.model.lowerPositionLimit
            upper = self.model.upperPositionLimit

            for q_idx in self.q_indices:
                if np.isfinite(lower[q_idx]):
                    q[q_idx] = max(
                        q[q_idx],
                        lower[q_idx],
                    )

                if np.isfinite(upper[q_idx]):
                    q[q_idx] = min(
                        q[q_idx],
                        upper[q_idx],
                    )

        q_arm = self._extract_arm_q(q)

        info = {
            "iterations": iteration + 1,
            "position_error": pos_error_norm,
            "orientation_error": rot_error_norm,
            "weighted_error": weighted_error_norm,
        }

        return q_arm, success, info