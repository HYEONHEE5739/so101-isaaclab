from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from .so101_ik_actions import SO101PinocchioIKAction


@configclass
class SO101PinocchioIKActionCfg(ActionTermCfg):
    """Configuration for SO101 Pinocchio task-space IK action."""

    class_type: type = SO101PinocchioIKAction

    # --------------------------------------------------------------
    # Robot
    # --------------------------------------------------------------

    asset_name: str = "robot"

    joint_names: list[str] = MISSING

    # Actual physical/link frame in USD + URDF
    body_name: str = "tool0"

    # --------------------------------------------------------------
    # Kinematic model
    # --------------------------------------------------------------

    urdf_path: str = MISSING

    # --------------------------------------------------------------
    # Action preprocessing
    #
    # 6D:
    # [dx, dy, dz, rx, ry, rz]
    # --------------------------------------------------------------

    scale: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    # --------------------------------------------------------------
    # IK
    # --------------------------------------------------------------

    position_weight: float = 1.0
    orientation_weight: float = 0.01

    max_iterations: int = 200

    tolerance: float = 1e-3

    damping: float = 1e-3

    ik_step_size: float = 0.3

    # --------------------------------------------------------------
    # Debug
    # --------------------------------------------------------------

    debug: bool = False