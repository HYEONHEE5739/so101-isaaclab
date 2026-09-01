# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.045,

    # 완전히 닫혔다고 판단할 비율
    gripper_closed_threshold: float = 0.3,

    # 충분히 열렸다고 판단할 비율
    gripper_open_threshold: float = 0.5,
) -> torch.Tensor:
    """Return True when:

    1. 그리퍼가 최소 한번 OPEN(물체 잡기위해)
    2. tool0가 물체에 CLOSE
    3. 그리퍼가 CLOSED된 상태

    초기에 그리퍼가 닫혀 있어서 물체에 접근만했을때, True가 되지 않도록 하기 위함
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]


    tool0_pos_b = ee_frame.data.target_pos_source[:, 0]

    obj_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )

    distance = torch.linalg.norm(obj_pos_b - tool0_pos_b, dim=-1)

    object_is_close = (distance < distance_threshold)

    joint_id = robot_cfg.joint_ids[0]
    gripper_pos = robot.data.joint_pos[:, joint_id]

    gripper_is_closed = (gripper_pos < gripper_closed_threshold)
    gripper_is_open = (gripper_pos > gripper_open_threshold)

    # ==========================================================
    #  "한 번이라도 OPEN 했는가?" 상태 저장
    # ==========================================================

    if not hasattr(env, "_gripper_opened_once"):
        env._gripper_opened_once = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=gripper_pos.device,
        )

    if not hasattr(env, "_grasp_completed"):
        env._grasp_completed = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=gripper_pos.device,
        )


    # if (gripper_is_open[0].item() and not env._gripper_opened_once[0].item()):
    #     print(
    #         f"🔵 GRIPPER OPEN"
    #         f" | gripper={gripper_pos[0].item():.4f}",
    #         flush=True,
    #     )

    # 한번 True가 되면 episode 동안 계속 True
    env._gripper_opened_once |= gripper_is_open

    # 현재 "잡고 있는 상태"
    grasp_state= (
        env._gripper_opened_once
        & object_is_close
        & gripper_is_closed
    )
    new_grasp = grasp_state & ~env._grasp_completed

    # if new_grasp[0].item():
    #     print(
    #         f"🟢 GRASP COMPLETED"
    #         f" | distance={distance[0].item():.4f} m"
    #         f" | gripper={gripper_pos[0].item():.4f}",
    #         flush=True,
    #     )

    # 한번 grasp 성공하면 episode 끝까지 True
    env._grasp_completed |= grasp_state

    # Mimic에는 event가 아니라 완료 상태를 전달
    return env._grasp_completed



def object_in_bin(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    bin_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float] = (-0.0255, 0.0255),
    y_bounds: tuple[float, float] = (-0.0255, 0.0255),
    z_bounds: tuple[float, float] = (0.00, 0.02),
) -> torch.Tensor:
    """Return True when the object origin lies inside configured bin-local bounds.

    This is NOT used as a Mimic subtask boundary in the first version.
    It is useful as a success/debug signal for generated demonstrations.

    Tune x/y/z bounds to the *inside* of your actual bin mesh.
    """
    obj = env.scene[object_cfg.name]
    bin_obj = env.scene[bin_cfg.name]

    # Object pose expressed in the bin frame.
    obj_pos_bin, _ = math_utils.subtract_frame_transforms(
        bin_obj.data.root_pos_w,
        bin_obj.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )

    x_ok = (obj_pos_bin[:, 0] >= x_bounds[0]) & (obj_pos_bin[:, 0] <= x_bounds[1])
    y_ok = (obj_pos_bin[:, 1] >= y_bounds[0]) & (obj_pos_bin[:, 1] <= y_bounds[1])
    z_ok = (obj_pos_bin[:, 2] >= z_bounds[0]) & (obj_pos_bin[:, 2] <= z_bounds[1])

    success = x_ok & y_ok & z_ok

    # if success:
    #     print("🟢🟢🟢 SUCCESS 🟢🟢🟢")

    return success
