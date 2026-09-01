# SPDX-License-Identifier: BSD-3-Clause

import torch


from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    gripper_pos = ee_frame.data.target_pos_source[:, 0]
    gripper_quat = ee_frame.data.target_quat_source[:, 0]

    return torch.cat([gripper_pos, gripper_quat],dim=-1)

def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:

    sensor = env.scene[sensor_cfg.name]
    images = sensor.data.output[data_type]

    return images.clone()