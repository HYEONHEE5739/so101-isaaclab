#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

"""
python scripts/inference/inference_act_sim.py \
    --policy /home/hyeonhee/soarm_isaaclab/soarm101_lab/policy/act_generated_50ep_0.4.1/checkpoints/last/pretrained_model \
    --num_episodes 1 \
    --max_steps 500 
"""

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_csv", type=str, default="./outputs/sim_eval/results.csv")
parser.add_argument("--approach_threshold", type=float, default=0.045)
parser.add_argument("--lift_threshold", type=float, default=0.03)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np

from isaaclab.envs import ManagerBasedEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
from soarm101_lab.utils.episode_randomizer import EpisodeRandomizer
from isaaclab.managers import SceneEntityCfg
from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_mdp import object_in_bin

def rgba_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def get_policy_obs(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return obs


def make_lerobot_observation(policy_obs: dict[str, torch.Tensor]) -> dict:
    """
    PyTorch vision model이 보통 channel-first 형식을 사용하기 때문.
    [0] : (1, H, W, C) -> (H, W, C)  permute(2, 0, 1) :  (H, W, C) -> (C, H, W)
    
    unit[0, 255] -> float32[0,1]

    """

    side_cam = rgba_to_rgb(policy_obs["side_cam"][0]).to(torch.float32) / 255.0
    wrist_cam = rgba_to_rgb(policy_obs["wrist_cam"][0]).to(torch.float32) / 255.0

    side_cam = side_cam.permute(2, 0, 1)
    wrist_cam = wrist_cam.permute(2, 0, 1)

    return {
        "observation.state": policy_obs["joint_pos"][0],
        "observation.images.side_cam": side_cam,
        "observation.images.wrist_cam": wrist_cam,
    }



# ==========================================================================================================
# [CLASS]                                  EpisodeAnalyzer
# ==========================================================================================================

class EpisodeAnalyzer:
    def __init__(self, approach_threshold: float, lift_threshold: float):
        self.approach_threshold = approach_threshold
        self.lift_threshold = lift_threshold
        self.reset()

    def reset(self):
        self.initial_object_z = None
        self.min_ee_object_dist = float("inf")
        self.max_object_lift = 0.0
        self.approached = False
        self.lifted = False
        self.success = False

    def update(self, ee_pos: torch.Tensor, object_pos: torch.Tensor, success: bool):
        ee_pos = ee_pos.detach().cpu()
        object_pos = object_pos.detach().cpu()

        if self.initial_object_z is None:
            self.initial_object_z = float(object_pos[2])

        dist = torch.linalg.vector_norm(ee_pos - object_pos).item()
        lift = float(object_pos[2]) - self.initial_object_z

        self.min_ee_object_dist = min(self.min_ee_object_dist, dist)
        self.max_object_lift = max(self.max_object_lift, lift)

        if dist < self.approach_threshold:
            self.approached = True

        if lift > self.lift_threshold:
            self.lifted = True

        if success:
            self.success = True

    def classify(self):
        if self.success:
            return "SUCCESS"
        if not self.approached:
            return "APPROACH_FAIL"
        if not self.lifted:
            return "GRASP_FAIL"
        return "PLACE_FAIL"

# ==========================================================================================================
# ==========================================================================================================

def get_success(env) -> bool:
    success = object_in_bin(
        env,
        object_cfg=SceneEntityCfg("cube_red"),
        bin_cfg=SceneEntityCfg("bin_a"),
    )
    return bool(success[0].item())

def reset_env(env, episode_random_state):
    cfg = env.event_manager.get_term_cfg("reset_episode")
    cfg.params["random_state"] = episode_random_state
    env.event_manager.set_term_cfg("reset_episode", cfg)
    obs, _ = env.reset()
    return obs

def main():

    # env 생성
    env_cfg = SO101TeleopEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=env_cfg)

    # env random state : randomizer
    randomizer = EpisodeRandomizer()

    # 학습된 모델 적용
    policy = ACTPolicy.from_pretrained(args.policy)
    policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    print("[INFO] pre/post processor done", flush=True)

    # Episode Sim 판단 하기 위한 asset 정보 불러오기
    robot = env.scene["robot"]
    object_asset = env.scene["cube_red"]

    ee_body_ids = robot.find_bodies("tool0")[0]
    if len(ee_body_ids) != 1:
        raise RuntimeError(f"Could not uniquely find EE body. Found: {ee_body_ids}")

    ee_body_id = ee_body_ids[0]


    # inference 결과 저장
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for episode in range(args.num_episodes):
        print()
        print("=" * 60)
        print(f"EPISODE {episode + 1}/{args.num_episodes}")
        print("=" * 60)

        torch.manual_seed(args.seed + episode)
        np.random.seed(args.seed + episode)

        episode_random_state = randomizer.sample()
        obs = reset_env(env, episode_random_state)

        policy.reset()

        analyzer = EpisodeAnalyzer(
            approach_threshold=args.approach_threshold,
            lift_threshold=args.lift_threshold,
        )

        episode_success = False

        for step in range(args.max_steps):
            policy_obs = get_policy_obs(obs)
            lerobot_obs = make_lerobot_observation(policy_obs)

            for key, value in lerobot_obs.items():
                print(key, value.shape, value.dtype, value.device, flush=True)

            import traceback
            with torch.inference_mode():

                #print(preprocessor, flush=True)

                try:
                    batch = preprocessor(lerobot_obs)
                except Exception:
                    #print("[ERROR] preprocessor failed", flush=True)
                    traceback.print_exc()
                    raise

                #print("[DEBUG] preprocessor done", flush=True)

                action = policy.select_action(batch)
                action = postprocessor(action)

            action = action.to(device=env.device, dtype=torch.float32)
            #print("[DEBUG] action.to done", flush=True)
            #print("[BEFORE STEP] action:", action, flush=True)

            obs, _ = env.step(action)

            ee_pos = robot.data.body_pos_w[0, ee_body_id]
            object_pos = object_asset.data.root_pos_w[0]

            success = get_success(env)
            analyzer.update(ee_pos, object_pos, success)

            # if step % 30 == 0:
            #     q = robot.data.joint_pos[0].detach().cpu().numpy()
            #     action_np = action[0].detach().cpu().numpy()
            #     print(f"[{step:04d}] action={np.round(action_np, 3)} q={np.round(q, 3)}")

            if success:
                episode_success = True
                print(f"[SUCCESS] step={step}")
                break

        result_type = analyzer.classify()

        result = {
            "episode": episode,
            "seed": args.seed + episode,
            "success": episode_success,
            "result": result_type,
            "steps": step + 1,
            "min_ee_object_dist": analyzer.min_ee_object_dist,
            "max_object_lift": analyzer.max_object_lift,
        }

        results.append(result)

        print(f"[RESULT] {result_type}")
        print(f"min EE-object distance: {analyzer.min_ee_object_dist:.4f} m")
        print(f"max object lift: {analyzer.max_object_lift:.4f} m")

    success_count = sum(int(result["success"]) for result in results)

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Success: {success_count}/{args.num_episodes}")
    print(f"Success rate: {100.0 * success_count / args.num_episodes:.1f}%")

    for category in ["SUCCESS", "APPROACH_FAIL", "GRASP_FAIL", "PLACE_FAIL"]:
        count = sum(result["result"] == category for result in results)
        print(f"{category:15s}: {count}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved: {output_csv}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()