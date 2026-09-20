#!/usr/bin/env python3
"""Non-hardware open-cup smoke check in an isolated package; never edits accepted data."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument('--workspace', required=True); p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.enable_cameras = True
app = AppLauncher(args).app
try:
    import json
    import torch
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.math import quat_apply
    from soarm101_lab.real2sim.workspaces.package import load, publish, ROOT as REPO
    from soarm101_lab.real2sim.workspaces.environment import make_config, task_success
    w = load(args.workspace); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    package = publish(w['root'] / 'profile.json', 'physics_validation', w['task'], registry=output / 'packages')
    checked = load(package); env = ManagerBasedEnv(cfg=make_config(package, args.device)); obs, _ = env.reset()
    cube = env.scene[w['task']['pick']]; cup = env.scene[w['task']['place']]
    height = w['profile']['objects'][w['task']['targets'][w['task']['place']]]['dimensions_m'][2]
    q = env.scene['robot'].data.joint_pos.clone()
    pose = cup.data.root_pose_w.clone()
    above = torch.tensor([[0., 0., height / 2 + .06]], device=env.device)
    pose[:, :3] += quat_apply(pose[:, 3:7], above)
    cube.write_root_pose_to_sim(pose); cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    initial_z = float(pose[0, 2])
    for _ in range(120):
        env.step(q)
    result = {'entered_open_cup': bool(task_success(env, checked)[0]), 'initial_cube_z_m': initial_z,
              'final_cube_z_m': float(cube.data.root_pos_w[0, 2]), 'steps': 120, 'hardware_tested': False}
    (output / 'physics_report.json').write_text(json.dumps(result, indent=2))
    print('PHYSICS_CHECK', result, flush=True)
    env.close()
    if not result['entered_open_cup']:
        raise RuntimeError('Cube failed to enter and settle in open cup')
finally:
    app.close()
