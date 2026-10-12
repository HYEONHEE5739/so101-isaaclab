#!/usr/bin/env python3
"""Read-only workspace physics/camera smoke check. No hardware or policy training."""
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
    from PIL import Image
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.math import quat_apply
    from soarm101_lab.real2sim.workspaces.package import load
    from soarm101_lab.real2sim.workspaces.environment import make_config, task_success
    from soarm101_lab.real2sim.workspaces.mimic import make_mimic_config
    w = load(args.workspace); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedEnv(cfg=make_config(w['root'], args.device)); obs, _ = env.reset()
    picked = env.scene[w['task']['pick']]; target = env.scene[w['task']['place']]
    height = w['profile']['objects'][w['task']['targets'][w['task']['place']]]['dimensions_m'][2]
    q = env.scene['robot'].data.joint_pos.clone()
    pose = target.data.root_pose_w.clone()
    above = torch.tensor([[0., 0., height / 2 + .06]], device=env.device)
    pose[:, :3] += quat_apply(pose[:, 3:7], above)
    picked.write_root_pose_to_sim(pose); picked.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    initial_z = float(pose[0, 2])
    for _ in range(120): obs, _ = env.step(q)
    success = bool(task_success(env, w)[0])
    cameras = w['task'].get('observation_cameras', list(w['profile']['cameras']))
    images = {}
    for name in cameras:
        image = env.scene['camera_' + name + 'view'].data.output['rgb'][0, ..., :3].cpu().numpy()
        Image.fromarray(image).save(output / ('isaac_' + name + '.png'))
        images[name] = list(image.shape)
        assert name + '_cam' in obs['policy'], 'Missing named policy camera'
    masses = {name: float(env.scene[name].root_physx_view.get_masses()[0,0]) for name in w['task']['dynamic']}
    mimic = make_mimic_config(w['root'], args.device)
    binding = [s.object_ref for s in mimic.subtask_configs['tool0']]
    assert binding == [w['task']['pick'], w['task']['place']], binding
    result = {'passed': success, 'success_relation': w['task']['success']['relation'],
              'entered_open_cup': success, 'initial_cube_z_m': initial_z,
              'final_cube_z_m': float(picked.data.root_pos_w[0, 2]), 'steps': 120,
              'cameras': images, 'policy_observations': list(obs['policy']), 'masses_kg': masses,
              'mimic_bindings': binding, 'hardware_tested': False,
              'manipulation_tested': False, 'workflow_end_to_end': False}
    (output / 'physics_report.json').write_text(json.dumps(result, indent=2))
    print('PHYSICS_CHECK', result, flush=True)
    env.close()
    if not success: raise RuntimeError('Picked object failed to enter and settle in target')
except BaseException:
    import traceback, json
    failure = {'passed':False,'error':traceback.format_exc()}
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/'failure.json').write_text(json.dumps(failure,indent=2))
    print(failure['error'],file=sys.stderr,flush=True)
    raise
finally:
    app.close()
