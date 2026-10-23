#!/usr/bin/env python3
"""ID-based workspace preview / source-demo / teleop / smoke launcher."""
from pathlib import Path
import argparse,sys,os,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/soarm101_lab'))
p=argparse.ArgumentParser();p.add_argument('command',choices=['preview','teleop','source-demo','smoke']);p.add_argument('workspace')
known,rest=p.parse_known_args()
if known.command in ('teleop','source-demo'):
    script='teleop_so101.py' if known.command=='teleop' else 'record_mimic_dataset.py'
    os.environ['PYTHONPATH']=str(ROOT/'source/soarm101_lab')+os.pathsep+os.environ.get('PYTHONPATH','')
    os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/teleoperation'/script),'--workspace',known.workspace,*rest])
p.add_argument('--report');p.add_argument('--dataset');p.add_argument('--steps',type=int,default=60)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=True
app=AppLauncher(args).app
try:
    import torch,numpy as np
    from PIL import Image
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.datasets import EpisodeData,HDF5DatasetFileHandler
    from soarm101_lab.real2sim.workspaces.environment import make_config
    from soarm101_lab.real2sim.workspaces.package import load,metadata
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    from soarm101_lab.so101_dataset_contract import write_contract
    sys.path.insert(0,str(ROOT/'scripts/teleoperation'))
    import record_mimic_dataset as recording
    w=load(args.workspace)
    # Workflow reports and preview images belong to the same run directory.
    out=Path(args.report).resolve().parent if args.report else ROOT/'outputs/workflow/runs'/('workspace_'+w['manifest']['workspace_id'])
    out.mkdir(parents=True,exist_ok=True)
    dataset=Path(args.dataset) if args.dataset else out/'smoke.hdf5'
    from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
    legacy_before=SO101TeleopEnvCfg().scene.to_dict()
    cfg=make_config(w,args.device,overview_camera=False, grasp_observations=False)
    w=cfg.workflow_workspace
    assert SO101TeleopEnvCfg().scene.to_dict()==legacy_before, 'Legacy scene mutated'
    from soarm101_lab.workflow.presentation import attach
    env=ManagerBasedEnv(cfg=cfg);attach(env);obs,_=env.reset()
    print('WORKSPACE_SPAWNED',metadata(w),list(env.scene.rigid_objects),flush=True)
    if args.command=='preview':
        indices=[env.scene['robot'].joint_names.index(n) for n in cfg.actions.joint_positions.joint_names]
        for _ in range(args.steps):obs,_=env.step(env.scene['robot'].data.joint_pos[:,indices].clone())
    elif args.command=='smoke':
        if dataset.exists():raise FileExistsError(dataset)
        episode=EpisodeData();recording.add_initial_state(episode,env);q0=env.scene['robot'].data.joint_pos[:,[env.scene['robot'].joint_names.index(n) for n in cfg.actions.joint_positions.joint_names]].clone();prev=obs['policy']['ee_state'][0].clone()
        for t in range(args.steps):
            recording.add_observation(episode,obs,env)
            action=q0.clone();action[:,3]+=.02*np.sin(2*np.pi*t/max(1,args.steps))
            next_obs,_=env.step(action);next_ee=next_obs['policy']['ee_state'][0].clone()
            episode.add('actions',recording.to_cpu(recording.pose_delta_action(prev,next_ee,action[0,-1])))
            episode.add('joint_targets',recording.to_cpu(action[0]));recording.add_post_state(episode,env)
            for name in env.scene.rigid_objects:episode.add('workspace_states/'+name,env.scene[name].data.root_pose_w[0].detach().cpu())
            prev=next_ee;obs=next_obs
        coordinates=coordinate_contract(w);write_contract(dataset,coordinates)
        writer=HDF5DatasetFileHandler();writer.create(str(dataset),env_name='SO101-Workspace-Source-v1')
        writer.add_env_args({'workspace':metadata(w),'workspace_path':str(w['root']),'so101_coordinates':coordinates,
            'grasp_object':w['task']['pick'],'place_bin':w['task']['place'],'control_dt':env.step_dt,
            'validation_only':True,'root_quaternion_order':'wxyz',
            'scene_configuration':cfg.workflow_workspace['scene_configuration']})
        from soarm101_lab.tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import telemetry_metadata
        writer.add_env_args(telemetry_metadata(w['task']['dynamic']))
        episode.success=False;episode.pre_export();writer.write_episode(episode);writer.flush();writer.close()
        print('SMOKE_RECORDED',dataset,flush=True)
    for view in [k for k in obs['policy'] if k.endswith('_cam')]:Image.fromarray(obs['policy'][view][0].detach().cpu().numpy().astype('uint8')).save(out/(view+'.png'))
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush()
    app.close()
    raise
else:app.close()
