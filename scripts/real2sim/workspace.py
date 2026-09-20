#!/usr/bin/env python3
"""ID-based workspace preview / source-demo / teleop / smoke / replay launcher."""
from pathlib import Path
import argparse,sys,os,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/soarm101_lab'))
p=argparse.ArgumentParser();p.add_argument('command',choices=['preview','teleop','source-demo','smoke','replay']);p.add_argument('workspace')
known,rest=p.parse_known_args()
if known.command in ('teleop','source-demo'):
    script='teleop_so101.py' if known.command=='teleop' else 'record_mimic_dataset.py'
    os.environ['PYTHONPATH']=str(ROOT/'source/soarm101_lab')+os.pathsep+os.environ.get('PYTHONPATH','')
    os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/envs/teleoperation'/script),'--workspace',known.workspace,*rest])
p.add_argument('--report');p.add_argument('--dataset');p.add_argument('--episode',default='demo_0');p.add_argument('--steps',type=int,default=60)
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
    from soarm101_lab.real2sim.workspaces.demo import verify_dataset,restore_initial_state,coordinate_contract,replay_fields
    from soarm101_lab.so101_dataset_contract import current_contract,write_contract,MIMIC_SPACE,require_joint_replay
    sys.path.insert(0,str(ROOT/'scripts/envs/teleoperation'))
    import record_mimic_dataset as recording
    w=load(args.workspace);out=ROOT/'outputs/real2sim/workspace_runs'/w['manifest']['workspace_id'];out.mkdir(parents=True,exist_ok=True)
    dataset=Path(args.dataset) if args.dataset else out/'smoke.hdf5'
    if args.command=='replay':verify_dataset(dataset,args.workspace);require_joint_replay(dataset)
    from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg
    legacy_before=SO101TeleopEnvCfg().scene.to_dict()
    cfg=make_config(args.workspace,args.device)
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
            recording.add_observation(episode,obs)
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
            'validation_only':True,'root_quaternion_order':'wxyz'})
        episode.success=False;episode.pre_export();writer.write_episode(episode);writer.flush();writer.close()
        print('SMOKE_RECORDED',dataset,flush=True)
    if args.command in ('smoke','replay'):
        _,meta=verify_dataset(dataset,args.workspace)
        reader=HDF5DatasetFileHandler();reader.open(str(dataset));ep=reader.load_episode(args.episode,env.device)
        if ep is None:raise ValueError('Episode missing')
        obs,_=restore_initial_state(env,ep,meta.get('root_quaternion_order','wxyz'))
        recorded_obs, recorded_joints, recorded_objects = replay_fields(ep.data)
        print(f'[WORKFLOW] Replay {args.episode} 시작: {len(ep.data["joint_targets"])} frames', flush=True)
        joint_errors=[];object_errors=[];camera_errors=[]
        for t,action in enumerate(ep.data['joint_targets']):
            if t % 100 == 0: print(f'[WORKFLOW] Replay {args.episode}: {t}/{len(ep.data["joint_targets"])}', flush=True)
            for camera in ['side_cam','wrist_cam']:
                if camera in recorded_obs:
                    expected=recorded_obs[camera][t];camera_errors.append(float((obs['policy'][camera][0].float()-expected.float()).abs().mean()))
            obs,_=env.step(action.unsqueeze(0));joint_errors.append(float((env.scene['robot'].data.joint_pos[0]-recorded_joints[t]).abs().max()))
            for name,target in recorded_objects.items():object_errors.append(float((env.scene[name].data.root_pose_w[0,:3]-target[t,:3]).abs().max()))
        result={'steps':len(joint_errors),'max_joint_error_rad':max(joint_errors),'max_object_position_error_m':max(object_errors) if object_errors else None,'mean_camera_rgb_abs_error':float(np.mean(camera_errors)),'hardware_tested':False,'legacy_scene_unchanged':True,'rigid_objects':list(env.scene.rigid_objects)}
        result['passed']=result['max_joint_error_rad']<.005 and (result['max_object_position_error_m'] is None or result['max_object_position_error_m']<.002) and result['mean_camera_rgb_abs_error']<3.
        result['object_trajectory_checked']=bool(object_errors)
        report_path=Path(args.report) if args.report else out/'replay_report.json'
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(result,indent=2));print('REPLAY_RESULT',result,flush=True);reader.close()
        print(f'[WORKFLOW] Replay {args.episode} 완료: passed={result["passed"]}', flush=True)
        if not result['passed']:raise RuntimeError('Workspace replay verification failed; see report')
    for view in ['side_cam','wrist_cam']:Image.fromarray(obs['policy'][view][0].detach().cpu().numpy().astype('uint8')).save(out/(view+'.png'))
    env.close()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush()
    app.close()
    raise
else:app.close()
