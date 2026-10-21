"""Workspace policy evaluation. Real motion is opt-in and never used by tests."""
import json
import time
import numpy as np
from pathlib import Path
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from .policy import load_policy, observation, minivla_root


def real_start_pose(native, mapper=None, raw=None):
    """Validate a measured hold in float64 before any torque activation.

    Float32 is the policy observation format, not an encoder round-trip format.
    Actual out-of-calibration measurements must still block activation.
    """
    from ..so101_joint_mapping import DEFAULT_MAPPER,JOINTS
    mapping=mapper or DEFAULT_MAPPER
    try:
        q=(mapping.calibrated_to_observation(native,raw=raw) if hasattr(mapping,'calibrated_to_observation')
           else mapping.calibrated_to_urdf(native))
        state=np.asarray([q[n] for n in JOINTS],dtype=np.float64)
        if not np.isfinite(state).all():raise ValueError('Non-finite measured joint state')
        mapping.urdf_to_calibrated(dict(zip(JOINTS,state.tolist())))
    except ValueError as exc:
        detail=str(exc).split('; no goal sent.')[0]
        raise ValueError('Real 평가 시작 자세/보정 검사 실패: '+detail+
                         '. 모터는 활성화하지 않았습니다. Follower를 보정 범위 안의 자세로 옮긴 뒤 다시 시작하세요.') from exc
    return state


class EvaluationRecorder:
    """Completed policy rollouts use the same numeric/video writer as collection."""
    def __init__(self, native, workspace, backend, checkpoint, output, hz, seed, *, index=0, writers=None, contact_views=None):
        import threading
        from .adapters import IsaacAdapter, MujocoAdapter
        from ..data.episodes import DatasetWriter, recording_args
        from ..so101_dataset_contract import read_contract, MIMIC_SPACE
        from ..real2sim.workspaces.tasks import configured, selected
        self.backend = backend
        self.writers=writers;self.output=Path(output);self.index=index
        self.recording=(workspace['task'].get('observation_cameras')==['side','wrist'] and
                        all(workspace['profile']['cameras'][c]['resolution']==[640,480] for c in ('side','wrist')))
        buffer_root=Path(output) if writers is not None else Path(output).parent
        if backend == 'isaac':
            self.adapter = IsaacAdapter(native, index, buffer_root, seed, threading.Event(),contact_views=contact_views,require_standard_cameras=False)
        else:
            self.adapter = MujocoAdapter(native, hz, buffer_root, seed, threading.Event(),require_standard_cameras=False)
        self.adapter.workspace = workspace
        contract = {**read_contract(minivla_root(checkpoint)), 'action_space': MIMIC_SPACE}
        from ..so101_joint_mapping import RobotCoordinates
        mapping=contract['mapping']
        coordinates=RobotCoordinates(mapping['gripper_zero_offset_rad'],
            mapping['gripper_zero_offset_rad']+100*mapping['gripper_rad_per_percent']).metadata()
        policy_provenance=minivla_root(checkpoint)/'project_provenance.json'
        if policy_provenance.is_file():
            coordinates=json.loads(policy_provenance.read_text()).get('coordinates',coordinates)
        definition = selected(workspace)
        scene = configured(workspace, definition, workspace['task']['dynamic'], backend)['scene_configuration']
        provenance = {'backend': backend, 'workspace': metadata(workspace), 'workspace_path': str(workspace['root']),
            'coordinate_contract': contract, 'coordinates':coordinates, 'scene_configuration': scene,
            'scene': {'task_definition': definition, 'pick': workspace['task']['pick'],
                'target': workspace['task']['place'], 'dynamic_objects': workspace['task']['dynamic']}}
        args = recording_args(provenance, {'control_hz': hz})
        args.update(evaluation_schema='so101.evaluation-episode/1', policy_checkpoint=str(checkpoint), seed=seed)
        self.args=args;self.contract=contract
        self.initial_physics_state=native.get_initial_state() if backend=='mujoco' else None
        self.writer = DatasetWriter(output, args, contract) if writers is None and self.recording else None
        self.adapter.after_reset()

    def append(self, target, tracker, observation=None, images=None):
        if not self.recording:return
        a = self.adapter
        a.tracker = tracker
        if self.backend == 'mujoco':
            goal = np.linalg.inv(a.base_workspace) @ a.backend.target_pose(target)
        else:
            goal = a.kinematics.forward_kinematics(target[:5])
        a.record_frame(target, goal,observation,images)

    def finish(self, row, tracker, max_steps):
        a = self.adapter
        a.tracker = tracker
        grasp = bool(tracker.completed)
        place = bool(a.place_success())
        if self.writers is not None and self.recording:
            from ..data.episodes import DatasetWriter
            path=self.output/('success' if row['success'] else 'failure')/'dataset.hdf5'
            if path not in self.writers:self.writers[path]=DatasetWriter(path,self.args,self.contract)
            self.writer=self.writers[path]
        row.update(evaluation_schema='so101.evaluation-episode/1', backend=self.backend,
            episode_id='demo_'+str(self.writer.count if self.writer else 0), policy_checkpoint=self.args['policy_checkpoint'],
            grasp_success=grasp, place_success=place, timeout=bool(not row['success'] and row['steps'] >= max_steps),
            ik_failure=False, controller_failure=False, episode_duration_s=row['steps'] / a.hz)
        row['initial_object_configuration'] = {k:v['root_pose'][0].tolist()
            for k,v in a.buffer.initial['rigid_object'].items()}
        row['failure_reason'] = '' if row['success'] else row.get('failure_reason') or a.failure_category()
        row['failure_diagnostics']=a.failure_diagnostics()
        row['success_condition'] = 'backend placement geometry and rest; stable grasp recorded separately'
        row['episode_dataset'] = str(self.writer.path) if self.writer else None
        row['recording_available']=self.recording
        if not self.recording:
            row['recording_unavailable_reason']='Legacy camera contract: numeric/video collection requires Side/Wrist 640x480'
            return row
        attrs = {'success':bool(row['success']), 'success_source':'simulation_place_condition',
            'failure_reason':row['failure_reason'], 'evaluation':json.dumps(row), 'source_group':'evaluation',
            'task_definition':json.dumps(self.writer.args['task_definitions'][0]),
            'scene_configuration':json.dumps(self.writer.args['scene_configuration'])}
        if self.initial_physics_state is not None:attrs['legacy_initial_state']=json.dumps(self.initial_physics_state)
        try:
            prepared=a.buffer.finish(attrs,success=a.check_success())
            if self.writers is not None:
                a.buffer.video.relocate(self.writer.path.parent)
                for spec in prepared[2].values():spec['file']=str(a.buffer.video.folder/Path(spec['file']).name)
            self.writer.commit(*prepared,video_staging=a.buffer.video)
            if self.writers is None:self.writer.close()
        finally:
            a.buffer.abort()
            if self.writers is None:self.writer.close(False)
        return row

    def close(self):
        self.adapter.buffer.abort()
        if self.writers is None and self.writer is not None:self.writer.close(False)


def evaluation_recorder(native, workspace, backend, checkpoint, output, hz, seed, stack):
    """Legacy one-camera/review workspaces keep their existing evaluation records."""
    cameras = workspace['task'].get('observation_cameras', [])
    if cameras != ['side', 'wrist'] or any(workspace['profile']['cameras'][c]['resolution'] != [640,480] for c in cameras):
        return None
    recorder = EvaluationRecorder(native,workspace,backend,checkpoint,output,hz,seed)
    stack.callback(recorder.close)
    return recorder



def evaluation_seed(options):
    mode = options.get('eval_seed_mode', 'fixed')
    if mode == 'random':
        import secrets
        return secrets.randbelow(2**31)
    if mode != 'fixed':
        raise ValueError('eval_seed_mode must be fixed or random')
    seed = options.get('eval_seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('eval_seed must be an integer in [0, 2147483647]')
    return seed


def episode_result(definition, episode, steps, success, details):
    checks = {name: bool(details[key][0]) for name, key in
              [('높이', 'height_ok'), ('컵 안쪽 벽', 'wall_ok'), ('정지 속도', 'speed_ok')]}
    failed = [name for name, passed in checks.items() if not passed]
    return dict(task_id=definition['task_id'], language_instruction=definition['language_instruction'],
                episode=episode, steps=steps, success=success, checks=checks,
                failure_conditions=failed,
                outcome_reason='컵 안 정지 조건 충족' if success else '최대 step 도달 · 미충족: ' + ', '.join(failed))


def log_evaluation(rows, task_id):
    selected = [r for r in rows if r['task_id'] == task_id]
    last = rows[-1]
    wins = sum(r['success'] for r in selected)
    total_wins = sum(r['success'] for r in rows)
    print(f"[WORKFLOW] Sim 평가 {task_id} Episode {last['episode'] + 1}: "
          f"{'성공' if last['success'] else '실패'} · {last['steps']} steps · {last['outcome_reason']}", flush=True)
    print(f"[WORKFLOW] Task 성공률 {wins}/{len(selected)} ({100*wins/len(selected):.1f}%) · "
          f"전체 누적 {total_wins}/{len(rows)} ({100*total_wins/len(rows):.1f}%)", flush=True)


def validate_evaluation_options(options):
    for key,default in (('episodes',1),('max_steps',300)):
        if type(options.get(key,default)) is not int or options.get(key,default)<1:
            raise ValueError('Positive integer required: '+key)
    count=options.get('num_envs',1)
    if type(count) is not int or count not in (1,2,4,8):
        raise ValueError('num_envs must be 1,2,4 or 8')
    return count


class EvaluationProgress:
    """One episode quota across all slots; failed episodes are completed evaluations."""
    def __init__(self,total,num_envs,started_at=None):
        self.total=total;self.started=time.monotonic()-(max(0,time.time()-started_at) if started_at is not None else 0)
        self.finished=None;self.rows=[];self.latest={}
        self.workers={i:dict(state='waiting',attempts=0,successes=0,failures={},steps=0) for i in range(num_envs)}

    def begin(self,index,episode,task):
        row=self.workers[index];row.update(state='running',steps=0,episode=episode,task_id=task)
        row['attempts']+=1

    def finish(self,index,row):
        self.rows.append(row);worker=self.workers[index]
        category='success' if row['success'] else row['failure_reason']
        worker.update(state=category,last_category=category,steps=row['steps'])
        if row['success']:worker['successes']+=1
        elif category!='interrupted':worker['failures'][category]=worker['failures'].get(category,0)+1
        self.latest=dict(latest_env_id=index,latest_attempt=row['evaluation_index']+1,latest_category=category,
                         episode_outcome='성공' if row['success'] else '실패',
                         outcome_reason=row.get('outcome_reason',''))

    def progress(self):
        import copy
        counts={}
        for worker in self.workers.values():
            for k,v in worker['failures'].items():counts[k]=counts.get(k,0)+v
        successes=sum(r['success'] for r in self.rows);failures=sum(counts.values());completed=successes+failures
        return dict(workflow_phase='evaluation',target_episodes=self.total,completed_episodes=completed,
            attempts=sum(w['attempts'] for w in self.workers.values()),successes=successes,failures=failures,
            interrupted_episodes=sum(r.get('failure_reason')=='interrupted' for r in self.rows),
            success_rate=successes/completed if completed else None,failure_breakdown=counts,
            env_status=copy.deepcopy(self.workers),wall_seconds=(self.finished or time.monotonic())-self.started,**self.latest)


def evaluate_sim(request, path, workspace, definitions, bridge=None, publish=None):
    """Parallel physics with one policy and batched, isolated action-queue columns.

    A group resets together after its episodes finish, preserving the official ACT/
    SmolVLA queues without copying models or clearing another slot's history.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .policy import batch_actions, MiniVLAJointPolicy
    from .tasks import object_metrics, evaluation_distribution, evaluation_metrics
    from ..real2sim.workspaces.tasks import configured
    from ..so101_dataset_contract import read_contract
    from .policy import ActionGuard
    opts=request['options'];count=validate_evaluation_options(opts);episodes=opts.get('episodes',1)
    limit=opts.get('max_steps',300);backend=request.get('backend','isaac')
    if backend=='common':backend='isaac'
    seed=opts['seed'] if 'seed' in opts else evaluation_seed(opts);device=opts.get('device','cuda:0')
    root=Path(request['output']);root=root.parent if root.suffix=='.json' else root
    root.mkdir(parents=True,exist_ok=True)
    if (root/'evaluation_datasets.json').exists():raise FileExistsError(root/'evaluation_datasets.json')
    progress=EvaluationProgress(episodes*len(definitions),count,getattr(bridge,'config',{}).get('started_at') if bridge else None)
    if bridge:
        bridge.progress_provider=progress.progress;bridge.status()
    policy,pre,post=load_policy(path,coordinate_contract(workspace),device)
    if hasattr(policy,'policy') and hasattr(policy.policy,'checkpoint_step'):
        atomic(root/'selected_checkpoint.json',{'path':str(path),'step':policy.policy.checkpoint_step})
    writers={};stopped=False;fatal=None;scene_reports=[]
    active=opts.get('active_object_ids',workspace['task']['dynamic'])
    try:
        for task_index,definition in enumerate(definitions):
            w=configured(workspace,definition,active,backend)
            native=None;parent=None;backends=[];executor=None
            scene_root=root if len(definitions)==1 else root/'tasks'/definition['task_id']
            try:
                if backend=='mujoco':
                    from .mujoco import _environment
                    from ..simulation.mujoco_backend import MujocoBackend
                    env_request={**request,'checkpoint':str(path),'options':{**opts,'task_definition':definition,
                        'instruction':definition['language_instruction'],'active_object_ids':active}}
                    hz=getattr(getattr(policy,'policy',None),'control_hz',None)
                    parent=_environment(env_request,scene_root,hz=hz)
                    backends=[parent.backend]+[MujocoBackend(render_backend=parent.backend.render_backend,
                        shared_backend=parent.backend) for _ in range(count-1)]
                    hz=parent.hz
                    executor=ThreadPoolExecutor(max_workers=count,thread_name_prefix='eval-mujoco')
                    scene_reports.append(parent.backend.metadata)
                else:
                    import torch
                    from isaaclab.envs import ManagerBasedEnv
                    from ..real2sim.workspaces.environment import make_config
                    cfg=make_config(w['root'],device,count,task_definition=definition,
                                    coordinates=read_contract(minivla_root(path)),active_object_ids=active,
                                    overview_camera=bridge is not None,grasp_observations=bridge is not None)
                    cfg.seed=seed;cfg.events.reset_episode.params['randomization_seed']=seed
                    native=ManagerBasedEnv(cfg=cfg);native._workflow_task_workspace=w
                    native.reset();hz=1/native.step_dt
                    if bridge:bridge.env=native;bridge.idle_render=native.sim.render
                    scene_reports.append(cfg.simulation_configuration)
                for first in range(0,episodes,count):
                    size=min(count,episodes-first);policy.reset();recorders=[];states=[]
                    for i in range(size,count):progress.workers[i]['state']='waiting'
                    if native is not None:
                        native.reset()
                        isaac_actions=native.scene['robot'].data.joint_pos[:,
                            [native.scene['robot'].joint_names.index(n) for n in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')]].clone()
                        for field in ('_grasp_completed','_gripper_opened_once'):
                            if hasattr(native,field):getattr(native,field).zero_()
                    metrics=[{} for _ in range(size)];initial=[];traces=[[] for _ in range(size)]
                    guards=[];done=[False]*size;latencies=[[] for _ in range(size)]
                    try:
                        for i in range(size):
                            episode=first+i;global_index=task_index*episodes+episode
                            views=None
                            if backend=='mujoco':backends[i].reset(seed=seed+global_index)
                            else:
                                pick=definition['pick_object_id']
                                views=[native.sim.physics_sim_view.create_rigid_contact_view(
                                    f'/World/envs/env_{i}/Robot/'+link,[f'/World/envs/env_{i}/'+pick]) for link in ('gripper','jaw')]
                            recorder=EvaluationRecorder(backends[i] if backend=='mujoco' else native,w,backend,path,
                                scene_root,hz,seed+global_index if backend=='mujoco' else seed,
                                index=i,writers=writers,contact_views=views)
                            recorders.append(recorder);a=recorder.adapter
                            state=a.state();states.append(state)
                            initial.append({n:state['object_poses'][n][:3,3].copy() for n in active})
                            bounds=a.backend.limits if backend=='mujoco' else a.limits
                            guards.append(ActionGuard(bounds[:,0],bounds[:,1],[2.]*6,hz))
                            progress.begin(i,episode,definition['task_id'])
                        # Completed slots keep their final measured target for the rest of the group.
                        holds=[s['joint_pos'].copy() for s in states]
                        for step in range(limit):
                            if bridge:bridge.before_step()
                            images=[r.adapter.images() for r in recorders]
                            begin=time.perf_counter()
                            commands=batch_actions(policy,pre,post,[s['joint_pos'] for s in states],images,
                                                   definition['language_instruction'],device).astype(np.float64)
                            duration=time.perf_counter()-begin
                            for i,r in enumerate(recorders):
                                if done[i]:continue
                                command=commands[i] if isinstance(policy,MiniVLAJointPolicy) or backend=='isaac' else guards[i].apply(commands[i],states[i]['joint_pos'])
                                commands[i]=command;r.append(command,r.adapter.tracker,states[i],images[i])
                                r.adapter.previous=command.copy();latencies[i].append(duration)
                            if backend=='mujoco':
                                futures={i:executor.submit(backends[i].step,commands[i]) for i in range(size) if not done[i]}
                                for i,future in futures.items():future.result()
                            else:
                                action=isaac_actions.clone()
                                for i in range(size):action[i]=torch.as_tensor(holds[i] if done[i] else commands[i],device=native.device,dtype=action.dtype)
                                native.step(action)
                            for i,r in enumerate(recorders):
                                if done[i]:continue
                                a=r.adapter;a.update_grasp();states[i]=a.state();success=bool(a.place_success())
                                measured=object_metrics(active,definition['pick_object_id'],
                                    {n:states[i]['object_poses'][n][:3,3] for n in active},states[i]['eef_pose'][:3,3],
                                    metrics[i],a.tracker.completed,success,initial[i])
                                traces[i].append(dict(step=step+1,**measured));progress.workers[i]['steps']=step+1
                                if success or step+1==limit:
                                    reason='' if success else a.failure_category()
                                    if backend=='mujoco':details=a.backend.success_details()
                                    else:
                                        from ..real2sim.workspaces.environment import task_success_details
                                        _,batch_details=task_success_details(native,w)
                                        details={k:bool(batch_details[k][i]) for k in ('height_ok','wall_ok','speed_ok')}
                                    checks={label:bool(details[key]) for label,key in [('높이','height_ok'),('컵 안쪽 벽','wall_ok'),('정지 속도','speed_ok')]}
                                    failed=[k for k,v in checks.items() if not v]
                                    from .presentation import failure_label
                                    row=dict(task_id=definition['task_id'],language_instruction=definition['language_instruction'],
                                        episode=first+i,evaluation_index=task_index*episodes+first+i,env_id=i,steps=step+1,
                                        success=success,failure_reason=reason,checks=checks,failure_conditions=failed,
                                        outcome_reason='컵 안 정지 조건 충족' if success else failure_label(reason)+' · 미충족: '+', '.join(failed),
                                        action_call_latency_s=latencies[i],object_metric_trace=traces[i],**measured)
                                    r.finish(row,a.tracker,limit);progress.finish(i,row);done[i]=True
                                    holds[i]=states[i]['joint_pos'].copy()
                                    atomic(root/f'episode_{row["evaluation_index"]:04d}.json',{'summary':row,'trace':traces[i]})
                                    log_evaluation(progress.rows,definition['task_id'])
                            if bridge:
                                bridge.steps=step+1
                                bridge.details['task_signals']={'grasp':recorders[0].adapter.tracker.completed,'place':bool(recorders[0].adapter.place_success())}
                                preview={'sim_'+k:v for k,v in images[0].items()}
                                if backend=='mujoco' and 'overview' in parent.backend.metadata.get('cameras',{}):
                                    preview['sim_overview']=parent.backend.get_rgb('overview')
                                bridge.publish(preview);bridge.status()
                            elif publish:publish({'sim_'+k:v for k,v in images[0].items()},progress.progress())
                            if all(done):break
                    finally:
                        for r in recorders:r.close()
            finally:
                if executor:executor.shutdown(wait=True)
                for b in backends[1:]:b.close()
                if parent:parent.close()
                if native:
                    native.close()
                    if bridge:bridge.env=None;bridge.idle_render=None
    except KeyboardInterrupt:stopped=True
    except BaseException as exc:
        fatal=exc
    finally:
        progress.finished=time.monotonic()
        for worker in progress.workers.values():
            if worker['state']=='waiting':worker['state']=worker.get('last_category','waiting')
            elif worker['state']=='running':worker['state']='interrupted' if stopped else 'simulation_failure' if fatal else 'waiting'
        datasets=[]
        for p,writer in writers.items():
            writer.close()
            if writer.count:datasets.append({'path':str(p.relative_to(root)),'outcome':p.parent.name,
                                             'episodes':writer.count})
        result=dict(schema='so101.sim-evaluation/1',passed=fatal is None,stopped=stopped,backend=backend,
            workspace=metadata(workspace),checkpoint_path=str(path),num_envs=count,evaluation_seed=seed,
            task_definitions=definitions,scene_configurations=[configured(workspace,d,active,backend)['scene_configuration'] for d in definitions],
            scenes=scene_reports,datasets=datasets,episodes=progress.rows,**progress.progress(),
            **evaluation_distribution({'path':str(minivla_root(path))},workspace,active),
            metrics_per_task=evaluation_metrics(definitions,progress.rows,True),
            task_success='backend geometric placement and rest; grasp/drop measured separately',
            batching='reset group together; completed slots hold until group ends; one shared policy')
        if fatal:result['error']=f'{type(fatal).__name__}: {fatal}'
        atomic(root/'evaluation_datasets.json',result)
        atomic(root/'report.json',result)
        if bridge:bridge.status('FINISHED')
    if fatal:raise fatal
    return result


def evaluate(request):
    from .presentation import current, attach
    bridge = current()
    stage, opts = request['stage'], request['options']
    if stage == 'real_eval' and not opts.get('enable_motion'):
        raise ValueError('Real motion must be explicitly enabled')
    app = None
    if stage == 'sim_eval':
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=opts.get('headless', True), enable_cameras=True).app
    try:
        import torch
        w = load(request['workspace']); device = opts.get('device', 'cuda:0')
        artifact = request['input']; path = artifact['path']
        if artifact['type'] == 'hf_policy':
            remote = json.loads(Path(path).read_text())
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=remote['repo_id'], revision=remote['commit'])
        if opts.get('checkpoint_file'):
            from .policy import selected_minivla_checkpoint
            path=str(selected_minivla_checkpoint(path,opts['checkpoint_file']))
        policy,pre,post=load_policy(path,coordinate_contract(w),device) if stage!='sim_eval' else (None,None,None)
        if hasattr(policy,'policy') and hasattr(policy.policy,'checkpoint_step'):
            atomic(Path(request['output']).with_suffix('')/'selected_checkpoint.json',{'path':path,'step':policy.policy.checkpoint_step})
        from .tasks import object_metrics, evaluation_distribution
        from ..real2sim.workspaces.tasks import catalog, selected, validate, configured, active_objects
        active = active_objects(w,opts.get('active_object_ids'))
        definitions = [t for t in catalog(w) if t['pick_object_id'] in active] if opts.get('task_scope') == 'all' else opts.get('tasks') or [selected(w)]
        definitions = [validate(t, w) for t in definitions]
        if not definitions or len({t['task_id'] for t in definitions}) != len(definitions):
            raise ValueError('Select distinct evaluation tasks')
        rows = []
        used_seed = None
        if stage == 'sim_eval':
            result=evaluate_sim(request,path,w,definitions,bridge)
            atomic(request['output'],{'status':'STOPPED' if result['stopped'] else 'SUCCEEDED',**result})
            return result
        else:
            from ..real2sim.hardware import Follower
            from ..real2sim.cameras import CameraReader
            from ..representation import real_observation
            from ..so101_joint_mapping import DEFAULT_MAPPER, JOINTS
            from ..so101_robot_spec import joint_limits
            from ..real2sim.core import values
            import hashlib
            model_hash = hashlib.sha256((w['root']/'assets/SO101/urdf/so101_isaaclab.urdf').read_bytes()).hexdigest()
            follower = Follower(opts['port'], opts['follower_id'], opts.get('calibration_dir'), model_hash=model_hash)
            cameras = {}
            try:
                for view in w['task'].get('observation_cameras',list(w['profile']['cameras'])):
                    width, height = w['profile']['cameras'][view]['resolution']
                    device_path=opts.get('camera_devices',{}).get(view) or opts.get(view+'_device')
                    if not device_path:raise ValueError('Real camera device required: '+view)
                    cameras[view] = CameraReader(device_path, width, height)
                    cameras[view].start()
                deadline = time.monotonic() + 10
                while not all(c.nearest() is not None for c in cameras.values()):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Cameras did not deliver frames; no motion started')
                    time.sleep(.02)
                follower.connect()
                mapper = getattr(follower, 'precision_mapper', None)
                from ..real2sim.precision import require_hardware_policy_frame
                from ..so101_dataset_contract import read_contract
                require_hardware_policy_frame(read_contract(minivla_root(path)), w['profile'], mapper)
                held, _ = follower.read()
                preflight={'joint_order':list(JOINTS),'measured_native':held,
                           'measured_raw':getattr(follower,'last_raw',None),'motors_activated':False}
                try:
                    held_q=real_start_pose(held,mapper,preflight['measured_raw'])
                    preflight.update(passed=True,measured_model_rad=held_q.tolist())
                except ValueError as exc:
                    preflight.update(passed=False,error=str(exc))
                    atomic(Path(request['output']).parent/'hardware_preflight.json',preflight)
                    raise
                atomic(Path(request['output']).parent/'hardware_preflight.json',preflight)
                print('[REAL EVAL] 시작 자세/보정 검사 통과 · Follower 활성화 준비',flush=True)
                bounds = np.array(list(joint_limits(w['root']/'assets/SO101/urdf/so101_isaaclab.urdf').values()))
                follower.activate(); policy.reset()
                for definition, ep in [(d, e) for d in definitions for e in range(int(opts.get('episodes', 1)))]:
                    task = definition['language_instruction']; policy.reset()
                    if bridge:
                        bridge.details.update(task_id=definition['task_id'], language_instruction=task, episode_index=ep + 1, episode_total=int(opts.get('episodes', 1)), episode_outcome='진행 중', outcome_reason='')
                    for step in range(int(opts.get('max_steps', 300))):
                        if bridge:
                            bridge.before_step()
                        begin = time.monotonic()
                        native, _ = follower.read()
                        frames = {v: c.nearest() for v, c in cameras.items()}
                        if any(time.monotonic_ns() - f['read_end']['monotonic_ns'] > 500_000_000 for f in frames.values()):
                            raise RuntimeError('Stale real camera frame')
                        if bridge:
                            bridge.steps = step
                            bridge.publish({'real_' + v: f['rgb'] for v, f in frames.items()})
                        q = real_observation({n + '.pos': v for n, v in native.items()}, mapper)
                        batch = observation(q, None, None, task, device, images={n:f['rgb'] for n,f in frames.items()})
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(batch))).reshape(6).detach().cpu().numpy()
                        if action.shape != (6,) or not np.isfinite(action).all():
                            raise ValueError('Non-finite or invalid policy goal')
                        bounded = np.clip(action,bounds[:,0],bounds[:,1])
                        command = (mapper or DEFAULT_MAPPER).urdf_to_calibrated(dict(zip(JOINTS,bounded.tolist())))
                        command, encoded = follower.encode_goal(command)
                        applied = np.array(list((mapper or DEFAULT_MAPPER).calibrated_to_urdf(encoded).values()))
                        if np.any(applied<bounds[:,0]-1e-6) or np.any(applied>bounds[:,1]+1e-6):
                            raise ValueError('Encoded policy goal outside URDF limits')
                        sent, _ = follower.send(command)
                        if not np.allclose(list(values(sent).values()),list(values(command).values()),atol=1e-10,rtol=0):
                            raise ValueError('Follower altered policy goal')
                        rows.append({'task_id': definition['task_id'], 'language_instruction': task, 'episode': ep, 'step': step, 'model_observation': q.tolist(), 'model_action': applied.tolist(), 'requested_model_action': action.tolist()})
                        time.sleep(max(0, 1 / int(opts.get('fps', 30)) - (time.monotonic() - begin)))
            finally:
                try:
                    follower.close()
                finally:
                    for camera in cameras.values():
                        camera.close()
        from .tasks import evaluation_metrics
        if stage == 'sim_eval':
            for definition in definitions:
                subset = [r for r in rows if r['task_id'] == definition['task_id']]
                wins = sum(r['success'] for r in subset)
                print(f"[WORKFLOW] 평가 완료 · {definition['task_id']} · 성공 {wins}/{len(subset)} · 성공률 {100*wins/len(subset):.1f}%", flush=True)
        atomic(request['output'], {'status': 'SUCCEEDED', 'mode': stage, 'workspace': metadata(w),
               'scene_configuration':configured(w,definitions[0],active, 'isaac' if stage=='sim_eval' else 'real')['scene_configuration'],
               'simulation_configuration':cfg.simulation_configuration if stage=='sim_eval' else read_contract(minivla_root(path)).get('simulation_model'),
               'scene_configurations':[configured(w,d,active, 'isaac' if stage=='sim_eval' else 'real')['scene_configuration'] for d in definitions],
               **evaluation_distribution(artifact,w,active), 'policy_artifact': artifact['id'], 'checkpoint_path': str(path),
               'checkpoint_step': getattr(getattr(policy,'policy',None),'checkpoint_step',None), 'task_definitions': definitions, 'evaluation_seed': used_seed,
               'hardware_mapping': follower.metadata.get('precision_mapping') if stage == 'real_eval' else None,
               'metrics_per_task': evaluation_metrics(definitions, rows, stage == 'sim_eval'), 'episodes' if stage == 'sim_eval' else 'steps': rows,
               'task_success': 'simulation geometric condition' if stage == 'sim_eval' else 'unmeasured; human review required'})
    except Exception:
        # Report before Isaac shutdown, which can itself take time after setup failure.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if bridge:
            bridge.close()
        if app:
            app.close()
