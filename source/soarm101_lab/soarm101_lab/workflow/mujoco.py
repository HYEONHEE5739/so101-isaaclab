"""Worker-only implementations. No model, simulation or hardware in the GUI process."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from ..simulation import environment as simulation_environment
from .dataset import recording_provenance
from .episodes import grasp_tracker, subtask_signals
from .worker_request import mujoco_request, prepare_mujoco
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata


def _preview(env, request, publish):
    """Reuse recorded camera frames; render only missing inspection feeds."""
    previous = [0.]
    def emit(frame, details):
        if request['options'].get('preview_cameras') and frame is not None and time.monotonic() - previous[0] >= .1:
            previous[0] = time.monotonic()
            images = ({'sim_' + name: rgb for name, rgb in frame.items()} if isinstance(frame, dict)
                      else {'sim_' + env.backend.metadata.get('primary_camera','side'): frame})
            for name in env.backend.metadata.get('cameras', {}):
                if 'sim_' + name not in images:
                    images['sim_' + name] = env.backend.get_rgb(name)
            publish(images, details)
        else:
            publish({'sim_' + name: rgb for name, rgb in frame.items()} if isinstance(frame, dict) else frame, details)
    return emit


def _verify_workspace(request, provenance):
    if provenance['workspace'] != metadata(load(request['workspace'])):
        raise ValueError('Dataset/checkpoint workspace identity differs from selected workspace')


def _forbidden():
    found = sorted(n for n in sys.modules if n.split('.')[0] in ('isaaclab', 'omni', 'pxr'))
    if found:
        raise RuntimeError('Forbidden engine imports in worker: ' + ', '.join(found))
    return found


def environment(request, output, publish, bridge):
    import psutil
    with simulation_environment.create_mujoco(**prepare_mujoco(request)) as env:
        publish = _preview(env, request, publish)
        physics_rss_mib = psutil.Process().memory_info().rss / 2**20
        env.reset(request['options'].get('seed', 0))
        steps = int(request['options'].get('steps', 150))
        if steps < 1:
            raise ValueError('Positive steps required')
        for i in range(steps):
            if bridge:
                bridge.before_step()
            state = env.backend.step(env.backend.home)
            if request['renderer'] != 'none':
                publish(env.backend.get_rgb(), {'step': i + 1, 'success': state['success']})
            if request['options'].get('realtime', True):
                time.sleep(1 / env.hz)
        return {'passed': True, 'steps': steps, 'scene': env.backend.metadata,
                'physics_only_rss_mib': physics_rss_mib,
                'final_rss_mib': psutil.Process().memory_info().rss / 2**20,
                'coordinates': env.coordinates.metadata()}


def source(request, output, publish, bridge):
    from ..utils.voice import log_say
    from ..real2sim.hardware import Leader
    from ..so101_joint_mapping import DEFAULT_MAPPER
    from . import source as source_stage
    from ..data.recording import SourceEpisode
    from ..data.episodes import DatasetWriter,LegacyEpisodeWriter,recording_args
    from ..so101_joint_mapping import observed_positions
    options = request['options']
    from ..real2sim.precision import approved_mappings, collection_coordinates
    options['_collection_mappings']=approved_mappings(request['workspace']) if request.get('workspace') else {}
    selected_precision=options['_collection_mappings'].get('leader',{}).get('profile')
    from ..real2sim.workspaces.reset import source_seed
    options['seed'] = source_seed(options.get('seed'))
    print('SOURCE_RANDOMIZATION_SEED', options['seed'], flush=True)
    count = int(options.get('episodes', 3))
    duration = float(options.get('episode_time_s', 20))
    if count < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Positive episode count/duration required')
    with simulation_environment.create_mujoco(**prepare_mujoco(request)) as env:
        publish = _preview(env, request, publish)
        if request.get('_script_bindings'):
            from ..simulation.task_context import TaskContext
            TaskContext(env.backend).validate_bindings(request['_script_bindings'])
        leader = Leader(None, options['port'], options.get('calibration_file_name', 'so101_leader.json'),
                        model_hash=selected_precision['urdf_sha256'] if selected_precision else None)
        try:
            # Fail rendering preflight before opening the physical leader bus.
            env.observe()
            if options.get('scene_only') and env.backend.workspace['task'].get('observation_cameras')!=['side','wrist']:
                raise NotImplementedError('Scene-only Source currently requires existing Side/Wrist Common recording')
            leader.connect()
            mapper = getattr(leader, 'precision_mapper', None) or DEFAULT_MAPPER
            if selected_precision and getattr(mapper, 'profile', {}).get('id') != selected_precision['id']:
                raise ValueError('Active calibration changed while preparing Source; start a new run')
            if selected_precision and mapper.metadata()['sha256']!=options['_collection_mappings']['leader']['sha256']:
                raise ValueError('Leader mapping changed before Source activation')
            leader.activate()
            storage='common' if env.backend.workspace['task'].get('observation_cameras')==['side','wrist'] else 'legacy'
            manifest,provenance=source_stage.metadata(env.recording_context(),options['instruction'])
            from ..real2sim.workspaces.demo import coordinate_contract
            from ..so101_dataset_contract import current_contract, MIMIC_SPACE
            contract = coordinate_contract(request['workspace']) if request.get('workspace') else current_contract(MIMIC_SPACE)
            provenance['coordinate_contract'] = collection_coordinates(contract, mapper if selected_precision else None,env.backend.metadata.get('simulation_configuration'))
            if options.get('resume_data'):
                previous=recording_provenance(options['resume_data']).get('coordinate_contract',{})
                if previous and 'simulation_model' not in previous:
                    # Keep a verified historical identity/legacy-mode contract unchanged.
                    provenance['coordinate_contract'].pop('simulation_model',None)
            if options.get('script_identity'):
                provenance['source_script'] = {**options['script_identity'], 'followup_contracts':request.get('_followup_contracts'),
                                               'source_reference':request.get('_source_reference'), 'declaration':request.get('_source_declaration')}
            provenance['leader'] = leader.metadata if hasattr(leader, 'metadata') else {'fixture': True}
            common=storage=='common'
            dataset_path=(output/'dataset').with_suffix('.hdf5') if common else output/'dataset'
            if common:
                writer=DatasetWriter(dataset_path,recording_args(provenance,manifest),provenance['coordinate_contract'])
            else:
                writer=LegacyEpisodeWriter(dataset_path,manifest,provenance);provenance=writer.provenance
            recording=SourceEpisode(dataset_path.parent if common else dataset_path,env.hz,manifest['cameras'],
                env.backend.metadata['dynamic_objects'],env.backend.metadata['pick'])
            if options.get('resume_data'):
                if common:source_stage.resume_common(writer,provenance,options['resume_data'])
                else:source_stage.resume_legacy(writer,options['resume_data'],env.backend.workspace)
            if writer.count >= count: raise ValueError('Episode target must exceed existing count')
            lower = np.maximum(env.coordinates.to_model(env.lower),env.backend.limits[:,0])
            upper = np.minimum(env.coordinates.to_model(env.upper),env.backend.limits[:,1])
            limit_counts = np.zeros(6,dtype=int)
            last_limit_log = -float('inf')
            episode = writer.count
            while writer.count < count:
                env.reset(options.get('seed', 0) + episode)
                if options.get('scene_only'):
                    from ..simulation.task_context import reset_objects
                    reset_objects(env.backend,options['script_scene']['objects'],request['_scene_resets'])
                episode += 1
                recording.begin(env.backend.get_initial_state())
                log_say(f'Episode {writer.count+1}',blocking=False)
                end = time.monotonic() + duration
                discard, quit_recording = False, False
                while time.monotonic() < end:
                    start = time.monotonic()
                    if bridge:
                        bridge.poll()
                        if 'quit' in bridge.pending:
                            quit_recording = True
                            break
                        if 'discard' in bridge.pending:
                            bridge.pending.remove('discard')
                            discard = True
                            break
                        if 'save' in bridge.pending:
                            bridge.pending.remove('save')
                            break
                    native, _ = leader.read()
                    target = observed_positions(mapper, native, raw=getattr(leader,'last_raw',None), motor_specs=getattr(leader,'motor_specs',None))
                    target = np.array([target[n] for n in env.backend.metadata['joint_order']])
                    requested = target.copy()
                    if requested.shape != (6,) or not np.isfinite(requested).all():
                        raise ValueError('Non-finite or invalid Leader target')
                    target = np.clip(requested,lower,upper)
                    limited = {n:{'requested_rad':float(requested[i]),'applied_rad':float(target[i])}
                               for i,n in enumerate(env.backend.metadata['joint_order']) if abs(requested[i]-target[i])>1e-8}
                    limit_counts += np.abs(requested-target)>1e-8
                    if limited and time.monotonic() - last_limit_log >= 1:
                        print('[WORKFLOW] Leader target limited: ' + json.dumps(limited), flush=True)
                        last_limit_log = time.monotonic()
                    env.backend._target(target)
                    observation = env.observe()
                    recording.append(env.recording_sample(observation,target,requested),observation['camera_images'])
                    env.backend.step(target)
                    publish(observation['camera_images'], {'episode_index': writer.count + 1,
                            'episode_total': count, 'attempt': episode, 'steps': len(recording.rows),
                            'task_id': (options.get('task_definition') or {}).get('task_id', ''),
                            'task_signals': {} if options.get('scene_only') else {'place': env.backend.is_success()},
                            'leader_limit_status': '관절 한계 적용: ' + ', '.join(limited) if limited else '',
                            'remaining_s': max(0., end - time.monotonic())})
                    time.sleep(max(0., 1 / env.hz - (time.monotonic() - start)))
                if not discard and not quit_recording and recording.rows:
                    provenance['leader_target_limits'] = {'policy':'URDF/backend target bounds', 'clamped_samples_by_joint':dict(zip(env.backend.metadata['joint_order'],limit_counts.tolist()))}
                    outcome=dict(success=None if options.get('scene_only') else env.backend.is_success(),success_source='recording_only' if options.get('scene_only') else 'simulation_place_condition')
                    if common:name=source_stage.save_common(writer,recording,env,provenance,outcome)
                    else:
                        provenance['coordinates']=env.coordinates.metadata()
                        name=writer.save(recording.rows,recording.initial,recording.video,outcome)
                    recording.abort()
                    print('Saved ' + name, flush=True)
                    log_say(f'save episode {writer.count}',blocking=False)
                if quit_recording:
                    break
                if writer.count < count:log_say('resetting',blocking=False)
                reset_end = time.monotonic() + float(options.get('reset_time_s', 0))
                while time.monotonic() < reset_end and writer.count < count:
                    if bridge:
                        bridge.poll()
                        if 'quit' in bridge.pending: quit_recording = True; break
                        bridge.pending.discard('save'); bridge.pending.discard('discard')
                    publish(env.backend.get_rgb(), {'episode_outcome': '저장 완료' if not discard else '폐기 완료',
                            'reset_remaining_s': max(0., reset_end - time.monotonic())})
                    time.sleep(.05)
                if quit_recording: break
            return {'passed': bool(writer.count), 'dataset': str(dataset_path),
                    'episode_count': writer.count,
                    'leader_target_limits': {'policy':'URDF/backend target bounds', 'clamped_samples_by_joint':dict(zip(env.backend.metadata['joint_order'],limit_counts.tolist()))},
                    'coordinates': env.coordinates.metadata(), 'hardware': 'leader input only; no follower commands'}
        except KeyboardInterrupt:
            if 'writer' not in locals(): raise
            return {'passed': bool(writer.count), 'dataset': str(dataset_path),
                    'episode_count': writer.count, 'stopped': True}
        finally:
            try:
                if 'recording' in locals():recording.abort()
            finally:
                try:
                    if 'writer' in locals() and common:writer.close()
                finally:
                    try:leader.close()
                    finally:log_say('stop recording',blocking=False)


def rollout(request, output, publish, bridge, policy_factory=None):
    from ..learning.minivla.policy import MiniVLA
    from .policy import ActionGuard
    checkpoint = Path(request['checkpoint'])
    if checkpoint.is_dir():
        return rollout_lerobot(request, output, publish, bridge)
    from .policy import minivla_root
    provenance = recording_provenance(minivla_root(checkpoint))
    _verify_workspace(request, provenance)
    policy = (policy_factory or MiniVLA)(checkpoint)
    rows = []
    options = request['options']
    atomic(output/'selected_checkpoint.json',{'path':str(checkpoint.resolve()),'step':getattr(policy,'checkpoint_step',None)})
    from .tasks import object_metrics,evaluation_distribution
    if int(options.get('episodes', 1)) < 1 or int(options.get('max_steps', 300)) < 1:
        raise ValueError('Positive rollout episodes/max_steps required')
    from contextlib import ExitStack
    from .evaluation import evaluation_recorder
    with simulation_environment.create_mujoco(**prepare_mujoco(request,coordinates=provenance['coordinates'],hz=policy.control_hz)) as env, ExitStack() as recordings:
        cameras = tuple(env.backend.workspace['task'].get('observation_cameras', ['side']))
        if not set(policy.cfg.cameras).issubset(cameras):
            raise ValueError('Checkpoint camera contract differs from workspace: '+str(policy.cfg.cameras)+' vs '+str(cameras))
        publish = _preview(env, request, publish)
        guard = ActionGuard(env.lower, env.upper, options.get('max_velocity', [2.] * 5 + [1.]), env.hz)
        for episode in range(int(options.get('episodes', 1))):
            env.reset(options.get('seed', 0) + episode)
            guard.reset()
            tracker = grasp_tracker(env)
            recorder=evaluation_recorder(env.backend,env.backend.workspace,'mujoco',checkpoint,
                output/f'evaluation_{episode:04d}.hdf5',env.hz,options.get('seed',0)+episode,recordings)
            remaining, trace = [], []
            metric_state = {};active=env.backend.workspace['task']['dynamic'];pick=env.backend.metadata['pick']
            initial_positions={n:env.backend.get_object_pose(n)[:3,3].copy() for n in active}
            for step in range(int(options.get('max_steps', 300))):
                if bridge:
                    bridge.before_step()
                observed = env.observe()
                replanning=not remaining
                if replanning:
                    time_before = env.backend.data.time
                    model_images = [observed['camera_images'][n] for n in policy.cfg.cameras] if 'camera_images' in observed else observed['images']
                    chunk, metrics = policy.predict(model_images, observed['state'], options['instruction'])
                    chunk = np.asarray(chunk)
                    if chunk.shape != (policy.cfg.horizon, 6) or not np.isfinite(chunk).all():
                        raise ValueError('Policy must output finite [H,6] chunk')
                    if env.backend.data.time != time_before:
                        raise RuntimeError('Simulator advanced during prediction; discard stale action chunk')
                    remaining = list(chunk[:policy.cfg.replan_steps])
                action = guard.apply(remaining.pop(0), observed['state'])
                if recorder:recorder.append(env.coordinates.to_model(action),tracker)
                result = env.step(action)
                trace.append({'timestamp_s': result['timestamp_s'], 'model_joint_rad': result['joint_pos'].tolist(),
                              'canonical_action': action.tolist(), 'success': result['success'], 'policy_metrics': metrics, 'replan':replanning})
                signals = subtask_signals(env, tracker, step)
                measured = object_metrics(active,pick,{n:env.backend.get_object_pose(n)[:3,3] for n in active},
                    env.backend.get_eef_pose()[:3,3],metric_state,signals['grasp'],signals['place'],initial_positions)
                trace[-1]['object_metrics']=measured
                publish(observed['images'][0], {'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                        'steps': step + 1, 'task_id': (options.get('task_definition') or {}).get('task_id', ''),
                        'task_signals': signals, 'success': result['success']})
                if result['success']:
                    break
            row = {'episode': episode, 'steps': len(trace), 'success': bool(trace[-1]['success']),
                   'limit_violations': guard.violations, 'rate_limited': guard.rate_limited,**measured,
                   'inference_latency_s':[r['policy_metrics'].get('latency_s') for r in trace if r.get('replan')]}
            rows.append(row)
            if recorder:recorder.finish(row,tracker,int(options.get('max_steps',300)))
            if bridge:
                bridge.details.update(episode_outcome='성공' if row['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if row['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': row, 'trace': trace})
        return {'passed': True, 'checkpoint_path':str(checkpoint.resolve()),
                'checkpoint_step':getattr(policy,'checkpoint_step',None), 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows),
                'scope': 'synchronous simulated rollout; passed means execution validation, not task success',
                'coordinates': env.coordinates.metadata(), 'scene': env.backend.metadata,
                'scene_configuration':env.backend.metadata.get('scene_configuration'),
                **evaluation_distribution({'provenance':provenance},env.backend.workspace,env.backend.workspace['task']['dynamic'])}


def rollout_lerobot(request, output, publish, bridge):
    import torch
    from .policy import load_policy, observation
    from ..real2sim.workspaces.demo import coordinate_contract
    from .policy import ActionGuard
    from .tasks import object_metrics,evaluation_distribution
    options = request['options']; w = load(request['workspace'])
    device = options.get('device', 'cuda:0')
    policy, pre, post = load_policy(request['checkpoint'], coordinate_contract(w), device)
    rows = []
    from contextlib import ExitStack
    from .evaluation import evaluation_recorder
    with simulation_environment.create_mujoco(**prepare_mujoco(request)) as env, ExitStack() as recordings:
        emit = _preview(env, request, publish)
        guard = ActionGuard(env.backend.limits[:, 0], env.backend.limits[:, 1], [2.] * 6, env.hz)
        for episode in range(int(options.get('episodes', 1))):
            env.reset(options.get('seed', 0) + episode); policy.reset(); guard.reset(); tracker = grasp_tracker(env)
            recorder=evaluation_recorder(env.backend,env.backend.workspace,'mujoco',request['checkpoint'],
                output/f'evaluation_{episode:04d}.hdf5',env.hz,options.get('seed',0)+episode,recordings)
            trace = [];metric_state={};latencies=[];inference_latencies=[];active=env.backend.workspace['task']['dynamic'];pick=env.backend.metadata['pick']
            initial_positions={n:env.backend.get_object_pose(n)[:3,3].copy() for n in active}
            for i in range(int(options.get('max_steps', 300))):
                if bridge: bridge.before_step()
                state = env.backend.get_state(); side = env.backend.get_rgb()
                images = {n:env.backend.get_rgb(n) for n in w['task'].get('observation_cameras',list(w['profile']['cameras']))}
                batch = observation(state['joint_pos'], None, None, options['instruction'], device, images=images)
                begin=time.perf_counter()
                with torch.inference_mode():
                    action = post(policy.select_action(pre(batch))).detach().cpu().numpy().reshape(6)
                latencies.append(time.perf_counter()-begin)
                inference=getattr(policy,'last_inference_metrics',None)
                if inference and inference.get('latency_s') is not None:inference_latencies.append(inference['latency_s'])
                action = guard.apply(action, state['joint_pos'])
                if recorder:recorder.append(action,tracker)
                env.backend.step(action)
                signals = subtask_signals(env, tracker, i)
                measured=object_metrics(active,pick,{n:env.backend.get_object_pose(n)[:3,3] for n in active},
                    env.backend.get_eef_pose()[:3,3],metric_state,signals['grasp'],signals['place'],initial_positions)
                trace.append({'action': action.tolist(), 'success': signals['place'],'object_metrics':measured})
                emit(side, {'steps': i + 1, 'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                    'task_id': (options.get('task_definition') or {}).get('task_id', ''), 'task_signals': signals})
                if signals['place']: break
            rows.append({'episode': episode, 'steps': len(trace), 'success': trace[-1]['success'],**measured,
                         'inference_latency_s':inference_latencies or None,'action_call_latency_s':latencies})
            if recorder:recorder.finish(rows[-1],tracker,int(options.get('max_steps',300)))
            if bridge:
                bridge.details.update(episode_outcome='성공' if rows[-1]['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if rows[-1]['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': rows[-1], 'trace': trace})
    return {'passed': True, 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows),
            'scene_configuration':env.backend.metadata.get('scene_configuration'),
            **evaluation_distribution({'path':str(request['checkpoint'])},w,active)}


def execute(request):
    if request['stage'] in ('annotate','datagen'):
        raise ValueError('Episode stages use workflow.operations.execute_request and the common dataset API')
    request = {**request, 'options': dict(request['options'])}
    if request['options'].get('remote_policy'):
        from huggingface_hub import snapshot_download
        remote = request['options']['remote_policy']
        checkpoint = Path(snapshot_download(repo_id=remote['repo_id'], revision=remote['commit']))
        if (checkpoint / 'best.pt').is_file():
            request['checkpoint'] = str(checkpoint / 'best.pt')
            request['config'] = str(checkpoint / 'config.json')
        else:
            request['checkpoint'] = str(checkpoint)
    if request.get('workspace'):
        workspace = load(request['workspace'])
        request['primary_camera'] = workspace['task'].get('observation_cameras', list(workspace['profile']['cameras']))[0]
    if request.get('workspace') and request['stage'] in ('source', 'rollout') and not request['options'].get('scene_only'):
        from ..real2sim.workspaces.tasks import configured
        w = configured(workspace, request['options'].get('task_definition'))
        request['options'].setdefault('instruction', w['task_definition']['language_instruction'])
    mujoco_request(request['stage'], sys.executable, request.get('workspace', ''), request.get('input', ''),
                         request.get('config', ''), request.get('checkpoint', ''), request['renderer'], request['options'])
    output = Path(request['output']).resolve()
    if request.get('workspace') and output.is_relative_to(Path(request['workspace']).resolve()):
        raise ValueError('Output must be outside immutable workspace')
    from .paths import resumes_in_place
    output.mkdir(parents=True, exist_ok=resumes_in_place(request['stage'],request['options'],output))
    from .presentation import current
    bridge = current()

    def publish(frame, details):
        images = frame if isinstance(frame, dict) else {'sim_' + request.get('primary_camera','side'): frame} if frame is not None else {}
        if bridge:
            progress = dict(details)
            bridge.steps = int(progress.pop('steps', progress.pop('step', bridge.steps)))
            bridge.details.update(progress)
            if images:
                bridge.publish(images)
            bridge.status()

    stage = request['stage']
    try:
        if stage == 'environment':
            result = environment(request, output, publish, bridge)
        elif stage == 'source':
            result = source(request, output, publish, bridge)
        elif stage == 'convert':
            from .episodes import convert_dataset
            result = {'passed': True, **convert_dataset(request['input'], output, {**request['options'], 'workspace': request['workspace']})}
        elif stage == 'import':
            from .episodes import convert_hdf5
            recipe = json.loads(Path(request['input']).read_text())
            if any(type(e.get('success')) is not bool for e in recipe['episodes']):
                raise ValueError('Import recipe must explicitly label success for every episode')
            convert_hdf5(request['input'], output / 'dataset')
            for e in recipe['episodes']:
                atomic(output / 'dataset' / e['id'] / 'outcome.json', {'success': e['success'], 'validation_only': False})
            from ..so101_joint_mapping import RobotCoordinates
            coordinates = RobotCoordinates(request['options'].get('gripper_closed_rad'), request['options'].get('gripper_open_rad'))
            atomic(output / 'dataset/project_provenance.json', {'schema': 'so101.minivla-dataset/1',
                 'workspace': metadata(load(request['workspace'])), 'coordinates': coordinates.metadata(),
                 'backend': 'explicit_import', 'recipe': recipe, 'source': request['input']})
            result = {'passed': True, 'dataset': str(output / 'dataset')}
        elif stage == 'rollout':
            from .evaluation import evaluate_sim
            from ..real2sim.workspaces.tasks import configured
            w=workspace
            definitions=request['options'].get('tasks') or [configured(w,request['options'].get('task_definition'))['task_definition']]
            result=evaluate_sim(request,request['checkpoint'],w,definitions,bridge,publish)

        else:
            raise ValueError(stage)
        result['forbidden_imports'] = _forbidden()
        result['stage'] = stage
        atomic(output / 'report.json', result)
        print(json.dumps({'passed': result['passed'], 'report': str(output / 'report.json')}, ensure_ascii=False), flush=True)
        if not result['passed']:
            raise RuntimeError('Stage validation did not pass')
    except BaseException as exc:
        if not (output / 'report.json').exists():
            atomic(output / 'report.json', {'passed': False, 'stage': stage, 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if bridge:
            bridge.close()
