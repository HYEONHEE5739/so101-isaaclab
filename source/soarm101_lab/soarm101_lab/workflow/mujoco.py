"""Worker-only implementations. No model, simulation or hardware in the GUI process."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..simulation.mujoco_backend import PolicyEnvironment
from .episodes import annotate, datagen, grasp_tracker, subtask_signals
from .operations import mujoco_request, MODEL_ROOT, ROOT
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata
from ..simulation.mjcf import RGB_WIDTH, RGB_HEIGHT


def _environment(request, output, coordinates=None, hz=None):
    options = request['options']
    from ..real2sim.precision import uses_urdf_frame
    recorded_contract = None
    source = options.get('resume_data') or request.get('input') or request.get('checkpoint')
    if isinstance(source, (str, Path)):
        path = Path(source); path = path.parent if path.is_file() else path
        if (path/'project_provenance.json').is_file():
            recorded_contract = _provenance(path).get('coordinate_contract')
        elif (path/'so101_coordinates.json').is_file():
            from ..so101_dataset_contract import read_contract
            recorded_contract = read_contract(path)
    coords = coordinates or {}
    physics_profile = options.get('physics_profile', 'isaac_matched')
    if options.get('resume_data'):
        previous_scene = _provenance(options['resume_data'])['scene']
        physics_profile = previous_scene.get('physics_profile', 'legacy_urdf')
    return PolicyEnvironment(output / 'scene', request['workspace'], request['renderer'],
                              control_hz=hz or options.get('control_hz', 30),
                              width=options.get('width', RGB_WIDTH), height=options.get('height', RGB_HEIGHT),
                              closed_rad=coords.get('gripper_closed_rad', options.get('gripper_closed_rad')),
                              open_rad=coords.get('gripper_open_rad', options.get('gripper_open_rad')),
                              task_definition=options.get('task_definition'),
                              full_workspace=options.get('full_workspace', True),
                              viewer=options.get('mode') == 'developer', physics_profile=physics_profile,
                              identity_sim=options.get('_identity_sim', False) or uses_urdf_frame(recorded_contract))


def _provenance(dataset):
    return json.loads((Path(dataset) / 'project_provenance.json').read_text())


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
    with _environment(request, output) as env:
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
    from ..real2sim.hardware import Leader
    from ..so101_joint_mapping import DEFAULT_MAPPER
    from .episodes import EpisodeWriter
    from ..so101_joint_mapping import observed_positions
    options = request['options']
    from ..real2sim.precision import active_precision, collection_coordinates, profile_for_model
    selected_precision = active_precision('leader') if request.get('workspace') else None
    if selected_precision:
        import hashlib
        robot = load(request['workspace'])['root']/'assets/SO101/urdf/so101_isaaclab.urdf'
        selected_precision = profile_for_model(selected_precision, hashlib.sha256(robot.read_bytes()).hexdigest())
    options['_identity_sim'] = selected_precision is not None
    from ..real2sim.workspaces.reset import source_seed
    options['seed'] = source_seed(options.get('seed'))
    print('SOURCE_RANDOMIZATION_SEED', options['seed'], flush=True)
    count = int(options.get('episodes', 3))
    duration = float(options.get('episode_time_s', 20))
    if count < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Positive episode count/duration required')
    with _environment(request, output) as env:
        publish = _preview(env, request, publish)
        leader = Leader(None, options['port'], options.get('calibration_file_name', 'so101_leader.json'),
                        model_hash=selected_precision['urdf_sha256'] if selected_precision else None)
        try:
            # Fail rendering preflight before opening the physical leader bus.
            env.observe()
            leader.connect()
            mapper = getattr(leader, 'precision_mapper', None) or DEFAULT_MAPPER
            if selected_precision and getattr(mapper, 'profile', {}).get('id') != selected_precision['id']:
                raise ValueError('Active calibration changed while preparing Source; start a new run')
            leader.activate()
            writer = EpisodeWriter(output / 'dataset', env, options['instruction'])
            from ..real2sim.workspaces.demo import coordinate_contract
            from ..so101_dataset_contract import current_contract, MIMIC_SPACE
            contract = coordinate_contract(request['workspace']) if request.get('workspace') else current_contract(MIMIC_SPACE)
            writer.provenance['coordinate_contract'] = collection_coordinates(contract, mapper if selected_precision else None)
            writer.provenance['leader'] = leader.metadata if hasattr(leader, 'metadata') else {'fixture': True}
            if options.get('resume_data'): writer.resume(options['resume_data'])
            if len(writer.manifest['episodes']) >= count: raise ValueError('Episode target must exceed existing count')
            lower = np.maximum(env.coordinates.to_model(env.lower),env.backend.limits[:,0])
            upper = np.minimum(env.coordinates.to_model(env.upper),env.backend.limits[:,1])
            limit_counts = np.zeros(6,dtype=int)
            last_limit_log = -float('inf')
            episode = len(writer.manifest['episodes'])
            while len(writer.manifest['episodes']) < count:
                env.reset(options.get('seed', 0) + episode)
                episode += 1
                writer.begin()
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
                    writer.append(observation, target, requested_target=requested)
                    env.backend.step(target)
                    publish(observation['camera_images'], {'episode_index': len(writer.manifest['episodes']) + 1,
                            'episode_total': count, 'attempt': episode, 'steps': len(writer.rows),
                            'task_id': options.get('task_definition', {}).get('task_id', ''),
                            'task_signals': {'place': env.backend.is_success()},
                            'leader_limit_status': '관절 한계 적용: ' + ', '.join(limited) if limited else '',
                            'remaining_s': max(0., end - time.monotonic())})
                    time.sleep(max(0., 1 / env.hz - (time.monotonic() - start)))
                if not discard and not quit_recording and writer.rows:
                    writer.provenance['leader_target_limits'] = {'policy':'URDF/backend target bounds', 'clamped_samples_by_joint':dict(zip(env.backend.metadata['joint_order'],limit_counts.tolist()))}
                    name = writer.save(env.backend.is_success())
                    print('Saved ' + name, flush=True)
                if quit_recording:
                    break
                reset_end = time.monotonic() + float(options.get('reset_time_s', 0))
                while time.monotonic() < reset_end and len(writer.manifest['episodes']) < count:
                    if bridge:
                        bridge.poll()
                        if 'quit' in bridge.pending: quit_recording = True; break
                        bridge.pending.discard('save'); bridge.pending.discard('discard')
                    publish(env.backend.get_rgb(), {'episode_outcome': '저장 완료' if not discard else '폐기 완료',
                            'reset_remaining_s': max(0., reset_end - time.monotonic())})
                    time.sleep(.05)
                if quit_recording: break
            return {'passed': bool(writer.manifest['episodes']), 'dataset': str(writer.root),
                    'episode_count': len(writer.manifest['episodes']),
                    'leader_target_limits': {'policy':'URDF/backend target bounds', 'clamped_samples_by_joint':dict(zip(env.backend.metadata['joint_order'],limit_counts.tolist()))},
                    'coordinates': env.coordinates.metadata(), 'hardware': 'leader input only; no follower commands'}
        except KeyboardInterrupt:
            if 'writer' not in locals(): raise
            return {'passed': bool(writer.manifest['episodes']), 'dataset': str(writer.root),
                    'episode_count': len(writer.manifest['episodes']), 'stopped': True}
        finally:
            leader.close()


def replay(request, output, publish, bridge):
    from ..learning.minivla.data import load_manifest, episode_arrays
    path = Path(request['input'])
    provenance = _provenance(path)
    _verify_workspace(request, provenance)
    manifest = load_manifest(path / 'manifest.json')
    name = request['options'].get('episode', manifest['episodes'][0]['id'])
    entry = next(e for e in manifest['episodes'] if e['id'] == name)
    arrays = episode_arrays(entry, manifest.get('cameras',['side']), manifest['control_hz'])
    scene = provenance['scene']
    request = {**request, 'options': {**request['options'], 'task_definition': scene.get('task_definition'),
                                    'physics_profile': scene.get('physics_profile', 'legacy_urdf'),
                                    'full_workspace': scene.get('full_workspace', False),
                                    'width': scene['camera']['resolution'][0], 'height': scene['camera']['resolution'][1]}}
    errors = []
    with _environment(request, output, provenance['coordinates'], manifest['control_hz']) as env:
        publish = _preview(env, request, publish)
        env.backend.restore_state(json.loads((Path(entry['_path']) / 'replay.json').read_text()))
        raw = np.load(Path(entry['_path']) / 'raw_state.npy', allow_pickle=False)
        actions = np.load(Path(entry['_path']) / 'raw_action.npy', allow_pickle=False)
        if raw.shape != actions.shape or raw.shape != arrays['state'].shape:
            raise ValueError('Raw replay state/action lengths differ')
        for i, target in enumerate(actions):
            if bridge:
                bridge.before_step()
            actual = env.backend.get_state()['joint_pos']
            errors.append(float(np.max(np.abs(actual - raw[i]))))
            publish(env.backend.get_rgb() if request['renderer'] != 'none' else None, {'frame': i, 'episode': name})
            env.backend.step(target)
            if request['options'].get('realtime', True): time.sleep(1 / env.hz)
        maximum = max(errors)
        return {'passed': maximum < 1e-5, 'max_joint_error_rad': maximum, 'episode': name,
                'frames': len(actions), 'scope': 'same MuJoCo model integration-state/action replay'}


def train(request, output):
    from ..learning.minivla.config import MiniVLAConfig
    from ..learning.minivla.data import Episodes
    from .artifacts import fingerprint
    dataset = Path(request['input'])
    provenance = _provenance(dataset)
    if request['options'].get('architecture', 'minivla') != 'minivla':
        from .artifacts import inspect_artifact
        from .policy import train as train_lerobot
        child = dataset / 'lerobot'
        artifact = dict(inspect_artifact('dataset', child), path=str(child), id='canonical-conversion')
        model = Path(request['options'].get('resume_output', output / 'policy'))
        train_lerobot(artifact, request['options'], model)
        return {'passed': True, 'checkpoint': str(model / 'checkpoints/last/pretrained_model')}
    cfg = MiniVLAConfig.load(request['config'])
    if 'steps' in request['options']:
        cfg.steps = int(request['options']['steps'])
        if cfg.steps < 1:
            raise ValueError('Positive training steps required')
    stats_path = dataset / 'statistics.json'
    stats = json.loads(stats_path.read_text())
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for entry in manifest['episodes']:
        outcome = json.loads((dataset / entry['path'] / 'outcome.json').read_text())
        if outcome.get('validation_only'):
            raise ValueError('Synthetic smoke data cannot be trained')
    Episodes(dataset / 'manifest.json', stats, cfg, 'train')
    Episodes(dataset / 'manifest.json', stats, cfg, 'val')
    identity = fingerprint(dataset)
    model = Path(request['options'].get('resume_output', output / 'policy'))
    cfg.save(output / 'training_config.json')
    argv = [sys.executable, '-m', 'soarm101_lab.learning.minivla.train', '--config', str(output / 'training_config.json'),
            '--manifest', str(dataset / 'manifest.json'), '--stats', str(stats_path), '--out', str(model)]
    if request['options'].get('resume_checkpoint'):
        # Let the model's exact config/statistics guard enforce resume compatibility.
        resume = Path(request['options']['resume_checkpoint'])
        if resume.is_dir(): resume = resume / 'last.pt'
        argv += ['--resume', str(resume)]
    from ..real2sim.launcher import worker_environment
    subprocess.run(argv, cwd=ROOT, env=worker_environment(ROOT), check=True)
    if fingerprint(dataset) != identity:
        raise ValueError('Training dataset changed during the run')
    if not (model / 'best.pt').is_file() or not (model / 'last.pt').is_file():
        raise ValueError('Model trainer did not produce complete checkpoints')
    import xml.etree.ElementTree as ET
    from ..so101_joint_mapping import JOINTS
    workspace = load(request['workspace'])
    urdf = ET.parse(workspace['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf').getroot()
    limits = {j.get('name'): [float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))]
              for j in urdf.findall('joint') if j.get('type') == 'revolute'}
    atomic(model / 'project_provenance.json', {**provenance, 'source_dataset': str(dataset.resolve()),
                                               'joint_limits_rad': [limits[n] for n in JOINTS],
                                               'dataset_sha256': identity,
                                               'trained_config': json.loads((model / 'config.json').read_text())})
    from ..so101_dataset_contract import write_contract, JOINT_SPACE
    from ..real2sim.workspaces.demo import coordinate_contract
    write_contract(model, {**provenance.get('coordinate_contract', coordinate_contract(workspace)), 'action_space': JOINT_SPACE})
    return {'passed': True, 'checkpoint': str(model / 'best.pt'), 'model': 'custom_so101_minivla'}


def train_from_lerobot(artifact, options, output):
    """Adapt the selected LeRobot frames; preserve measured state and action alignment."""
    from ..so101_joint_mapping import RobotCoordinates
    from .episodes import prepare_dataset
    from .artifacts import fingerprint
    root = Path(artifact['path'])
    if options.get('resume_checkpoint'):
        from .policy import resume_output
        model = resume_output(options['resume_checkpoint'], 'minivla')
        previous = _provenance(model)
        prepared = Path(previous['source_dataset'])
        original = previous.get('lerobot_source_dataset') or _provenance(prepared).get('source_dataset')
        if not original or Path(original).resolve() != root.resolve() or previous.get('source_sha256') != fingerprint(root):
            raise ValueError('MiniVLA resume requires the original unchanged LeRobot dataset')
        request = {'input': str(prepared), 'workspace': options['workspace'],
                   'config': str(model / 'config.json'), 'options': {**options, 'resume_output': str(model)}}
        result = train(request, output)
        atomic(model / 'project_provenance.json', {**_provenance(model), 'lerobot_source_dataset': str(root.resolve())})
        return result
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((root / 'meta/info.json').read_text())
    ds = LeRobotDataset(options.get('dataset_repo_id', info.get('repo_id', 'local/so101')), root=root)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'canonical_source'; source.mkdir()
    coords = RobotCoordinates()
    manifest = {'schema_version': 1, 'joint_names': list(coords.metadata()['joint_order']),
        'units': ['rad'] * 5 + ['fraction'], 'control_hz': info['fps'],
        'action_semantics': 'absolute_target_after_observation', 'episodes': [],
        'cameras':[k.removeprefix('observation.images.').removesuffix('_cam') for k in info['features'] if k.startswith('observation.images.')]}
    # Converted generated variants need explicit source groups, never random frame splitting.
    provenance = artifact.get('provenance', {})
    generated = bool(provenance.get('generation_method') or provenance.get('source_env_args', {}).get('generation')) or 'datagen' in str(provenance.get('source_dataset', '')).lower()
    source_groups = options.get('source_groups') or provenance.get('source_groups', {})
    if generated and not source_groups:
        raise ValueError('MiniVLA split requires source_groups for generated LeRobot episodes')
    by_episode = {}
    for i in range(len(ds)):
        index = int(ds.hf_dataset[i]['episode_index'])
        by_episode.setdefault(index, []).append(i)
    for index, indices in sorted(by_episode.items()):
        if source_groups and str(index) not in source_groups:
            raise ValueError('Source group mapping must cover every episode')
        rows = [ds[i] for i in indices]
        name = f'episode_{index:04d}'; folder = source / name; folder.mkdir()
        states = np.stack([r['observation.state'].numpy() for r in rows])
        actions = np.stack([r['action'].numpy() for r in rows])
        np.save(folder / 'state.npy', coords.to_policy(states, measured=True))
        np.save(folder / 'action.npy', coords.to_policy(actions))
        np.save(folder / 'raw_state.npy', states); np.save(folder / 'raw_action.npy', actions)
        times = np.arange(len(rows), dtype=float) / info['fps']
        for camera in manifest['cameras']:
            key='observation.images.'+camera+'_cam'
            frames=np.stack([np.clip(r[key].numpy().transpose(1,2,0)*255,0,255).astype(np.uint8) for r in rows])
            np.save(folder/(camera+'.npy'),frames);np.save(folder/(camera+'_timestamp.npy'),times)
        for key in ('timestamp','action_timestamp'):np.save(folder/(key+'.npy'),times)
        atomic(folder / 'outcome.json', {'success': False, 'success_known': False, 'validation_only': False})
        manifest['episodes'].append({'id': name, 'path': name, 'split': 'train', 'instruction': rows[0]['task'],
            'group_id': source_groups.get(str(index), name)})
    atomic(source / 'manifest.json', manifest)
    atomic(source / 'project_provenance.json', {'schema': 'so101.minivla-dataset/1',
        'workspace': artifact['workspace'], 'backend': artifact.get('backend', 'isaac'),
        'coordinates': coords.metadata(), 'task_definitions': artifact.get('tasks', []),
        'coordinate_contract': artifact['coordinates'],
        'source_dataset': str(root), 'source_sha256': fingerprint(root), 'outcomes': 'unknown; preserving selected training data'})
    prepared = output / 'dataset'
    prepare_dataset(source, prepared, include_failed=True)
    workspace = options['workspace']
    request = {'input': str(prepared), 'workspace': workspace,
        'config': options.get('model_config', str(MODEL_ROOT / 'configs/default.json')), 'options': options}
    result = train(request, output)
    model = output / 'policy'
    atomic(model / 'project_provenance.json', {**_provenance(model), 'lerobot_source_dataset': str(root.resolve())})
    return result


def rollout(request, output, publish, bridge, policy_factory=None):
    from ..learning.minivla.policy import MiniVLA, ActionGuard
    checkpoint = Path(request['checkpoint'])
    if checkpoint.is_dir():
        return rollout_lerobot(request, output, publish, bridge)
    provenance = _provenance(checkpoint.parent)
    _verify_workspace(request, provenance)
    policy = (policy_factory or MiniVLA)(checkpoint)
    rows = []
    options = request['options']
    if int(options.get('episodes', 1)) < 1 or int(options.get('max_steps', 300)) < 1:
        raise ValueError('Positive rollout episodes/max_steps required')
    with _environment(request, output, provenance['coordinates'], policy.control_hz) as env:
        cameras = tuple(env.backend.workspace['task'].get('observation_cameras', ['side']))
        if not set(policy.cfg.cameras).issubset(cameras):
            raise ValueError('Checkpoint camera contract differs from workspace: '+str(policy.cfg.cameras)+' vs '+str(cameras))
        publish = _preview(env, request, publish)
        guard = ActionGuard(env.lower, env.upper, options.get('max_velocity', [2.] * 5 + [1.]), env.hz)
        for episode in range(int(options.get('episodes', 1))):
            env.reset(options.get('seed', 0) + episode)
            guard.reset()
            tracker = grasp_tracker(env)
            remaining, trace = [], []
            for step in range(int(options.get('max_steps', 300))):
                if bridge:
                    bridge.before_step()
                observed = env.observe()
                if not remaining:
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
                result = env.step(action)
                trace.append({'timestamp_s': result['timestamp_s'], 'model_joint_rad': result['joint_pos'].tolist(),
                              'canonical_action': action.tolist(), 'success': result['success'], 'policy_metrics': metrics})
                signals = subtask_signals(env, tracker, step)
                publish(observed['images'][0], {'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                        'steps': step + 1, 'task_id': options.get('task_definition', {}).get('task_id', ''),
                        'task_signals': signals, 'success': result['success']})
                if result['success']:
                    break
            row = {'episode': episode, 'steps': len(trace), 'success': bool(trace[-1]['success']),
                   'limit_violations': guard.violations, 'rate_limited': guard.rate_limited}
            rows.append(row)
            if bridge:
                bridge.details.update(episode_outcome='성공' if row['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if row['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': row, 'trace': trace})
        return {'passed': True, 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows),
                'scope': 'synchronous simulated rollout; passed means execution validation, not task success',
                'coordinates': env.coordinates.metadata(), 'scene': env.backend.metadata}


def rollout_lerobot(request, output, publish, bridge):
    import torch
    from .policy import load_policy, observation
    from ..real2sim.workspaces.demo import coordinate_contract
    from ..learning.minivla.policy import ActionGuard
    options = request['options']; w = load(request['workspace'])
    device = options.get('device', 'cuda:0')
    policy, pre, post = load_policy(request['checkpoint'], coordinate_contract(w), device)
    rows = []
    with _environment(request, output) as env:
        emit = _preview(env, request, publish)
        guard = ActionGuard(env.backend.limits[:, 0], env.backend.limits[:, 1], [2.] * 6, env.hz)
        for episode in range(int(options.get('episodes', 1))):
            env.reset(options.get('seed', 0) + episode); policy.reset(); guard.reset(); tracker = grasp_tracker(env)
            trace = []
            for i in range(int(options.get('max_steps', 300))):
                if bridge: bridge.before_step()
                state = env.backend.get_state(); side = env.backend.get_rgb()
                images = {n:env.backend.get_rgb(n) for n in w['task'].get('observation_cameras',list(w['profile']['cameras']))}
                batch = observation(state['joint_pos'], None, None, options['instruction'], device, images=images)
                with torch.inference_mode():
                    action = post(policy.select_action(pre(batch))).detach().cpu().numpy().reshape(6)
                action = guard.apply(action, state['joint_pos'])
                env.backend.step(action)
                signals = subtask_signals(env, tracker, i)
                trace.append({'action': action.tolist(), 'success': signals['place']})
                emit(side, {'steps': i + 1, 'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                    'task_id': options.get('task_definition', {}).get('task_id', ''), 'task_signals': signals})
                if signals['place']: break
            rows.append({'episode': episode, 'steps': len(trace), 'success': trace[-1]['success']})
            if bridge:
                bridge.details.update(episode_outcome='성공' if rows[-1]['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if rows[-1]['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': rows[-1], 'trace': trace})
    return {'passed': True, 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows)}


def execute(request):
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
    if request.get('workspace') and request['stage'] in ('source', 'rollout'):
        from .tasks import configured
        w = configured(load(request['workspace']), request['options'].get('task_definition'))
        request['options'].setdefault('instruction', w['task_definition']['language_instruction'])
    mujoco_request(request['stage'], sys.executable, request.get('workspace', ''), request.get('input', ''),
                         request.get('config', ''), request.get('checkpoint', ''), request['renderer'], request['options'])
    output = Path(request['output']).resolve()
    if request.get('workspace') and output.is_relative_to(Path(request['workspace']).resolve()):
        raise ValueError('Output must be outside immutable workspace')
    output.mkdir(parents=True, exist_ok=False)
    from .presentation import current
    bridge = current()

    def publish(frame, details):
        images = frame if isinstance(frame, dict) else {'sim_' + request.get('primary_camera','side'): frame} if frame is not None else {}
        if images and not (output / 'rgb.png').exists():
            from PIL import Image
            Image.fromarray(images.get('sim_' + request.get('primary_camera', 'side'), next(iter(images.values())))).save(output / 'rgb.png')
            for name, rgb in images.items(): Image.fromarray(rgb).save(output / (name + '.png'))
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
        elif stage == 'replay':
            result = replay(request, output, publish, bridge)
        elif stage == 'annotate' and 'episode' not in request['options']:
            result = annotate(request, output, publish, bridge, _environment, _preview)
        elif stage == 'datagen':
            result = datagen(request, output, publish, bridge, _environment, _preview)
        elif stage == 'convert':
            from .episodes import convert_dataset
            result = {'passed': True, **convert_dataset(request['input'], output, {**request['options'], 'workspace': request['workspace']})}
        elif stage == 'prepare':
            from .episodes import prepare_dataset
            result = {'passed': True, **prepare_dataset(request['input'], output / 'dataset',
                                                        request['options'].get('splits'), request['options'].get('include_failed', False))}
        elif stage == 'annotate':
            from .episodes import annotate_episode
            result = {'passed': True, **annotate_episode(request['input'], request['options']['episode'], request['options']['boundaries'])}
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
            result = {'passed': True, 'dataset': str(output / 'dataset'), 'replay_available': False}
        elif stage == 'train':
            result = train(request, output)
        elif stage == 'rollout':
            definitions = request['options'].get('tasks', [])
            if len(definitions) <= 1:
                result = rollout(request, output, publish, bridge)
            else:
                reports = []
                for definition in definitions:
                    folder = output / definition['task_id']; folder.mkdir()
                    opts = {**request['options'], 'task_definition': definition, 'instruction': definition['language_instruction']}
                    if bridge: bridge.details['task_id'] = definition['task_id']
                    reports.append({'task_definition': definition, **rollout({**request, 'options': opts}, folder, publish, bridge)})
                result = {'passed': all(r['passed'] for r in reports), 'tasks': reports, 'task_definitions': definitions}

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
