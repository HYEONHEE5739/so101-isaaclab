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
    from ..real2sim.precision import simulation_configuration,calibration_status
    recorded_contract = None
    source = options.get('resume_data') or request.get('input') or request.get('checkpoint')
    if isinstance(source, (str, Path)):
        path = Path(source); path = path.parent if path.is_file() else path
        if (path/'project_provenance.json').is_file():
            recorded_contract = _provenance(path).get('coordinate_contract')
        elif (path/'so101_coordinates.json').is_file():
            from ..so101_dataset_contract import read_contract
            recorded_contract = read_contract(path)
    w=load(request['workspace'])
    from .tasks import configured
    w=configured(w,options.get('task_definition'),options.get('active_object_ids'),backend='mujoco')
    if request.get('stage')=='source':
        from ..real2sim.workspaces.reset import source_workspace
        w=source_workspace(w,options.get('resume_data'))
    elif recorded_contract and recorded_contract.get('simulation_model'):
        w=configured(w,reset=recorded_contract['simulation_model'].get('scene_configuration',{}).get('reset'))
    simulation=simulation_configuration(w,coordinates=recorded_contract,
        mappings=options.get('_collection_mappings'),identity=options.get('identity_sim_for_validation',False),
        backend='mujoco',scene_configuration=w['scene_configuration'],hardware_role='leader' if '_collection_mappings' in options else None)
    print('[WORKFLOW] '+calibration_status(simulation),flush=True)
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
                              simulation_configuration=simulation,
                              active_object_ids=options.get('active_object_ids'))


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
    from ..utils.voice import log_say
    from ..real2sim.hardware import Leader
    from ..so101_joint_mapping import DEFAULT_MAPPER
    from .episodes import EpisodeWriter
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
            if selected_precision and mapper.metadata()['sha256']!=options['_collection_mappings']['leader']['sha256']:
                raise ValueError('Leader mapping changed before Source activation')
            leader.activate()
            writer = EpisodeWriter(output / 'dataset', env, options['instruction'])
            from ..real2sim.workspaces.demo import coordinate_contract
            from ..so101_dataset_contract import current_contract, MIMIC_SPACE
            contract = coordinate_contract(request['workspace']) if request.get('workspace') else current_contract(MIMIC_SPACE)
            writer.provenance['coordinate_contract'] = collection_coordinates(contract, mapper if selected_precision else None,env.backend.metadata.get('simulation_configuration'))
            if options.get('resume_data'):
                previous=_provenance(options['resume_data']).get('coordinate_contract',{})
                if previous and 'simulation_model' not in previous:
                    # Keep a verified historical identity/legacy-mode contract unchanged.
                    writer.provenance['coordinate_contract'].pop('simulation_model',None)
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
                log_say(f'Episode {len(writer.manifest["episodes"])+1}',blocking=False)
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
                    name = writer.save(env.backend.is_success(),success_source='simulation_place_condition')
                    print('Saved ' + name, flush=True)
                    log_say(f'save episode {len(writer.manifest["episodes"])}',blocking=False)
                if quit_recording:
                    break
                if len(writer.manifest['episodes']) < count:log_say('resetting',blocking=False)
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
            if 'writer' in locals():writer.close()
            leader.close()
            log_say('stop recording',blocking=False)


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
    from .tasks import recorded_scene
    inherited = recorded_scene({'provenance':provenance,'scene_configuration':entry.get('scene_configuration')},load(request['workspace']))
    request = {**request,'options':{**request['options'],'active_object_ids':inherited['active_object_ids']}}
    request = {**request, 'options': {**request['options'], 'task_definition': entry.get('task_definition') or inherited.get('task_definition') or scene.get('task_definition'),
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
    if not manifest.get('storage'):
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
    from ..learning.minivla.data import lerobot_storages
    if any(fingerprint(st['root'])!=st['sha256'] for st in lerobot_storages(manifest)):
        raise ValueError('LeRobot dataset changed during the run')
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


def _lerobot_index(artifact,options,prefix=''):
    from ..so101_joint_mapping import RobotCoordinates
    from .artifacts import fingerprint
    root=Path(artifact['path'])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((root / 'meta/info.json').read_text())
    ds = LeRobotDataset(options.get('dataset_repo_id', info.get('repo_id', 'local/so101')), root=root)
    provenance = artifact.get('provenance', {})
    coordinate_contract=artifact['coordinates']
    from ..representation import describe
    from ..so101_dataset_contract import JOINT_SPACE,read_contract
    describe(coordinate_contract)
    if coordinate_contract!=read_contract(root) or coordinate_contract['action_space']!=JOINT_SPACE:
        raise ValueError('LeRobot training requires unchanged model-radian coordinates')
    endpoints=provenance.get('canonical_coordinates') or {}
    coords = RobotCoordinates(endpoints.get('gripper_closed_rad'),endpoints.get('gripper_open_rad'))
    manifest = {'schema_version': 1, 'joint_names': list(coords.metadata()['joint_order']),
        'units': ['rad'] * 5 + ['fraction'], 'control_hz': info['fps'],
        'action_semantics': 'absolute_target_after_observation', 'episodes': [],
        'cameras':[k.removeprefix('observation.images.').removesuffix('_cam') for k in info['features'] if k.startswith('observation.images.')]}
    source_identity=fingerprint(root)
    manifest['storage']={'format':'lerobot','root':str(root.resolve()),'repo_id':options.get('dataset_repo_id',info.get('repo_id','local/so101')),
        'sha256':source_identity,'coordinate_contract':coordinate_contract,'canonical_coordinates':coords.metadata(),
        'action_source':coordinate_contract.get('action_source') or provenance.get('action_source'),
        'camera_keys':{k.removeprefix('observation.images.').removesuffix('_cam'):k for k in info['features'] if k.startswith('observation.images.')},
        'decode':'torchcodec_cpu_exact','num_workers':0,'decoder_cache_limit':2,'rgb_cache_frames':8,'rgb_cache_bytes':16*2**20}
    # Converted generated variants need explicit source groups, never random frame splitting.
    source_groups = options.get('source_groups') or provenance.get('source_groups', {})
    source_tasks = {str(e['episode_index']):e.get('task_definition') for e in provenance.get('episodes',[])}
    partition=provenance.get('source_partition')
    if partition:
        from .episodes import validate_partition
        validate_partition(partition)
    if not partition and (not source_groups or any(str(g).startswith('UNKNOWN:') for g in source_groups.values())):
        raise ValueError('MiniVLA Direct split requires verified source_groups; unknown generated lineage cannot be split independently')
    for meta in ds.meta.episodes:
        index=int(meta['episode_index']);start,end=int(meta['dataset_from_index']),int(meta['dataset_to_index'])
        if source_groups and str(index) not in source_groups:
            raise ValueError('Source group mapping must cover every episode')
        row=ds.hf_dataset[start];instruction=str(ds.meta.tasks.iloc[int(row['task_index'])].name)
        name = prefix+f'episode_{index:04d}'
        review=provenance.get('episode_review',{}).get(str(index),{})
        if review.get('validation_only') or provenance.get('source_env_args',{}).get('validation_only'):
            raise ValueError('Synthetic smoke data cannot be trained')
        scene_configuration=provenance.get('episode_scene_configurations',{}).get(str(index),provenance.get('scene_configuration'))
        manifest['episodes'].append({'id': name,'dataset_episode_index':index,'from_index':start,'to_index':end,'frames':end-start,
            'split': 'train', 'instruction': instruction,
            'group_id': source_groups.get(str(index),'UNKNOWN'),
            'task_definition':source_tasks.get(str(index)) or (scene_configuration or {}).get('task_definition'),
            'scene_configuration':scene_configuration,'recorded_review':review,
            **({'source_partition':partition,'source_lineage':provenance.get('episode_source_lineage',{}).get(str(index)),
                'source_partition_input':provenance.get('source_partition_input')} if partition else {})})
    return manifest,coords


def train_from_lerobot(artifact, options, output):
    """Publish small index/statistics only; Episodes reads original Parquet/MP4 lazily."""
    from .episodes import split_episodes
    from ..learning.minivla.data import fit_statistics
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
        validation=options.get('validation_artifact');saved=previous.get('validation_source')
        if validation and (not saved or Path(validation['path']).resolve()!=Path(saved['path']).resolve() or fingerprint(validation['path'])!=saved['sha256']):
            raise ValueError('Resume Validation dataset differs from the original Source Partition')
        request = {'input': str(prepared), 'workspace': options['workspace'],
                   'config': str(model / 'config.json'), 'options': {**options, 'resume_output': str(model)}}
        result = train(request, output)
        atomic(model / 'project_provenance.json', {**_provenance(model), 'lerobot_source_dataset': str(root.resolve())})
        return result
    manifest,coords=_lerobot_index(artifact,options)
    provenance=artifact.get('provenance',{});source_identity=fingerprint(root)
    validation=options.get('validation_artifact')
    if validation:
        other,_=_lerobot_index(validation,options,prefix='validation_')
        from ..representation import require_compatible
        from ..so101_dataset_contract import JOINT_SPACE
        require_compatible(other['storage']['coordinate_contract'],manifest['storage']['coordinate_contract'],JOINT_SPACE)
        def calibration(c):
            model=c.get('simulation_model')
            return {k:c.get(k) for k in ('hardware_mapping','simulation_joint_frame')}, ({k:model.get(k) for k in ('workspace','robot_urdf_sha256','precision_mappings','simulation_residual','model_mode','robot_model')} if model else None)
        if (artifact.get('workspace')!=validation.get('workspace')
                or calibration(other['storage']['coordinate_contract'])!=calibration(manifest['storage']['coordinate_contract'])
                or other['control_hz']!=manifest['control_hz'] or other['cameras']!=manifest['cameras']
                or any(other['storage']['canonical_coordinates'][k]!=manifest['storage']['canonical_coordinates'][k] for k in ('gripper_closed_rad','gripper_open_rad'))):
            raise ValueError('Train/Validation Workspace, calibration, camera, frequency or gripper contract differs')
        if (provenance.get('source_partition') or {}).get('split')!='train' or (validation.get('provenance',{}).get('source_partition') or {}).get('split')!='val':
            raise ValueError('Separate validation artifact requires verified train and val Source Partitions')
        storages=[manifest['storage'],other['storage']]
        for e in other['episodes']:e['source_index']=1
        manifest['episodes']+=other['episodes']
        from ..real2sim.precision import fingerprint as contract_hash
        manifest['storage']={'format':'lerobot','sources':storages,'sha256':contract_hash(storages)}
    manifest['episodes'],_=split_episodes(manifest['episodes'])
    counts={s:sum(e['split']==s for e in manifest['episodes']) for s in ('train','val')}
    if any(e.get('source_partition') for e in manifest['episodes']):
        pools={s:{key for e in manifest['episodes'] if e['split']==s for key in e['source_partition']['source_ids']} for s in ('train','val')}
        print(f'[WORKFLOW] Source lineage verified · Train Source {len(pools["train"])} / Validation Source {len(pools["val"])} · episodes {counts} · leakage NONE',flush=True)
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / 'dataset';prepared.mkdir()
    atomic(prepared / 'manifest.json', manifest)
    atomic(prepared / 'project_provenance.json', {'schema': 'so101.minivla-dataset/1',
        'workspace': artifact['workspace'], 'backend': artifact.get('backend', 'isaac'),
        'coordinates': coords.metadata(), 'task_definitions': artifact.get('tasks', []),
        'coordinate_contract': artifact['coordinates'],
        'source_dataset': str(root.resolve()), 'source_sha256': source_identity,'loader':'lerobot_direct',
        'source_groups':{e['id']:e['group_id'] for e in manifest['episodes']},
        'source_partitions':{e['source_partition']['sha256']:e['source_partition'] for e in manifest['episodes'] if e.get('source_partition')},
        'validation_source':{'path':validation['path'],'sha256':fingerprint(validation['path'])} if validation else None,
        'scene_configuration':provenance.get('scene_configuration'),
        'scene_configurations':provenance.get('scene_configurations',[]),
        'episode_scene_configurations':provenance.get('episode_scene_configurations',{}),
        'episode_review':provenance.get('episode_review',{}), 'outcomes': 'unknown; preserving selected training data'})
    fit_statistics(prepared/'manifest.json',prepared/'statistics.json',manifest['cameras'])
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
    from .tasks import object_metrics,evaluation_distribution
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
                result = env.step(action)
                trace.append({'timestamp_s': result['timestamp_s'], 'model_joint_rad': result['joint_pos'].tolist(),
                              'canonical_action': action.tolist(), 'success': result['success'], 'policy_metrics': metrics, 'replan':replanning})
                signals = subtask_signals(env, tracker, step)
                measured = object_metrics(active,pick,{n:env.backend.get_object_pose(n)[:3,3] for n in active},
                    env.backend.get_eef_pose()[:3,3],metric_state,signals['grasp'],signals['place'],initial_positions)
                trace[-1]['object_metrics']=measured
                publish(observed['images'][0], {'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                        'steps': step + 1, 'task_id': options.get('task_definition', {}).get('task_id', ''),
                        'task_signals': signals, 'success': result['success']})
                if result['success']:
                    break
            row = {'episode': episode, 'steps': len(trace), 'success': bool(trace[-1]['success']),
                   'limit_violations': guard.violations, 'rate_limited': guard.rate_limited,**measured,
                   'inference_latency_s':[r['policy_metrics'].get('latency_s') for r in trace if r.get('replan')]}
            rows.append(row)
            if bridge:
                bridge.details.update(episode_outcome='성공' if row['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if row['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': row, 'trace': trace})
        return {'passed': True, 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows),
                'scope': 'synchronous simulated rollout; passed means execution validation, not task success',
                'coordinates': env.coordinates.metadata(), 'scene': env.backend.metadata,
                'scene_configuration':env.backend.metadata.get('scene_configuration'),
                **evaluation_distribution({'provenance':provenance},env.backend.workspace,env.backend.workspace['task']['dynamic'])}


def rollout_lerobot(request, output, publish, bridge):
    import torch
    from .policy import load_policy, observation
    from ..real2sim.workspaces.demo import coordinate_contract
    from ..learning.minivla.policy import ActionGuard
    from .tasks import object_metrics,evaluation_distribution
    options = request['options']; w = load(request['workspace'])
    device = options.get('device', 'cuda:0')
    policy, pre, post = load_policy(request['checkpoint'], coordinate_contract(w), device)
    rows = []
    with _environment(request, output) as env:
        emit = _preview(env, request, publish)
        guard = ActionGuard(env.backend.limits[:, 0], env.backend.limits[:, 1], [2.] * 6, env.hz)
        for episode in range(int(options.get('episodes', 1))):
            env.reset(options.get('seed', 0) + episode); policy.reset(); guard.reset(); tracker = grasp_tracker(env)
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
                env.backend.step(action)
                signals = subtask_signals(env, tracker, i)
                measured=object_metrics(active,pick,{n:env.backend.get_object_pose(n)[:3,3] for n in active},
                    env.backend.get_eef_pose()[:3,3],metric_state,signals['grasp'],signals['place'],initial_positions)
                trace.append({'action': action.tolist(), 'success': signals['place'],'object_metrics':measured})
                emit(side, {'steps': i + 1, 'episode_index': episode + 1, 'episode_total': int(options.get('episodes', 1)),
                    'task_id': options.get('task_definition', {}).get('task_id', ''), 'task_signals': signals})
                if signals['place']: break
            rows.append({'episode': episode, 'steps': len(trace), 'success': trace[-1]['success'],**measured,
                         'inference_latency_s':inference_latencies or None,'action_call_latency_s':latencies})
            if bridge:
                bridge.details.update(episode_outcome='성공' if rows[-1]['success'] else '실패',
                    outcome_reason='컵 안 정지 조건 충족' if rows[-1]['success'] else '최대 step 도달 · 컵 안 정지 조건 미충족')
                bridge.status()
            atomic(output / f'episode_{episode:04d}.json', {'summary': rows[-1], 'trace': trace})
    return {'passed': True, 'episodes': rows, 'success_rate': sum(r['success'] for r in rows) / len(rows),
            'scene_configuration':env.backend.metadata.get('scene_configuration'),
            **evaluation_distribution({'path':str(request['checkpoint'])},w,active)}


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
