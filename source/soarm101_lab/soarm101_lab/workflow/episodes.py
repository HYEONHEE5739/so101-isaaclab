"""Canonical MiniVLA episodes plus lossless model-radian/replay telemetry."""
import copy
import json
from pathlib import Path
import shutil
import uuid
import time

import numpy as np

from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata
from ..so101_joint_mapping import JOINTS, POLICY_UNITS


class EpisodeWriter:
    def __init__(self, root, env, instruction):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.env = env
        self.rows = []
        self.initial = None
        self.manifest = {'schema_version': 1, 'joint_names': list(JOINTS),
                         'units': list(POLICY_UNITS), 'control_hz': env.hz,
                         'action_semantics': 'absolute_target_after_observation', 'episodes': [],
                         'cameras': env.backend.workspace['task'].get('observation_cameras', ['side'])}
        self.provenance = {'schema': 'so101.minivla-dataset/1', 'backend': 'mujoco',
                           'workspace': env.backend.metadata['source'],
                           'scene': copy.deepcopy(env.backend.metadata), 'coordinates': env.coordinates.metadata(),
                           'instruction': instruction, 'source_group_prefix': uuid.uuid4().hex}
        from .tasks import selected
        if not self.provenance['scene'].get('task_definition'):
            self.provenance['scene']['task_definition'] = selected(env.backend.workspace)
        self.instruction = instruction
        self.flush_manifest()

    def resume(self, source):
        source = Path(source).resolve()
        manifest = json.loads((source / 'manifest.json').read_text())
        provenance = json.loads((source / 'project_provenance.json').read_text())
        if (provenance['workspace'] != self.provenance['workspace'] or
                provenance['scene']['model_sha256'] != self.provenance['scene']['model_sha256'] or
                manifest['units'] != self.manifest['units'] or manifest['control_hz'] != self.env.hz or
                any(provenance['coordinates'].get(k) != self.provenance['coordinates'].get(k)
                    for k in ('joint_order', 'gripper_closed_rad', 'gripper_open_rad')) or
                provenance['scene'].get('task_definition') != self.provenance['scene'].get('task_definition')):
            raise ValueError('Resume workspace/model/task/coordinates differ')
        previous_contract = provenance.get('coordinate_contract') or {}
        current_contract = self.provenance.get('coordinate_contract') or {}
        if any(previous_contract.get(key) != current_contract.get(key)
               for key in ('hardware_mapping', 'simulation_joint_frame')):
            raise ValueError('Resume hardware calibration/collection coordinates differ; use a new dataset')
        for entry in manifest['episodes']:
            folder = (source / entry['path']).resolve()
            if not folder.is_relative_to(source): raise ValueError('Episode outside dataset')
            name = f'episode_{len(self.manifest["episodes"]):04d}'
            shutil.copytree(folder, self.root / name)
            self.manifest['episodes'].append({**entry, 'id': name, 'path': name})
        self.provenance['resumed_from'] = str(source)
        self.flush_manifest()

    def begin(self):
        self.rows = []
        self.initial = self.env.backend.get_initial_state()

    def append(self, observation, model_target, requested_target=None):
        # obs comes before applying this target. Both timestamps are simulation time at that boundary.
        self.rows.append({'state': observation['state'].copy(),
                          'action': self.env.coordinates.to_policy(model_target),
                          'raw_state': observation['model_joint_rad'].copy(),
                          'raw_action': np.array(model_target, copy=True),
                          'requested_action': np.array(model_target if requested_target is None else requested_target, copy=True),
                          'camera_images': {n:a.copy() for n,a in observation.get('camera_images', {'side':observation['images'][0]}).items()},
                          'timestamp': observation['timestamp'],
                          'eef_pose': observation['eef_pose'].copy(),
                          'object_pose': observation['object_poses'][self.env.backend.metadata['pick']].copy()})

    def save(self, success, validation_only=False):
        if not self.rows:
            raise ValueError('Cannot save an empty episode')
        name = f'episode_{len(self.manifest["episodes"]):04d}'
        folder = self.root / name
        folder.mkdir()
        for key in ('state', 'action'):
            np.save(folder / f'{key}.npy', np.asarray([r[key] for r in self.rows], dtype=np.float32))
        for key in ('raw_state', 'raw_action', 'requested_action', 'eef_pose', 'object_pose'):
            np.save(folder / f'{key}.npy', np.asarray([r[key] for r in self.rows], dtype=np.float64))
        times = np.asarray([r['timestamp'] for r in self.rows], dtype=np.float64)
        for camera in self.manifest['cameras']:
            np.save(folder / (camera+'.npy'), np.asarray([r['camera_images'][camera] for r in self.rows], dtype=np.uint8))
            np.save(folder / (camera+'_timestamp.npy'), times)
        for key in ('timestamp', 'action_timestamp'):
            np.save(folder / f'{key}.npy', times)
        atomic(folder / 'replay.json', self.initial)
        atomic(folder / 'outcome.json', {'success': bool(success), 'validation_only': validation_only,
                                       'coordinates': self.env.coordinates.metadata()})
        entry = {'id': name, 'path': name, 'split': 'train', 'instruction': self.instruction,
                 'group_id': self.provenance['source_group_prefix'] + '/' + name}
        self.manifest['episodes'].append(entry)
        self.flush_manifest()
        self.begin()
        return name

    def flush_manifest(self):
        atomic(self.root / 'manifest.json', self.manifest)
        self.provenance['coordinates'] = self.env.coordinates.metadata()
        atomic(self.root / 'project_provenance.json', self.provenance)


def prepare_dataset(source, output, splits=None, include_failed=False, require_validation=True):
    from ..learning.minivla.data import load_manifest, episode_arrays, fit_statistics
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new prepared dataset output')
    manifest = load_manifest(source / 'manifest.json')
    provenance = json.loads((source / 'project_provenance.json').read_text())
    eligible = []
    for entry in manifest['episodes']:
        folder = Path(entry['_path'])
        outcome = json.loads((folder / 'outcome.json').read_text())
        if outcome.get('validation_only'):
            raise ValueError('Synthetic smoke episodes cannot enter training')
        if outcome['success'] or include_failed:
            episode_arrays(entry, manifest.get('cameras',['side']), manifest['control_hz'])
            eligible.append(entry)
    if not eligible or (require_validation and len(eligible) < 2):
        raise ValueError('At least two eligible independent source groups for train/val required')
    groups = sorted({e['group_id'] for e in eligible})
    if splits is None:
        if require_validation and len(groups) < 2:
            raise ValueError('Need independent train/val source groups')
        # Deterministic group split. No frame split or generated-variant leakage.
        assignment = {g: 'train' for g in groups}
        if len(groups) >= 2: assignment[groups[-1]] = 'val'
        if len(groups) >= 3:
            assignment[groups[-2]] = 'test'
        splits = {e['id']: assignment[e['group_id']] for e in eligible}
    if set(splits) != {e['id'] for e in eligible}:
        raise ValueError('Split map must cover exactly the eligible episode IDs')
    entries = [{**{k: v for k, v in e.items() if k != '_path'}, 'split': splits[e['id']]} for e in eligible]
    grouped = {}
    for entry in entries:
        if entry['split'] not in ('train', 'val', 'test'):
            raise ValueError('Prepared training split must be train/val/test')
        if entry['group_id'] in grouped and grouped[entry['group_id']] != entry['split']:
            raise ValueError('Source group leakage')
        grouped[entry['group_id']] = entry['split']
    if require_validation and not {'train', 'val'}.issubset(grouped.values()):
        raise ValueError('Training and validation groups both required')
    output.mkdir(parents=True)
    for original, entry in zip(eligible, entries):
        shutil.copytree(original['_path'], output / entry['path'])
    cleaned = {**manifest, 'episodes': entries}
    atomic(output / 'manifest.json', cleaned)
    atomic(output / 'project_provenance.json', {**provenance, 'prepared_from': str(source.resolve()),
                                               'include_failed': include_failed})
    fit_statistics(output / 'manifest.json', output / 'statistics.json', manifest.get('cameras',['side']))
    return {'dataset': str(output), 'episodes': len(entries), 'groups': grouped}


def annotate_episode(dataset, episode, boundaries):
    """Video/state index annotation only; never claims to produce MimicGen DatagenInfo."""
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text())
    if episode not in {e['id'] for e in manifest['episodes']} or Path(episode).name != episode:
        raise ValueError('Unknown episode ID')
    path = Path(dataset) / episode
    length = len(np.load(path / 'state.npy', mmap_mode='r', allow_pickle=False))
    expected = ('approach', 'grasp', 'transport', 'place')
    if list(boundaries) != list(expected):
        raise ValueError('Boundary order must be approach, grasp, transport, place')
    values = list(boundaries.values())
    if any(type(v) is not int or not 0 <= v < length for v in values) or values != sorted(set(values)):
        raise ValueError('Strictly increasing integer frame boundaries inside episode required')
    destination = path / 'annotation.json'
    if destination.exists():
        raise FileExistsError('Annotation exists; keep originals and use a dataset copy for revisions')
    atomic(destination, {'schema': 'so101.subtask-boundaries/1', 'frame_count': length,
                         'boundaries': copy.deepcopy(boundaries), 'scope': 'manual dataset-only boundaries'})
    return {'annotation': str(destination)}


def convert_dataset(source, output, options):
    """Preserve canonical model input and reuse the existing LeRobot converter."""
    import subprocess
    import sys
    import h5py
    from ..real2sim.workspaces.package import ROOT, load
    from ..real2sim.workspaces.demo import coordinate_contract
    from ..so101_dataset_contract import write_contract, JOINT_SPACE
    result = prepare_dataset(source, output / 'dataset', include_failed=options.get('include_failed', False), require_validation=False)
    dataset = Path(result['dataset'])
    provenance = json.loads((dataset / 'project_provenance.json').read_text())
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['control_hz'] != round(manifest['control_hz']):
        raise ValueError('LeRobot conversion requires an integer camera/control FPS')
    contract = provenance.get('coordinate_contract', coordinate_contract(load(options['workspace'])))
    contract = {**contract, 'action_space': JOINT_SPACE}
    intermediate = output / 'conversion_input.hdf5'
    definition = provenance['scene']['task_definition']
    with h5py.File(intermediate, 'w') as h:
        data = h.create_group('data')
        data.attrs['env_args'] = json.dumps({'workspace': provenance['workspace'], 'task_definitions': [definition]})
        for i, entry in enumerate(manifest['episodes']):
            folder = dataset / entry['path']; group = data.create_group(f'demo_{i}')
            group.attrs['task_definition'] = json.dumps(definition)
            actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
            group.create_dataset('actions', data=actions)
            group.create_dataset('joint_targets', data=actions)
            group.create_dataset('obs/policy/joint_pos', data=np.load(folder / 'raw_state.npy', allow_pickle=False))
            for view in manifest.get('cameras',['side','wrist']):
                path = folder / (view + '.npy')
                if path.exists(): group.create_dataset('obs/policy/' + view + '_cam', data=np.load(path, allow_pickle=False),
                                                       compression='gzip', compression_opts=1)
    write_contract(intermediate, contract)
    subprocess.run([sys.executable, str(ROOT / 'scripts/tools/convert_isaac2lerobot.py'),
        '--input_file', str(intermediate), '--root', str(dataset / 'lerobot'),
        '--repo_id', options['repo_id'], '--task', definition['language_instruction'],
        '--fps', str(round(manifest['control_hz'])), '--action_source', 'joint_targets'], check=True)
    converted_provenance = dataset / 'lerobot/workflow_provenance.json'
    atomic(converted_provenance, {**json.loads(converted_provenance.read_text()), 'workspace': provenance['workspace'],
        'task_definitions': [definition], 'backend': 'mujoco', 'source': str(Path(source).resolve()),
        'source_groups': {str(i): e['group_id'] for i, e in enumerate(manifest['episodes'])},
        'generation_method': provenance.get('generation_method')})
    # Conversion adapter is temporary, never a registered artifact.
    intermediate.unlink()
    from ..so101_dataset_contract import sidecar
    sidecar(intermediate).unlink(missing_ok=True)
    return {**result, 'lerobot': str(dataset / 'lerobot')}

def grasp_tracker(env):
    from ..real2sim.workspaces.grasp import StableGrasp
    return StableGrasp(1 / env.hz, env.backend.workspace['profile']['objects'][env.backend.metadata['pick']]['dimensions_m'])


def subtask_signals(env, tracker, tick):
    from scipy.spatial.transform import Rotation
    eef = env.backend.get_eef_pose()
    obj = env.backend.get_object_pose(env.backend.metadata['pick'])
    relative = np.linalg.inv(eef) @ obj
    left, right = env.backend.finger_contacts()
    grasp = tracker.update(tick, relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_quat(),
                           eef[:3, 3], obj[:3, 3], left, right)
    return {'grasp': bool(grasp), 'place': env.backend.is_success(),
            'tcp_distance_mm': float(np.linalg.norm(eef[:3, 3] - obj[:3, 3]) * 1000)}


def annotate(request, output, publish, bridge, environment_factory, preview):
    """Replay and annotate grasp/place, using the shared stable-grasp detector."""
    import shutil
    from ..learning.minivla.data import load_manifest
    source = Path(request['input'])
    provenance = json.loads((source / 'project_provenance.json').read_text())
    if request.get('workspace') and provenance['workspace'] != metadata(load(request['workspace'])):
        raise ValueError('Dataset workspace identity differs')
    manifest = load_manifest(source / 'manifest.json')
    destination = output / 'dataset'; destination.mkdir()
    accepted, rejected = [], []
    scene = provenance['scene']
    request = {**request, 'options': {**request['options'], 'task_definition': scene.get('task_definition'),
        'physics_profile': scene.get('physics_profile', 'legacy_urdf'),
        'full_workspace': scene.get('full_workspace', False),
        'width': scene['camera']['resolution'][0], 'height': scene['camera']['resolution'][1]}}
    with environment_factory(request, output, provenance['coordinates'], manifest['control_hz']) as env:
        emit = preview(env, request, publish)
        for episode_index, entry in enumerate(manifest['episodes']):
            folder = Path(entry['_path'])
            env.backend.restore_state(json.loads((folder / 'replay.json').read_text()))
            actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
            raw = np.load(folder / 'raw_state.npy', allow_pickle=False)
            tracker = grasp_tracker(env)
            marks, skipped, position = [], [], [0]
            if bridge:
                bridge.callbacks['mark'] = lambda: marks.append(position[0])
                bridge.callbacks['skip'] = lambda: skipped.append(True)
                bridge.details['marks'] = marks
            grasp_at = place_at = None
            for i, target in enumerate(actions):
                position[0] = i
                if bridge: bridge.before_step()
                if skipped: break
                if not np.allclose(env.backend.get_state()['joint_pos'], raw[i], atol=1e-5):
                    raise ValueError('Annotation replay diverged from recorded state')
                env.backend.step(target)
                signals = subtask_signals(env, tracker, i)
                if signals['grasp'] and grasp_at is None: grasp_at = i
                if signals['place'] and grasp_at is not None: place_at = i
                emit(env.backend.get_rgb(), {'steps': i + 1, 'episode_index': episode_index + 1,
                     'episode_total': len(manifest['episodes']), 'task_signals': signals, 'marks': list(marks)})
                if request['options'].get('realtime', True): time.sleep(1 / env.hz)
            if not request['options'].get('auto', True):
                grasp_at = marks[0] if len(marks) == 1 else None
                place_at = len(actions) - 1 if env.backend.is_success() else None
            if skipped or not env.backend.is_success() or grasp_at is None or place_at is None or not 0 <= grasp_at < place_at < len(actions):
                rejected.append({'episode': entry['id'], 'reason': 'ordered grasp/place boundaries or successful placement missing'})
                continue
            dest = destination / entry['path']
            shutil.copytree(folder, dest)
            atomic(dest / 'annotation.json', {'schema': 'so101.mujoco-subtasks/1', 'grasp': grasp_at,
                'place': place_at, 'frame_count': len(actions),
                'method': 'stable_contact_replay' if request['options'].get('auto', True) else 'manual_grasp_and_measured_place'})
            accepted.append({k: v for k, v in entry.items() if k != '_path'})
    atomic(destination / 'manifest.json', {**manifest, 'episodes': accepted})
    atomic(destination / 'project_provenance.json', {**provenance, 'annotated_from': str(source.resolve())})
    return {'passed': bool(accepted), 'dataset': str(destination), 'accepted': len(accepted), 'rejected': rejected}


def datagen(request, output, publish, bridge, environment_factory, preview):
    """Object-relative retargeting with bounded IK and physical success filtering."""
    from ..learning.minivla.data import load_manifest
    from ..real2sim.workspaces.reset import datagen_seed
    source = Path(request['input']); provenance = json.loads((source / 'project_provenance.json').read_text())
    if request.get('workspace') and provenance['workspace'] != metadata(load(request['workspace'])):
        raise ValueError('Dataset workspace identity differs')
    manifest = load_manifest(source / 'manifest.json')
    options = request['options']
    count = int(options.get('num_successful_demos', 10))
    seed = datagen_seed(load(request['workspace'])['task']['reset']['seed'], options.get('generation_seed'))
    rng = np.random.default_rng(seed)
    attempts = []
    with environment_factory(request, output, provenance['coordinates'], manifest['control_hz']) as env:
        emit = preview(env, request, publish)
        writer = EpisodeWriter(output / 'dataset', env, options['instruction'])
        if 'coordinate_contract' in provenance:
            writer.provenance['coordinate_contract'] = copy.deepcopy(provenance['coordinate_contract'])
        if options.get('resume_data'):
            resumed = json.loads((Path(options['resume_data']) / 'project_provenance.json').read_text())
            if resumed.get('source_dataset') != str(source.resolve()) or resumed.get('generation_method') != 'object_relative_bounded_ik':
                raise ValueError('Resume Datagen requires the same annotated source and generation method')
            writer.resume(options['resume_data'])
        if len(writer.manifest['episodes']) >= count:
            raise ValueError('Datagen target must exceed resumed episode count')
        writer.provenance.update(generation_method='object_relative_bounded_ik', source_dataset=str(source.resolve()), generation_seed=seed)
        stopped = False
        try:
            for attempt in range(int(options.get('max_attempts', max(100, count * 100)))):
                if len(writer.manifest['episodes']) >= count: break
                entry = manifest['episodes'][int(rng.integers(len(manifest['episodes'])))]
                folder = Path(entry['_path'])
                annotation = json.loads((folder / 'annotation.json').read_text())
                if annotation.get('schema') != 'so101.mujoco-subtasks/1':
                    raise ValueError('Run workflow Annotation before Datagen')
                actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
                original_object = np.load(folder / 'object_pose.npy', allow_pickle=False)[0]
                env.reset(int(rng.integers(2**31)))
                writer.begin(); tracker = grasp_tracker(env)
                moved_object = env.backend.get_object_pose(env.backend.metadata['pick'])
                transform_pick = moved_object @ np.linalg.inv(original_object)
                previous = env.backend.home.copy(); reason = ''
                try:
                    for i, raw_target in enumerate(actions):
                        if bridge: bridge.before_step()
                        goal = env.backend.target_pose(raw_target)
                        # Blend object-relative pick displacement back to the fixed destination.
                        blend = np.clip((i - annotation['grasp']) / max(1, annotation['place'] - annotation['grasp']), 0., 1.)
                        shifted = transform_pick @ goal
                        goal[:3, 3] = (1 - blend) * shifted[:3, 3] + blend * goal[:3, 3]
                        command = env.backend.solve_pose(goal, np.r_[previous[:5], raw_target[5]])
                        observation = env.observe()
                        writer.append(observation, command)
                        env.backend.step(command); previous = command
                        signals = subtask_signals(env, tracker, i)
                        emit(observation['images'][0], {'steps': i + 1, 'attempts': attempt + 1,
                             'successes': len(writer.manifest['episodes']), 'failures': attempt - len(writer.manifest['episodes']),
                             'task_signals': signals})
                    success = env.backend.is_success() and tracker.completed
                    if not success: reason = 'physical grasp/place validation failed'
                except ValueError as exc:
                    success = False; reason = str(exc)
                if success:
                    name = writer.save(True)
                    writer.manifest['episodes'][-1]['group_id'] = entry['group_id']
                    writer.flush_manifest()
                    print('[WORKFLOW] Generated ' + name, flush=True)
                attempts.append({'attempt': attempt + 1, 'source': entry['id'], 'success': bool(success), 'reason': reason})
        except KeyboardInterrupt:
            stopped = True
            print("[WORKFLOW] Datagen 중지: 완료된 episode를 보존합니다.", flush=True)
        atomic(output / 'attempts.json', attempts)
        return {'passed': len(writer.manifest['episodes']) == count or (stopped and bool(writer.manifest['episodes'])), 'dataset': str(writer.root),
                'successes': len(writer.manifest['episodes']), 'requested': count, 'attempts': len(attempts),
                'stopped': stopped, 'generation_seed': seed, 'method': 'object_relative_bounded_ik', 'is_isaac_mimicgen': False}


def convert_hdf5(recipe_file, output):
    import h5py
    from ..learning.minivla.data import JOINTS, POLICY_UNITS, UNITS, episode_arrays, load_manifest
    recipe_file=Path(recipe_file); r=json.loads(recipe_file.read_text()); root=Path(output)
    if root.exists(): raise FileExistsError('Use a new output directory')
    if r['action_semantics']!='absolute_target_after_observation': raise ValueError('Re-align action timestamps before import')
    if r['raw_joint_names']!=JOINTS: raise ValueError('Reorder joints explicitly before import')
    if r['arm_units'] not in ('rad','deg'): raise ValueError('Arm units must be declared')
    low,high=r['raw_gripper_range']
    if not np.isfinite([low,high]).all() or low==high: raise ValueError('Invalid gripper calibration endpoints')
    root.mkdir(parents=True); entries=[]
    for spec in r['episodes']:
        name=spec['id']
        if not name or Path(name).name!=name or name in ('.','..'): raise ValueError('Invalid episode id')
        dest=root/name; dest.mkdir()
        with h5py.File(recipe_file.parent/spec['file'],'r') as h:
            for key in ('state','action'):
                a=np.asarray(h[spec[key+'_key']],dtype=np.float32)
                if a.ndim!=2 or a.shape[1]!=6: raise ValueError('Expected [T,6] state/action')
                if r['arm_units']=='deg': a[:,:5]=np.deg2rad(a[:,:5])
                a[:,-1]=(a[:,-1]-low)/(high-low)
                np.save(dest/f'{key}.npy',a)
            for key in ('timestamp','action_timestamp'):
                np.save(dest/f'{key}.npy',np.asarray(h[spec[key+'_key']],dtype=np.float64))
            for camera,mapping in spec['cameras'].items():
                images=np.asarray(h[mapping['image_key']])
                if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3:
                    raise ValueError('Decode frames into RGB uint8 [T,H,W,3] before import')
                if r['image_order']=='BGR': images=images[...,::-1].copy()
                elif r['image_order']!='RGB': raise ValueError('image_order must be RGB or BGR')
                np.save(dest/f'{camera}.npy',images)
                np.save(dest/f'{camera}_timestamp.npy',np.asarray(h[mapping['timestamp_key']],dtype=np.float64))
        e={k:spec[k] for k in ('id','split','group_id','instruction')}; e['path']=name; entries.append(e)
    m=dict(schema_version=1,joint_names=JOINTS,units=UNITS,control_hz=r['control_hz'],
           action_semantics=r['action_semantics'],episodes=entries)
    path=root/'manifest.json'; path.write_text(json.dumps(m,indent=2))
    loaded=load_manifest(path)
    for e,spec in zip(loaded['episodes'],r['episodes']): episode_arrays(e,list(spec['cameras']),r['control_hz'])
    return path
