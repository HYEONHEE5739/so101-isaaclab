"""Typed, validated local artifacts; immutable entries and explicit parent identities."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import os
import shutil
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from ..so101_dataset_contract import read_contract
from ..representation import describe

TYPES = {'workspace', 'source', 'replay', 'annotated', 'datagen', 'dataset', 'policy', 'sim_eval', 'real_eval', 'hf_dataset', 'hf_policy'}
DEFAULTS = {'profile': 'outputs/real2sim', 'workspace': 'outputs/real2sim/environments',
            'source': 'datasets', 'annotated': 'datasets/annotated', 'datagen': 'datasets/generated',
            'dataset': 'datasets/LerobotDataset', 'policy': 'policy', 'replay': 'outputs/workflow',
            'sim_eval': 'outputs/sim_eval', 'real_eval': 'outputs/real_eval'}


def fingerprint(path):
    p = Path(path).expanduser().resolve()
    files = [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file() and '.cache' not in f.parts)
    if p.is_file() and p.suffix in ('.hdf5','.h5'):
        import h5py
        from ..utils.async_video_writer import external_video_path
        with h5py.File(p,'r') as h:
            videos=[]
            if 'data' in h:
                for name,g in h['data'].items():
                    if name.startswith('demo_'):
                        videos.extend(external_video_path(p,spec) for spec in json.loads(g.attrs.get('external_videos','{}')).values())
        files += sorted(set(videos))
    if not files:
        raise ValueError('Empty artifact')
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(p) if p.is_dir() else f.name if f==p else f.relative_to(p.parent)).encode())
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def inspect_artifact(kind, path):
    if kind not in TYPES:
        raise ValueError('Unknown artifact type')
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir() and (p / 'project_provenance.json').is_file() and kind in ('source', 'annotated', 'datagen', 'dataset', 'policy'):
        return inspect_canonical(kind, p)
    result = {}
    coordinates = None
    if kind == 'workspace':
        w = load(p); result['workspace'] = metadata(w); coordinates = coordinate_contract(w)
    elif kind in ('source', 'annotated', 'datagen'):
        import h5py
        with h5py.File(p, 'r') as h:
            args = json.loads(h['data'].attrs.get('env_args', '{}'))
            episodes = [g for n, g in h['data'].items() if n.startswith('demo_')]
            if not episodes or any('actions' not in g or not len(g['actions']) for g in episodes):
                raise ValueError('No nonempty demonstration episodes')
            if kind in ('source', 'datagen'):
                for g in episodes:
                    state_key = 'obs/policy/joint_pos' if 'obs/policy/joint_pos' in g else 'obs/joint_pos'
                    if 'joint_targets' not in g or state_key not in g:
                        raise ValueError('Model joint observations/targets missing')
                    if g['joint_targets'].shape[-1] != 6 or g[state_key].shape[-1] != 6:
                        raise ValueError('Expected six model joints')
            if args.get('validation_only'):
                raise ValueError('Smoke data cannot enter the training workflow')
            if any(g.attrs.get('validation_only',False) for g in episodes):raise ValueError('Smoke episodes cannot enter the training workflow')
            if kind == 'annotated' and any('obs/datagen_info/subtask_term_signals/grasp' not in g for g in episodes):
                raise ValueError('Missing Mimic annotation')
            from .tasks import episode_definition
            definitions = [episode_definition(g, args) for g in episodes]
            unique = {json.dumps(d, sort_keys=True): d for d in definitions if d is not None}
            if unique and any(d is None for d in definitions):
                raise ValueError('Some episodes have no task definition')
            result.update(workspace=args.get('workspace'), episode_count=len(episodes), env_args=args, tasks=list(unique.values()))
            if args.get('dataset_schema'):
                from .dataset import SCHEMA
                if args['dataset_schema']!=SCHEMA:raise ValueError('Unknown episode dataset schema')
                result.update(format=SCHEMA,backend=args.get('backend'),scene_configuration=args.get('scene_configuration'))
            if args.get('source_partition'):
                from .episodes import validate_partition,validate_partition_episode
                partition=validate_partition(args['source_partition'])
                for g in episodes:validate_partition_episode(partition,json.loads(g.attrs.get('source_lineage','null')),args.get('source_partition_input'))
                result.update(source_partition=partition,lineage_status='verified_partition',source_count=len(partition['source_ids']))
        coordinates = read_contract(p)
        if result.get('format')=='so101.common-episode/2':
            from .dataset import EpisodeDataset,validate_dataset
            validate_dataset(EpisodeDataset(p),require_annotations=kind in ('annotated','datagen'))
    elif kind in ('dataset', 'policy'):
        marker = p / ('meta/info.json' if kind == 'dataset' else 'config.json')
        result['format'] = json.loads(marker.read_text())
        if kind == 'policy':
            if not (p / 'model.safetensors').exists():
                raise ValueError('Policy weights missing')
            result['processors'] = {n: json.loads((p / (n + '.json')).read_text())
                                    for n in ('policy_preprocessor', 'policy_postprocessor')}
        if kind == 'dataset' and not list((p / 'data').rglob('*.parquet')):
            raise ValueError('Dataset parquet data missing')
        provenance = p / 'workflow_provenance.json'
        if provenance.exists():
            result['provenance'] = json.loads(provenance.read_text())
            result['workspace'] = result['provenance'].get('workspace')
            result['tasks'] = result['provenance'].get('task_definitions', [])
            result['backend'] = result['provenance'].get('backend', 'isaac')
        coordinates = read_contract(p)
    else:
        result['report'] = json.loads(p.read_text())
        if result['report'].get('task_definitions'):
            result['tasks'] = result['report']['task_definitions']
        if kind == 'replay' and not result['report'].get('passed'):
            raise ValueError('Replay verification failed')
        if kind.startswith('hf_') and not result['report'].get('commit'):
            raise ValueError('Remote publication not verified')
    if coordinates:
        result['coordinates'] = coordinates
        result['representation'] = describe(coordinates)
    return result


def inspect_canonical(kind, path):
    """Inspect the existing canonical format without importing ML or simulation."""
    import numpy as np
    p = Path(path)
    provenance = json.loads((p / 'project_provenance.json').read_text())
    definition = provenance.get('scene', {}).get('task_definition')
    result = {'format': 'so101.canonical/1', 'backend': provenance.get('backend'),
              'workspace': provenance['workspace'], 'provenance': provenance,
              'tasks': provenance.get('task_definitions') or ([definition] if definition else []),
              'scene_configuration':provenance.get('scene_configuration') or provenance.get('scene',{}).get('scene_configuration'),
              'coordinates': provenance['coordinates'],
              'representation': {'action_space': 'arm_rad_gripper_fraction', 'backend': provenance.get('backend')}}
    if kind == 'policy':
        if not (p / 'best.pt').is_file() or not (p / 'config.json').is_file():
            raise ValueError('MiniVLA checkpoint weights/config missing')
        result['format'] = 'so101.minivla-policy/1'
        result['model_config'] = json.loads((p / 'config.json').read_text())
        result['coordinates'] = read_contract(p)
        result['representation'] = describe(result['coordinates'])
        return result
    manifest = json.loads((p / 'manifest.json').read_text())
    episodes = manifest.get('episodes', [])
    if not episodes:
        raise ValueError('No nonempty demonstration episodes')
    definitions=[e.get('task_definition') or definition for e in episodes]
    if any(definitions):result['tasks']=list({t['task_id']:t for t in definitions if t}.values())
    scenes=[e.get('scene_configuration') or result['scene_configuration'] for e in episodes]
    if any(scenes):
        result['provenance']={**provenance,'scene_configurations':list({s['sha256']:s for s in scenes if s}.values())}
    for entry in episodes:
        folder = (p / entry['path']).resolve()
        if not folder.is_relative_to(p.resolve()):
            raise ValueError('Episode outside dataset')
        state = np.load(folder / 'state.npy', mmap_mode='r', allow_pickle=False)
        action = np.load(folder / 'action.npy', mmap_mode='r', allow_pickle=False)
        if state.ndim != 2 or state.shape != action.shape or state.shape[1] != 6 or not len(state):
            raise ValueError('Invalid six-dimensional state/action episode')
        if json.loads((folder / 'outcome.json').read_text()).get('validation_only'):
            raise ValueError('Smoke data cannot enter the training workflow')
        if kind == 'annotated':
            annotation = json.loads((folder / 'annotation.json').read_text())
            grasp, place = annotation.get('grasp'), annotation.get('place')
            if (annotation.get('schema') != 'so101.mujoco-subtasks/1' or
                    annotation.get('frame_count') != len(state) or
                    type(grasp) is not int or type(place) is not int or
                    not 0 <= grasp < place < len(state)):
                raise ValueError('Invalid ordered grasp/place annotation')
    if kind == 'dataset' and not (p / 'statistics.json').is_file():
        raise ValueError('Convert the demonstrations before training')
    result['episode_count'] = len(episodes)
    return result


class Registry:
    def __init__(self, root=None):
        self.root = Path(root or ROOT / 'outputs/workflow')
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, kind, path, parents=(), extra=None):
        details = inspect_artifact(kind, path)
        lineage = [self.get(i) for i in parents]
        workspaces = [a.get('workspace') for a in lineage if a.get('workspace')]
        if details.get('workspace'):
            workspaces.append(details['workspace'])
        if workspaces and any(w != workspaces[0] for w in workspaces):
            raise ValueError('Cross-workspace artifact dependency')
        resolved = str(Path(path).resolve())
        content_hash = fingerprint(path)
        if not extra:
            for existing in reversed(self.all()):
                if (existing['type'] == kind and existing['path'] == resolved and
                        existing.get('sha256') == content_hash and existing.get('status') == 'SUCCEEDED'
                        and (not parents or existing.get('parents') == list(parents))):
                    return existing
        artifact = {**details, 'id': uuid.uuid4().hex, 'type': kind, 'path': resolved,
                    'created_at': time.time(), 'status': 'SUCCEEDED', 'parents': list(parents),
                    'sha256': content_hash, 'execution': extra or {}}
        if workspaces:
            artifact['workspace'] = workspaces[0]
        for key in ('coordinates', 'representation', 'tasks', 'backend'):
            if key not in artifact and lineage and lineage[0].get(key):
                artifact[key] = lineage[0][key]
        atomic(self.root / 'artifacts' / (artifact['id'] + '.json'), artifact)
        if kind == 'policy':
            for old in self.all():
                if (old['id'] != artifact['id'] and old['path'] == resolved and
                        old.get('status') == 'SUCCEEDED' and old.get('sha256') != content_hash):
                    atomic(self.root / 'artifacts' / (old['id'] + '.json'),
                           {**old, 'status': 'SUPERSEDED', 'superseded_by': artifact['id'],
                            'superseded_reason': 'policy updated by resumed training'})
        return artifact

    def get(self, identifier):
        return json.loads((self.root / 'artifacts' / (identifier + '.json')).read_text())

    def all(self):
        return sorted((json.loads(p.read_text()) for p in (self.root / 'artifacts').glob('*.json')), key=lambda a: a['created_at'])

    def summary(self):
        """Explain records and dependencies without hashing or changing stored data."""
        from collections import Counter
        artifacts = self.all()
        children = {a['id']: [] for a in artifacts}
        for a in artifacts:
            for parent in a.get('parents', []):
                if parent in children: children[parent].append(a['id'])
        records = [{'id': a['id'], 'type': a['type'], 'backend': a.get('backend', 'isaac'),
                    'tasks': [t['task_id'] for t in a.get('tasks', [])], 'path': a['path'],
                    'status': a.get('status'), 'exists': Path(a['path']).exists(),
                    'parents': a.get('parents', []), 'used_by': children[a['id']]} for a in artifacts]
        return {'registry': str(self.root), 'count': len(records),
                'by_type': dict(Counter(a['type'] for a in artifacts)),
                'missing_paths': sum(not a['exists'] for a in records),
                'records': records}

    def latest(self, kinds, workspace=None):
        for a in reversed(self.all()):
            if a.get('status') == 'SUCCEEDED' and a['type'] in kinds and (workspace is None or a.get('workspace') == workspace) and Path(a['path']).exists():
                return a
        return None

    def verify(self, artifact):
        if artifact.get('status') == 'SUPERSEDED':
            raise ValueError('Source was replaced by its validated annotation; select the annotated artifact')
        if fingerprint(artifact['path']) != artifact['sha256']:
            raise ValueError('Artifact changed since registration; import it again explicitly')
        current = inspect_artifact(artifact['type'], artifact['path'])
        for key in ('coordinates', 'workspace'):
            if key in current and current[key] != artifact.get(key):
                raise ValueError('Artifact metadata changed since registration')
        return current

    def promote_annotation(self, source, completed, extra=None):
        """Validate first, replace source transactionally, retain lineage as superseded records."""
        from ..so101_dataset_contract import sidecar
        current = self.verify(source)
        if current.get('source_partition') or current.get('provenance', {}).get('source_partition'):
            raise ValueError('Source Partition is immutable; register annotation as a separate artifact')
        completed, destination = Path(completed).resolve(), Path(source['path']).resolve()
        if completed == destination or source['type'] != 'source':
            raise ValueError('Annotation replacement requires a distinct completed source result')
        annotation = inspect_artifact('annotated', completed)
        if annotation['episode_count'] != source['episode_count']:
            raise ValueError('Partial annotation cannot replace the complete source dataset')
        backup = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.annotation_backup')
        previous = [a for a in self.all() if a['path'] == str(destination) and a.get('status') == 'SUCCEEDED']
        file_sidecar = destination.is_file()
        old_sidecar, new_sidecar = sidecar(destination), sidecar(completed)
        backup_sidecar = Path(str(backup) + '.so101.json')
        result = None
        moved = old_sidecar_moved = new_sidecar_moved = False
        try:
            os.replace(destination, backup)
            moved = True
            if file_sidecar and old_sidecar.exists():
                os.replace(old_sidecar, backup_sidecar)
                old_sidecar_moved = True
            os.replace(completed, destination)
            if file_sidecar and new_sidecar.exists():
                os.replace(new_sidecar, old_sidecar)
                new_sidecar_moved = True
            result = self.register('annotated', destination, parents=[source['id']], extra=extra)
            for item in previous:
                atomic(self.root / 'artifacts' / (item['id'] + '.json'),
                       {**item, 'status': 'SUPERSEDED', 'superseded_by': result['id'],
                        'superseded_reason': 'validated annotation replaced source at the same path'})
        except Exception:
            if moved:
                if destination.exists(): os.replace(destination, completed)
                os.replace(backup, destination)
            if new_sidecar_moved: os.replace(old_sidecar, new_sidecar)
            if old_sidecar_moved: os.replace(backup_sidecar, old_sidecar)
            if result:
                (self.root / 'artifacts' / (result['id'] + '.json')).unlink(missing_ok=True)
            for item in previous:
                atomic(self.root / 'artifacts' / (item['id'] + '.json'), item)
            raise
        if backup.is_dir(): shutil.rmtree(backup)
        else: backup.unlink()
        backup_sidecar.unlink(missing_ok=True)
        return result


def browse_directory(kind, current='', registry=None, workspace=None):
    p = Path(current).expanduser() if current else None
    if p and p.exists():
        return str(p if p.is_dir() else p.parent)
    if registry:
        a = registry.latest({kind}, workspace)
        if a:
            p = Path(a['path']); return str(p if p.is_dir() else p.parent)
    p = ROOT / DEFAULTS.get(kind, 'outputs/workflow')
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
