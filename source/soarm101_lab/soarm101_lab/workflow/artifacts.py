"""Typed, validated local artifacts; immutable entries and explicit parent identities."""
import hashlib
import json
from pathlib import Path
import time
import uuid
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from ..so101_dataset_contract import read_contract
from ..representation import describe

TYPES = {'revision', 'workspace', 'source', 'replay', 'annotated', 'datagen', 'dataset', 'policy', 'sim_eval', 'real_eval', 'hf_dataset', 'hf_policy'}
DEFAULTS = {'revision': 'outputs/real2sim/revisions', 'workspace': 'outputs/real2sim/environments',
            'source': 'datasets', 'annotated': 'datasets/annotated', 'datagen': 'datasets/generated',
            'dataset': 'datasets/LerobotDataset', 'policy': 'policy', 'replay': 'outputs/workflow',
            'sim_eval': 'outputs/sim_eval', 'real_eval': 'outputs/real_eval'}


def fingerprint(path):
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file() and '.cache' not in f.parts)
    if not files:
        raise ValueError('Empty artifact')
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(p) if p.is_dir() else f.name).encode())
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
    result = {}
    coordinates = None
    if kind == 'workspace':
        w = load(p); result['workspace'] = metadata(w); coordinates = coordinate_contract(w)
    elif kind == 'revision':
        from ..real2sim.profile import load as read_profile
        profile = read_profile(p); result['revision'] = profile.get('revision', {})
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
            if kind == 'annotated' and any('obs/datagen_info/subtask_term_signals/grasp' not in g for g in episodes):
                raise ValueError('Missing Mimic annotation')
            from .tasks import episode_definition
            definitions = [episode_definition(g, args) for g in episodes]
            unique = {json.dumps(d, sort_keys=True): d for d in definitions if d is not None}
            if unique and any(d is None for d in definitions):
                raise ValueError('Some episodes have no task definition')
            result.update(workspace=args.get('workspace'), episode_count=len(episodes), env_args=args, tasks=list(unique.values()))
        coordinates = read_contract(p)
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
        artifact = {**details, 'id': uuid.uuid4().hex, 'type': kind, 'path': str(Path(path).resolve()),
                    'created_at': time.time(), 'status': 'SUCCEEDED', 'parents': list(parents),
                    'sha256': fingerprint(path), 'execution': extra or {}}
        if workspaces:
            artifact['workspace'] = workspaces[0]
        for key in ('coordinates', 'representation', 'tasks'):
            if key not in artifact and lineage and lineage[0].get(key):
                artifact[key] = lineage[0][key]
        atomic(self.root / 'artifacts' / (artifact['id'] + '.json'), artifact)
        return artifact

    def get(self, identifier):
        return json.loads((self.root / 'artifacts' / (identifier + '.json')).read_text())

    def all(self):
        return sorted((json.loads(p.read_text()) for p in (self.root / 'artifacts').glob('*.json')), key=lambda a: a['created_at'])

    def latest(self, kinds, workspace=None):
        for a in reversed(self.all()):
            if a['type'] in kinds and (workspace is None or a.get('workspace') == workspace) and Path(a['path']).exists():
                return a
        return None

    def verify(self, artifact):
        if fingerprint(artifact['path']) != artifact['sha256']:
            raise ValueError('Artifact changed since registration; import it again explicitly')
        current = inspect_artifact(artifact['type'], artifact['path'])
        for key in ('coordinates', 'workspace'):
            if key in current and current[key] != artifact.get(key):
                raise ValueError('Artifact metadata changed since registration')
        return current


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
