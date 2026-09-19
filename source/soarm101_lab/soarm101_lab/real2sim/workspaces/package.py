"""Immutable, checksummed workspace packages. No Isaac runtime or hardware imports."""
from pathlib import Path
import copy
import hashlib
import json
import re
import shutil
import tempfile
import numpy as np
from ..profile import load as load_profile, validate as validate_profile
from ...so101_joint_mapping import DEFAULT_MAPPER

ROOT = Path(__file__).resolve().parents[5]
REGISTRY = ROOT / 'outputs/real2sim/environments'
SCHEMA = 'so101.workspace/1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def dump(path, value):
    Path(path).write_bytes(canonical(value))


def default_task(profile):
    """Explicit existing pick-red/place-first-receptacle recipe; not inferred from photos."""
    dynamic = [n for n in ('cube_red', 'cube_green', 'cube_blue') if n in profile['objects']]
    targets = {n: n + '_visual' for n in ('cup_a', 'cup_b') if n + '_visual' in profile['objects']}
    return {'schema': 'so101.workspace-task/1', 'version': 1,
            'semantics_source': 'existing pick-red/place-first recipe; editable before publish',
            'dynamic': dynamic, 'targets': targets, 'pick': 'cube_red', 'place': 'cup_a',
            'success': {'relation': 'contained_and_settled', 'max_speed_m_s': .03},
            'reset': {'seed': 0, 'margin_m': .01, 'clearance_m': .003, 'separation_m': .005,
                      'support_object': 'marker', 'yaw_rad': [0., 0.]},
            'physics': {'mass_kg': .02, 'static_friction': 1., 'dynamic_friction': 1.,
                        'restitution': 0., 'provenance': 'provisional task simulation parameters'},
            'robot_initial_joint_rad': {'shoulder_pan': 0., 'shoulder_lift': -1.7453,
                'elbow_flex': 1.5708, 'wrist_flex': 1.2217, 'wrist_roll': 0., 'gripper': 0.}}


def validate_task(task, profile):
    if task.get('schema') != 'so101.workspace-task/1' or task.get('version') != 1:
        raise ValueError('Unsupported task schema/version')
    dynamic, targets = task['dynamic'], task['targets']
    if not dynamic or len(set(dynamic)) != len(dynamic) or set(dynamic) & set(targets):
        raise ValueError('Invalid dynamic/target classification')
    if task['pick'] not in dynamic or task['place'] not in targets:
        raise ValueError('Task pick/place not classified')
    for name in dynamic:
        obj = profile['objects'][name]
        if obj['shape'] != 'box' or not obj.get('render_enabled', True):
            raise ValueError('Phase 1 dynamic objects must be rendered boxes')
    for name, visual in targets.items():
        if name == visual or visual in dynamic or visual not in profile['objects']:
            raise ValueError('Invalid target visual link')
        if profile['objects'][visual]['shape'] not in ('cup', 'cup_proxy'):
            raise ValueError('Phase 1 targets must be cups')
    reset = task['reset']
    if reset['yaw_rad'] != [0., 0.]:
        raise ValueError('Phase 1 supports fixed zero workspace yaw')
    if any(not np.isfinite(reset[k]) or reset[k] < 0 for k in ('margin_m', 'clearance_m', 'separation_m')):
        raise ValueError('Invalid reset margin/clearance')
    support = profile['objects'][reset['support_object']]
    if support['shape'] != 'box' or not np.allclose(np.array(support['T_workspace'])[:3,:3], np.eye(3)):
        raise ValueError('Support must be workspace-aligned box')
    for name in dynamic:
        if np.any(np.array(profile['objects'][name]['dimensions_m'][:2]) + 2*reset['margin_m'] >= np.array(support['dimensions_m'][:2])):
            raise ValueError('Object does not fit spawn region')
    if task['success']['relation'] != 'contained_and_settled' or task['success']['max_speed_m_s'] <= 0:
        raise ValueError('Invalid success relation')
    if set(task['robot_initial_joint_rad']) != set(DEFAULT_MAPPER.metadata()['joint_order']):
        raise ValueError('Initial joint order/names missing')
    for k in ('mass_kg', 'static_friction', 'dynamic_friction', 'restitution'):
        if not np.isfinite(task['physics'][k]) or task['physics'][k] < 0:
            raise ValueError('Invalid physics value')
    if task['physics']['mass_kg'] <= 0:
        raise ValueError('Mass must be positive')
    return task


def resolve(workspace, registry=REGISTRY):
    p = Path(workspace)
    if p.is_dir():
        return p.resolve()
    # A short ID resolves only when unambiguous; never silently switch versions.
    found = list((Path(registry) / workspace).glob('v*/manifest.json'))
    if len(found) != 1:
        raise ValueError('Workspace ID missing/ambiguous; specify explicit package path/version')
    return found[0].parent.resolve()


def load(workspace, registry=REGISTRY):
    root = resolve(workspace, registry)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'READY':
        raise ValueError('Workspace is not READY')
    if not re.fullmatch(r'[a-zA-Z0-9_]+', manifest['workspace_id']) or not isinstance(manifest['version'], int) or manifest['version'] < 1:
        raise ValueError('Invalid workspace ID/version')
    for relative, digest in manifest['files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise ValueError('Missing/modified package asset: '+relative)
    required={'profile.json','task.json','provenance.json','scene/static.usda',manifest['robot_asset']}
    if not required.issubset(manifest['files']):raise ValueError('Required package files missing from manifest')
    if hashlib.sha256(canonical(manifest['files'])).hexdigest()!=manifest['content_sha256']:
        raise ValueError('Package content hash mismatch')
    if sha(root/'task.json')!=manifest['task_sha256']:raise ValueError('Task hash mismatch')
    robot_files={k:v for k,v in manifest['files'].items() if k.startswith('assets/SO101/')}
    if hashlib.sha256(canonical(robot_files)).hexdigest()!=manifest['robot_asset_sha256']:raise ValueError('Robot asset hash mismatch')
    profile = json.loads((root/'profile.json').read_text())
    resolved=copy.deepcopy(profile);asset_map=json.loads((root/'provenance.json').read_text())['asset_map']
    def remap(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='print_texture':
                    relative=asset_map[v['path']]
                    if relative not in manifest['files']:raise ValueError('Untracked texture')
                    v['path']=str(root/relative)
                else:remap(v)
        elif isinstance(value,list):
            for v in value:remap(v)
    remap(resolved);validate_profile(resolved)
    if sha(root/'profile.json') != manifest['source_profile_sha256']:
        raise ValueError('Profile hash mismatch')
    if manifest['mapper'] != DEFAULT_MAPPER.metadata():
        raise ValueError('Current joint mapper differs from workspace')
    task = validate_task(json.loads((root/'task.json').read_text()), profile)
    for name in task['targets']:
        if f'scene/{name}.usda' not in manifest['files']:raise ValueError('Target asset missing')
    if profile.get('wrist_mount') and 'scene/wrist_mount.usda' not in manifest['files']:raise ValueError('Mount asset missing')
    return {'root': root, 'manifest': manifest, 'profile': profile, 'task': task}


def metadata(workspace):
    w = load(workspace) if not isinstance(workspace, dict) else workspace
    m = w['manifest']
    return {k:m[k] for k in ('schema','workspace_id','version','source_revision','source_profile_sha256',
                            'robot_asset_sha256','mapper','task_sha256','content_sha256')}


def publish(revision, workspace_id, task, version=1, registry=REGISTRY):
    if not re.fullmatch(r'[a-zA-Z0-9_]+', workspace_id) or type(version) is not int or version < 1:
        raise ValueError('Invalid workspace ID/version')
    source = Path(revision).resolve();profile = load_profile(source)
    if profile.get('revision', {}).get('status', '').upper() != 'ACCEPTED':
        raise ValueError('Only explicitly ACCEPTED revisions may be published')
    validate_task(task, profile)
    destination = Path(registry)/workspace_id/f'v{version}'
    if destination.exists():
        existing = load(destination)
        if existing['manifest']['source_profile_sha256'] == sha(source) and existing['task'] == task:
            return destination  # Idempotent immutable publish.
        raise FileExistsError('Workspace version already exists; publish a new version')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.publish_', dir=destination.parent))
    try:
        shutil.copyfile(source, staging/'profile.json');dump(staging/'task.json', task)
        shutil.copytree(ROOT/'assets/SO101', staging/'assets/SO101')
        asset_map = {}
        def collect(value):
            if isinstance(value, dict):
                for k,v in value.items():
                    if k == 'print_texture':
                        src = Path(v['path']).resolve()
                        if not src.is_file():raise FileNotFoundError(src)
                        relative = 'assets/textures/'+sha(src)+src.suffix
                        (staging/relative).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, staging/relative);asset_map[str(src)] = relative
                    else:collect(v)
            elif isinstance(value, list):
                for v in value:collect(v)
        collect(profile)
        from .scene_assets import build_assets
        build_assets(staging, profile, task, asset_map)
        dump(staging/'provenance.json', {'source_revision_path':str(source),'asset_map':asset_map,
             'task_semantics_source':task['semantics_source'], 'physics':task['physics']['provenance']})
        files = {str(f.relative_to(staging)):sha(f) for f in sorted(staging.rglob('*')) if f.is_file()}
        robot_files = {k:v for k,v in files.items() if k.startswith('assets/SO101/')}
        manifest = {'schema':SCHEMA,'status':'READY','workspace_id':workspace_id,'version':version,
            'source_revision':source.parent.name,'source_profile_sha256':sha(source),
            'robot_asset':'assets/SO101/usd/so101_isaaclab.usd',
            'robot_asset_sha256':hashlib.sha256(canonical(robot_files)).hexdigest(),
            'mapper':DEFAULT_MAPPER.metadata(),'task_sha256':sha(staging/'task.json'),
            'files':files,'content_sha256':hashlib.sha256(canonical(files)).hexdigest()}
        dump(staging/'manifest.json',manifest);load(staging)
        staging.rename(destination)
        return destination
    except Exception as exc:
        dump(staging/'manifest.json',{'schema':SCHEMA,'status':'INVALID','error':str(exc)})
        raise
