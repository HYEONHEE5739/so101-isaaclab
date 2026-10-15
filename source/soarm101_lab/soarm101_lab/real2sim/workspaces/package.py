"""Immutable, checksummed workspace packages. No Isaac runtime or hardware imports."""
from pathlib import Path
import copy
import hashlib
import json
import re
import numpy as np
from ..profile import validate as validate_profile
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
            'semantics_source': 'existing pick-red/place-first recipe; editable before Builder import',
            'dynamic': dynamic, 'targets': targets, 'pick': 'cube_red', 'place': 'cup_a',
            'success': {'relation': 'contained_and_settled', 'max_speed_m_s': .03},
            'reset': {'seed': 0, 'margin_m': .01, 'clearance_m': .003, 'separation_m': .005,
                      'support_object': 'marker', 'yaw_rad': [0., 0.], 'mode': 'four_anchors', 'xy_jitter_m': .02},
            'physics': {'mass_kg': .02, 'static_friction': 1., 'dynamic_friction': 1.,
                        'restitution': 0., 'provenance': 'default_template'},
            'robot_initial_joint_rad': {'shoulder_pan': 0., 'shoulder_lift': -1.7453,
                'elbow_flex': 1.5708, 'wrist_flex': 1.2217, 'wrist_roll': 0., 'gripper': 0.}}


def validate_task(task, profile):
    if task.get('schema') != 'so101.workspace-task/1' or task.get('version') != 1:
        raise ValueError('Unsupported task schema/version')
    if task.get('mode') == 'geometry_review':
        dynamic, targets = task['dynamic'], task['targets']
        if len(set(dynamic)) != len(dynamic) or set(dynamic) & set(targets):
            raise ValueError('Invalid review physics classification')
        for name in dynamic:
            if name not in profile['objects'] or profile['objects'][name].get('geometry_source') != 'builder_export':
                raise ValueError('Review dynamics require explicit Builder collision geometry')
        for name, visual in targets.items():
            if name == visual or visual not in profile['objects'] or visual in dynamic or profile['objects'][visual]['shape'] != 'cup_proxy':
                raise ValueError('Review receptacles require explicit open round collision geometry')
        physics = task['physics']
        for key in ('static_friction', 'dynamic_friction', 'restitution'):
            if not np.isfinite(physics[key]) or physics[key] < 0:
                raise ValueError('Invalid review physics value')
        if physics['restitution'] > 1 or dynamic and (not np.isfinite(physics.get('mass_kg', np.nan)) or physics['mass_kg'] <= 0):
            raise ValueError('Invalid review dynamics mass/restitution')
        for name, row in task.get('object_physics', {}).items():
            if name not in [*dynamic, *targets]: raise ValueError('Unknown review physics object')
            for key in ('mass_kg', 'static_friction', 'dynamic_friction', 'restitution'):
                if key in row and (not np.isfinite(row[key]) or row[key] < 0 or key == 'mass_kg' and row[key] <= 0 or key == 'restitution' and row[key] > 1):
                    raise ValueError('Invalid review object physics')
        expected = review_contract(profile, task['reset']['support_object'], dynamic, targets)
        expected['physics'] = copy.deepcopy(physics)
        if 'object_physics' in task: expected['object_physics'] = copy.deepcopy(task['object_physics'])
        if task != expected:
            raise ValueError('Invalid geometry review contract; task semantics require explicit workflow metadata')
        return task
    dynamic, targets = task['dynamic'], task['targets']
    if not dynamic or len(set(dynamic)) != len(dynamic) or set(dynamic) & set(targets):
        raise ValueError('Invalid dynamic/target classification')
    if task['pick'] not in dynamic or task['place'] not in targets:
        raise ValueError('Task pick/place not classified')
    for name in dynamic:
        obj = profile['objects'][name]
        if (obj['shape'] != 'box' and obj.get('geometry_source') != 'builder_export') or not obj.get('render_enabled', True):
            raise ValueError('Phase 1 dynamic objects must be rendered boxes')
    for name, visual in targets.items():
        if name == visual or visual in dynamic or visual not in profile['objects']:
            raise ValueError('Invalid target visual link')
        if profile['objects'][visual]['shape'] not in ('cup', 'cup_proxy') and profile['objects'][visual].get('geometry_source') != 'builder_export':
            raise ValueError('Phase 1 targets must be cups')
    reset = task['reset']
    if reset.get('mode', 'uniform') not in ('uniform', 'four_anchors'):
        raise ValueError('Unknown reset mode')
    if reset.get('mode') == 'four_anchors':
        if len(dynamic) > 4 or not np.isfinite(reset.get('xy_jitter_m', -1)) or reset.get('xy_jitter_m', -1) < 0:
            raise ValueError('Invalid four-anchor reset settings')
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
    def check_success(success, target):
        if success.get('relation') not in ('contained_and_settled', 'inside_region_and_settled') or not np.isfinite(success.get('max_speed_m_s', np.nan)) or success['max_speed_m_s'] <= 0:
            raise ValueError('Invalid success relation')
        if success['relation'] == 'inside_region_and_settled':
            bounds = np.asarray(success.get('bounds_m'), dtype=float)
            if bounds.shape != (2,3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError('Success requires target-local bounds_m [min,max]')
        elif profile['objects'][targets[target]]['shape'] not in ('cup','cup_proxy'):
            raise ValueError('Non-round target requires explicit success region')
    check_success(task['success'], task['place'])
    seen = set()
    for item in task.get('tasks', []):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', item.get('id','')) or item['id'] in seen or item.get('pick') not in dynamic or item.get('place') not in targets:
            raise ValueError('Invalid/duplicate explicit workflow task')
        if not isinstance(item.get('instruction'),str) or not item['instruction'].strip(): raise ValueError('Task instruction required')
        seen.add(item['id']); check_success(item.get('success', task['success']), item['place'])
    cameras = task.get('observation_cameras', list(profile['cameras']))
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= profile['cameras'].keys():
        raise ValueError('Unknown/duplicate observation camera')
    bindings = task.get('real_camera_bindings', {})
    if not set(bindings) <= profile['cameras'].keys() or len(set(bindings.values())) != len(bindings):
        raise ValueError('Unknown/duplicate real camera bindings')
    if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',v) for v in bindings.values()):
        raise ValueError('Real camera bindings must reference config field IDs')
    for name, values in task.get('object_physics', {}).items():
        if name not in dynamic + list(targets): raise ValueError('Unknown object physics ID')
        for key in ('mass_kg','static_friction','dynamic_friction','restitution'):
            if key in values and (not np.isfinite(values[key]) or values[key] < 0 or (key=='mass_kg' and values[key]==0)):
                raise ValueError('Invalid per-object physics')
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
    builder = manifest.get('builder')
    if builder:
        expected = {'scene': 'builder/scene.yaml', 'mjcf': 'builder/scene.xml', 'usd': 'builder/scene.usd'}
        if any(builder.get(k) != v for k,v in expected.items()): raise ValueError('Invalid Builder asset mapping')
        required.update([*expected.values(), 'builder/scene_manifest.yaml', 'scene/wrist_mount.usda'])
        for link in builder.get('robot_visual_links', []):
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', link): raise ValueError('Invalid Builder robot visual link')
            required.add(f'scene/robot_visual_{link}.usda')
        if builder.get('manifest_sha256') != sha(root/'builder/scene_manifest.yaml'):
            raise ValueError('Builder source manifest hash mismatch')
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
    validate_mapping(manifest['mapper'], root/'assets/SO101/urdf/so101_isaaclab.urdf')
    task = validate_task(json.loads((root/'task.json').read_text()), profile)
    for name in [*task['targets'], *(task['dynamic'] if builder else [])]:
        if f'scene/{name}.usda' not in manifest['files']:raise ValueError('Task object asset missing')
    if profile.get('wrist_mount') and 'scene/wrist_mount.usda' not in manifest['files']:raise ValueError('Mount asset missing')
    return {'root': root, 'manifest': manifest, 'profile': profile, 'task': task}


def metadata(workspace):
    w = load(workspace) if not isinstance(workspace, dict) else workspace
    m = w['manifest']
    result = {k:m[k] for k in ('schema','workspace_id','version','source_revision','source_profile_sha256',
                            'robot_asset_sha256','mapper','task_sha256','content_sha256')}
    if m.get('builder'): result['builder_scene'] = copy.deepcopy(m['builder'])
    return result


def review_contract(profile, support, dynamic=(), targets=None):
    """Fixed geometry inspection, with no manipulation goal or success predicate."""
    if support not in profile['objects'] and support != 'table':
        raise ValueError('Unknown review support')
    return {'schema': 'so101.workspace-task/1', 'version': 1, 'mode': 'geometry_review',
            'semantics_source': 'geometry review only; no task/reset distribution/success supplied',
            'dynamic': list(dynamic), 'targets': dict(targets or {}), 'pick': None, 'place': None, 'success': None,
            'reset': {'mode': 'fixed', 'seed': 0, 'support_object': support},
            'physics': {'static_friction': 1., 'dynamic_friction': 1., 'restitution': 0.,
                        'provenance': 'provisional', 'note': 'Fixed review geometry; contact fidelity unvalidated'},
            'robot_initial_joint_rad': {n: 0. for n in DEFAULT_MAPPER.metadata()['joint_order']},
            'observation_cameras': list(profile['cameras']),
            'real_camera_bindings': {n: n+'_camera' for n in profile['cameras'] if n in ('side','wrist')}}


def validate_mapping(mapping, urdf):
    """Validate capture coordinate provenance; never activate or rewrite a device profile."""
    if mapping == DEFAULT_MAPPER.metadata():
        return
    from ..precision import MAPPING, mapper_from_metadata, profile_for_model
    if not isinstance(mapping, dict) or mapping.get('schema') != MAPPING:
        raise ValueError('Builder robot mapping differs from supported SO-101 mappings')
    if mapping.get('joint_order') != DEFAULT_MAPPER.metadata()['joint_order'] or mapping.get('output') != 'urdf_joint_radian':
        raise ValueError('Precision joint order/unit mismatch')
    mapper = mapper_from_metadata(mapping)
    identity = mapper.profile.get('identity', {})
    if mapper.profile.get('role') not in ('leader', 'follower') or identity.get('role') != mapper.profile.get('role') or not identity.get('calibration_sha256'):
        raise ValueError('Precision device/calibration identity missing or conflicting')
    for joint, entry in mapper.profile['joints'].items():
        motor = identity.get('motors', {}).get(joint, {})
        if (motor.get('id') != entry['motor']['calibration']['id'] or motor.get('model') != 'sts3215'
                or entry['motor'].get('calibration_hash') != identity['calibration_sha256']):
            raise ValueError('Precision motor/calibration identity mismatch: '+joint)
        for reference in entry['references']:
            if reference.get('device_identity') != identity or reference.get('calibration_hash') != identity['calibration_sha256']:
                raise ValueError('Precision reference motor/calibration identity mismatch: '+joint)
            if reference.get('motor') != entry['motor']:
                raise ValueError('Precision reference motor mismatch: '+joint)
    profile_for_model(mapper.profile, sha(urdf))
