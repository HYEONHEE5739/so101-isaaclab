"""Workspace Task selection and immutable scene composition; no Workflow dependency."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

TASK_ENV = 'SO101_TASK_SELECTION'


def catalog(workspace):
    task = workspace['task']
    if task.get('mode') == 'geometry_review': return []
    reset_hash = hashlib.sha256(json.dumps(workspace.get('original_reset',task['reset']), sort_keys=True).encode()).hexdigest()
    if task.get('tasks'):
        return [dict(schema='so101.task-definition/1', task_id=t['id'], pick_object_id=t['pick'],
                     target_object_id=t['place'], language_instruction=t['instruction'],
                     success_condition=copy.deepcopy(t.get('success',task['success'])),
                     randomization_spec={'path':'task.json#/reset','sha256':reset_hash}) for t in task['tasks'] if t['pick'] in task['dynamic']]
    return [dict(schema='so101.task-definition/1', task_id=f'{pick}_to_{target}',
                 pick_object_id=pick, target_object_id=target,
                 language_instruction=f"Pick up {pick} and place it in {target}.",
                 success_condition=copy.deepcopy(task['success']),
                 randomization_spec={'path': 'task.json#/reset', 'sha256': reset_hash})
            for pick in task['dynamic']
            for target in task['targets']]


def validate(definition, workspace):
    choices = {t['task_id']: t for t in catalog(workspace)}
    expected = choices.get(definition.get('task_id'))
    if expected is None:
        raise ValueError('Unknown workspace task_id')
    for key in ('schema', 'pick_object_id', 'target_object_id', 'success_condition', 'randomization_spec'):
        if definition.get(key) != expected[key]:
            raise ValueError('Task semantic mismatch: ' + key)
    if not isinstance(definition.get('language_instruction'), str) or not definition['language_instruction'].strip():
        raise ValueError('Task language_instruction must be nonempty')
    return copy.deepcopy(definition)


def selected(workspace):
    if workspace['task'].get('mode') == 'geometry_review':
        raise ValueError('Geometry review has no workflow task; import explicit workflow metadata first')
    path = os.environ.get(TASK_ENV)
    if path:
        definitions = json.loads(Path(path).read_text())['tasks']
        if len(definitions) != 1:
            raise ValueError('This stage requires exactly one task')
        return validate(definitions[0], workspace)
    # Compatibility: the package's explicit pick/place fields, never language text.
    return next(t for t in catalog(workspace) if t['pick_object_id'] == workspace['task']['pick']
                and t['target_object_id'] == workspace['task']['place'])


def object_ids(workspace):
    return list(workspace.get('all_dynamic_object_ids', workspace['task']['dynamic']))


def active_objects(workspace, requested=None):
    universe = object_ids(workspace)
    values = list(workspace['task']['dynamic']) if requested is None else requested
    if not isinstance(values, (list, tuple)) or not values or any(not isinstance(n, str) for n in values):
        raise ValueError('Select at least one active dynamic object')
    if len(set(values)) != len(values): raise ValueError('Duplicate active object IDs')
    if set(values) - set(universe): raise ValueError('Unknown/non-dynamic active object IDs: '+str(sorted(set(values)-set(universe))))
    result = [n for n in universe if n in values]
    for name in result:
        obj = workspace['profile']['objects'].get(name)
        if obj is None or len(obj['dimensions_m']) != 3 or any(not math.isfinite(v) or v <= 0 for v in obj['dimensions_m']):
            raise ValueError('Missing/invalid dimensions for active object: '+name)
    if workspace['task']['reset'].get('mode') == 'four_anchors' and len(result) > 4:
        raise ValueError('Four-anchor sampling supports up to four active objects')
    return result


def workspace_key(workspace):
    from .package import metadata
    if workspace.get('scene_configuration'):return copy.deepcopy(workspace['scene_configuration']['workspace'])
    if workspace.get('manifest', {}).get('content_sha256'):
        return metadata(workspace)
    # Existing in-memory profile/test workflows have no published manifest.
    return {'profile_sha256': hashlib.sha256(json.dumps(workspace['profile'],sort_keys=True).encode()).hexdigest()}


def configured(workspace, definition=None, active_object_ids=None, backend=None, reset=None):
    review = workspace['task'].get('mode') == 'geometry_review'
    definition = definition or workspace.get('task_definition')
    definition = validate(definition, workspace) if definition is not None else (None if review else selected(workspace))
    if active_object_ids is None and not workspace.get('scene_configuration') and os.environ.get(TASK_ENV):
        active_object_ids = json.loads(Path(os.environ[TASK_ENV]).read_text()).get('active_object_ids')
    active = active_objects(workspace, active_object_ids) if workspace['task']['dynamic'] else []
    if definition is not None and definition['pick_object_id'] not in active:
        raise ValueError('Pick target is not in active object set: '+definition['pick_object_id'])
    w = copy.deepcopy(workspace)
    original = copy.deepcopy(workspace.get('original_reset',workspace['task']['reset']))
    reset = reset or workspace.get('scene_configuration',{}).get('reset')
    if reset is not None:
        uniform = {**original,'mode':'uniform','yaw_rad':[0.,0.],'xy_jitter_m':0.}
        edge_uniform = {**uniform,'margin_m':0.}
        if reset != original and reset != uniform and reset != edge_uniform:
            raise ValueError('Unsupported runtime reset policy')
        w['original_reset'] = original
        w['task']['reset'] = copy.deepcopy(reset)
    w['all_dynamic_object_ids'] = object_ids(workspace)
    excluded = set(w['all_dynamic_object_ids']) - set(active)
    w['excluded_object_ids'] = sorted(excluded)
    if excluded and w.get('manifest', {}).get('builder'):
        import xml.etree.ElementTree as ET
        source = ET.parse(w['root']/w['manifest']['builder']['mjcf']).getroot()
        separate = {b.get('name') for b in source.findall('worldbody/body')}
        for name in excluded:
            if name not in separate or not (w['root']/f'scene/{name}.usda').is_file():
                raise ValueError('Builder object is not independently removable: '+name)
    # Remove objects rather than relabel them as static. Both scene compilers use this profile.
    w['profile']['objects'] = {n:o for n,o in w['profile']['objects'].items() if n not in excluded}
    w['task']['dynamic'] = active
    for key in ('roles', 'object_physics'):
        if key in w['task']:w['task'][key] = {n:v for n,v in w['task'][key].items() if n not in excluded}
    if definition is not None:
        w['task'].update(pick=definition['pick_object_id'], place=definition['target_object_id'],
                         success=copy.deepcopy(definition['success_condition']))
        w['task_definition'] = definition
    from .physics import compile_physics
    physics = compile_physics(w['profile'], w['task'])['objects']
    if any(not physics[n].get('collision') or not physics[n].get('mass_kg', 0) > 0 for n in active):
        raise ValueError('Active objects require collision and positive physics mass')
    scene = dict(workspace=workspace_key(workspace), active_object_ids=active,
        pick_object_id=w['task']['pick'], place_target_id=w['task']['place'],
        task_definition=copy.deepcopy(definition), reset=copy.deepcopy(w['task']['reset']),
        backend=backend or workspace.get('scene_configuration',{}).get('backend'))
    scene['sha256'] = hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    w['scene_configuration'] = scene
    return w
