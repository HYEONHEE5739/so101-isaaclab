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


def placement_condition(workspace, object_id, target_id, condition=None):
    """Resolve a unique published pair condition; never invent a speed or geometry."""
    task = workspace['task']
    if object_id not in task['dynamic'] or target_id not in task['targets']:
        raise ValueError('Placement requires existing active dynamic object and target')
    if condition is None:
        matches = [d['success_condition'] for d in catalog(workspace)
                   if d['pick_object_id'] == object_id and d['target_object_id'] == target_id
                   and (task.get('tasks') or (object_id == task['pick'] and target_id == task['place']))]
        unique = {json.dumps(c, sort_keys=True): c for c in matches}
        if len(unique) != 1:
            raise ValueError('Metadata has no unique placement condition for this pair; specify condition explicitly')
        condition = next(iter(unique.values()))
    if not isinstance(condition, dict): raise ValueError('Explicit placement condition required')
    relation = condition.get('relation')
    if relation not in ('contained_and_settled', 'inside_region_and_settled'):
        raise ValueError('Unsupported placement relation')
    speed = condition.get('max_speed_m_s')
    if not isinstance(speed, (float, int)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError('Placement requires positive finite max_speed_m_s')
    if relation == 'inside_region_and_settled':
        bounds = condition.get('bounds_m')
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or
                any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in bounds) or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in bounds for v in row) or
                any(a >= b for a, b in zip(*bounds))):
            raise ValueError('Explicit target-local region requires ordered 2x3 bounds_m')
    elif 'wall_m' not in workspace['profile']['objects'][task['targets'][target_id]]:
        raise ValueError('contained_and_settled requires supported cup geometry; use explicit region for other targets')
    return copy.deepcopy(condition)


def task_binding(workspace, object_id, target_id, condition=None):
    """Reuse exact Metadata contracts, or explicitly bind a supported non-catalog pair.

    This reference serves existing recording/grasp diagnostics, not Scene membership
    or the user's evaluation ID. Native definitions keep their historical schema.
    """
    condition = placement_condition(workspace, object_id, target_id, condition)
    matches = [d for d in catalog(workspace) if d['pick_object_id'] == object_id and
               d['target_object_id'] == target_id and d['success_condition'] == condition]
    if len(matches) == 1: return matches[0]
    return dict(schema='so101.task-binding/1', task_id='explicit_placement',
                pick_object_id=object_id, target_object_id=target_id,
                language_instruction='Explicit placement reference', success_condition=condition,
                randomization_spec={'path':'task.json#/reset', 'sha256':hashlib.sha256(
                    json.dumps(workspace.get('original_reset',workspace['task']['reset']),sort_keys=True).encode()).hexdigest()})


def scene_selection(workspace, scene):
    """Select existing movable/target bodies; structural support remains mandatory."""
    if not isinstance(scene, dict) or set(scene)-{'objects','reset','diagnostics'}:
        raise ValueError('Scene accepts objects, reset and optional diagnostics only')
    ids = scene.get('objects')
    if not isinstance(ids, list) or not ids or any(not isinstance(n,str) for n in ids) or len(set(ids)) != len(ids):
        raise ValueError('Scene requires unique Object IDs')
    known = set(object_ids(workspace)) | set(workspace['task']['targets'])
    if set(ids)-known: raise ValueError('Unknown/unsupported Scene Object IDs: '+str(sorted(set(ids)-known)))
    active = active_objects(workspace, [n for n in ids if n in object_ids(workspace)])
    return active


def validate(definition, workspace):
    if definition.get('schema') == 'so101.task-binding/1':
        expected = task_binding(workspace, definition.get('pick_object_id'), definition.get('target_object_id'), definition.get('success_condition'))
        if ({k:v for k,v in definition.items() if k!='language_instruction'} !=
                {k:v for k,v in expected.items() if k!='language_instruction'}):
            raise ValueError('Explicit placement binding differs from validated Metadata')
        if not isinstance(definition.get('language_instruction'),str) or not definition['language_instruction'].strip():
            raise ValueError('Task language_instruction must be nonempty')
        return copy.deepcopy(definition)
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


def configured(workspace, definition=None, active_object_ids=None, backend=None, reset=None, *, scene_objects=None, scene_only=False, object_configuration=None):
    scene_only = scene_only or workspace.get('scene_configuration',{}).get('scene_only',False)
    review = workspace['task'].get('mode') == 'geometry_review'
    definition = None if scene_only else (definition or workspace.get('task_definition'))
    definition = validate(definition, workspace) if definition is not None else (None if review or scene_only else selected(workspace))
    if active_object_ids is None and not workspace.get('scene_configuration') and os.environ.get(TASK_ENV):
        active_object_ids = json.loads(Path(os.environ[TASK_ENV]).read_text()).get('active_object_ids')
    active = active_objects(workspace, active_object_ids) if workspace['task']['dynamic'] else []
    scene_objects = scene_objects if scene_objects is not None else workspace.get('scene_configuration',{}).get('object_ids')
    if scene_objects is not None:
        chosen = scene_selection(workspace, {'objects': scene_objects})
        if chosen != active: raise ValueError('Scene objects and active dynamic IDs conflict')
        if definition is not None and definition['target_object_id'] not in scene_objects:
            raise ValueError('Placement target absent from selected Scene')
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
    excluded = (set(w['all_dynamic_object_ids']) - set(active)) | set(workspace.get('excluded_object_ids', []))
    excluded_profiles = set()
    if scene_objects is not None:
        excluded |= set(w['task']['targets']) - set(scene_objects)
        if w['task']['reset']['support_object'] in excluded:
            raise ValueError('Cannot remove required reset support')
        excluded_profiles = {w['task']['targets'][n] for n in excluded if n in w['task']['targets']}
        if w['task']['reset']['support_object'] in excluded_profiles:
            raise ValueError('Cannot remove required reset support profile')
        if excluded_profiles & {v for n,v in w['task']['targets'].items() if n not in excluded}:
            raise ValueError('Scene targets sharing one physical profile cannot be removed independently')
        w['task']['targets'] = {n:v for n,v in w['task']['targets'].items() if n not in excluded}
    w['excluded_object_ids'] = sorted(excluded)
    if excluded and w.get('manifest', {}).get('builder'):
        import xml.etree.ElementTree as ET
        source = ET.parse(w['root']/w['manifest']['builder']['mjcf']).getroot()
        separate = {b.get('name') for b in source.findall('worldbody/body')}
        for name in excluded:
            if name not in separate or not (w['root']/f'scene/{name}.usda').is_file():
                raise ValueError('Builder object is not independently removable: '+name)
    # Remove objects rather than relabel them as static. Both scene compilers use this profile.
    w['profile']['objects'] = {n:o for n,o in w['profile']['objects'].items() if n not in excluded and n not in excluded_profiles}
    w['task']['dynamic'] = active
    for key in ('roles', 'object_physics'):
        if key in w['task']:w['task'][key] = {n:v for n,v in w['task'][key].items() if n not in excluded and n not in excluded_profiles}
    if definition is not None:
        w['task'].update(pick=definition['pick_object_id'], place=definition['target_object_id'],
                         success=copy.deepcopy(definition['success_condition']))
        w['task_definition'] = definition
    elif scene_only:
        w['task'].update(mode='scene_recording',pick=None,place=None)
        w['task_definition']=None
    from .physics import compile_physics
    physics = compile_physics(w['profile'], w['task'])['objects']
    if any(not physics[n].get('collision') or not physics[n].get('mass_kg', 0) > 0 for n in active):
        raise ValueError('Active objects require collision and positive physics mass')
    scene = dict(workspace=workspace_key(workspace), active_object_ids=active,
        pick_object_id=w['task']['pick'], place_target_id=w['task']['place'],
        task_definition=copy.deepcopy(definition), reset=copy.deepcopy(w['task']['reset']),
        backend=backend or workspace.get('scene_configuration',{}).get('backend'))
    # Existing active-dynamic + all-target scenes retain their exact recorded hash.
    # Only target subsets need the extra membership field for faithful restoration.
    if scene_objects is not None and (workspace.get('scene_configuration',{}).get('object_ids') is not None
                                     or set(workspace['task']['targets']) - set(scene_objects)):
        scene['object_ids'] = list(active) + list(w['task']['targets'])
    if scene_only:
        scene.update(scene_only=True,object_ids=list(active)+list(w['task']['targets']),
                     objects=copy.deepcopy(object_configuration if object_configuration is not None else workspace.get('scene_configuration',{}).get('objects',{})))
    scene['sha256'] = hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    w['scene_configuration'] = scene
    return w


def generate_template(workspace, output, definition=None, *, script_stage=None, checkpoint=None, evaluation_definitions=None):
    """Generate trusted Python source from one validated Workspace, without executing it.

    Generic success is deliberately unimplemented. An explicitly selected existing
    definition reuses the Backend's Pick & Place criterion; it is not Mimic success.
    """
    import ast
    from pprint import pformat
    output=Path(output).expanduser().resolve()
    if workspace.get('root') and output.is_relative_to(Path(workspace['root']).resolve()):
        raise ValueError('Task output must be outside the immutable Workspace')
    if output.suffix!='.py':raise ValueError('Task output must be a .py file')
    if evaluation_definitions is not None and script_stage is None:
        raise ValueError('Multiple definitions require a Sim Evaluation Script')
    definition=validate(definition,workspace) if definition is not None else None
    objects=tuple(workspace['task']['dynamic'])+tuple(workspace['task']['targets'])
    cameras=tuple(workspace['profile']['cameras'])
    if definition:
        objects=(definition['pick_object_id'],definition['target_object_id'])
    if script_stage is not None:
        if evaluation_definitions is not None:
            if script_stage!='sim_eval' or definition is not None or not isinstance(evaluation_definitions,list) or not evaluation_definitions:
                raise ValueError('evaluation_definitions requires a nonempty list for Sim Evaluation only')
            choices=[validate(d,workspace) for d in evaluation_definitions]
            if len({d['task_id'] for d in choices})!=len(choices):raise ValueError('Duplicate evaluation Task IDs')
        else: choices=definition
        source = _execution_template(workspace, output, choices, script_stage, checkpoint)
        ast.parse(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream: stream.write(source)
        return output
    source='''"""Generated Task: edit semantics explicitly; no implicit reset or Mimic support.

Object poses: workspace frame, metres, 4x4 matrix. Cameras below are Metadata
references only. No camera is created or rendered by this Task.
"""\n'''
    source+=f'TASK_ID = {(definition["task_id"] if definition else "my_task")!r}\n'
    source+=f'INSTRUCTION = {(definition["language_instruction"] if definition else "Describe the task")!r}\n'
    source+=f'OBJECT_IDS = {objects!r}\nCAMERA_IDS = {cameras!r}\n'
    source+=f'ROBOT_ASSET = {workspace["profile"].get("robot_asset")!r}\n'
    source+=f'ROBOT_ASSET_SHA256 = {workspace.get("manifest",{}).get("robot_asset_sha256")!r}\n'
    source+=f'TASK_DEFINITION = {pformat(definition,sort_dicts=False)}\n'
    if definition:source+=f'TASK_CONTRACTS = {pformat(pick_place_contracts(),sort_dicts=False)}\n'
    source+='\n'
    source+='''def setup(ctx):
    ctx.require(objects=OBJECT_IDS, cameras=CAMERA_IDS)
    if TASK_DEFINITION is not None and ctx.workspace.get('task_definition') != TASK_DEFINITION:
        raise ValueError('Task definition differs from the configured runtime')


def observe(ctx):
    return {name: ctx.object_pose(name) for name in OBJECT_IDS}


def success(ctx):
'''
    source+=('    return bool(ctx.placement_details(TASK_DEFINITION)["success"])\n' if definition else
             "    raise NotImplementedError('Define task success explicitly; Mimic hooks are not implemented')\n")
    ast.parse(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: never replace an existing user task.
    with output.open('x',encoding='utf-8') as stream:stream.write(source)
    return output


def validate_task_module(namespace):
    """Check the minimal function-based Task contract without calling its hooks.

    Call after explicit worker-side Python loading, before acquiring an engine.
    Runtime binding must be checked by TaskContext.validate_bindings before setup.
    This does not certify Python safety, success semantics or Mimic support.
    """
    for name in ('TASK_ID','INSTRUCTION'):
        value=namespace.get(name)
        if not isinstance(value,str) or not value.strip():
            raise ValueError('Task requires a nonempty string: '+name)
    for name in ('OBJECT_IDS','CAMERA_IDS'):
        values=namespace.get(name)
        if (not isinstance(values,(tuple,list)) or
                any(not isinstance(v,str) or not v.strip() for v in values) or
                len(set(values))!=len(values)):
            raise ValueError('Task requires unique explicit ID strings: '+name)
    import inspect
    for name in ('setup','observe','success'):
        if not callable(namespace.get(name)):
            raise ValueError('Task requires callable '+name+'(ctx)')
        try:inspect.signature(namespace[name]).bind(object())
        except (TypeError,ValueError) as exc:
            raise ValueError('Task hook must accept one context: '+name) from exc
    definition=namespace.get('TASK_DEFINITION')
    if definition is not None and not isinstance(definition,dict):
        raise ValueError('TASK_DEFINITION must be a dict or None')

    if namespace.get('ROBOT_ASSET') is not None and not isinstance(namespace['ROBOT_ASSET'],dict):
        raise ValueError('ROBOT_ASSET must be a dict or None')

    digest=namespace.get('ROBOT_ASSET_SHA256')
    if digest is not None and (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('ROBOT_ASSET_SHA256 must be a SHA256 hex string or None')


def pick_place_contracts():
    """References to existing implementations, not replacement algorithms or a registry.

    Only placement/evaluation is dispatched by the current SDK worker. The other
    references document the unchanged contracts for subsequent Annotation/Mimic wiring.
    """
    return {
        'placement': 'simulation.mujoco_backend.MujocoBackend.success_details',
        'stable_grasp': 'real2sim.workspaces.grasp.StableGrasp',
        'grasp_drop_release': 'workflow.adapters.SimulationAdapter.update_grasp',
        'annotation': 'workflow.annotation.annotate_episode',
        'mimic': 'workflow.mimic.configuration',
        'evaluation': 'so101.placement/1',
    }


def validate_task_scene(task, workspace):
    """Pure pre-engine check against an explicitly configured Workspace."""
    objects=set(workspace['task']['dynamic']) | set(workspace['task']['targets'])
    missing=set(task['OBJECT_IDS'])-objects
    if missing:raise ValueError('Task objects absent from selected Scene: '+str(sorted(missing)))
    missing=set(task['CAMERA_IDS'])-set(workspace['profile']['cameras'])
    if missing:raise ValueError('Unknown Task camera IDs: '+str(sorted(missing)))
    if task.get('TASK_DEFINITION') is not None and task['TASK_DEFINITION']!=workspace.get('task_definition'):
        raise ValueError('Task definition differs from explicitly configured scene')
    if task.get('ROBOT_ASSET') is not None and task['ROBOT_ASSET']!=workspace['profile']['robot_asset']:
        raise ValueError('Task robot asset reference differs')
    digest=task.get('ROBOT_ASSET_SHA256')
    if digest is not None and digest!=workspace.get('manifest',{}).get('robot_asset_sha256'):
        raise ValueError('Task robot asset identity differs or is unavailable')
    contracts=task.get('TASK_CONTRACTS')
    if contracts is not None:
        if contracts!=pick_place_contracts() or task.get('TASK_DEFINITION') is None:
            raise ValueError('Unsupported Task contracts; existing Pick & Place contracts require TASK_DEFINITION')


def _execution_template(workspace, output, definition, stage, checkpoint):
    """Emit editable intent only; Geometry, identities and conditions stay in Metadata."""
    from pprint import pformat
    if stage not in ('source','sim_eval'): raise ValueError('Choose Source or Sim Evaluation script')
    definitions=definition if isinstance(definition,list) else ([definition] if definition else [])
    definition=definitions[0] if definitions else None
    opts=dict(backend='mujoco',storage_name=output.stem,num_envs=1)
    if stage=='source':opts.update(port='/dev/so101_leader',episodes=3,episode_time_s=20,reset_time_s=5)
    else:opts.update(device='cpu',max_steps=1000,eval_seed_mode='fixed',eval_seed=1047)
    config=dict(stage=stage,workspace=str(workspace['root']),
                scene={'objects':list(workspace['task']['dynamic'])+list(workspace['task']['targets'])},options=opts)
    if stage=='source':
        config['scene']['objects']={n:{} for n in config['scene']['objects']}
        if definition:
            config['source']={'instruction':definition['language_instruction']}
            config['annotation']={'diagnostics':{'object_id':definition['pick_object_id'],'target_id':definition['target_object_id'],'condition':definition['success_condition']},
                'events':{'grasp':'grasp_event','place':'place_event'},'signals':['grasp'],'final_events':['place'],'manual_event':'grasp','accept':'native_acceptance'}
            config['mimic']={'recipe':'pick_place/1','accept':'native_acceptance'}
    else:
        config['checkpoint']=str(checkpoint) if checkpoint else 'REPLACE_CHECKPOINT_PATH'
        config['evaluations']=[dict(id=d['task_id'],instruction=d['language_instruction'],
            success='placement_success' if i==0 else f'placement_success_{i+1}',episodes=10) for i,d in enumerate(definitions)] if definitions else [
            dict(id='my_task',instruction='EDIT_INSTRUCTION',success='custom_success',episodes=10)]
    text='"""Edit Scene IDs, intent and run options. Paths are relative to this file.\n'
    text+='Geometry/Physics/Camera/Robot and unique pair conditions come from the Workspace.\n'
    text+=('Source records a Scene without Task/Target requirements; default Pose comes from Builder.\n' if stage=='source' else 'Custom Evaluation terms require explicit scene.diagnostics for the current Policy Recorder.\n')+'"""\n'
    text+='SCRIPT = '+pformat(config,sort_dicts=False)+'\n'
    if stage=='source' and definition:
        text+='\nfrom soarm101_lab.task_terms.diagnostics import grasp_completed as grasp_event\nfrom soarm101_lab.task_terms.annotation import placed as place_event, native_acceptance\n'
    if stage=='sim_eval' and definition:
        text+='\nfrom functools import partial\nfrom soarm101_lab.task_terms.placement import placement\n'
        for i,d in enumerate(definitions):
            name='placement_success' if i==0 else f'placement_success_{i+1}'
            text+=name+' = partial(placement, object_id='+repr(d['pick_object_id'])+', target_id='+repr(d['target_object_id'])+')\n'
    elif stage=='sim_eval':
        text+='\ndef custom_success(ctx):\n    # Add explicit scene.diagnostics and implement this condition before execution.\n    raise NotImplementedError("Define an evaluation term")\n'
    return text
