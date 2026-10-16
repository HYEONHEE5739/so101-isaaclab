"""Explicit manipulation semantics; language is never parsed for identities."""
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
    from ..real2sim.workspaces.package import metadata
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
    from ..real2sim.workspaces.physics import compile_physics
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


def recorded_scene(artifact, workspace, required=True):
    """Read explicit evidence only; never infer three objects from today's defaults."""
    prov = artifact.get('provenance', artifact)
    scene = (artifact.get('scene_configuration') or prov.get('scene_configuration') or
             prov.get('scene', {}).get('scene_configuration') or artifact.get('env_args', {}).get('scene_configuration'))
    if scene:
        if scene.get('workspace') != workspace_key(workspace): raise ValueError('Recorded scene workspace identity differs')
        ids = active_objects(workspace, scene['active_object_ids'])
        if ids != scene['active_object_ids']:raise ValueError('Recorded active object order differs from Workspace')
        payload = {k:v for k,v in scene.items() if k != 'sha256'}
        digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if scene.get('sha256') != digest:raise ValueError('Recorded scene configuration hash mismatch')
        bound=configured(workspace,scene.get('task_definition'),ids,scene.get('backend'),reset=scene.get('reset'))['scene_configuration']
        for key in ('pick_object_id','place_target_id','reset'):
            if scene.get(key)!=bound[key]:raise ValueError('Recorded scene semantic mismatch: '+key)
        return copy.deepcopy(scene)
    old = prov.get('scene', {})
    # Historical MuJoCo compiler explicitly recorded actual bodies, even for minimal scenes.
    ids = old.get('dynamic_objects')
    if ids is None and artifact.get('path') and Path(artifact['path']).is_file() and Path(artifact['path']).suffix in ('.hdf5','.h5'):
        import h5py
        with h5py.File(artifact['path'],'r') as h:
            args = json.loads(h['data'].attrs.get('env_args','{}'))
            if args.get('workspace') == workspace_key(workspace):
                sets = [set(g['initial_state/rigid_object']) & set(object_ids(workspace))
                        for g in h['data'].values() if 'initial_state/rigid_object' in g]
                if sets and all(v==sets[0] for v in sets):ids = list(sets[0])
    if ids:
        definitions = from_artifact(artifact,workspace)
        definition = definitions[0] if len(definitions)==1 else old.get('task_definition')
        return configured(workspace,definition,ids,artifact.get('backend',old.get('backend')))['scene_configuration']
    if required:raise ValueError('Recorded active object set is UNKNOWN; scene/model evidence is required for replay')
    return None


def bind_scene_options(stage, artifact, workspace, options):
    """Bind run choices or inherited evidence into existing request options."""
    opts = dict(options)
    if workspace['task'].get('mode')=='geometry_review':
        raise ValueError('Geometry review requires explicit workflow metadata for task execution')
    if stage in ('source','sim_eval','real_eval'):
        ids = active_objects(workspace, opts.get('active_object_ids'))
    elif stage in ('replay','annotate','datagen'):
        scenes = artifact.get('provenance',{}).get('scene_configurations',[])
        if len({tuple(s['active_object_ids']) for s in scenes}) > 1:
            raise ValueError('Mixed scene compositions: select a homogeneous source subset for replay/annotation/datagen')
        scene = recorded_scene(artifact,workspace)
        ids = scene['active_object_ids']
        if 'active_object_ids' in opts and set(opts['active_object_ids']) != set(ids):
            raise ValueError('Derived stages must inherit the recorded active object set')
    else:
        return opts
    opts['active_object_ids'] = ids
    return opts


def from_artifact(artifact, workspace=None):
    definitions = copy.deepcopy(artifact.get('tasks') or artifact.get('env_args', {}).get('task_definitions') or
                                artifact.get('provenance', {}).get('task_definitions') or [])
    if not definitions and workspace:
        args = artifact.get('env_args') or artifact.get('provenance', {}).get('source_env_args', {})
        if args.get('grasp_object') and args.get('place_bin'):
            definitions = [t for t in catalog(workspace) if t['pick_object_id'] == args['grasp_object']
                           and t['target_object_id'] == args['place_bin']]
    return definitions


def episode_definition(group, env_args):
    if 'task_definition' in group.attrs:
        return json.loads(group.attrs['task_definition'])
    definitions = env_args.get('task_definitions', [])
    return definitions[0] if len(definitions) == 1 else None


def stamp_group(group, definition, scene_configuration=None):
    if scene_configuration:group.attrs['scene_configuration']=json.dumps(scene_configuration)
    group.attrs['task_definition'] = json.dumps(definition)
    group.attrs['task_id'] = definition['task_id']
    group.attrs['language_instruction'] = definition['language_instruction']


def stamp_file(path, definition):
    import h5py
    with h5py.File(path, 'a') as h:
        group = h['data']; args = json.loads(group.attrs.get('env_args', '{}'))
        args['task_definitions'] = [definition]
        group.attrs['env_args'] = json.dumps(args)
        for name, episode in group.items():
            if name.startswith('demo_'):
                stamp_group(episode, definition, args.get('scene_configuration'))


def evaluation_metrics(definitions, rows, simulation):
    result = {}
    for definition in definitions:
        samples = [r for r in rows if r['task_id'] == definition['task_id']]
        episodes = len(samples) if simulation else len({r['episode'] for r in samples})
        successes = sum(bool(r['success']) for r in samples) if simulation else None
        result[definition['task_id']] = {
            'episodes': episodes, 'steps': sum(r['steps'] for r in samples) if simulation else len(samples),
            'successes': successes, 'success_rate': successes / episodes if simulation and episodes else None,
            'success_measurement': definition['success_condition']['relation'] if simulation else 'unmeasured',
            'language_instruction': definition['language_instruction']}
    return result


def evaluation_distribution(artifact, workspace, active):
    """Object-set OOD is reported, while the original workspace contract remains mandatory."""
    prov = artifact.get('provenance', artifact)
    path = artifact.get('path')
    if path:
        for name in ('project_provenance.json','workflow_provenance.json'):
            p = Path(path)/name
            if p.is_file():prov=json.loads(p.read_text());break
    scenes = prov.get('scene_configurations') or [recorded_scene({'provenance':prov},workspace,required=False)]
    sets = [s['active_object_ids'] for s in scenes if s]
    return {'training_active_object_sets':sets or None, 'evaluation_active_object_ids':list(active),
            'scene_composition_distribution_shift':(all(set(ids)!=set(active) for ids in sets) if sets else None)}


def object_metrics(active, target, positions, eef, state, grasp=False, place=False, initial_positions=None):
    """Approach identity proxy; single-object selection accuracy is N/A, never automatic success."""
    import numpy as np
    distance = {name:float(np.linalg.norm(np.asarray(pos)-eef)) for name,pos in positions.items() if name in active}
    if set(distance)!=set(active):raise ValueError('Evaluation object states differ from active scene')
    if 'selected_object_id' not in state:state['selected_object_id']='UNKNOWN'
    if not state.get('grasp_success') and state['selected_object_id']=='UNKNOWN' and distance:
        closest=min(distance,key=distance.get)
        if distance[closest] < .04:state['selected_object_id']=closest
    state['grasp_success']=bool(state.get('grasp_success') or grasp)
    if initial_positions and state['grasp_success']:
        state['transport_success']=bool(state.get('transport_success') or np.linalg.norm(
            np.asarray(positions[target])-initial_positions[target])>.04)
    state['place_success']=bool(state.get('place_success') or place)
    minimum = state.setdefault('object_min_distances_m', {})
    for name,value in distance.items():minimum[name]=min(value,minimum.get(name,float('inf')))
    choice = state['selected_object_id']
    discriminable = len(active)>1 and choice!='UNKNOWN'
    return {'scene_active_object_count':len(active),'active_object_ids':list(active),'target_object_id':target,
        'distractor_ids':[n for n in active if n!=target], 'selected_object_id':choice,
        'target_selection_accuracy':(choice==target if discriminable else None),
        'wrong_object_approach':(choice!=target if discriminable else None),
        'object_distances_m':distance, 'object_min_distances_m':dict(minimum),
        'object_approached':{n:v<.04 for n,v in minimum.items()}, 'grasp_success':state['grasp_success'],
        'transport_success':bool(state.get('transport_success')), 'place_success':state['place_success'],
        'full_task_success':bool(state['grasp_success'] and state.get('transport_success') and state['place_success']),
        'selection_measurement':'closest object within 40mm before grasp; geometric proxy, not verified semantic recognition'}
