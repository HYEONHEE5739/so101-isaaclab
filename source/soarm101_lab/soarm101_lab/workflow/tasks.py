"""Recorded Task metadata, Stage bindings and evaluation diagnostics."""
import copy
import hashlib
import json
from pathlib import Path

from ..real2sim.workspaces import tasks as workspace_tasks

def recorded_scene(artifact, workspace, required=True):
    """Read explicit evidence only; never infer three objects from today's defaults."""
    prov = artifact.get('provenance', artifact)
    scene = (artifact.get('scene_configuration') or prov.get('scene_configuration') or
             prov.get('scene', {}).get('scene_configuration') or artifact.get('env_args', {}).get('scene_configuration'))
    if scene:
        if scene.get('workspace') != workspace_tasks.workspace_key(workspace): raise ValueError('Recorded scene workspace identity differs')
        ids = workspace_tasks.active_objects(workspace, scene['active_object_ids'])
        if ids != scene['active_object_ids']:raise ValueError('Recorded active object order differs from Workspace')
        payload = {k:v for k,v in scene.items() if k != 'sha256'}
        digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if scene.get('sha256') != digest:raise ValueError('Recorded scene configuration hash mismatch')
        bound=workspace_tasks.configured(workspace,scene.get('task_definition'),ids,scene.get('backend'),reset=scene.get('reset'),scene_objects=scene.get('object_ids'),scene_only=scene.get('scene_only',False),object_configuration=scene.get('objects'))['scene_configuration']
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
            if args.get('workspace') == workspace_tasks.workspace_key(workspace):
                sets = [set(g['initial_state/rigid_object']) & set(workspace_tasks.object_ids(workspace))
                        for g in h['data'].values() if 'initial_state/rigid_object' in g]
                if sets and all(v==sets[0] for v in sets):ids = list(sets[0])
    if ids:
        definitions = from_artifact(artifact,workspace)
        definition = definitions[0] if len(definitions)==1 else old.get('task_definition')
        return workspace_tasks.configured(workspace,definition,ids,artifact.get('backend',old.get('backend')))['scene_configuration']
    if required:raise ValueError('Recorded active object set is UNKNOWN; scene/model evidence is required for replay')
    return None


def bind_scene_options(stage, artifact, workspace, options):
    """Bind run choices or inherited evidence into existing request options."""
    from .task_scripts import resolve_script
    opts = dict(resolve_script(workspace,options))
    if opts.get('scene_only'):return opts
    if workspace['task'].get('mode')=='geometry_review':
        raise ValueError('Geometry review requires explicit workflow metadata for task execution')
    if stage in ('source','sim_eval','real_eval'):
        ids = workspace_tasks.active_objects(workspace, opts.get('active_object_ids'))
    elif stage in ('annotate','datagen'):
        scenes = artifact.get('provenance',{}).get('scene_configurations',[])
        if len({tuple(s['active_object_ids']) for s in scenes}) > 1:
            raise ValueError('Mixed scene compositions: select a homogeneous source subset for annotation/datagen')
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
            definitions = [t for t in workspace_tasks.catalog(workspace) if t['pick_object_id'] == args['grasp_object']
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
