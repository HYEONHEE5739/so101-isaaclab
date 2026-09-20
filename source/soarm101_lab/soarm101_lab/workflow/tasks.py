"""Explicit manipulation semantics; language is never parsed for identities."""
import copy
import hashlib
import json
import os
from pathlib import Path

TASK_ENV = 'SO101_TASK_SELECTION'


def catalog(workspace):
    task = workspace['task']
    reset_hash = hashlib.sha256(json.dumps(task['reset'], sort_keys=True).encode()).hexdigest()
    return [dict(schema='so101.task-definition/1', task_id=f'{pick}_to_{target}',
                 pick_object_id=pick, target_object_id=target,
                 language_instruction=f"Pick up the {pick.removeprefix('cube_')} cube and place it in cup {target.removeprefix('cup_').upper()}.",
                 success_condition=copy.deepcopy(task['success']),
                 randomization_spec={'path': 'task.json#/reset', 'sha256': reset_hash})
            for pick in ('cube_red', 'cube_blue', 'cube_green') if pick in task['dynamic']
            for target in ('cup_a', 'cup_b') if target in task['targets']]


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
    path = os.environ.get(TASK_ENV)
    if path:
        definitions = json.loads(Path(path).read_text())['tasks']
        if len(definitions) != 1:
            raise ValueError('This stage requires exactly one task')
        return validate(definitions[0], workspace)
    # Compatibility: the package's explicit pick/place fields, never language text.
    return next(t for t in catalog(workspace) if t['pick_object_id'] == workspace['task']['pick']
                and t['target_object_id'] == workspace['task']['place'])


def configured(workspace, definition=None):
    definition = validate(definition, workspace) if definition is not None else selected(workspace)
    w = copy.deepcopy(workspace)
    w['task'].update(pick=definition['pick_object_id'], place=definition['target_object_id'],
                     success=copy.deepcopy(definition['success_condition']))
    w['task_definition'] = definition
    return w


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


def stamp_group(group, definition):
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
                stamp_group(episode, definition)


def evaluation_metrics(definitions, rows, simulation):
    result = {}
    for definition in definitions:
        samples = [r for r in rows if r['task_id'] == definition['task_id']]
        episodes = len(samples) if simulation else len({r['episode'] for r in samples})
        successes = sum(bool(r['success']) for r in samples) if simulation else None
        result[definition['task_id']] = {
            'episodes': episodes, 'steps': sum(r['steps'] for r in samples) if simulation else len(samples),
            'successes': successes, 'success_rate': successes / episodes if simulation and episodes else None,
            'success_measurement': 'contained_and_settled' if simulation else 'unmeasured',
            'language_instruction': definition['language_instruction']}
    return result
