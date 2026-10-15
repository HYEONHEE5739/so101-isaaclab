"""Workspace provenance checks and replay support for existing source-demo HDF5."""
import json
from .package import load, metadata


def verify_dataset(path, workspace):
    import h5py
    w=load(workspace)
    with h5py.File(path,'r') as f:
        args=json.loads(f['data'].attrs['env_args'])
    if args.get('workspace')!=metadata(w):raise ValueError('Demo workspace identity/hash differs; refusing replay')
    if args.get('so101_coordinates') is not None:
        from ...representation import require_compatible
        require_compatible(args['so101_coordinates'], coordinate_contract(w))
    from ...workflow.tasks import recorded_scene, configured, from_artifact
    artifact = {'path':str(path),'env_args':args,'workspace':args.get('workspace'),'backend':'isaac'}
    scene = recorded_scene(artifact,w)
    definitions = from_artifact(artifact,w)
    definition = scene.get('task_definition') or (definitions[0] if len(definitions)==1 else None)
    w = configured(w,definition,scene['active_object_ids'],backend='isaac')
    return w,args


def restore_initial_state(env, episode, quaternion_order='wxyz'):
    import copy
    from isaaclab.utils.math import convert_quat
    state=copy.deepcopy(episode.get_initial_state())
    if quaternion_order=='xyzw':
        for kind in ('articulation','rigid_object'):
            for obj in state.get(kind,{}).values():obj['root_pose'][...,3:7]=convert_quat(obj['root_pose'][...,3:7],to='wxyz')
    elif quaternion_order!='wxyz':raise ValueError('Unsupported root quaternion order')
    import torch
    def on_device(value):
        if isinstance(value, torch.Tensor): return value.to(env.device)
        if isinstance(value, dict): return {k: on_device(v) for k, v in value.items()}
        return value
    return env.reset_to(on_device(state),env_ids=None,is_relative=False)


def coordinate_contract(workspace):
    import hashlib
    from ...so101_dataset_contract import current_contract,MIMIC_SPACE
    w=load(workspace) if not isinstance(workspace,dict) else workspace
    result=current_contract(MIMIC_SPACE)
    files={k:v for k,v in w['manifest']['files'].items() if k.endswith(('.urdf','.usd')) and k.startswith('assets/SO101/')}
    result['robot_asset']={'scope':'URDF and composed USD files from immutable workspace; full assets tracked by workspace manifest',
                          'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    return result


def replay_fields(data):
    """Read both source-recorder and Isaac ActionStateRecorder layouts, without remapping."""
    observations = data['obs'].get('policy', data['obs'])
    states = data['states']
    joints = states['joint_pos'] if 'joint_pos' in states else states['articulation']['robot']['joint_position']
    objects = data.get('workspace_states')
    if objects is None:
        objects = {name: state['root_pose'] for name, state in states.get('rigid_object', {}).items()}
    count = len(data['joint_targets'])
    if len(joints) != count or not count:
        raise ValueError('Replay state/action lengths differ or are empty')
    cameras = [n for n in observations if n.endswith('_cam')]
    if not cameras: raise ValueError('Replay requires camera observations')
    for name in cameras:
        if name not in observations or len(observations[name]) != count:
            raise ValueError('Replay requires synchronized camera observations')
    if any(len(value) != count for value in objects.values()):
        raise ValueError('Replay object/action lengths differ')
    return observations, joints, objects
