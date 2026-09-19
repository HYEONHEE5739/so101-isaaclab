"""Workspace provenance checks and replay support for existing source-demo HDF5."""
import json
from .package import load, metadata


def verify_dataset(path, workspace):
    import h5py
    w=load(workspace)
    with h5py.File(path,'r') as f:
        args=json.loads(f['data'].attrs['env_args'])
    if args.get('workspace')!=metadata(w):raise ValueError('Demo workspace identity/hash differs; refusing replay')
    return w,args


def restore_initial_state(env, episode, quaternion_order='wxyz'):
    import copy
    from isaaclab.utils.math import convert_quat
    state=copy.deepcopy(episode.get_initial_state())
    if quaternion_order=='xyzw':
        for kind in ('articulation','rigid_object'):
            for obj in state.get(kind,{}).values():obj['root_pose'][...,3:7]=convert_quat(obj['root_pose'][...,3:7],to='wxyz')
    elif quaternion_order!='wxyz':raise ValueError('Unsupported root quaternion order')
    return env.reset_to(state,env_ids=None,is_relative=False)


def coordinate_contract(workspace):
    import hashlib
    from ...so101_dataset_contract import current_contract,MIMIC_SPACE
    w=load(workspace) if not isinstance(workspace,dict) else workspace
    result=current_contract(MIMIC_SPACE)
    files={k:v for k,v in w['manifest']['files'].items() if k.endswith(('.urdf','.usd')) and k.startswith('assets/SO101/')}
    result['robot_asset']={'scope':'URDF and composed USD files from immutable workspace; full assets tracked by workspace manifest',
                          'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    return result
