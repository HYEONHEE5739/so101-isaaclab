"""Workspace provenance and coordinate checks for source-demo HDF5."""
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
    from ...workflow.tasks import recorded_scene, from_artifact
    from .tasks import configured
    artifact = {'path':str(path),'env_args':args,'workspace':args.get('workspace'),'backend':'isaac'}
    scene = recorded_scene(artifact,w)
    definitions = from_artifact(artifact,w)
    definition = scene.get('task_definition') or (definitions[0] if len(definitions)==1 else None)
    w = configured(w,definition,scene['active_object_ids'],backend='isaac',reset=scene['reset'])
    return w,args




def coordinate_contract(workspace):
    import hashlib
    from ...so101_dataset_contract import current_contract,MIMIC_SPACE
    w=load(workspace) if not isinstance(workspace,dict) else workspace
    result=current_contract(MIMIC_SPACE)
    files={k:v for k,v in w['manifest']['files'].items() if k.endswith(('.urdf','.usd')) and k.startswith('assets/SO101/')}
    result['robot_asset']={'scope':'URDF and composed USD files from immutable workspace; full assets tracked by workspace manifest',
                          'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    return result
