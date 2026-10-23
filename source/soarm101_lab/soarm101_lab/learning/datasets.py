"""Native learning input contracts, independent of historical split roles."""
import json
from pathlib import Path
from ..representation import describe, require_compatible
from ..so101_dataset_contract import read_contract, JOINT_SPACE


def inspect_dataset(path, *, full=True):
    ref=hub_reference(str(path))
    identity=None
    if ref:
        from huggingface_hub import HfApi,snapshot_download
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata,LeRobotDataset
        repo,revision=ref
        info=HfApi().dataset_info(repo,revision=revision)
        # A commit-specific official snapshot avoids reusing mutable LeRobot root
        # metadata from a different revision of the same repository.
        path=snapshot_download(repo,repo_type='dataset',revision=info.sha,
            allow_patterns=None if full else ['meta/**'])
        meta=LeRobotDatasetMetadata(repo,root=path,revision=info.sha)
        identity=dict(repo_id=repo,revision=info.sha)
        if full:
            dataset=LeRobotDataset(repo,root=path,revision=info.sha)
            dataset[0]
            del dataset
    path = Path(path).expanduser().resolve()
    if any(p.with_name('.'+p.name+'.workflow-lock').exists() for p in (path,*path.parents) if p.name):
        raise ValueError('Dataset is still being written')
    info = json.loads((path/'meta/info.json').read_text())
    coordinates = read_contract(path) if (path/'so101_coordinates.json').is_file() else None
    if coordinates is not None:describe(coordinates)
    features = info['features']
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('Installed LeRobot training requires Dataset v3.0; convert older versions with official dataset tools')
    for name in ('observation.state','action'):
        feature=features.get(name,{})
        shape=feature.get('shape',[])
        if len(shape)!=1 or not isinstance(shape[0],int) or shape[0]<1 or feature.get('dtype')!='float32':
            raise ValueError('Native training requires a nonempty float32 vector: '+name)
    if coordinates is not None:
        from ..so101_joint_mapping import JOINTS
        if coordinates['action_space']!=JOINT_SPACE:raise ValueError('Declared SO-101 actions must be joint radians')
        for name in ('observation.state','action'):
            if features[name].get('names')!=list(JOINTS) or features[name]['shape']!=[6]:
                raise ValueError('Declared SO-101 joint order/shape differs: '+name)
    cameras = [k for k,v in features.items() if k.startswith('observation.') and v.get('dtype') in ('video','image')]
    if any(features[k].get('dtype') not in ('video','image') or len(features[k].get('shape',[])) != 3 for k in cameras):
        raise ValueError('Dataset camera features missing or invalid')
    if float(info.get('fps',0)) <= 0 or int(info.get('total_frames',0)) <= 0 or int(info.get('total_episodes',0)) <= 0:
        raise ValueError('Dataset requires positive FPS, frames and episodes')
    if int(info.get('total_tasks',0))<1:raise ValueError('Dataset Task instructions missing')
    if full and not list((path/'data').rglob('*.parquet')):raise ValueError('Dataset Parquet missing')
    provenance = path/'workflow_provenance.json'
    provenance = json.loads(provenance.read_text()) if provenance.is_file() else {}
    return dict(path=str(path),format=info,coordinates=coordinates,provenance=provenance,
                tasks=provenance.get('task_definitions',[]),workspace=provenance.get('workspace'),type='dataset',hub=identity)


def leakage(train, validation):
    """Report only evidenced overlap; absence of evidence never proves independence."""
    same_path=Path(train['path']).resolve() == Path(validation['path']).resolve()
    a,b = train.get('provenance',{}),validation.get('provenance',{})
    def entries(p):
        result=set()
        def visit(value):
            if isinstance(value,dict):
                # These identify an original episode, unlike a partition or group digest.
                for key in ('source_id','curation_origin'):
                    if isinstance(value.get(key),str):result.add(value[key])
                for identity in value.get('exact_source_ids') or []:
                    if isinstance(identity,str):result.add(identity)
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(p.get('episode_source_lineage',{}))
        visit(p.get('episodes',[]))
        return result
    shared=entries(a)&entries(b)
    groups={g for g in set(a.get('source_groups',{}).values())&set(b.get('source_groups',{}).values())
            if isinstance(g,str) and g and not g.startswith('UNKNOWN')}
    warnings=[]
    if same_path:warnings.append('Train and Validation are the same Dataset Path; this is not held-out evaluation')
    if shared:warnings.append('Overlapping episode lineage detected: held-out independence is not established')
    if groups:warnings.append('Shared historical Source Groups; allocation is allowed but leakage is possible')
    warnings.append('Lineage coverage may be incomplete; no independence guarantee')
    return dict(warnings=warnings,shared_lineages=len(shared),shared_episode_origins=sorted(shared),shared_groups=sorted(groups),independence_proven=False)


def validate_datasets(train_path, validation_path=None, *, full=True):
    train=inspect_dataset(train_path,full=full)
    validation=inspect_dataset(validation_path,full=full) if validation_path else None
    evidence=None
    if validation:
        if validation['coordinates'] is not None and train['coordinates'] is not None:
            require_compatible(validation['coordinates'],train['coordinates'],JOINT_SPACE)
        if validation['format']['fps']!=train['format']['fps']:raise ValueError('Train/Validation FPS differs')
        def schema(record):
            return {k:{field:v.get(field) for field in ('dtype','shape','names')} for k,v in record['format']['features'].items()}
        if schema(validation)!=schema(train):raise ValueError('Train/Validation feature schema differs')
        evidence=leakage(train,validation)
    return train,validation,evidence


def inspect_policy(path, *, owned_lock=None):
    """Native LeRobot files only. This does not authorize SO-101 motion."""
    root=Path(path).expanduser().resolve()
    if (root/'pretrained_model').is_dir():root=root/'pretrained_model'
    if any(p.name and p.with_name('.'+p.name+'.workflow-lock').exists() and
           p.with_name('.'+p.name+'.workflow-lock')!=owned_lock for p in (root,*root.parents)):
        raise ValueError('Policy is still being written')
    for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (root/name).is_file():raise ValueError('Incomplete native Policy: '+name)
    config=json.loads((root/'config.json').read_text())
    if config.get('type') not in installed_policies():raise ValueError('Policy type not registered in installed LeRobot: '+str(config.get('type')))
    for name in ('policy_preprocessor','policy_postprocessor'):
        for step in json.loads((root/(name+'.json')).read_text())['steps']:
            if step.get('state_file') and not (root/step['state_file']).is_file():raise ValueError('Missing saved processor state: '+step['state_file'])
    coordinates=read_contract(root) if (root/'so101_coordinates.json').is_file() else None
    if coordinates is not None:describe(coordinates)
    prov=root/'workflow_provenance.json'
    provenance=json.loads(prov.read_text()) if prov.is_file() else {}
    return dict(path=str(root),type='policy',format=config,coordinates=coordinates,provenance=provenance,
                workspace=provenance.get('workspace'),tasks=provenance.get('task_definitions',[]))


def hub_reference(value):
    """Explicit remote identity; plain paths are always local."""
    if not isinstance(value,str) or not value.startswith('hf://'):return None
    from huggingface_hub.utils import validate_repo_id
    repo,sep,revision=value[5:].partition('@')
    validate_repo_id(repo)
    if sep and not revision:raise ValueError('Hub revision must be nonempty')
    return repo,revision or None

def resolve_model(value, revision=None):
    """Pin a complete native model snapshot in the official cache; no source edits."""
    ref=hub_reference(value)
    if ref is None and Path(value).expanduser().exists():
        if revision:raise ValueError('Revision applies to Hub repositories, not local model paths')
        return str(Path(value).expanduser().resolve()),None
    from huggingface_hub import HfApi,snapshot_download
    repo,selected=ref if ref else (value,revision)
    if revision and selected and selected!=revision:raise ValueError('Conflicting model revisions')
    info=HfApi().model_info(repo,revision=selected or revision)
    folder=snapshot_download(repo,repo_type='model',revision=info.sha,
        ignore_patterns=['.git/**','.env*','*.pt','training_state/**'])
    inspect_policy(folder)
    return str(folder),dict(repo_id=repo,revision=info.sha)


def validate_sim_features(config, path=None):
    """Narrow SO-101 runner capability check, separate from native Training support."""
    from lerobot.configs.types import FeatureType
    state=config.input_features.get('observation.state')
    action=config.output_features.get('action')
    if state is None or list(state.shape)!=[6] or action is None or list(action.shape)!=[6]:
        raise ValueError('SO-101 Sim Evaluation requires 6D state/action features')
    rename={}
    if path is not None:
        steps=json.loads((Path(path)/'policy_preprocessor.json').read_text())['steps']
        for step in steps:
            if step.get('registry_name')=='rename_observations_processor':rename=step.get('config',{}).get('rename_map',{})
    supplied={rename.get(k,k) for k in ('observation.images.side_cam','observation.images.wrist_cam')}
    present=[]
    for key,feature in config.input_features.items():
        if feature.type==FeatureType.VISUAL:
            if key not in supplied:
                if config.type=='smolvla':continue  # Upstream prepare_images owns missing-camera handling.
                raise ValueError('Policy camera key absent from SO-101 runtime/processor: '+key)
            shape=list(feature.shape)
            if len(shape)!=3 or shape[0]!=3 or (shape!=[3,480,640] and not (config.type=='smolvla' and config.resize_imgs_with_padding)):
                raise ValueError('Policy camera shape differs without an official resize path: '+key)
            present.append(key)
        elif key!='observation.state':raise ValueError('Unsupported Sim observation feature: '+key)
    if config.image_features and not present:raise ValueError('No Policy camera is provided by runtime/saved processor')

def installed_policies(python=None):
    """Read the installed official factory's dispatch table without importing models.

    Availability here means an upstream factory entry exists, not that optional
    model dependencies or a particular Dataset have been tested.
    """
    import ast,importlib.util
    if python is not None:
        candidates=sorted({p.resolve() for p in (Path(python).parent.parent/'lib').glob('python*/site-packages/lerobot/policies/factory.py')})
        if len(candidates)!=1:raise ValueError('Cannot locate installed LeRobot factory for '+str(python))
        factory_path=candidates[0]
    else:
        root=Path(importlib.util.find_spec('lerobot').origin).parent
        factory_path=root/'policies/factory.py'
    tree=ast.parse(factory_path.read_text())
    factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_policy_class')
    return sorted({node.comparators[0].value for node in ast.walk(factory)
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='name'
        and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)
        and isinstance(node.comparators[0].value,str)})
