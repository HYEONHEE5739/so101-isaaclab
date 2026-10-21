"""Native learning input contracts, independent of historical split roles."""
import json
from pathlib import Path
from ..representation import describe, require_compatible
from ..so101_dataset_contract import read_contract, JOINT_SPACE


def inspect_dataset(path):
    path = Path(path).expanduser().resolve()
    if any(p.with_name('.'+p.name+'.workflow-lock').exists() for p in (path,*path.parents) if p.name):
        raise ValueError('Dataset is still being written')
    info = json.loads((path/'meta/info.json').read_text())
    coordinates = read_contract(path)
    describe(coordinates)
    if coordinates['action_space'] != JOINT_SPACE:raise ValueError('Training requires model joint-radian actions')
    features = info['features']
    from ..so101_joint_mapping import JOINTS
    for name in ('observation.state','action'):
        if features.get(name,{}).get('names')!=list(JOINTS):raise ValueError('Dataset joint order differs: '+name)
        if features.get(name,{}).get('shape') != [6] or features[name].get('dtype') != 'float32':
            raise ValueError('Expected float32 six-joint feature: '+name)
    cameras = [k for k in features if k.startswith('observation.images.')]
    if not cameras or any(features[k].get('dtype') not in ('video','image') or len(features[k].get('shape',[])) != 3 for k in cameras):
        raise ValueError('Dataset camera features missing or invalid')
    if float(info.get('fps',0)) <= 0 or int(info.get('total_frames',0)) <= 0 or int(info.get('total_episodes',0)) <= 0:
        raise ValueError('Dataset requires positive FPS, frames and episodes')
    if int(info.get('total_tasks',0))<1:raise ValueError('Dataset Task instructions missing')
    if not list((path/'data').rglob('*.parquet')):raise ValueError('Dataset Parquet missing')
    provenance = path/'workflow_provenance.json'
    provenance = json.loads(provenance.read_text()) if provenance.is_file() else {}
    return dict(path=str(path),format=info,coordinates=coordinates,provenance=provenance,
                tasks=provenance.get('task_definitions',[]),workspace=provenance.get('workspace'),type='dataset')


def leakage(train, validation):
    """Report only evidenced overlap; absence of evidence never proves independence."""
    if Path(train['path']).resolve() == Path(validation['path']).resolve():
        raise ValueError('Train and Validation must be different Dataset Paths')
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
    if shared:warnings.append('Overlapping episode lineage detected: held-out independence is not established')
    if groups:warnings.append('Shared historical Source Groups; allocation is allowed but leakage is possible')
    warnings.append('Lineage coverage may be incomplete; no independence guarantee')
    return dict(warnings=warnings,shared_lineages=len(shared),shared_episode_origins=sorted(shared),shared_groups=sorted(groups),independence_proven=False)


def validate_datasets(train_path, validation_path=None):
    train=inspect_dataset(train_path)
    validation=inspect_dataset(validation_path) if validation_path else None
    evidence=None
    if validation:
        require_compatible(validation['coordinates'],train['coordinates'],JOINT_SPACE)
        if validation['format']['fps']!=train['format']['fps']:raise ValueError('Train/Validation FPS differs')
        def schema(record):
            return {k:{field:v.get(field) for field in ('dtype','shape','names')} for k,v in record['format']['features'].items()}
        if schema(validation)!=schema(train):raise ValueError('Train/Validation feature schema differs')
        evidence=leakage(train,validation)
    return train,validation,evidence
