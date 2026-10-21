"""Existing Source lineage/split policy, conversion and training input checks."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from ..real2sim.storage import atomic
from ..so101_joint_mapping import JOINTS


def validate_partition(partition):
    """Verify Source-only partition evidence; does not attribute individual Mimic subtasks."""
    from ..real2sim.precision import fingerprint
    p=partition
    if (not isinstance(p,dict) or p.get('schema')!='so101.source-partition/1'
            or p.get('sha256')!=fingerprint({k:v for k,v in p.items() if k!='sha256'})):
        raise ValueError('Invalid Source Partition evidence/hash')
    inventory=p['inventory'];assignments=p['assignments'];ids=p['source_ids']
    if (p['split'] not in ('train','val') or len(set(ids))!=len(ids) or not ids
            or set(inventory)!=set(assignments) or not set(ids).issubset(inventory)
            or set(assignments.values())!={'train','val'} or any(assignments[i]!=p['split'] for i in ids)):
        raise ValueError('Source Partition membership/split conflict')
    groups={}
    for key,record in inventory.items():
        if not record.get('dataset_sha256') or not record.get('episode_id'):raise ValueError('Source identity evidence missing')
        if record.get('payload_sha256') and key!='source:'+record['payload_sha256']:raise ValueError('Source payload identity conflict')
        for group in record.get('groups',[]):
            if group in groups and groups[group]!=assignments[key]:raise ValueError('Source group leakage in partition assignment')
            groups[group]=assignments[key]
    return p


def validate_partition_episode(partition,lineage,input_evidence=None):
    p=validate_partition(partition)
    if not isinstance(lineage,dict):raise ValueError('Source Partition episode lineage evidence missing')
    if lineage.get('kind')=='source_episode':
        key=lineage['source_id']
        if key not in p['source_ids'] or p['inventory'][key]!=lineage['source']:raise ValueError('Episode Source is outside declared partition')
    elif lineage.get('kind')=='partition':
        from ..real2sim.precision import fingerprint
        proof=input_evidence or {};ids=proof.get('source_ids',[])
        if (lineage.get('partition_sha256')!=p['sha256'] or lineage.get('exact_source_ids') is not None
                or proof.get('verification')!='source_only_input' or not proof.get('sha256') or not ids or not set(ids).issubset(p['source_ids'])
                or proof.get('evidence_sha256')!=fingerprint({k:v for k,v in proof.items() if k!='evidence_sha256'})):
            raise ValueError('Generated Source Partition input/lineage is unverifiable')
    elif lineage.get('kind')=='mimic_segments':
        segments=lineage.get('segments',[])
        if not segments or lineage.get('partition_sha256')!=p['sha256']:raise ValueError('Mimic segment evidence missing')
        for segment in segments:
            original=segment.get('source',{})
            if original.get('kind')!='source_episode':raise ValueError('Unknown Mimic subtask Source')
            validate_partition_episode(p,original)
    else:raise ValueError('Unknown Source lineage kind')


def source_catalog(source,allow_unknown=False):
    """Read original identities; exported/reindexed subsets keep the recorded identities."""
    import hashlib
    from ..utils.artifact_identity import fingerprint
    from .curation import _group_id
    source=Path(source).resolve()
    if source.is_dir():
        manifest=json.loads((source/'manifest.json').read_text());args=json.loads((source/'project_provenance.json').read_text())
        rows=[(e['id'],e.get('source_lineage'),e['group_id']) for e in manifest['episodes']]
        partition=args.get('source_partition');derived=bool(args.get('source_dataset') or args.get('generation_method'))
        identity=fingerprint(source)
    else:
        import h5py
        digest=hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
        identity=digest.hexdigest()
        with h5py.File(source,'r') as h:
            args=json.loads(h['data'].attrs.get('env_args','{}'));partition=args.get('source_partition')
            derived=bool(args.get('source_dataset') or h['data'].attrs.get('generation_provenance'))
            rows=[(n,json.loads(g.attrs['source_lineage']) if 'source_lineage' in g.attrs else None,_group_id(source,n,g.attrs,args))
                  for n,g in h['data'].items() if n.startswith('demo_')]
    if partition:validate_partition(partition)
    records={}
    for name,lineage,group in rows:
        if lineage and lineage.get('kind')=='source_episode':record=lineage['source'];key=lineage['source_id']
        elif partition:
            if allow_unknown:return {},partition
            raise ValueError('Generated partition cannot be split again as independent Source episodes')
        elif derived:
            if allow_unknown:return {},None
            raise ValueError('Original Source lineage unavailable; split the original Source dataset first')
        else:
            # Content identity survives copying into another file, renumbering, and Source append.
            payload=hashlib.sha256()
            if source.is_dir():
                entry=next(e for e in manifest['episodes'] if e['id']==name);folder=(source/entry['path']).resolve()
                if not folder.is_relative_to(source):raise ValueError('Source episode outside dataset')
                for file in sorted(f for f in folder.iterdir() if f.suffix in ('.npy','.mp4')):
                    payload.update(file.name.encode())
                    with file.open('rb') as stream:
                        for block in iter(lambda:stream.read(2**20),b''):payload.update(block)
            else:
                with h5py.File(source,'r') as h:
                    datasets=[]
                    h['data'][name].visititems(lambda key,value:datasets.append((key,value)) if isinstance(value,h5py.Dataset) and not key.startswith(('obs/datagen_info/','obs/events/')) else None)
                    for key,value in sorted(datasets):
                        if value.dtype.kind not in 'biufc':continue
                        payload.update((key+str(value.shape)+str(value.dtype)).encode())
                        if not value.ndim:payload.update(np.asarray(value[()]).tobytes());continue
                        stride=max(1,int(8*2**20/max(1,np.prod(value.shape[1:])*value.dtype.itemsize)))
                        for start in range(0,len(value),stride):payload.update(np.ascontiguousarray(value[start:start+stride]).tobytes())
            if not source.is_dir():
                from ..utils.async_video_writer import external_video_path
                with h5py.File(source,'r') as h:
                    for camera,spec in sorted(json.loads(h['data'][name].attrs.get('external_videos','{}')).items()):
                        payload.update(camera.encode())
                        with external_video_path(source,spec).open('rb') as stream:
                            for block in iter(lambda:stream.read(2**20),b''):payload.update(block)
            payload_sha=payload.hexdigest()
            record={'dataset_sha256':identity,'episode_id':name,'payload_sha256':payload_sha,'groups':[group] if not group.startswith('UNKNOWN') else []}
            key='source:'+payload_sha
        if not partition:
            previous=next((r['source'] for r in records.values() if r['source_id']==key),None)
            if previous is not None:
                previous['groups']=sorted(set(previous.get('groups',[]))|set(record.get('groups',[])));record=previous
        records[name]={'kind':'source_episode','source_id':key,'source':record}
        if partition and (key not in partition['source_ids'] or partition['inventory'][key]!=record):
            raise ValueError('Source episode differs from partition inventory')
    return records,partition


def convert_dataset(source, output, options):
    """Compatibility API for legacy training callers; one converter owns all mapping."""
    import subprocess,sys
    from ..real2sim.workspaces.package import ROOT
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,str(ROOT/'scripts/tools/convert_isaac2lerobot.py'),
        '--input_file',str(Path(source).resolve()),'--root',str(output/'lerobot'),
        '--repo_id',options['repo_id'],'--action_source','joint_targets']
    if options.get('include_failed'):argv.append('--include_failed')
    subprocess.run(argv,check=True)
    info=json.loads((output/'lerobot/meta/info.json').read_text())
    return {'dataset':str(output/'lerobot'),'lerobot':str(output/'lerobot'),'episodes':info['total_episodes']}

def grasp_tracker(env):
    from ..real2sim.workspaces.grasp import StableGrasp
    return StableGrasp(1 / env.hz, env.backend.workspace['profile']['objects'][env.backend.metadata['pick']]['dimensions_m'])


def subtask_signals(env, tracker, tick):
    from scipy.spatial.transform import Rotation
    eef = env.backend.get_eef_pose()
    obj = env.backend.get_object_pose(env.backend.metadata['pick'])
    relative = np.linalg.inv(eef) @ obj
    left, right = env.backend.finger_contacts()
    grasp = tracker.update(tick, relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_quat(),
                           eef[:3, 3], obj[:3, 3], left, right)
    return {'grasp': bool(grasp), 'place': env.backend.is_success(),
            'tcp_distance_mm': float(np.linalg.norm(eef[:3, 3] - obj[:3, 3]) * 1000)}


def convert_hdf5(recipe_file, output):
    import h5py
    from ..data.legacy import JOINTS, POLICY_UNITS, UNITS, episode_arrays, load_manifest
    recipe_file=Path(recipe_file); r=json.loads(recipe_file.read_text()); root=Path(output)
    if root.exists(): raise FileExistsError('Use a new output directory')
    if r['action_semantics']!='absolute_target_after_observation': raise ValueError('Re-align action timestamps before import')
    if r['raw_joint_names']!=JOINTS: raise ValueError('Reorder joints explicitly before import')
    if r['arm_units'] not in ('rad','deg'): raise ValueError('Arm units must be declared')
    low,high=r['raw_gripper_range']
    if not np.isfinite([low,high]).all() or low==high: raise ValueError('Invalid gripper calibration endpoints')
    root.mkdir(parents=True); entries=[]
    for spec in r['episodes']:
        name=spec['id']
        if not name or Path(name).name!=name or name in ('.','..'): raise ValueError('Invalid episode id')
        dest=root/name; dest.mkdir()
        with h5py.File(recipe_file.parent/spec['file'],'r') as h:
            for key in ('state','action'):
                a=np.asarray(h[spec[key+'_key']],dtype=np.float32)
                if a.ndim!=2 or a.shape[1]!=6: raise ValueError('Expected [T,6] state/action')
                if r['arm_units']=='deg': a[:,:5]=np.deg2rad(a[:,:5])
                a[:,-1]=(a[:,-1]-low)/(high-low)
                np.save(dest/f'{key}.npy',a)
            for key in ('timestamp','action_timestamp'):
                np.save(dest/f'{key}.npy',np.asarray(h[spec[key+'_key']],dtype=np.float64))
            from ..utils.async_video_writer import VideoEpisode
            if r['image_order'] not in ('RGB','BGR'):raise ValueError('image_order must be RGB or BGR')
            cameras={camera:h[mapping['image_key']] for camera,mapping in spec['cameras'].items()}
            for images in cameras.values():
                if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3:
                    raise ValueError('RGB uint8 [T,H,W,3] required')
                if len(images)!=len(a):raise ValueError('RGB/numeric frame count mismatch')
            video=VideoEpisode(root,r['control_hz'])
            try:
                for index in range(len(a)):
                    video.append({camera:(images[index][...,::-1] if r['image_order']=='BGR' else images[index]) for camera,images in cameras.items()})
                video.finish(len(a))
                for camera,mapping in spec['cameras'].items():
                    (video.folder/(camera+'.mp4')).rename(dest/(camera+'.mp4'))
                    np.save(dest/f'{camera}_timestamp.npy',np.asarray(h[mapping['timestamp_key']],dtype=np.float64))
            finally:video.abort()
        e={k:spec[k] for k in ('id','split','group_id','instruction')}; e['path']=name; entries.append(e)
    m=dict(schema_version=1,joint_names=JOINTS,units=UNITS,control_hz=r['control_hz'],
           action_semantics=r['action_semantics'],episodes=entries,cameras=list(r['episodes'][0]['cameras']))
    path=root/'manifest.json'; path.write_text(json.dumps(m,indent=2))
    loaded=load_manifest(path)
    for e,spec in zip(loaded['episodes'],r['episodes']): episode_arrays(e,list(spec['cameras']),r['control_hz'])
    return path


def training_coordinates(artifact, workspace=None):
    """Separate measured joint limits from nominal gripper command endpoints."""
    from ..so101_joint_mapping import RobotCoordinates
    import hashlib
    import xml.etree.ElementTree as ET
    prov=artifact.get('provenance',{});endpoints=prov.get('canonical_coordinates') or {}
    model=(artifact.get('coordinates') or {}).get('simulation_model')
    limits=None
    if model:
        robot=model.get('robot_model') or {}
        workspace=workspace or prov.get('workspace_path') or prov.get('source_env_args',{}).get('workspace_path')
        if not workspace or not robot.get('urdf') or not robot.get('urdf_sha256'):
            raise ValueError('측정 관절 범위 확인에 필요한 기록된 Workspace/URDF 근거가 없습니다.')
        root=Path(workspace).expanduser().resolve();path=(root/robot['urdf']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=robot['urdf_sha256']:
            raise ValueError('기록된 실행 URDF와 현재 파일의 hash가 다릅니다. 측정 범위를 추정하지 않습니다.')
        limit=ET.parse(path).getroot().find("joint[@name='gripper']/limit")
        if limit is None:raise ValueError('기록된 URDF에 gripper 관절 범위가 없습니다.')
        limits=[float(limit.get(key)) for key in ('lower','upper')]
    return RobotCoordinates(endpoints.get('gripper_closed_rad'),endpoints.get('gripper_open_rad'),measured_gripper_limits=limits)


def partition_evidence(provenance, episode_index):
    """Converted merges retain the original partition/proof of every episode."""
    key = str(episode_index)
    return (provenance.get('episode_source_partitions', {}).get(key, provenance.get('source_partition')),
            provenance.get('episode_source_partition_inputs', {}).get(key, provenance.get('source_partition_input')))
