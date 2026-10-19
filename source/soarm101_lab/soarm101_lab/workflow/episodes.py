"""Streaming RGB episodes plus lossless model-radian/replay telemetry."""
import copy
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from ..real2sim.storage import atomic
from ..so101_joint_mapping import JOINTS, POLICY_UNITS


class EpisodeWriter:
    def __init__(self, root, env, instruction, video_crf=None,storage='legacy'):
        self.common=storage=='common';self.dataset_writer=None
        self.root = Path(root).with_suffix('.hdf5') if self.common else Path(root)
        if self.common:
            if self.root.exists():raise FileExistsError(self.root)
            self.root.parent.mkdir(parents=True,exist_ok=True)
        else:self.root.mkdir(parents=True, exist_ok=False)
        self.env = env
        self.rows = []
        self.video = None
        import os
        self.video_crf = int(os.environ.get('SO101_VIDEO_CRF','18')) if video_crf is None else video_crf
        self.initial = None
        self.manifest = {'schema_version': 1, 'joint_names': list(JOINTS),
                         'units': list(POLICY_UNITS), 'control_hz': env.hz,
                         'action_semantics': 'absolute_target_after_observation', 'episodes': [],
                         'cameras': env.backend.workspace['task'].get('observation_cameras', ['side'])}
        self.provenance = {'schema': 'so101.minivla-dataset/1', 'backend': 'mujoco',
                           'workspace': env.backend.metadata['source'],
                           'scene': copy.deepcopy(env.backend.metadata), 'coordinates': env.coordinates.metadata(),
                           'instruction': instruction, 'source_group_prefix': uuid.uuid4().hex,
                           'scene_configuration':copy.deepcopy(env.backend.metadata.get('scene_configuration'))}
        from .tasks import selected, configured
        if not self.provenance['scene'].get('task_definition'):
            self.provenance['scene']['task_definition'] = selected(env.backend.workspace)
        if not self.provenance['scene_configuration']:
            scene = configured(env.backend.workspace, self.provenance['scene']['task_definition'],
                               env.backend.metadata['dynamic_objects'], backend='mujoco')['scene_configuration']
            self.provenance['scene_configuration'] = scene
            self.provenance['scene']['scene_configuration'] = copy.deepcopy(scene)
        w=env.backend.workspace
        self.provenance['workspace_path']=str(w['root']) if w.get('root') else None
        self.provenance['scene']['robot_base_T_world']=copy.deepcopy(w['profile']['robot_base']['T_world'])
        self.instruction = instruction
        self.flush_manifest()

    def resume(self, source):
        source = Path(source).resolve()
        if self.common:
            from .dataset import EpisodeDataset,DatasetWriter,recording_args
            previous=EpisodeDataset(source)
            if previous.args['workspace']!=self.provenance['workspace'] or previous.contract!=self.provenance['coordinate_contract']:
                raise ValueError('Resume Workspace/calibration/model differs')
            if previous.args.get('source_partition'):raise ValueError('Cannot append new Source to an immutable partition')
            self.dataset_writer=DatasetWriter(self.root,recording_args(self.provenance,self.manifest),self.provenance['coordinate_contract'])
            records=source_catalog(source,allow_unknown=True)[0]
            for old in previous.names:
                values,attrs,videos=previous.episode(old)
                if not attrs.get('source_lineage') and old in records:attrs['source_lineage']=json.dumps(records[old])
                if not previous.legacy:videos={k:{**v,'file':str(previous.path.parent/v['file'])} for k,v in videos.items()}
                name=self.dataset_writer.commit(values,attrs,videos)
                self.manifest['episodes'].append({'id':name,'path':name})
            self.provenance['resumed_from']=str(source);return
        manifest = json.loads((source / 'manifest.json').read_text())
        provenance = json.loads((source / 'project_provenance.json').read_text())
        from .tasks import recorded_scene
        old = recorded_scene({'provenance':provenance},self.env.backend.workspace)
        current = self.provenance.get('scene_configuration')
        if current and old['active_object_ids'] != current['active_object_ids']:
            raise ValueError('Resume active object set differs')
        if (provenance['workspace'] != self.provenance['workspace'] or
                provenance['scene']['model_sha256'] != self.provenance['scene']['model_sha256'] or
                manifest['units'] != self.manifest['units'] or manifest['control_hz'] != self.env.hz or
                any(provenance['coordinates'].get(k) != self.provenance['coordinates'].get(k)
                    for k in ('joint_order', 'gripper_closed_rad', 'gripper_open_rad')) or
                provenance['scene'].get('task_definition') != self.provenance['scene'].get('task_definition')):
            raise ValueError('Resume workspace/model/task/coordinates differ')
        previous_contract = provenance.get('coordinate_contract') or {}
        current_contract = self.provenance.get('coordinate_contract') or {}
        if any(previous_contract.get(key) != current_contract.get(key)
               for key in ('hardware_mapping', 'simulation_joint_frame','simulation_model')):
            raise ValueError('Resume hardware calibration/collection coordinates differ; use a new dataset')
        for entry in manifest['episodes']:
            folder = (source / entry['path']).resolve()
            if not folder.is_relative_to(source): raise ValueError('Episode outside dataset')
            name = f'episode_{len(self.manifest["episodes"]):04d}'
            shutil.copytree(folder, self.root / name)
            self.manifest['episodes'].append({**entry, 'id': name, 'path': name})
        self.provenance['resumed_from'] = str(source)
        self.flush_manifest()

    def close(self):
        self._close_video()
        if self.dataset_writer is not None:self.dataset_writer.close()

    def _close_video(self):
        if self.video is not None:
            self.video.abort(); self.video = None

    def begin(self):
        self._close_video()
        self.rows = []
        self.initial = self.env.backend.get_initial_state()

    def append(self, observation, model_target, requested_target=None):
        from ..utils.async_video_writer import VideoEpisode
        if self.video is None:self.video=VideoEpisode(self.root.parent if self.common else self.root,self.env.hz,self.video_crf)
        images=observation.get('camera_images')
        if images is None:images={'side':observation['images'][0]}
        if set(images)!=set(self.manifest['cameras']):raise ValueError('Recorded camera set differs')
        try:self.video.append(images)
        except BaseException:self.close();raise
        # obs comes before applying this target. Both timestamps are simulation time at that boundary.
        self.rows.append({'state': observation['state'].copy(),
                          'action': self.env.coordinates.to_policy(model_target),
                          'raw_state': observation['model_joint_rad'].copy(),
                          'raw_action': np.array(model_target, copy=True),
                          'requested_action': np.array(model_target if requested_target is None else requested_target, copy=True),
                          'timestamp': observation['timestamp'],
                          'eef_pose': observation['eef_pose'].copy(),
                          'object_poses': {n:observation['object_poses'][n].copy() for n in self.env.backend.metadata['dynamic_objects']},
                          'object_pose': observation['object_poses'][self.env.backend.metadata['pick']].copy()})

    def save(self, success, validation_only=False, success_source='unknown'):
        if not self.rows:
            raise ValueError('Cannot save an empty episode')
        if self.common:
            from .dataset import DatasetWriter,recording_args,recorded_mujoco_episode
            args=recording_args(self.provenance,self.manifest)
            if self.dataset_writer is None:self.dataset_writer=DatasetWriter(self.root,args,self.provenance['coordinate_contract'])
            else:self.dataset_writer.args.update(args)
            specs=self.video.finish(len(self.rows));videos={name+'_cam':{**s,'file':str((self.video.folder/s['file']).resolve())} for name,s in specs.items()}
            values=recorded_mujoco_episode(self.env,self.rows)
            logical=f'episode_{len(self.manifest["episodes"]):04d}'
            attrs={'episode_id':logical,'source_group':self.provenance['source_group_prefix']+'/'+logical,
                'success':bool(success),'validation_only':validation_only,'success_source':success_source,
                'legacy_initial_state':json.dumps(self.initial),'task_definition':json.dumps(self.provenance['scene']['task_definition']),
                'scene_configuration':json.dumps(self.provenance['scene_configuration'])}
            name=self.dataset_writer.commit(values,attrs,videos,video_staging=self.video)
            self.manifest['episodes'].append({'id':name,'path':name});self._close_video();self.begin();return name
        name = f'episode_{len(self.manifest["episodes"]):04d}'
        folder = self.video.folder
        try:videos=self.video.finish(len(self.rows))
        except BaseException:self.close();raise
        for key in ('state', 'action'):
            np.save(folder / f'{key}.npy', np.asarray([r[key] for r in self.rows], dtype=np.float32))
        for key in ('raw_state', 'raw_action', 'requested_action', 'eef_pose', 'object_pose'):
            np.save(folder / f'{key}.npy', np.asarray([r[key] for r in self.rows], dtype=np.float64))
        times = np.asarray([r['timestamp'] for r in self.rows], dtype=np.float64)
        for camera in self.manifest['cameras']:
            np.save(folder / (camera+'_timestamp.npy'), times)
        for key in ('timestamp', 'action_timestamp'):
            np.save(folder / f'{key}.npy', times)
        # Reuse the existing target matrix/timestamps; only distractors need another compact array.
        active=self.env.backend.metadata['dynamic_objects'];pick=self.env.backend.metadata['pick']
        telemetry={pick:{'file':'object_pose.npy','pose_format':'matrix4','frame':'workspace','timestamp_file':'timestamp.npy'}}
        others=[n for n in active if n!=pick]
        if others:
            from scipy.spatial.transform import Rotation
            poses=np.array([[r['object_poses'][n] for n in others] for r in self.rows])
            quat=Rotation.from_matrix(poses[...,:3,:3].reshape(-1,3,3)).as_quat().reshape(len(poses),len(others),4)[...,[3,0,1,2]]
            np.save(folder/'object_poses.npy',np.concatenate([poses[...,:3,3],quat],axis=-1).astype(np.float32))
            telemetry.update({n:{'file':'object_poses.npy','index':i,'pose_format':'xyz_wxyz','frame':'workspace','timestamp_file':'timestamp.npy'} for i,n in enumerate(others)})
        atomic(folder / 'replay.json', self.initial)
        atomic(folder / 'outcome.json', {'success': bool(success), 'validation_only': validation_only,
                                       'success_source':success_source,
                                       'coordinates': self.env.coordinates.metadata()})
        folder.rename(self.root / name)
        self.video = None
        entry = {'id': name, 'path': name, 'split': 'train', 'instruction': self.instruction, 'videos':videos,
            'object_telemetry':telemetry,
            'scene_configuration':copy.deepcopy(self.provenance.get('scene_configuration')),
            'task_definition':copy.deepcopy(self.provenance['scene'].get('task_definition')),
                 'group_id': self.provenance['source_group_prefix'] + '/' + name}
        self.manifest['episodes'].append(entry)
        try:self.flush_manifest()
        except BaseException:
            self.manifest['episodes'].pop();shutil.rmtree(self.root/name);raise
        self.begin()
        return name

    def flush_manifest(self):
        self.provenance['coordinates'] = self.env.coordinates.metadata()
        if self.common:return
        atomic(self.root / 'project_provenance.json', self.provenance)
        # Publish the episode index last, after finalized videos, numeric arrays and provenance.
        atomic(self.root / 'manifest.json', self.manifest)


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


def source_split(source,train_fraction=.8,seed=0,assignments=None,names=None):
    """Plan disjoint original Source pools; never modify the input dataset."""
    from ..real2sim.precision import fingerprint
    if not 0<float(train_fraction)<1 or type(seed) is not int or seed<0:raise ValueError('Invalid split ratio/seed')
    records,existing=source_catalog(source)
    if existing:raise ValueError('Recorded partition is locked; use the original Source for a new split')
    names=list(records) if names is None else list(names)
    if len(names)<2 or len(set(names))!=len(names) or not set(names).issubset(records):raise ValueError('Select at least two Source episodes')
    components=[]
    for name in sorted(names):
        lineage=records[name];keys={lineage['source_id'],*('group:'+g for g in lineage['source'].get('groups',[]))}
        matches=[c for c in components if c[1]&keys];members={name}
        for c in matches:members|=c[0];keys|=c[1];components.remove(c)
        components.append((members,keys))
    if len(components)<2:raise ValueError('Need at least two independent Source groups')
    components.sort(key=lambda component:min(records[n]['source_id'] for n in component[0]))
    if assignments is None:
        order=np.random.default_rng(seed).permutation(len(components));target=max(1,min(len(names)-1,int(len(names)*float(train_fraction)+.5)));chosen=set()
        for i in order[:-1]:
            if len(chosen)<target:chosen|=components[i][0]
        assignments={n:'train' if n in chosen else 'val' for n in names}
    if set(assignments)!=set(names) or set(assignments.values())!={'train','val'}:raise ValueError('Assignments must cover selected episodes with train and val')
    for members,_ in components:
        if len({assignments[n] for n in members})!=1:raise ValueError('Source leakage: duplicates cross Train/Validation')
    inventory={records[n]['source_id']:records[n]['source'] for n in names};by_id={records[n]['source_id']:assignments[n] for n in names};partitions={}
    for split in ('train','val'):
        payload={'schema':'so101.source-partition/1','split':split,'seed':seed,'train_fraction':float(train_fraction),
                 'inventory':inventory,'assignments':by_id,'source_ids':sorted(k for k,v in by_id.items() if v==split)}
        partitions[split]=validate_partition({**payload,'sha256':fingerprint(payload)})
    return {'assignments':assignments,'records':{n:records[n] for n in names},'partitions':partitions,
            'group_members':{n:sorted(members) for members,_ in components for n in members},
            'counts':{s:sum(v==s for v in assignments.values()) for s in ('train','val')},'leakage':False}


def export_source_splits(source,destination,train_fraction=.8,seed=0,assignments=None,names=None,*,nested=False):
    from .curation import export_subset
    plan=source_split(source,train_fraction,seed,assignments,names);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False);outputs={}
    try:
        for split in ('train','val'):
            path=destination/split if Path(source).is_dir() else destination/split/'dataset.hdf5' if nested else destination/(split+'.hdf5')
            chosen=[n for n,v in plan['assignments'].items() if v==split]
            outputs[split]=str(export_subset(source,path,chosen,partition=plan['partitions'][split],source_records=plan['records']))
        atomic(destination/'source_split.json',plan)
    except BaseException:
        shutil.rmtree(destination);raise
    return {**plan,'outputs':outputs}


def split_episodes(eligible,splits=None,require_validation=True):
    """Existing deterministic source-group split, shared by NPY and direct LeRobot."""
    if not eligible or (require_validation and len(eligible) < 2):
        raise ValueError('At least two eligible independent source groups for train/val required')
    eligible=copy.deepcopy(eligible);seen={};explicit=any(e.get('source_partition') for e in eligible)
    if explicit:
        if not all(e.get('source_partition') for e in eligible):raise ValueError('Cannot mix verified partition and unknown Source lineage')
        for entry in eligible:
            p=validate_partition(entry['source_partition']);split=p['split']
            validate_partition_episode(p,entry.get('source_lineage'),entry.get('source_partition_input'))
            entry['source_group']=entry.get('source_group',entry['group_id']);entry['group_id']='partition:'+p['sha256'];entry['split']=split
            for key in [*p['source_ids'],*('group:'+g for i in p['source_ids'] for g in p['inventory'][i].get('groups',[]))]:
                if key in seen and seen[key]!=split:raise ValueError('Source leakage: Train/Validation share original Source episodes/groups')
                seen[key]=split
        recorded={e['id']:e['split'] for e in eligible}
        if splits is not None and splits!=recorded:raise ValueError('Explicit Source split cannot be overwritten')
        splits=recorded
    elif any((e.get('source_lineage') or {}).get('kind')=='mimic_segments' for e in eligible):
        # Different subtask combinations may share one original demonstration.
        # Assign connected Source components together before any train/val split.
        parent=list(range(len(eligible)));owner={}
        def find(index):
            while parent[index]!=index:
                parent[index]=parent[parent[index]];index=parent[index]
            return index
        for index,entry in enumerate(eligible):
            lineage=entry.get('source_lineage') or {}
            sources=[s.get('source',{}) for s in lineage.get('segments',[])] if lineage.get('kind')=='mimic_segments' else [lineage]
            if not sources or any(s.get('kind')!='source_episode' for s in sources):
                raise ValueError('Unknown original Source lineage cannot be split independently')
            keys={entry['group_id']}
            for source in sources:
                keys.add(source['source_id']);keys.update('group:'+g for g in source['source'].get('groups',[]))
            for key in keys:
                if key in owner:parent[find(index)]=find(owner[key])
                owner[key]=index
        components={}
        for index,entry in enumerate(eligible):components.setdefault(find(index),[]).append(entry['group_id'])
        for index,entry in enumerate(eligible):
            entry['source_group']=entry['group_id'];entry['group_id']='sources:'+min(components[find(index)])
    groups = sorted({e['group_id'] for e in eligible})
    if splits is None:
        if require_validation and len(groups) < 2:
            raise ValueError('Need independent train/val source groups: 독립 원본 Group이 2개 이상 필요합니다. 같은 Source에서 생성한 episode는 나눌 수 없습니다.')
        # Deterministic group split. No frame split or generated-variant leakage.
        assignment = {g: 'train' for g in groups}
        if len(groups) >= 2: assignment[groups[-1]] = 'val'
        if len(groups) >= 3:
            assignment[groups[-2]] = 'test'
        splits = {e['id']: assignment[e['group_id']] for e in eligible}
    if set(splits) != {e['id'] for e in eligible}:
        raise ValueError('Split map must cover exactly the eligible episode IDs')
    entries = [{**{k: v for k, v in e.items() if not k.startswith('_')}, 'split': splits[e['id']]} for e in eligible]
    grouped = {}
    for entry in entries:
        if entry['split'] not in ('train', 'val', 'test'):
            raise ValueError('Prepared training split must be train/val/test')
        if entry['group_id'] in grouped and grouped[entry['group_id']] != entry['split']:
            raise ValueError('Source group leakage')
        grouped[entry['group_id']] = entry['split']
    if require_validation and not {'train', 'val'}.issubset(grouped.values()):
        raise ValueError('Training and validation groups both required: 학습·검증 데이터가 모두 필요합니다. Train 전용 입력은 Validation Dataset에 별도로 변환한 Val 데이터를 선택하세요. 이미 분할된 Train을 자동 재분할하지 않습니다.')
    return entries,grouped


def prepare_dataset(source, output, splits=None, include_failed=False, require_validation=True):
    from ..learning.minivla.data import load_manifest, episode_arrays, fit_statistics
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new prepared dataset output')
    manifest = load_manifest(source / 'manifest.json')
    provenance = json.loads((source / 'project_provenance.json').read_text())
    eligible = []
    for entry in manifest['episodes']:
        folder = Path(entry['_path'])
        outcome = json.loads((folder / 'outcome.json').read_text())
        if outcome.get('validation_only'):
            raise ValueError('Synthetic smoke episodes cannot enter training')
        if outcome['success'] or include_failed:
            episode_arrays(entry, manifest.get('cameras',['side']), manifest['control_hz'])
            eligible.append(entry)
    entries,grouped=split_episodes(eligible,splits,require_validation)
    output.mkdir(parents=True)
    for original, entry in zip(eligible, entries):
        shutil.copytree(original['_path'], output / entry['path'])
    cleaned = {**manifest, 'episodes': entries}
    atomic(output / 'manifest.json', cleaned)
    scenes = [e.get('scene_configuration') or provenance.get('scene_configuration') for e in entries]
    scenes = list({s['sha256']:s for s in scenes if s}.values())
    atomic(output / 'project_provenance.json', {**provenance, 'scene_configurations':scenes, 'prepared_from': str(source.resolve()),
                                               'include_failed': include_failed})
    if any(e['split']=='train' for e in entries):
        fit_statistics(output / 'manifest.json', output / 'statistics.json', manifest.get('cameras',['side']))
    else:
        atomic(output/'statistics.json',{'status':'not_fitted','reason':'Validation-only partition; normalization is fitted from Train at training time'})
    return {'dataset': str(output), 'episodes': len(entries), 'groups': grouped}


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
    from ..learning.minivla.data import JOINTS, POLICY_UNITS, UNITS, episode_arrays, load_manifest
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


def validate_training_inputs(artifact, options):
    """Metadata-only preflight shared by readiness and worker; never invent lineage."""
    from ..representation import require_compatible
    from ..so101_dataset_contract import JOINT_SPACE
    validation=options.get('validation_artifact');records=[artifact]+([validation] if validation else [])
    evidence=[]
    for source,record in enumerate(records):
        prov=record.get('provenance',{});partition=prov.get('source_partition');groups=prov.get('source_groups',{})
        if options.get('source_groups') and options['source_groups']!=groups:
            raise ValueError('수동 Source Group은 기록된 원본 계보와 일치해야 합니다.')
        if not partition and (not groups or any(str(g).startswith('UNKNOWN') for g in groups.values())):
            raise ValueError('MiniVLA Direct split requires verified source_groups; 원본 Group 근거가 없습니다.')
        if not partition and not options.get('legacy_group_split'):
            raise ValueError('명시적 Source Partition이 없습니다. 독립 원본 Group 기록이 있는 데이터만 고급 Legacy 단일 데이터 Group 분리를 사용하세요.')
        info=json.loads((Path(record['path'])/'meta/info.json').read_text())
        require_compatible(record['coordinates'],record['coordinates'],JOINT_SPACE)
        coords=prov.get('canonical_coordinates') or {}
        endpoints=[coords.get(k) for k in ('gripper_closed_rad','gripper_open_rad')]
        if coords.get('joint_order')!=list(JOINTS) or any(v is None for v in endpoints) or not np.isfinite(endpoints).all() or endpoints[1]<=endpoints[0]:
            raise ValueError('관절 순서·Gripper 정규화 계약이 없거나 잘못되었습니다. 검증된 원본을 다시 변환하세요.')
        training_coordinates(record,options.get('workspace'))
        if not np.isfinite(info['fps']) or info['fps']<=0:
            raise ValueError('Dataset FPS가 유효하지 않습니다.')
        for key in ('observation.state','action'):
            feature=info.get('features',{}).get(key,{})
            if feature.get('shape')!=[6] or feature.get('names')!=list(JOINTS):
                raise ValueError('관측·action의 6축 관절 순서가 맞지 않습니다: '+key)
        for i in range(info['total_episodes']):
            episode_partition, input_proof = partition_evidence(prov, i)
            evidence.append(dict(id=f'{source}_{i}',group_id=groups.get(str(i),'UNKNOWN'),
                source_partition=episode_partition,source_lineage=prov.get('episode_source_lineage',{}).get(str(i)),
                source_partition_input=input_proof))
    if validation:
        a,b=records
        if (a.get('provenance',{}).get('source_partition') or {}).get('split')!='train' or (b.get('provenance',{}).get('source_partition') or {}).get('split')!='val':
            raise ValueError('Train/Val 역할에 맞는 Source Partition을 선택하세요.')
        require_compatible(a['coordinates'],b['coordinates'],JOINT_SPACE)
        infos=[json.loads((Path(x['path'])/'meta/info.json').read_text()) for x in records]
        cameras=lambda info:{k:{f:v.get(f) for f in ('dtype','shape','names')} for k,v in info['features'].items() if k.startswith('observation.images.')}
        coords=lambda x:{k:x.get('provenance',{}).get('canonical_coordinates',{}).get(k) for k in ('joint_order','gripper_closed_rad','gripper_open_rad')}
        contract=lambda x:{k:v for k,v in x['coordinates'].items() if k!='source_dataset'}
        if a.get('workspace')!=b.get('workspace') or contract(a)!=contract(b) or infos[0]['fps']!=infos[1]['fps'] or cameras(infos[0])!=cameras(infos[1]) or coords(a)!=coords(b):
            raise ValueError('Train/Val의 Workspace·Robot/관절·카메라·FPS·Gripper 계약이 다릅니다.')
        if a.get('tasks')!=b.get('tasks'):raise ValueError('Train/Val Task 계약이 다릅니다.')
    split_episodes(evidence)
    return True
