"""Streaming RGB episodes plus lossless model-radian/replay telemetry."""
import copy
import json
from pathlib import Path
import shutil
import uuid
import time

import numpy as np

from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata
from ..so101_joint_mapping import JOINTS, POLICY_UNITS


class EpisodeWriter:
    def __init__(self, root, env, instruction, video_crf=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
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
        if self.video is not None:
            self.video.abort(); self.video = None

    def begin(self):
        self.close()
        self.rows = []
        self.initial = self.env.backend.get_initial_state()

    def append(self, observation, model_target, requested_target=None):
        from ..utils.async_video_writer import VideoEpisode
        if self.video is None:self.video=VideoEpisode(self.root,self.env.hz,self.video_crf)
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
    else:raise ValueError('Unknown Source lineage kind')


def source_catalog(source,allow_unknown=False):
    """Read original identities; exported/reindexed subsets keep the recorded identities."""
    import hashlib
    from .artifacts import fingerprint
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


def export_source_splits(source,destination,train_fraction=.8,seed=0,assignments=None,names=None):
    from .curation import export_subset
    plan=source_split(source,train_fraction,seed,assignments,names);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False);outputs={}
    for split in ('train','val'):
        path=destination/(split if Path(source).is_dir() else split+'.hdf5')
        chosen=[n for n,v in plan['assignments'].items() if v==split]
        outputs[split]=str(export_subset(source,path,chosen,partition=plan['partitions'][split],source_records=plan['records']))
    atomic(destination/'source_split.json',plan)
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
    groups = sorted({e['group_id'] for e in eligible})
    if splits is None:
        if require_validation and len(groups) < 2:
            raise ValueError('Need independent train/val source groups')
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
        raise ValueError('Training and validation groups both required')
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


def annotate_episode(dataset, episode, boundaries):
    """Video/state index annotation only; never claims to produce MimicGen DatagenInfo."""
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text())
    if episode not in {e['id'] for e in manifest['episodes']} or Path(episode).name != episode:
        raise ValueError('Unknown episode ID')
    path = Path(dataset) / episode
    length = len(np.load(path / 'state.npy', mmap_mode='r', allow_pickle=False))
    expected = ('approach', 'grasp', 'transport', 'place')
    if list(boundaries) != list(expected):
        raise ValueError('Boundary order must be approach, grasp, transport, place')
    values = list(boundaries.values())
    if any(type(v) is not int or not 0 <= v < length for v in values) or values != sorted(set(values)):
        raise ValueError('Strictly increasing integer frame boundaries inside episode required')
    destination = path / 'annotation.json'
    if destination.exists():
        raise FileExistsError('Annotation exists; keep originals and use a dataset copy for revisions')
    atomic(destination, {'schema': 'so101.subtask-boundaries/1', 'frame_count': length,
                         'boundaries': copy.deepcopy(boundaries), 'scope': 'manual dataset-only boundaries'})
    return {'annotation': str(destination)}


def convert_dataset(source, output, options):
    """Read canonical input without RGB copies and reuse the existing LeRobot converter."""
    import subprocess
    import sys
    import h5py
    from ..real2sim.workspaces.package import ROOT, load
    from ..real2sim.workspaces.demo import coordinate_contract
    from ..so101_dataset_contract import write_contract, JOINT_SPACE
    # Select read-only source entries without creating another canonical RGB dataset.
    from ..learning.minivla.data import load_manifest,episode_arrays
    dataset=Path(source).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(dataset/'manifest.json')
    manifest.setdefault('cameras',['side'])
    eligible=[]
    for entry in manifest['episodes']:
        outcome=json.loads((Path(entry['_path'])/'outcome.json').read_text())
        if outcome.get('validation_only'):raise ValueError('Synthetic smoke episodes cannot enter training')
        if outcome['success'] or options.get('include_failed',False):
            episode_arrays(entry,manifest['cameras'],manifest['control_hz']);eligible.append(entry)
    entries,groups=split_episodes(eligible,require_validation=False)
    manifest={**manifest,'episodes':entries}
    result={'episodes':len(entries),'groups':groups}
    provenance = json.loads((dataset / 'project_provenance.json').read_text())
    if manifest['control_hz'] != round(manifest['control_hz']):
        raise ValueError('LeRobot conversion requires an integer camera/control FPS')
    contract = provenance.get('coordinate_contract', coordinate_contract(load(options['workspace'])))
    contract = {**contract, 'action_space': JOINT_SPACE}
    intermediate = output / 'conversion_input.hdf5'
    definition = provenance['scene']['task_definition']
    definitions = list({e.get('task_definition',definition)['task_id']:e.get('task_definition',definition) for e in manifest['episodes']}.values())
    with h5py.File(intermediate, 'w') as h:
        data = h.create_group('data')
        data.attrs['env_args'] = json.dumps({'workspace': provenance['workspace'], 'task_definitions': definitions,
            'source_partition':provenance.get('source_partition'),'source_partition_input':provenance.get('source_partition_input'),
            'scene_configuration':provenance.get('scene_configuration'),'canonical_coordinates':provenance['coordinates'],'control_dt':1/manifest['control_hz'],'observation_timestamp_path':'obs/timestamp'})
        for i, entry in enumerate(manifest['episodes']):
            folder = dataset / entry['path']; group = data.create_group(f'demo_{i}')
            group.attrs['task_definition'] = json.dumps(entry.get('task_definition') or definition)
            group.attrs['scene_configuration'] = json.dumps(entry.get('scene_configuration') or provenance.get('scene_configuration'))
            outcome=json.loads((folder/'outcome.json').read_text())
            group.attrs['success']=bool(outcome['success']);group.attrs['success_source']=outcome.get('success_source','unknown')
            group.attrs['source_group']=entry.get('source_group',entry['group_id'])
            if entry.get('source_lineage'):group.attrs['source_lineage']=json.dumps(entry['source_lineage'])
            if (folder/'annotation.json').is_file():group.attrs['annotation']= (folder/'annotation.json').read_text()
            group.attrs['pose_reference']=json.dumps({'source_dataset':str(dataset.resolve()),'episode':entry['path'],'object_telemetry':entry.get('object_telemetry')})
            actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
            group.create_dataset('actions', data=actions)
            group.create_dataset('joint_targets', data=actions)
            group.create_dataset('obs/policy/joint_pos', data=np.load(folder / 'raw_state.npy', allow_pickle=False))
            group.create_dataset('obs/timestamp',data=np.load(folder/'timestamp.npy',allow_pickle=False))
            # Adapter contains numeric arrays and read-only camera references, never RGB copies.
            from ..utils.async_video_writer import VideoFrames
            refs={}
            for view in manifest.get('cameras',['side','wrist']):
                video=folder/(view+'.mp4');npy=folder/(view+'.npy')
                path=video if video.exists() else npy
                value=VideoFrames(path,len(actions),manifest['control_hz']) if video.exists() else np.load(path,mmap_mode='r')
                refs[view+'_cam']={'file':str(path.resolve()),'source_reference':True,
                    'format':'mp4' if video.exists() else 'npy','frames':len(actions),'fps':manifest['control_hz'],
                    'shape':list(value.shape[1:])}
            group.attrs['external_videos']=json.dumps(refs)
    write_contract(intermediate, contract)
    subprocess.run([sys.executable, str(ROOT / 'scripts/tools/convert_isaac2lerobot.py'),
        '--input_file', str(intermediate), '--root', str(output / 'lerobot'),
        '--repo_id', options['repo_id'], '--task', definition['language_instruction'],
        '--fps', str(round(manifest['control_hz'])), '--action_source', 'joint_targets'], check=True)
    converted_provenance = output / 'lerobot/workflow_provenance.json'
    atomic(converted_provenance, {**json.loads(converted_provenance.read_text()), 'workspace': provenance['workspace'],
        'task_definitions': definitions, 'backend': 'mujoco', 'source': str(Path(source).resolve()),
        'canonical_episodes':manifest['episodes'],
        'source_groups': {str(i): e.get('source_group',e['group_id']) for i, e in enumerate(manifest['episodes'])},
        'generation_method': provenance.get('generation_method'),
        'scene_configuration':provenance.get('scene_configuration'),
        'scene_configurations':provenance.get('scene_configurations',[]),
        'episode_scene_configurations':{str(i):e.get('scene_configuration') or provenance.get('scene_configuration') for i,e in enumerate(manifest['episodes'])}})
    # Conversion adapter is temporary, never a registered artifact.
    intermediate.unlink()
    from ..so101_dataset_contract import sidecar
    sidecar(intermediate).unlink(missing_ok=True)
    return {**result, 'dataset': str(output / 'lerobot'), 'lerobot': str(output / 'lerobot')}

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


def annotate(request, output, publish, bridge, environment_factory, preview):
    """Replay and annotate grasp/place, using the shared stable-grasp detector."""
    import shutil
    from ..learning.minivla.data import load_manifest
    source = Path(request['input'])
    provenance = json.loads((source / 'project_provenance.json').read_text())
    if request.get('workspace') and provenance['workspace'] != metadata(load(request['workspace'])):
        raise ValueError('Dataset workspace identity differs')
    manifest = load_manifest(source / 'manifest.json')
    destination = output / 'dataset'; destination.mkdir()
    accepted, rejected = [], []
    scene = provenance['scene']
    from .tasks import recorded_scene
    if request.get('workspace'):
        inherited = recorded_scene({'provenance':provenance},load(request['workspace']))
        request = {**request,'options':{**request['options'],'active_object_ids':inherited['active_object_ids']}}
    request = {**request, 'options': {**request['options'], 'task_definition': scene.get('task_definition'),
        'physics_profile': scene.get('physics_profile', 'legacy_urdf'),
        'full_workspace': scene.get('full_workspace', False),
        'width': scene['camera']['resolution'][0], 'height': scene['camera']['resolution'][1]}}
    with environment_factory(request, output, provenance['coordinates'], manifest['control_hz']) as env:
        emit = preview(env, request, publish)
        for episode_index, entry in enumerate(manifest['episodes']):
            folder = Path(entry['_path'])
            env.backend.restore_state(json.loads((folder / 'replay.json').read_text()))
            actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
            raw = np.load(folder / 'raw_state.npy', allow_pickle=False)
            tracker = grasp_tracker(env)
            marks, skipped, position = [], [], [0]
            if bridge:
                bridge.callbacks['mark'] = lambda: marks.append(position[0])
                bridge.callbacks['skip'] = lambda: skipped.append(True)
                bridge.details['marks'] = marks
            grasp_at = place_at = None
            for i, target in enumerate(actions):
                position[0] = i
                if bridge: bridge.before_step()
                if skipped: break
                if not np.allclose(env.backend.get_state()['joint_pos'], raw[i], atol=1e-5):
                    raise ValueError('Annotation replay diverged from recorded state')
                env.backend.step(target)
                signals = subtask_signals(env, tracker, i)
                if signals['grasp'] and grasp_at is None: grasp_at = i
                if signals['place'] and grasp_at is not None: place_at = i
                emit(env.backend.get_rgb(), {'steps': i + 1, 'episode_index': episode_index + 1,
                     'episode_total': len(manifest['episodes']), 'task_signals': signals, 'marks': list(marks)})
                if request['options'].get('realtime', True): time.sleep(1 / env.hz)
            if not request['options'].get('auto', True):
                grasp_at = marks[0] if len(marks) == 1 else None
                place_at = len(actions) - 1 if env.backend.is_success() else None
            if skipped or not env.backend.is_success() or grasp_at is None or place_at is None or not 0 <= grasp_at < place_at < len(actions):
                rejected.append({'episode': entry['id'], 'reason': 'ordered grasp/place boundaries or successful placement missing'})
                continue
            dest = destination / entry['path']
            shutil.copytree(folder, dest)
            atomic(dest / 'annotation.json', {'schema': 'so101.mujoco-subtasks/1', 'grasp': grasp_at,
                'place': place_at, 'frame_count': len(actions),
                'method': 'stable_contact_replay' if request['options'].get('auto', True) else 'manual_grasp_and_measured_place'})
            accepted.append({k: v for k, v in entry.items() if k != '_path'})
    atomic(destination / 'manifest.json', {**manifest, 'episodes': accepted})
    atomic(destination / 'project_provenance.json', {**provenance, 'annotated_from': str(source.resolve())})
    return {'passed': bool(accepted), 'dataset': str(destination), 'accepted': len(accepted), 'rejected': rejected}


def datagen(request, output, publish, bridge, environment_factory, preview):
    """Object-relative retargeting with bounded IK and physical success filtering."""
    from ..learning.minivla.data import load_manifest
    from ..real2sim.workspaces.reset import datagen_seed
    source = Path(request['input']); provenance = json.loads((source / 'project_provenance.json').read_text())
    if request.get('workspace') and provenance['workspace'] != metadata(load(request['workspace'])):
        raise ValueError('Dataset workspace identity differs')
    from .tasks import recorded_scene
    inherited = recorded_scene({'provenance':provenance},load(request['workspace']))
    request = {**request,'options':{**request['options'],'active_object_ids':inherited['active_object_ids'],
        'task_definition':inherited['task_definition'], 'physics_profile':provenance['scene'].get('physics_profile','legacy_urdf'),
        'full_workspace':provenance['scene'].get('full_workspace',False),
        'width':provenance['scene']['camera']['resolution'][0], 'height':provenance['scene']['camera']['resolution'][1],
        'instruction':inherited['task_definition']['language_instruction']}}
    manifest = load_manifest(source / 'manifest.json')
    options = request['options']
    count = int(options.get('num_successful_demos', 10))
    seed = datagen_seed(load(request['workspace'])['task']['reset']['seed'], options.get('generation_seed'))
    rng = np.random.default_rng(seed)
    attempts = []
    with environment_factory(request, output, provenance['coordinates'], manifest['control_hz']) as env:
        emit = preview(env, request, publish)
        writer = EpisodeWriter(output / 'dataset', env, options['instruction'])
        if 'coordinate_contract' in provenance:
            writer.provenance['coordinate_contract'] = copy.deepcopy(provenance['coordinate_contract'])
        if options.get('resume_data'):
            resumed = json.loads((Path(options['resume_data']) / 'project_provenance.json').read_text())
            if resumed.get('source_dataset') != str(source.resolve()) or resumed.get('generation_method') != 'object_relative_bounded_ik':
                raise ValueError('Resume Datagen requires the same annotated source and generation method')
            writer.resume(options['resume_data'])
        if len(writer.manifest['episodes']) >= count:
            raise ValueError('Datagen target must exceed resumed episode count')
        writer.provenance.update(generation_method='object_relative_bounded_ik', source_dataset=str(source.resolve()), generation_seed=seed)
        if provenance.get('source_partition'):writer.provenance['source_partition']=provenance['source_partition']
        stopped = False
        try:
            for attempt in range(int(options.get('max_attempts', max(100, count * 100)))):
                if len(writer.manifest['episodes']) >= count: break
                entry = manifest['episodes'][int(rng.integers(len(manifest['episodes'])))]
                folder = Path(entry['_path'])
                annotation = json.loads((folder / 'annotation.json').read_text())
                if annotation.get('schema') != 'so101.mujoco-subtasks/1':
                    raise ValueError('Run workflow Annotation before Datagen')
                actions = np.load(folder / 'raw_action.npy', allow_pickle=False)
                original_object = np.load(folder / 'object_pose.npy', allow_pickle=False)[0]
                env.reset(int(rng.integers(2**31)))
                writer.begin(); tracker = grasp_tracker(env)
                moved_object = env.backend.get_object_pose(env.backend.metadata['pick'])
                transform_pick = moved_object @ np.linalg.inv(original_object)
                previous = env.backend.home.copy(); reason = ''
                try:
                    for i, raw_target in enumerate(actions):
                        if bridge: bridge.before_step()
                        goal = env.backend.target_pose(raw_target)
                        # Blend object-relative pick displacement back to the fixed destination.
                        blend = np.clip((i - annotation['grasp']) / max(1, annotation['place'] - annotation['grasp']), 0., 1.)
                        shifted = transform_pick @ goal
                        goal[:3, 3] = (1 - blend) * shifted[:3, 3] + blend * goal[:3, 3]
                        command = env.backend.solve_pose(goal, np.r_[previous[:5], raw_target[5]])
                        observation = env.observe()
                        writer.append(observation, command)
                        env.backend.step(command); previous = command
                        signals = subtask_signals(env, tracker, i)
                        emit(observation['images'][0], {'steps': i + 1, 'attempts': attempt + 1,
                             'successes': len(writer.manifest['episodes']), 'failures': attempt - len(writer.manifest['episodes']),
                             'task_signals': signals})
                    success = env.backend.is_success() and tracker.completed
                    if not success: reason = 'physical grasp/place validation failed'
                except ValueError as exc:
                    success = False; reason = str(exc)
                if success:
                    name = writer.save(True,success_source='simulation_stable_grasp_and_place')
                    writer.manifest['episodes'][-1]['group_id'] = entry['group_id']
                    for key in ('source_partition','source_lineage','split'):
                        if key in entry:writer.manifest['episodes'][-1][key]=copy.deepcopy(entry[key])
                    writer.flush_manifest()
                    print('[WORKFLOW] Generated ' + name, flush=True)
                attempts.append({'attempt': attempt + 1, 'source': entry['id'], 'success': bool(success), 'reason': reason})
        except KeyboardInterrupt:
            stopped = True
            print("[WORKFLOW] Datagen 중지: 완료된 episode를 보존합니다.", flush=True)
        finally:
            writer.close()
        atomic(output / 'attempts.json', attempts)
        return {'passed': len(writer.manifest['episodes']) == count or (stopped and bool(writer.manifest['episodes'])), 'dataset': str(writer.root),
                'successes': len(writer.manifest['episodes']), 'requested': count, 'attempts': len(attempts),
                'stopped': stopped, 'generation_seed': seed, 'method': 'object_relative_bounded_ik', 'is_isaac_mimicgen': False}


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
