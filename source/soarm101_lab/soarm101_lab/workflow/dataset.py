"""Common/legacy readers, native recording normalization and explicit migration.

Common file validation/commit belongs to data.episodes. Legacy formats and
Dataset-level historical partition checks remain here until a separate unit.
"""
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from ..so101_joint_mapping import JOINTS
from ..so101_dataset_contract import MIMIC_SPACE, read_contract
from .robot_preview import RobotPreview
from ..data import episodes as episode_io
from ..utils import transforms


def _json(value,default=None):
    return json.loads(value) if value is not None else default


def _read(group):
    return {key:_read(value) if isinstance(value,h5py.Group) else np.asarray(value) for key,value in group.items()
            if isinstance(value,h5py.Group) or not (value.ndim==4 and value.shape[-1] in (3,4) and value.dtype==np.uint8)}


def isaac_recording_episode(values,attrs,videos,args,contract,kinematics=None,*,preserve_original=False):
    """Normalize native in-memory recording or read-only Legacy input into one contract."""
    from ..real2sim.workspaces.package import load,metadata
    if not args.get('workspace_path'):raise ValueError('Isaac recording/import requires the recorded Workspace path')
    w=load(args['workspace_path'])
    if metadata(w)!=args.get('workspace'):raise ValueError('Legacy Isaac Workspace identity differs')
    original=copy.deepcopy(values) if preserve_original else None;policy=values['obs'].get('policy',values['obs']);target=values['joint_targets'];n=len(target)
    if kinematics is None:
        model=contract.get('simulation_model',{}).get('robot_model',{})
        kinematics=RobotPreview(w['root']/model.get('urdf','assets/SO101/urdf/so101_isaaclab.urdf'),load_visuals=False)
    goals=np.array([kinematics.transforms(v)['tool0'] for v in target]);ee=transforms.pose_matrix(policy['ee_state'])
    if args.get('ee_frame','robot_base')!='robot_base':raise ValueError('Explicit legacy EE frame conversion is required')
    root=values['initial_state']['articulation']['robot']['root_pose']
    order=args.get('root_quaternion_order')
    if order not in ('wxyz','xyzw'):raise ValueError('Explicit legacy root quaternion order required')
    base=transforms.pose_matrix(np.asarray(root).reshape(-1,7)[0],order);objects={};world={}
    initial=copy.deepcopy(values['initial_state'])
    for group in ('articulation','rigid_object'):
        for entry in initial.get(group,{}).values():
            entry['root_pose']=transforms.as_pose7(transforms.pose_matrix(entry['root_pose'],order))
    for name,entry in initial['rigid_object'].items():
        observed=values['obs'].get('object_poses',{}).get(name)
        if observed is not None:poses=transforms.pose_matrix(observed)
        else:
            post=values.get('workspace_states',{}).get(name)
            if post is None:post=values.get('states',{}).get('rigid_object',{}).get(name,{}).get('root_pose')
            if post is None:
                if name in w['task']['dynamic']:raise ValueError('Moving object observations missing: '+name)
                poses=np.repeat(transforms.pose_matrix(entry['root_pose']).reshape(-1,4,4)[0][None],n,axis=0)
            else:poses=np.concatenate([transforms.pose_matrix(entry['root_pose']).reshape(-1,4,4)[:1],transforms.pose_matrix(post,order)[:-1]])
        world[name]=transforms.as_pose7(poses);objects[name]=np.linalg.inv(base)@poses
    stamp=values['obs'].get('timestamp')
    ts=np.arange(n,dtype=np.float64)*float(args['control_dt']) if stamp is None else np.asarray(stamp).reshape(-1)
    signals=copy.deepcopy(values['obs'].get('datagen_info',{}).get('subtask_term_signals',{}))
    normalized={'actions':transforms.cartesian_actions(goals,ee[0],target[:,-1]),'joint_targets':target,
        'action_timestamp':np.asarray(values.get('action_timestamp',ts)).reshape(-1),
        'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':policy['joint_pos'],'ee_state':transforms.as_pose7(ee)},
            'object_poses':world,'datagen_info':{'eef_pose':{'tool0':ee},'target_eef_pose':{'tool0':goals},
                'object_pose':objects,'subtask_term_signals':signals},
            'camera_timestamps':{name.removesuffix('_cam'):ts for name in videos}},
        'initial_state':initial}
    attrs={**attrs,'cartesian_action_derivation':'FK of recorded joint_targets'+('; originals retained under legacy/' if preserve_original else ''),
        'pose_alignment':'pre-step observations; legacy initial[0]+post[:-1] when required'}
    if preserve_original:normalized['legacy']=original
    else:
        normalized['collection']={'recorded_mimic_actions':values['actions']}
        if 'joint_vel' in policy:normalized['obs']['policy']['joint_vel']=policy['joint_vel']
        for key in ('states','workspace_states'):
            if key in values:normalized[key]=values[key]
    normalized['obs']['events']=copy.deepcopy(values['obs'].get('events',{}))
    return normalized,attrs,videos,kinematics


class EpisodeDataset:
    """One reader for common HDF5, legacy Isaac HDF5 and legacy Canonical folders."""
    def __init__(self,path):
        self.path=Path(path).resolve();self.legacy=self.path.is_dir()
        self._fk=None
        if self.legacy:
            self.manifest=json.loads((self.path/'manifest.json').read_text())
            self.provenance=json.loads((self.path/'project_provenance.json').read_text())
            self.args=self._legacy_args();self.names=[e['id'] for e in self.manifest['episodes']]
        else:
            with h5py.File(self.path,'r') as h:
                self.args=_json(h['data'].attrs.get('env_args'),{})
                self.names=sorted((n for n in h['data'] if n.startswith('demo_')),key=lambda n:int(n[5:]))
            self.provenance=self.args.get('legacy_provenance',{})
        if not self.args.get('control_dt'):
            sim=self.args.get('sim_args',{});self.args['control_dt']=float(sim.get('dt',0))*int(sim.get('decimation',1))
        if self.args['control_dt']<=0:raise ValueError('Recorded control_dt is required')
        self.hz=1/float(self.args['control_dt'])
        self.backend=self.args.get('backend','isaac')
        self.contract=copy.deepcopy(self.args['so101_coordinates']) if self.legacy else read_contract(self.path)
        self.common=self.args.get('dataset_schema')==episode_io.SCHEMA

    def _legacy_args(self):
        provenance=copy.deepcopy(self.provenance)
        provenance['coordinate_contract']=provenance.get('coordinate_contract') or read_contract(self.path)
        provenance['scene_configuration']=provenance.get('scene_configuration') or provenance['scene'].get('scene_configuration')
        return episode_io.recording_args(provenance,self.manifest)

    @property
    def mimic_names(self):
        """Match upstream HDF5 group iteration; UI may still use numeric order."""
        if self.legacy:return list(self.names)
        with h5py.File(self.path,'r') as dataset:
            return [name for name in dataset['data'].keys() if name in self.names]

    def episode(self,name):
        if name not in self.names:raise KeyError(name)
        if self.legacy:return self._legacy_episode(name)
        with h5py.File(self.path,'r') as h:
            g=h['data'][name];attrs={k:v for k,v in g.attrs.items()}
            attrs.setdefault('simulation_backend',self.backend)
            values=_read(g);videos=_json(g.attrs.get('external_videos'),{})
            if self.common:return values,attrs,videos
            for prefix in ('obs/policy','obs','obs/camera'):
                if prefix not in g:continue
                for key,value in g[prefix].items():
                    if isinstance(value,h5py.Dataset) and value.ndim==4 and value.shape[-1] in (3,4):
                        camera=key if key.endswith('_cam') else key+'_cam'
                        videos[camera]={'file':str(self.path),'internal_dataset':prefix+'/'+key,'frames':len(value),
                            'fps':self.hz,'shape':list(value.shape[1:3])+[3]}
            return self._isaac_episode(values,attrs,videos)

    def mimic_episode(self,name):
        """Load only the official pool fields, excluding raw telemetry and prior plans."""
        if name not in self.names:raise KeyError(name)
        if self.common:
            with h5py.File(self.path,'r') as dataset:
                group=dataset['data'][name]
                return {'actions':group['actions'][...],
                    'obs':{'datagen_info':_read(group['obs/datagen_info'])}}
        values,_,_=self.episode(name)
        return {'actions':values['actions'],'obs':{'datagen_info':values['obs']['datagen_info']}}

    def episode_attributes(self,name):
        if name not in self.names:raise KeyError(name)
        if self.legacy:return self.episode(name)[1]
        with h5py.File(self.path,'r') as dataset:
            return dict(dataset['data'][name].attrs)

    def _isaac_episode(self,values,attrs,videos):
        values,attrs,videos,self._fk=isaac_recording_episode(
            values,attrs,videos,self.args,self.contract,self._fk,preserve_original=True)
        return values,attrs,videos

    def _legacy_episode(self,name):
        e=next(e for e in self.manifest['episodes'] if e['id']==name);p=(self.path/e['path']).resolve()
        if not p.is_relative_to(self.path):raise ValueError('Episode outside dataset')
        pvd=self.provenance;scene=pvd['scene']
        def arr(key):return np.load(p/(key+'.npy'),allow_pickle=False)
        q=arr('raw_state');target=arr('raw_action');ts=arr('timestamp');ats=arr('action_timestamp');n=len(q)
        if not pvd.get('workspace_path'):raise ValueError('Legacy migration requires its recorded Workspace')
        from ..real2sim.workspaces.package import load,metadata
        w=load(pvd['workspace_path'])
        if metadata(w)!=pvd['workspace']:raise ValueError('Legacy Workspace identity differs')
        sim=pvd.get('coordinate_contract',{}).get('simulation_model',{})
        urdf=w['root']/sim.get('robot_model',{}).get('urdf','assets/SO101/urdf/so101_isaaclab.urdf')
        if self._fk is None:self._fk=RobotPreview(urdf,load_visuals=False)
        targets=np.array([self._fk.transforms(row)['tool0'] for row in target])
        workspace_world=np.asarray(scene['workspace_T_world']);base_world=np.asarray(scene['robot_base_T_world'])
        base_workspace=np.linalg.inv(base_world)@workspace_world
        measured=base_workspace@arr('eef_pose')
        objects={};world_objects={};legacy={}
        for obj in scene['dynamic_objects']+scene['target_objects']:
            spec=e.get('object_telemetry',{}).get(obj)
            if spec:
                values=arr(Path(spec['file']).stem)
                if 'index' in spec:values=values[:,spec['index']]
                values=transforms.pose_matrix(values)
            elif obj==scene['pick']:values=arr('object_pose')
            elif obj in scene['dynamic_objects']:
                raise ValueError('Dynamic object telemetry missing: '+obj)
            else:
                visual=w['task']['targets'].get(obj,obj)
                values=np.repeat(np.asarray(w['profile']['objects'][visual]['T_workspace'])[None],n,axis=0)
            legacy[obj]=values;objects[obj]=base_workspace@values;world_objects[obj]=transforms.as_pose7(workspace_world@values)
        annotation=json.loads((p/'annotation.json').read_text()) if (p/'annotation.json').exists() else None
        info={'eef_pose':{'tool0':measured},'target_eef_pose':{'tool0':targets},'object_pose':objects,'subtask_term_signals':{}}
        events={}
        if annotation:
            if annotation.get('frame_count')!=n or not 0<=annotation['grasp']<annotation['place']<n:
                raise ValueError('Invalid ordered legacy annotation')
            for label in ('grasp','place'):events[label]=(np.arange(n)>=annotation[label]).astype(np.uint8)[:,None]
            info['subtask_term_signals']['grasp']=events['grasp']
        outcome=json.loads((p/'outcome.json').read_text())
        values={'actions':transforms.cartesian_actions(targets,measured[0],target[:,5]),'joint_targets':target,
            'action_timestamp':ats,'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':q,'ee_state':transforms.as_pose7(measured)},
                'datagen_info':info,'object_poses':world_objects,'events':events,
                'ee_pose_workspace':arr('eef_pose'),'object_pose_workspace':legacy},
            'initial_state':{'articulation':{'robot':{'root_pose':transforms.as_pose7(base_world)[None],
                'root_velocity':np.zeros((1,6)),'joint_position':q[0:1],'joint_velocity':np.zeros((1,6))}},
                'rigid_object':{obj:{'root_pose':v[0:1],'root_velocity':np.zeros((1,6))} for obj,v in world_objects.items()}},
            'legacy':{f.stem:np.load(f,allow_pickle=False) for f in p.glob('*.npy')}}
        attrs={'episode_id':name,'source_group':e.get('source_group',e['group_id']),'success':bool(outcome['success']),
            'simulation_backend':self.backend,
            'validation_only':bool(outcome.get('validation_only',False)),
            'success_source':outcome.get('success_source','unknown'),'num_samples':n,
            'task_definition':json.dumps(e.get('task_definition') or scene['task_definition']),
            'scene_configuration':json.dumps(e.get('scene_configuration') or self.args['scene_configuration']),
            'legacy_initial_state':(p/'replay.json').read_text(),'initial_velocity_evidence':'unrecorded legacy velocity; replay validation required',
            'outcome':json.dumps(outcome),'annotation':json.dumps(annotation),'split':(e.get('source_partition') or pvd.get('source_partition') or {}).get('split','')}
        if e.get('source_lineage'):attrs['source_lineage']=json.dumps(e['source_lineage'])
        videos={}
        for camera in self.manifest['cameras']:
            spec=copy.deepcopy(e.get('videos',{}).get(camera,{}))
            file=p/spec.get('file',camera+'.mp4')
            if not file.is_file():raise ValueError('Legacy MP4 missing: '+str(file))
            values['obs'].setdefault('camera_timestamps',{})[camera]=arr(camera+'_timestamp')
            videos[camera+'_cam']={**spec,'file':str(file),'frames':n,'fps':self.hz,'shape':spec.get('shape',[480,640,3])}
        return values,attrs,videos


def validate_dataset(reader,*,require_annotations=False):
    """Validate semantic declarations before any simulator or source pool is loaded."""
    if reader.legacy or not reader.common:raise ValueError('Import Legacy data before common dataset validation')
    if reader.backend not in ('mujoco','isaac'):raise ValueError('Common dataset execution backend is missing or unknown')
    if reader.args.get('action_semantics')!=episode_io.SEMANTICS:raise ValueError('Common action semantics differ')
    expected={'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base',
        'pose_format':'matrix4','quaternion_order':'wxyz','action_semantics':episode_io.SEMANTICS,'actions':MIMIC_SPACE}
    declared=reader.args.get('schema_contract',{})
    for key,value in expected.items():
        if declared.get(key)!=value:raise ValueError('Common dataset contract mismatch: '+key)
    if reader.contract.get('joint_order')!=list(JOINTS) or reader.contract.get('action_space')!=MIMIC_SPACE:
        raise ValueError('Unverified joint/action coordinate contract')
    if reader.args.get('so101_coordinates')!=reader.contract:raise ValueError('Embedded/sidecar coordinate contracts differ')
    if not reader.names:raise ValueError('No committed episodes')
    for name in reader.names:
        values,attrs,videos=reader.episode(name)
        episode_io.validate_episode(values,attrs,videos,reader.hz,require_annotations=require_annotations,video_root=reader.path.parent)
        if reader.args.get('source_partition'):
            from .episodes import validate_partition_episode
            validate_partition_episode(reader.args['source_partition'],_json(attrs.get('source_lineage')),
                reader.args.get('source_partition_input'))
    return {'episodes':len(reader.names),'schema':episode_io.SCHEMA,'backend':reader.backend}


def migrate(source,destination,*,names=None):
    """Lossless numeric import; video bytes copied without decoding/re-encoding."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    reader=EpisodeDataset(source)
    if names is not None:
        if not names or len(set(names))!=len(names) or not set(names).issubset(reader.names):raise ValueError('Invalid import episode selection')
        reader.names=[name for name in reader.names if name in names]
    if reader.contract.get('joint_order')!=list(JOINTS) or reader.contract.get('action_space')!=MIMIC_SPACE:
        raise ValueError('Recorded coordinate/action contract required for common import; legacy unknown cannot be relabeled')
    from ..utils.artifact_identity import fingerprint
    from .episodes import source_catalog
    before=fingerprint(source);records,partition=source_catalog(source,allow_unknown=True)
    args=copy.deepcopy(reader.args);args['migration']={'source':str(source),'source_sha256':before,'numeric_comparison':'exact; original numeric arrays retained under legacy/','video_encoding':'byte copy'}
    args['so101_coordinates']=copy.deepcopy(reader.contract)
    args.update(backend=reader.backend,root_quaternion_order='wxyz',ee_frame='robot_base',observation_timestamp_path='obs/timestamp')
    first_values,_,_=reader.episode(reader.names[0])
    args['object_telemetry']={name:{'path':'obs/object_poses/'+name,'frame':'world','pose_format':'xyz_wxyz',
        'timestamp_path':'obs/timestamp'} for name in first_values['obs']['object_poses']}
    args.setdefault('schema_contract',{'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base','pose_format':'matrix4',
        'quaternion_order':'wxyz','action_semantics':episode_io.SEMANTICS,'actions':MIMIC_SPACE,
        'joint_targets':'absolute model rad applied after obs[t]','target_eef_pose':'controller target; joint-only inputs reconstructed by FK'})
    writer=episode_io.DatasetWriter(destination,args,reader.contract)
    try:
        for name in reader.names:
            values,attrs,videos=reader.episode(name)
            if name in records:attrs['source_lineage']=json.dumps(records[name])
            staging=None
            try:
                internal={k:spec for k,spec in videos.items() if 'internal_dataset' in spec}
                if internal:
                    from ..utils.async_video_writer import VideoEpisode
                    staging=VideoEpisode(destination.parent,reader.hz,18)
                    with h5py.File(source,'r') as raw:
                        for index in range(len(values['actions'])):
                            staging.append({key:np.asarray(raw['data'][name][spec['internal_dataset']][index])[...,:3] for key,spec in internal.items()})
                    specs=staging.finish(len(values['actions']))
                    videos={**videos,**{k:{**v,'file':str((staging.folder/v['file']).resolve())} for k,v in specs.items()}}
                    args['migration']['video_encoding']='legacy internal RGB streamed to H.264; existing MP4 byte copied'
                    writer.args['migration']=copy.deepcopy(args['migration'])
                if not reader.legacy:
                    from ..utils.async_video_writer import external_video_path
                    videos={k:{**v,'file':str(Path(v['file']) if k in internal else external_video_path(source,v))} for k,v in videos.items()}
                writer.commit(values,attrs,videos)
            finally:
                if staging:staging.abort()
        if fingerprint(source)!=before:raise ValueError('Migration source changed during import')
        writer.close()
    except BaseException:writer.close(False);raise
    validated=EpisodeDataset(destination)
    for name in validated.names:
        values,attrs,videos=validated.episode(name);episode_io.validate_episode(values,attrs,videos,validated.hz,video_root=destination.parent)
    return destination
