"""Engine-independent, versioned numeric HDF5 + relative MP4 episode contract.

The layout consumed by Isaac Lab DataGenInfoPool is preserved verbatim. All poses
in obs/datagen_info are robot-base SE(3); joint_targets and gripper are model rad.
Legacy readers never modify their source. Import is an explicit new artifact.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import h5py
import numpy as np

from ..so101_joint_mapping import JOINTS
from ..so101_dataset_contract import MIMIC_SPACE, read_contract, write_contract
from .robot_preview import RobotPreview, pose_matrix

SCHEMA = 'so101.common-episode/2'
SEMANTICS = 'absolute_target_after_observation'


def as_pose7(matrix):
    from scipy.spatial.transform import Rotation
    matrix=np.asarray(matrix)
    q=Rotation.from_matrix(matrix[..., :3, :3].reshape(-1,3,3)).as_quat()
    return np.concatenate([matrix[..., :3,3].reshape(-1,3),q[:,[3,0,1,2]]],-1).reshape(matrix.shape[:-2]+(7,))


def cartesian_actions(targets,initial,gripper):
    from scipy.spatial.transform import Rotation
    previous=np.concatenate([np.asarray(initial)[None],targets[:-1]])
    rotation=Rotation.from_matrix(targets[:,:3,:3]@previous[:,:3,:3].transpose(0,2,1)).as_rotvec()
    return np.column_stack([targets[:,:3,3]-previous[:,:3,3],rotation,gripper])


def _json(value,default=None):
    return json.loads(value) if value is not None else default


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def _write(group,values):
    for key,value in values.items():
        if isinstance(value,dict):_write(group.require_group(key),value)
        else:group.create_dataset(key,data=np.asarray(value),compression='gzip' if np.asarray(value).ndim else None)


def _read(group):
    return {key:_read(value) if isinstance(value,h5py.Group) else np.asarray(value) for key,value in group.items()
            if isinstance(value,h5py.Group) or not (value.ndim==4 and value.shape[-1] in (3,4) and value.dtype==np.uint8)}


def recording_args(provenance,manifest):
    """The same metadata contract for live writers and legacy migration."""
    p=provenance;s=p['scene']
    return dict(env_name='SO101-Workspace-Mimic-v1',type=2,dataset_schema=SCHEMA,backend=p.get('backend','mujoco'),
        workspace=p['workspace'],workspace_path=p.get('workspace_path'),control_dt=1/float(manifest['control_hz']),
        action_semantics=SEMANTICS,so101_coordinates=p['coordinate_contract'],scene_configuration=p['scene_configuration'],
        canonical_coordinates=copy.deepcopy(p['coordinates']),
        task_definitions=[s['task_definition']],grasp_object=s['pick'],place_bin=s['target'],source_partition=p.get('source_partition'),
        source_partition_input=p.get('source_partition_input'),legacy_provenance=copy.deepcopy(p),ee_frame='robot_base',root_quaternion_order='wxyz',
        observation_timestamp_path='obs/timestamp',object_telemetry={n:{'path':'obs/object_poses/'+n,'frame':'world',
            'pose_format':'xyz_wxyz','timestamp_path':'obs/timestamp'} for n in s['dynamic_objects']},
        schema_contract={'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base','pose_format':'matrix4',
            'quaternion_order':'wxyz','action_semantics':SEMANTICS,'actions':MIMIC_SPACE,
            'joint_targets':'absolute model rad applied after obs[t]','target_eef_pose':'controller target; legacy joint-only targets reconstructed by FK'})


def recorded_mujoco_episode(env,rows):
    """Live numeric samples directly to HDF5 values; no temporary NPY dataset."""
    backend=env.backend;world_workspace=backend.world_from_workspace
    world_base=np.asarray(backend.workspace['profile']['robot_base']['T_world']);base_workspace=np.linalg.inv(world_base)@world_workspace
    target=np.array([r['raw_action'] for r in rows]);q=np.array([r['raw_state'] for r in rows]);n=len(rows)
    ee=base_workspace@np.array([r['eef_pose'] for r in rows]);goals=base_workspace@np.array([backend.target_pose(v) for v in target])
    objects={name:base_workspace@np.array([r['object_poses'][name] for r in rows]) for name in backend.metadata['dynamic_objects']}
    objects.update({name:np.repeat((base_workspace@backend.get_object_pose(name))[None],n,axis=0) for name in backend.metadata['target_objects']})
    world={name:as_pose7(world_base@value) for name,value in objects.items()};ts=np.array([r['timestamp'] for r in rows])
    return {'actions':cartesian_actions(goals,ee[0],target[:,5]),'joint_targets':target,'action_timestamp':ts,
        'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':q,'ee_state':as_pose7(ee)},'object_poses':world,
            'datagen_info':{'eef_pose':{'tool0':ee},'target_eef_pose':{'tool0':goals},'object_pose':objects,'subtask_term_signals':{}},
            'camera_timestamps':{c:ts for c in backend.workspace['task']['observation_cameras']}},
        'initial_state':{'articulation':{'robot':{'root_pose':as_pose7(world_base)[None],'root_velocity':np.zeros((1,6)),
            'joint_position':q[0:1],'joint_velocity':np.zeros((1,6))}},
            'rigid_object':{name:{'root_pose':v[0:1],'root_velocity':np.zeros((1,6))} for name,v in world.items()}},
        'collection':{key:np.array([r[key] for r in rows]) for key in ('state','action','requested_action')}}


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
    goals=np.array([kinematics.transforms(v)['tool0'] for v in target]);ee=pose_matrix(policy['ee_state'])
    if args.get('ee_frame','robot_base')!='robot_base':raise ValueError('Explicit legacy EE frame conversion is required')
    root=values['initial_state']['articulation']['robot']['root_pose']
    order=args.get('root_quaternion_order')
    if order not in ('wxyz','xyzw'):raise ValueError('Explicit legacy root quaternion order required')
    base=pose_matrix(np.asarray(root).reshape(-1,7)[0],order);objects={};world={}
    initial=copy.deepcopy(values['initial_state'])
    for group in ('articulation','rigid_object'):
        for entry in initial.get(group,{}).values():
            entry['root_pose']=as_pose7(pose_matrix(entry['root_pose'],order))
    for name,entry in initial['rigid_object'].items():
        observed=values['obs'].get('object_poses',{}).get(name)
        if observed is not None:poses=pose_matrix(observed)
        else:
            post=values.get('workspace_states',{}).get(name)
            if post is None:post=values.get('states',{}).get('rigid_object',{}).get(name,{}).get('root_pose')
            if post is None:
                if name in w['task']['dynamic']:raise ValueError('Moving object observations missing: '+name)
                poses=np.repeat(pose_matrix(entry['root_pose']).reshape(-1,4,4)[0][None],n,axis=0)
            else:poses=np.concatenate([pose_matrix(entry['root_pose']).reshape(-1,4,4)[:1],pose_matrix(post,order)[:-1]])
        world[name]=as_pose7(poses);objects[name]=np.linalg.inv(base)@poses
    stamp=values['obs'].get('timestamp')
    ts=np.arange(n,dtype=np.float64)*float(args['control_dt']) if stamp is None else np.asarray(stamp).reshape(-1)
    signals=copy.deepcopy(values['obs'].get('datagen_info',{}).get('subtask_term_signals',{}))
    normalized={'actions':cartesian_actions(goals,ee[0],target[:,-1]),'joint_targets':target,
        'action_timestamp':np.asarray(values.get('action_timestamp',ts)).reshape(-1),
        'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':policy['joint_pos'],'ee_state':as_pose7(ee)},
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
        self.common=self.args.get('dataset_schema')==SCHEMA

    def _legacy_args(self):
        provenance=copy.deepcopy(self.provenance)
        provenance['coordinate_contract']=provenance.get('coordinate_contract') or read_contract(self.path)
        provenance['scene_configuration']=provenance.get('scene_configuration') or provenance['scene'].get('scene_configuration')
        return recording_args(provenance,self.manifest)

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
                values=pose_matrix(values)
            elif obj==scene['pick']:values=arr('object_pose')
            elif obj in scene['dynamic_objects']:
                raise ValueError('Dynamic object telemetry missing: '+obj)
            else:
                visual=w['task']['targets'].get(obj,obj)
                values=np.repeat(np.asarray(w['profile']['objects'][visual]['T_workspace'])[None],n,axis=0)
            legacy[obj]=values;objects[obj]=base_workspace@values;world_objects[obj]=as_pose7(workspace_world@values)
        annotation=json.loads((p/'annotation.json').read_text()) if (p/'annotation.json').exists() else None
        info={'eef_pose':{'tool0':measured},'target_eef_pose':{'tool0':targets},'object_pose':objects,'subtask_term_signals':{}}
        events={}
        if annotation:
            if annotation.get('frame_count')!=n or not 0<=annotation['grasp']<annotation['place']<n:
                raise ValueError('Invalid ordered legacy annotation')
            for label in ('grasp','place'):events[label]=(np.arange(n)>=annotation[label]).astype(np.uint8)[:,None]
            info['subtask_term_signals']['grasp']=events['grasp']
        outcome=json.loads((p/'outcome.json').read_text())
        values={'actions':cartesian_actions(targets,measured[0],target[:,5]),'joint_targets':target,
            'action_timestamp':ats,'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':q,'ee_state':as_pose7(measured)},
                'datagen_info':info,'object_poses':world_objects,'events':events,
                'ee_pose_workspace':arr('eef_pose'),'object_pose_workspace':legacy},
            'initial_state':{'articulation':{'robot':{'root_pose':as_pose7(base_world)[None],
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


def validate_episode(values,attrs,videos,hz,*,require_annotations=False,video_root=None,allow_external_sources=False):
    q=np.asarray(values['obs']['policy']['joint_pos']);target=np.asarray(values['joint_targets']);n=len(q)
    if n<1 or q.shape!=(n,6) or target.shape!=(n,6) or not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError('Finite [T,6] model-radian joint observations/targets required')
    if attrs.get('num_samples',n)!=n:raise ValueError('Episode sample count differs from numeric data')
    actions=np.asarray(values['actions'])
    if actions.shape!=(n,7) or not np.isfinite(actions).all():raise ValueError('Finite [T,7] Cartesian/gripper actions required')
    if not np.array_equal(actions[:,-1],target[:,-1]):raise ValueError('Gripper action differs from model target')
    ts=np.asarray(values['obs']['timestamp']).reshape(-1);ats=np.asarray(values['action_timestamp']).reshape(-1)
    if ts.shape!=(n,) or ats.shape!=(n,) or not np.isfinite(ts).all() or not np.isfinite(ats).all():raise ValueError('Timestamp length/finiteness mismatch')
    if n>1 and (not np.allclose(np.diff(ts),1/hz,atol=1e-6,rtol=0) or np.any(np.diff(ats)<=0)):raise ValueError('Timestamp/control frequency mismatch')
    if not np.allclose(ats,ts,atol=1e-7,rtol=0):raise ValueError('Action/observation timestamp mismatch; commands must share the recorded control tick')
    info=values['obs']['datagen_info']
    for category in ('eef_pose','target_eef_pose','object_pose'):
        if not info[category]:raise ValueError('Missing Mimic '+category)
        for name,v in info[category].items():
            v=np.asarray(v)
            if v.shape!=(n,4,4) or not np.isfinite(v).all() or not np.allclose(v[:,3,:],[0,0,0,1],atol=1e-7):raise ValueError('Invalid SE(3): '+name)
            if not np.allclose(v[:,:3,:3]@v[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-5) or np.any(np.linalg.det(v[:,:3,:3])<.999):raise ValueError('Invalid rotation: '+name)
    initial=values['initial_state']
    robot=initial['articulation']['robot']
    initial_q=np.asarray(robot['joint_position'])
    if initial_q.shape!=(1,6) or not np.allclose(initial_q[0],q[0],atol=1e-6,rtol=0):
        raise ValueError('Initial joint state differs from first observation')
    for entry in [robot,*initial.get('rigid_object',{}).values()]:
        pose=np.asarray(entry['root_pose']);velocity=np.asarray(entry['root_velocity'])
        if pose.shape!=(1,7) or not np.isfinite(pose).all() or not np.allclose(np.linalg.norm(pose[:,3:],axis=-1),1,atol=1e-5):
            raise ValueError('Initial root pose requires xyz + normalized wxyz')
        if velocity.shape!=(1,6) or not np.isfinite(velocity).all():raise ValueError('Invalid initial root velocity')
    velocity=np.asarray(robot['joint_velocity'])
    if velocity.shape!=(1,6) or not np.isfinite(velocity).all():raise ValueError('Invalid initial joint velocity')
    for name,poses in values['obs']['object_poses'].items():
        poses=np.asarray(poses)
        if poses.shape!=(n,7) or not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[:,3:],axis=-1),1,atol=1e-5):
            raise ValueError('Invalid world object pose: '+name)
        if name not in initial['rigid_object'] or not np.allclose(pose_matrix(initial['rigid_object'][name]['root_pose'][0]),pose_matrix(poses[0]),atol=1e-5):
            raise ValueError('Initial object state differs from first observation: '+name)
    signal=info.get('subtask_term_signals',{}).get('grasp')
    if require_annotations and signal is None:raise ValueError('Mimic grasp annotation missing')
    if signal is not None:
        signal=np.asarray(signal).reshape(-1)
        if len(signal)!=n or not np.isin(signal,[0,1]).all() or signal[0]!=0 or np.any(np.diff(signal.astype(int))<0):raise ValueError('Grasp must have one increasing 0→1 boundary')
        if require_annotations and not np.any(signal):raise ValueError('Grasp boundary missing')
    if set(videos)!={'side_cam','wrist_cam'}:raise ValueError('Side and Wrist videos required')
    from ..utils.async_video_writer import VideoFrames
    for camera,spec in videos.items():
        if spec.get('shape')!=[480,640,3] or spec.get('frames')!=n or not np.isclose(spec['fps'],hz):raise ValueError('Camera shape/FPS/count mismatch')
        path=Path(spec['file'])
        if not path.is_absolute():path=Path(video_root)/path
        if video_root is not None and not allow_external_sources and not path.resolve().is_relative_to(Path(video_root).resolve()):raise ValueError('Video path escapes dataset')
        video=VideoFrames(path,n,hz)
        try:
            if video.shape!=(n,480,640,3):raise ValueError('MP4 dimensions differ')
            if spec.get('sha256') and file_sha256(path)!=spec['sha256']:raise ValueError('Video bytes changed')
        finally:video.close()
        import av
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            if stream.codec_context.name!='h264':raise ValueError('Common RGB video must use H.264')
            points=sorted(float(packet.pts*stream.time_base) for packet in container.demux(stream) if packet.pts is not None)
            if len(points)!=n or not np.allclose(points,np.arange(n)/hz,atol=1e-6,rtol=0):
                raise ValueError('Video presentation timestamps do not match observation frames')
        stamp=values['obs'].get('camera_timestamps',{}).get(camera.removesuffix('_cam'),ts)
        if not np.array_equal(np.asarray(stamp).reshape(-1),ts):raise ValueError('Video/observation timestamp mismatch')
    return n


def validate_dataset(reader,*,require_annotations=False):
    """Validate semantic declarations before any simulator or source pool is loaded."""
    if reader.legacy or not reader.common:raise ValueError('Import Legacy data before common dataset validation')
    if reader.backend not in ('mujoco','isaac'):raise ValueError('Common dataset execution backend is missing or unknown')
    if reader.args.get('action_semantics')!=SEMANTICS:raise ValueError('Common action semantics differ')
    expected={'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base',
        'pose_format':'matrix4','quaternion_order':'wxyz','action_semantics':SEMANTICS,'actions':MIMIC_SPACE}
    declared=reader.args.get('schema_contract',{})
    for key,value in expected.items():
        if declared.get(key)!=value:raise ValueError('Common dataset contract mismatch: '+key)
    if reader.contract.get('joint_order')!=list(JOINTS) or reader.contract.get('action_space')!=MIMIC_SPACE:
        raise ValueError('Unverified joint/action coordinate contract')
    if reader.args.get('so101_coordinates')!=reader.contract:raise ValueError('Embedded/sidecar coordinate contracts differ')
    if not reader.names:raise ValueError('No committed episodes')
    for name in reader.names:
        values,attrs,videos=reader.episode(name)
        validate_episode(values,attrs,videos,reader.hz,require_annotations=require_annotations,video_root=reader.path.parent)
        if reader.args.get('source_partition'):
            from .episodes import validate_partition_episode
            validate_partition_episode(reader.args['source_partition'],_json(attrs.get('source_lineage')),
                reader.args.get('source_partition_input'))
    return {'episodes':len(reader.names),'schema':SCHEMA,'backend':reader.backend}


class DatasetWriter:
    """One owner serializes commits; unpublished groups never appear as episodes."""
    def __init__(self,path,args,contract):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():raise FileExistsError(self.path)
        self.temp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        self.h=h5py.File(self.temp,'x');self.data=self.h.create_group('data');self.count=0;self.total=0
        self.args=copy.deepcopy(args);self.args.update(dataset_schema=SCHEMA,env_name='SO101-Workspace-Mimic-v1',type=2,action_semantics=SEMANTICS)
        self.args.setdefault('so101_coordinates',copy.deepcopy(contract))
        self.args.setdefault('schema_contract',{'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base',
            'pose_format':'matrix4','quaternion_order':'wxyz','action_semantics':SEMANTICS,'actions':MIMIC_SPACE,
            'joint_targets':'absolute model rad applied after obs[t]',
            'target_eef_pose':'controller target; joint-only inputs reconstructed by FK'})
        from importlib.metadata import version,PackageNotFoundError
        versions={}
        for package in ('mujoco','isaacsim','torch','numpy','h5py','av'):
            try:versions[package]=version(package)
            except PackageNotFoundError:pass
        inherited_physics=self.args.get('physics_metadata')
        self.args['physics_metadata']={'execution_backend':self.args.get('backend'),
            'package_versions':versions,'simulation_model':contract.get('simulation_model'),
            'physics_step_semantics':'one control action; backend substeps at recorded control_dt'}
        if inherited_physics:self.args['physics_metadata']['source_physics_metadata']=inherited_physics
        self.contract=copy.deepcopy(contract);self.h.attrs['dataset_schema']=SCHEMA;self.flush()
        self.owned_video_dirs=[]

    def flush(self):
        self.data.attrs['env_args']=json.dumps(self.args,allow_nan=False);self.data.attrs['total']=self.total;self.h.flush()

    def record_attempts(self, records):
        payload=json.dumps(records,allow_nan=False).encode()
        group=self.h.require_group('generation')
        if 'attempts_json' in group:del group['attempts_json']
        group.create_dataset('attempts_json',data=np.frombuffer(payload,dtype=np.uint8),compression='gzip')
        self.args['generation_attempt_history']='generation/attempts_json'
        self.flush()

    def commit(self,values,attrs,videos,*,video_staging=None):
        n=validate_episode(values,attrs,videos,1/float(self.args['control_dt']),video_root=self.path.parent,allow_external_sources=True)
        info=copy.deepcopy(values['obs']['datagen_info'])
        for category in ('eef_pose','target_eef_pose','object_pose'):
            info[category]={name:np.asarray(value,dtype=np.float32) for name,value in info[category].items()}
        values={**values,'obs':{**values['obs'],'datagen_info':info}}
        self.args.setdefault('schema_contract',{})['mimic_pose_dtype']='float32; original numeric telemetry retained separately'
        folder=self.path.parent/(self.path.stem+'.videos')/uuid.uuid4().hex
        if video_staging is not None:
            from ..utils.async_video_writer import VideoEpisode
            if not isinstance(video_staging,VideoEpisode) or not video_staging.finalized:
                raise ValueError('Only a finalized writer-owned VideoEpisode can be adopted')
            staged=video_staging.folder.resolve()
            sources={Path(spec['file']).resolve() for spec in videos.values()}
            if not staged.is_relative_to(self.path.parent) or staged.is_symlink() or any(p.parent!=staged or p.is_symlink() for p in sources):
                raise ValueError('Video adoption requires private staging inside this output bundle')
            if sources!={p.resolve() for p in staged.iterdir()}:
                raise ValueError('Staging must contain exactly the committed camera videos')
        folder.parent.mkdir(parents=True,exist_ok=True)
        self.owned_video_dirs.append(folder);pending='_pending_'+uuid.uuid4().hex
        episode_name=f'demo_{self.count}';old_count=self.count;old_total=self.total
        try:
            if video_staging is not None:video_staging.folder.rename(folder)
            else:folder.mkdir(exist_ok=False)
            g=self.h.create_group(pending)
            copied={}
            for name,spec in videos.items():
                src=Path(spec['file']);src=src if src.is_absolute() else self.path.parent/src
                if video_staging is not None:dest=folder/src.name
                else:
                    dest=folder/(name+'.mp4');shutil.copyfile(src,dest)
                copied[name]={**spec,'file':str(dest.relative_to(self.path.parent)),'sha256':file_sha256(dest)}
            _write(g,values)
            for k,v in attrs.items():
                if v is not None:g.attrs[k]=v
            g.attrs.update(num_samples=n,external_videos=json.dumps(copied),committed=True)
            self.h.flush();self.h.move(pending,'data/'+episode_name)
            self.count+=1;self.total+=n;self.flush();return episode_name
        except BaseException:
            if pending in self.h:del self.h[pending]
            if episode_name in self.data:del self.data[episode_name]
            self.count=old_count;self.total=old_total
            shutil.rmtree(folder,ignore_errors=True);raise

    def close(self,publish=True):
        if self.h is None:return
        self.flush();self.h.close();self.h=None
        if publish and self.count:
            write_contract(self.path,self.contract)
            os.link(self.temp,self.path)
        else:
            for folder in self.owned_video_dirs:shutil.rmtree(folder,ignore_errors=True)
        self.temp.unlink(missing_ok=True)

    def __enter__(self):return self
    def __exit__(self,kind,*args):self.close(publish=kind is None)


def migrate(source,destination,*,names=None):
    """Lossless numeric import; video bytes copied without decoding/re-encoding."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    reader=EpisodeDataset(source)
    if names is not None:
        if not names or len(set(names))!=len(names) or not set(names).issubset(reader.names):raise ValueError('Invalid import episode selection')
        reader.names=[name for name in reader.names if name in names]
    if reader.contract.get('joint_order')!=list(JOINTS) or reader.contract.get('action_space')!=MIMIC_SPACE:
        raise ValueError('Recorded coordinate/action contract required for common import; legacy unknown cannot be relabeled')
    from .artifacts import fingerprint
    from .episodes import source_catalog
    before=fingerprint(source);records,partition=source_catalog(source,allow_unknown=True)
    args=copy.deepcopy(reader.args);args['migration']={'source':str(source),'source_sha256':before,'numeric_comparison':'exact; original numeric arrays retained under legacy/','video_encoding':'byte copy'}
    args['so101_coordinates']=copy.deepcopy(reader.contract)
    args.update(backend=reader.backend,root_quaternion_order='wxyz',ee_frame='robot_base',observation_timestamp_path='obs/timestamp')
    first_values,_,_=reader.episode(reader.names[0])
    args['object_telemetry']={name:{'path':'obs/object_poses/'+name,'frame':'world','pose_format':'xyz_wxyz',
        'timestamp_path':'obs/timestamp'} for name in first_values['obs']['object_poses']}
    args.setdefault('schema_contract',{'joint_order':list(JOINTS),'joint_unit':'rad','pose_frame':'robot_base','pose_format':'matrix4',
        'quaternion_order':'wxyz','action_semantics':SEMANTICS,'actions':MIMIC_SPACE,
        'joint_targets':'absolute model rad applied after obs[t]','target_eef_pose':'controller target; joint-only inputs reconstructed by FK'})
    writer=DatasetWriter(destination,args,reader.contract)
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
        values,attrs,videos=validated.episode(name);validate_episode(values,attrs,videos,validated.hz,video_root=destination.parent)
    return destination
