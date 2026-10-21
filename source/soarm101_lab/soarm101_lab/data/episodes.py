"""Common HDF5 and Legacy NPY/MP4 commits, metadata and lazy episode reads.

No training-role policy, Workspace loading, simulation or UI ownership.
"""
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import uuid
import numpy as np
from ..utils.transforms import pose_matrix
from ..so101_joint_mapping import JOINTS
from ..so101_dataset_contract import MIMIC_SPACE, write_contract
from ..real2sim.storage import atomic


SCHEMA = 'so101.common-episode/2'
SEMANTICS = 'absolute_target_after_observation'


def dataset_metadata(path, format: str) -> dict:
    """Header-only description for one explicitly selected path in a worker.

    This is not readiness/Coordinate validation. No arrays, videos, parent
    provenance or Workspace are loaded. Large metadata is left to full readers.
    """
    if format not in ('hdf5','canonical','lerobot','evaluation'):raise ValueError('Unknown dataset format')
    path=Path(path).expanduser().resolve();result={'path':str(path),'format':format}
    if format=='hdf5':
        import h5py
        with h5py.File(path,'r') as h:
            data=h['data'];args=data.attrs.get('env_args','{}')
            if len(args)>1024*1024:raise ValueError('Episode header exceeds display limit')
            args=json.loads(args)
            if not isinstance(args,dict):raise ValueError('Episode header must be an object')
            result.update(format=args.get('dataset_schema','isaac-hdf5'),backend=args.get('backend'),
                          control_hz=args.get('control_hz'),action_semantics=args.get('action_semantics'))
        return result
    marker=path/('meta/info.json' if format=='lerobot' else 'manifest.json' if format=='canonical' else 'evaluation_datasets.json')
    with marker.open('rb') as stream:raw=stream.read(1024*1024+1)
    if len(raw)>1024*1024:raise ValueError('Dataset header exceeds display limit')
    header=json.loads(raw)
    if not isinstance(header,dict):raise ValueError('Dataset header must be an object')
    if format=='canonical':result.update(episodes=len(header.get('episodes',[])),control_hz=header.get('control_hz'))
    elif format=='lerobot':result.update(episodes=header.get('total_episodes'),fps=header.get('fps'),features=header.get('features'))
    else:result.update(schema=header.get('schema'),datasets=header.get('datasets'))
    return result


class LegacyEpisodeWriter:
    """Own Legacy NPY/MP4 metadata and manifest publication, never simulation.

    Inputs are snapshots/rows from Recording. Video staging is finalized and
    renamed before provenance, then manifest, are published with existing atomic
    writes. A failed commit removes only its own unpublished episode. No engine,
    FK, control or Source-policy callback is accepted.
    """
    def __init__(self,root,manifest:dict,provenance:dict):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.manifest=copy.deepcopy(manifest);self.provenance=copy.deepcopy(provenance)
        try:self.flush_manifest()
        except BaseException:shutil.rmtree(self.root);raise

    @property
    def count(self):return len(self.manifest['episodes'])

    def flush_manifest(self):
        atomic(self.root/'project_provenance.json',self.provenance)
        atomic(self.root/'manifest.json',self.manifest)

    def copy_episodes(self,source,manifest:dict)->None:
        """Copy validated Legacy entries to a new bundle; source stays immutable."""
        source=Path(source).resolve();start=self.count;owned=[]
        try:
            for entry in manifest['episodes']:
                folder=(source/entry['path']).resolve()
                if not folder.is_relative_to(source):raise ValueError('Episode outside dataset')
                name=f'episode_{self.count:04d}';dest=self.root/name
                if dest.exists():raise FileExistsError(dest)
                staging=self.root/('.resume-'+uuid.uuid4().hex);owned.append(staging)
                shutil.copytree(folder,staging)
                dest.mkdir(exist_ok=False);owned.append(dest)
                staging.rename(dest);owned.remove(staging)
                self.manifest['episodes'].append({**entry,'id':name,'path':name})
            self.provenance['resumed_from']=str(source);self.flush_manifest()
        except BaseException:
            del self.manifest['episodes'][start:]
            for folder in owned:shutil.rmtree(folder,ignore_errors=True)
            raise

    def save(self,rows,initial:dict,video,outcome:dict)->str:
        """Borrow frozen rows/initial until return; publish one episode name.

        The caller retains VideoEpisode ownership and aborts it in finally.
        Empty/foreign staging raises ValueError; collisions FileExistsError;
        encoding/file errors propagate after rolling back this new episode.
        """
        if not rows or video is None:raise ValueError('Cannot save an empty episode')
        from ..utils.async_video_writer import VideoEpisode
        if not isinstance(video,VideoEpisode):raise ValueError('Legacy commit requires private video staging inside this dataset')
        name=f'episode_{self.count:04d}';dest=self.root/name;folder=video.folder;renamed=False
        if folder.is_symlink() or folder.resolve().parent!=self.root.resolve():
            raise ValueError('Legacy commit requires private video staging inside this dataset')
        if dest.exists():raise FileExistsError(dest)
        try:
            videos=video.finish(len(rows))
            for key in ('state','action'):
                np.save(folder/f'{key}.npy',np.asarray([r[key] for r in rows],dtype=np.float32))
            for key in ('raw_state','raw_action','requested_action','eef_pose','object_pose'):
                np.save(folder/f'{key}.npy',np.asarray([r[key] for r in rows],dtype=np.float64))
            times=np.asarray([r['timestamp'] for r in rows],dtype=np.float64)
            for camera in self.manifest['cameras']:np.save(folder/(camera+'_timestamp.npy'),times)
            for key in ('timestamp','action_timestamp'):np.save(folder/f'{key}.npy',times)
            active=self.provenance['scene']['dynamic_objects'];pick=self.provenance['scene']['pick']
            telemetry={pick:{'file':'object_pose.npy','pose_format':'matrix4','frame':'workspace','timestamp_file':'timestamp.npy'}}
            others=[n for n in active if n!=pick]
            if others:
                from scipy.spatial.transform import Rotation
                poses=np.array([[r['object_poses'][n] for n in others] for r in rows])
                quat=Rotation.from_matrix(poses[...,:3,:3].reshape(-1,3,3)).as_quat().reshape(len(poses),len(others),4)[...,[3,0,1,2]]
                np.save(folder/'object_poses.npy',np.concatenate([poses[...,:3,3],quat],axis=-1).astype(np.float32))
                telemetry.update({n:{'file':'object_poses.npy','index':i,'pose_format':'xyz_wxyz','frame':'workspace','timestamp_file':'timestamp.npy'} for i,n in enumerate(others)})
            atomic(folder/'replay.json',initial)
            atomic(folder/'outcome.json',dict(success=bool(outcome['success']),validation_only=outcome.get('validation_only',False),
                success_source=outcome.get('success_source','unknown'),coordinates=self.provenance['coordinates']))
            dest.mkdir(exist_ok=False);renamed=True
            folder.rename(dest)
            entry={'id':name,'path':name,'split':'train','instruction':self.provenance['instruction'],'videos':videos,
                'object_telemetry':telemetry,'scene_configuration':copy.deepcopy(self.provenance.get('scene_configuration')),
                'task_definition':copy.deepcopy(self.provenance['scene'].get('task_definition')),
                'group_id':self.provenance['source_group_prefix']+'/'+name}
            self.manifest['episodes'].append(entry)
            try:self.flush_manifest()
            except BaseException:self.manifest['episodes'].pop();raise
            return name
        except BaseException:
            if renamed:shutil.rmtree(dest,ignore_errors=True)
            video.abort();raise

class EpisodeReader:
    """Read inspected HDF5/Canonical rows without opening payloads at construction.

    The caller owns this reader and must close it (or use a context manager).
    First numeric read opens one read-only HDF5 handle; cameras open on demand.
    Returned numeric arrays do not depend on open file handles. RGB LRU is at most 8 frames/16MiB;
    at most two video readers remain open. Calls must be sequential, including
    when an existing Qt job queue hands this reader between worker threads.
    Close only after the last worker finishes; a closed reader cannot reopen.

    File/format errors propagate as OSError/KeyError/ValueError. Timestamp,
    frame and path violations raise ValueError; use after close is RuntimeError.
    No source file is written and no full RGB series is read implicitly.
    """
    def __init__(self,row:dict):
        self.row=row;self.ref=row['data_ref'];self.path=Path(self.ref['path']);self.canonical=self.path.is_dir()
        self.cache=OrderedDict();self.cache_bytes=0;self.bytes_read=0;self.video_readers=OrderedDict()
        self.frames=row['frames'];self.dt=row['dt'];self.specs=row.get('object_specs',{})
        self.objects=row.get('active_object_ids') or row.get('recorded_object_ids',list(self.specs))
        self.transforms=row['pose_transforms'];self.display_frame=row['display_frame']
        self._file=None;self._episode=None;self._closed=False

    def __enter__(self)->'EpisodeReader':
        if self._closed:raise RuntimeError('EpisodeReader is closed')
        return self

    def __exit__(self,exc_type,exc,tb):
        self.close()

    def close(self)->None:
        """Release owned HDF5/video handles and cached RGB; idempotent."""
        if self._closed:return
        self._closed=True;errors=[]
        for resource in [*self.video_readers.values(),self._file]:
            if resource is not None and hasattr(resource,'close'):
                try:resource.close()
                except Exception as exc:errors.append(exc)
        self._episode=None;self._file=None;self.video_readers.clear();self.cache.clear();self.cache_bytes=0
        if errors:raise errors[0]

    def read(self,key:str,selection=slice(None))->np.ndarray:
        if self._closed:raise RuntimeError('EpisodeReader is closed')
        if self.canonical:
            path=(self.path/self.ref['episode']/key).resolve()
            if not path.is_relative_to(self.path.resolve()):raise ValueError('Episode telemetry outside dataset')
            value=np.array(np.load(path,mmap_mode='r',allow_pickle=False)[selection],copy=True)
        else:
            import h5py
            if self._file is None:self._file=h5py.File(self.path,'r')
            if self._episode is None:self._episode=self._file['data'][self.ref['episode']]
            value=np.asarray(self._episode[key][selection])
        self.bytes_read+=value.nbytes
        return value

    def timestamps(self,selection=slice(None))->np.ndarray:
        key=self.row.get('timestamp_key')
        if key:return self.read(key,selection).reshape(-1)
        return self.indices(selection)*self.dt

    def action_timestamp(self,selection,*,observation_timestamps=None)->np.ndarray:
        key=self.row.get('action_timestamp_key')
        observed=self.timestamps(selection) if observation_timestamps is None else observation_timestamps
        stamp=self.read(key,selection).reshape(-1) if key and key!=self.row.get('timestamp_key') else observed
        if not np.allclose(stamp,observed,atol=1e-7,rtol=0):
            raise ValueError('Action/observation timestamp mismatch')
        return stamp

    def indices(self,selection):
        if isinstance(selection,slice):return np.arange(*selection.indices(self.frames))
        return np.asarray(selection).reshape(-1)

    def initial_pose(self,name:str)->np.ndarray|None:
        desc=self.row.get('object_telemetry',{}).get(name,{})
        if desc.get('initial_pose') is None or self.transforms.get(desc.get('frame')) is None:return None
        return np.asarray(self.transforms[desc['frame']])@pose_matrix(desc['initial_pose'],'xyzw' if desc.get('pose_format')=='xyz_xyzw' else 'wxyz')

    def ee(self,selection)->np.ndarray:
        value=self.read(self.row['ee_key'],selection)
        if value.shape[-2:]==(4,4):xyz=value[...,:3,3]
        else:xyz=value[...,:3]
        t=np.asarray(self.transforms[self.row['ee_frame']])
        return xyz@t[:3,:3].T+t[:3,3]

    def pose(self,name:str,selection,*,observation_timestamps=None)->np.ndarray|None:
        desc=self.row.get('object_telemetry',{}).get(name)
        if not desc or desc.get('unavailable'):return None
        t=self.transforms.get(desc['frame'])
        if t is None:return None
        indices=self.indices(selection)
        if desc.get('post_step'):
            # Existing states/workspace_states are AFTER action[t]. RGB/state[t] are BEFORE it.
            unique,inverse=np.unique(np.maximum(indices-1,0),return_inverse=True)
            old=self.read(desc['path'],unique)[inverse]
            if 0 in indices:old[indices==0]=np.asarray(desc['initial_pose'])
        else:
            old=self.read(desc.get('file',desc.get('path')),selection)
            if 'index' in desc:old=old[...,desc['index'],:]
        tskey=desc.get('timestamp_file') or desc.get('timestamp_path')
        if tskey:
            observed=self.timestamps(selection) if observation_timestamps is None else observation_timestamps
            stamp=observed if tskey==self.row.get('timestamp_key') else self.read(tskey,selection).reshape(-1)
            if not np.allclose(stamp,observed,atol=1e-7,rtol=0):raise ValueError('Object/observation timestamp mismatch: '+name)
        matrices=pose_matrix(old,'xyzw' if desc.get('pose_format')=='xyz_xyzw' else 'wxyz')
        return np.asarray(t)@matrices

    def frame(self,index:int,camera:str|Sequence[str]|None=None)->dict:
        if self._closed:raise RuntimeError('EpisodeReader is closed')
        if not 0<=index<self.frames:raise ValueError('Frame outside episode')
        timestamps=self.timestamps(index)
        poses={n:self.pose(n,index,observation_timestamps=timestamps) for n in self.objects}
        result={'index':index,'timestamp':float(timestamps[0]),'robot_q':self.read(self.row['q_key'],index) if self.row.get('q_key') else None,
                'action':self.read(self.row['action_key'],index),'action_timestamp':float(self.action_timestamp(index,observation_timestamps=timestamps)[0]),
                'ee':self.ee(index),'object_poses':{n:(v[0] if v is not None and v.ndim==3 else v) for n,v in poses.items()},
                'initial_object_poses':{n:self.initial_pose(n) for n,v in poses.items() if v is None},'rgb':None}
        cameras=[camera] if isinstance(camera,str) else list(camera or [])
        result['rgb_frames']={};result['rgb_timestamps']={}
        for camera in cameras:
            spec=self.row['cameras'][camera];key=(camera,index)
            if key not in self.cache:
                if spec.get('file') or spec['path'].endswith('.mp4'):
                    from ..utils.async_video_writer import VideoFrames,external_camera
                    if camera not in self.video_readers:
                        self.video_readers[camera]=(VideoFrames(self.path/self.ref['episode']/spec['path'],self.frames,1/self.dt)
                            if self.canonical else external_camera(self.path,spec,self.frames,1/self.dt))
                        while len(self.video_readers)>2:
                            _,old=self.video_readers.popitem(last=False)
                            if hasattr(old,'close'):old.close()
                    self.video_readers.move_to_end(camera)
                    rgb=self.video_readers[camera][index][...,:3]
                else:rgb=self.read(spec['path'],index)[...,:3]
                if rgb.dtype!=np.uint8:rgb=np.clip(rgb*(255 if rgb.max(initial=0)<=1.5 else 1),0,255).astype(np.uint8)
                rgb=np.ascontiguousarray(rgb)
                self.cache[key]=rgb;self.cache_bytes+=rgb.nbytes
                while len(self.cache)>8 or self.cache_bytes>16*2**20:
                    _,old=self.cache.popitem(last=False);self.cache_bytes-=old.nbytes
                result['rgb']=rgb
            else:self.cache.move_to_end(key);result['rgb']=self.cache[key]
            tskey=spec.get('timestamp_key')
            stamp=float(self.read(tskey,index).reshape(-1)[0]) if tskey and tskey!=self.row.get('timestamp_key') else result['timestamp']
            result['rgb_timestamp']=stamp
            result['rgb_frames'][camera]=result['rgb'];result['rgb_timestamps'][camera]=stamp
            age=result['timestamp']-stamp
            if age < -1e-7 or age > 2*self.dt+1e-7:raise ValueError('RGB frame is not aligned with the observation')
        return result


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def _write(group,values):
    for key,value in values.items():
        if isinstance(value,dict):_write(group.require_group(key),value)
        else:group.create_dataset(key,data=np.asarray(value),compression='gzip' if np.asarray(value).ndim else None)


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


class DatasetWriter:
    """One caller serializes commits into a new Common HDF5/MP4 bundle.

    Borrow values/attrs/video descriptors only until commit returns; the caller
    must not mutate them concurrently. Numeric input is never changed. Shape,
    units declared by the format, timestamp/pose and video contracts are checked
    before a private group is committed. No observation or simulation is read.
    A finalized writer-owned VideoEpisode may transfer its MP4 files by rename;
    external input videos are copied and kept unchanged. No re-encoding occurs.

    close(publish=True) publishes completed episodes with an exclusive hard link;
    close(False) discards this writer's private files. Context exceptions discard,
    while a Stage may explicitly publish completed episodes after cancellation.
    close is idempotent. Existing paths raise FileExistsError; contract violations
    raise ValueError; I/O/encoding failures propagate after commit rollback.
    """
    def __init__(self,path,args,contract):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():raise FileExistsError(self.path)
        self.temp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        import h5py
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
        # Only the converted pose maps are replaced. Borrow unchanged signals;
        # recursively copying all telemetry would duplicate large numeric arrays.
        info=dict(values['obs']['datagen_info'])
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
