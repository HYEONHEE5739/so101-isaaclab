"""Read-only, bounded-memory episode inspection and non-destructive selection."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .tasks import episode_definition
from .robot_preview import pose_matrix


def _json(value, default=None):
    if value is None:return default
    if isinstance(value,bytes):value=value.decode()
    return json.loads(value) if isinstance(value,str) else value


def _workspace(args):
    from ..real2sim.workspaces.package import load,metadata,ROOT
    identity=args.get('workspace',{}) or {};path=args.get('workspace_path')
    if not path and identity.get('workspace_id'):path=ROOT/'outputs/real2sim/environments'/identity['workspace_id']/('v'+str(identity['version']))
    if not path or not (Path(path)/'manifest.json').is_file():return None
    w=load(path)
    if identity and identity!=metadata(w):raise ValueError('Curation Workspace identity differs from recorded dataset')
    return w


def _group_id(path, name, attrs, args):
    explicit=attrs.get('source_group') or attrs.get('group_id') or args.get('source_groups',{}).get(name)
    if explicit:return str(explicit)
    generated=args.get('source_dataset') or args.get('source_env_args')
    if generated:return 'UNKNOWN:'+str(args.get('source_dataset',Path(path).resolve()))
    return str(Path(path).resolve())+'/'+name


class EpisodeReader:
    """Open one episode on demand. No RGB series reads; LRU caches at most 8 frames/16MiB."""
    def __init__(self,row):
        self.row=row;self.ref=row['data_ref'];self.path=Path(self.ref['path']);self.canonical=self.path.is_dir()
        self.cache=OrderedDict();self.cache_bytes=0;self.bytes_read=0;self.video_readers=OrderedDict()
        self.frames=row['frames'];self.dt=row['dt'];self.specs=row.get('object_specs',{})
        self.objects=row.get('active_object_ids') or row.get('recorded_object_ids',list(self.specs))
        self.transforms=row['pose_transforms'];self.display_frame=row['display_frame']

    def read(self,key,selection=slice(None)):
        if self.canonical:
            path=(self.path/self.ref['episode']/key).resolve()
            if not path.is_relative_to(self.path.resolve()):raise ValueError('Episode telemetry outside dataset')
            value=np.asarray(np.load(path,mmap_mode='r',allow_pickle=False)[selection])
        else:
            import h5py
            with h5py.File(self.path,'r') as h:value=np.asarray(h['data'][self.ref['episode']][key][selection])
        self.bytes_read+=value.nbytes
        return value

    def timestamps(self,selection=slice(None)):
        key=self.row.get('timestamp_key')
        if key:return self.read(key,selection).reshape(-1)
        return self.indices(selection)*self.dt

    def action_timestamp(self,selection):
        key=self.row.get('action_timestamp_key')
        stamp=self.read(key,selection).reshape(-1) if key else self.timestamps(selection)
        if not np.allclose(stamp,self.timestamps(selection),atol=1e-7,rtol=0):
            raise ValueError('Action/observation timestamp mismatch')
        return stamp

    def indices(self,selection):
        if isinstance(selection,slice):return np.arange(*selection.indices(self.frames))
        return np.asarray(selection).reshape(-1)

    def initial_pose(self,name):
        desc=self.row.get('object_telemetry',{}).get(name,{})
        if desc.get('initial_pose') is None or self.transforms.get(desc.get('frame')) is None:return None
        return np.asarray(self.transforms[desc['frame']])@pose_matrix(desc['initial_pose'],'xyzw' if desc.get('pose_format')=='xyz_xyzw' else 'wxyz')

    def ee(self,selection):
        value=self.read(self.row['ee_key'],selection)
        if value.shape[-2:]==(4,4):xyz=value[...,:3,3]
        else:xyz=value[...,:3]
        t=np.asarray(self.transforms[self.row['ee_frame']])
        return xyz@t[:3,:3].T+t[:3,3]

    def pose(self,name,selection):
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
        if tskey and not np.allclose(self.read(tskey,selection).reshape(-1),self.timestamps(selection),atol=1e-7,rtol=0):
            raise ValueError('Object/observation timestamp mismatch: '+name)
        matrices=pose_matrix(old,'xyzw' if desc.get('pose_format')=='xyz_xyzw' else 'wxyz')
        return np.asarray(t)@matrices

    def frame(self,index,camera=None):
        if not 0<=index<self.frames:raise ValueError('Frame outside episode')
        poses={n:self.pose(n,index) for n in self.objects}
        result={'index':index,'timestamp':float(self.timestamps(index)[0]),'robot_q':self.read(self.row['q_key'],index) if self.row.get('q_key') else None,
                'action':self.read(self.row['action_key'],index),'action_timestamp':float(self.action_timestamp(index)[0]),
                'ee':self.ee(index),'object_poses':{n:(v[0] if v is not None and v.ndim==3 else v) for n,v in poses.items()},
                'initial_object_poses':{n:self.initial_pose(n) for n,v in poses.items() if v is None},'rgb':None}
        if camera:
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
            stamp=float(self.read(spec['timestamp_key'],index).reshape(-1)[0]) if spec.get('timestamp_key') else result['timestamp']
            result['rgb_timestamp']=stamp
            age=result['timestamp']-stamp
            if age < -1e-7 or age > 2*self.dt+1e-7:raise ValueError('RGB frame is not aligned with the observation')
        return result


def _context(w,scene,canonical):
    world_ws=scene.get('workspace_T_world') or (w['profile']['workspace']['T_world'] if w else None)
    world_base=scene.get('robot_base_T_world') or (w['profile']['robot_base']['T_world'] if w else None)
    if world_base is not None:
        base_world=np.linalg.inv(np.asarray(world_base));display='robot_base'
        transforms={'robot_base':np.eye(4).tolist(),'world':base_world.tolist()}
        if world_ws is not None:transforms['workspace']=(base_world@world_ws).tolist()
        up=np.asarray(world_base)[:3,:3].T@np.array([0,0,1.])
    else:
        display='workspace' if canonical else 'robot_base';transforms={display:np.eye(4).tolist()};up=np.array([0,0,1.])
    return display,transforms,up


def _events(reader,annotation=None,signal_keys=None):
    events={}
    if annotation:
        if annotation.get('boundaries'):events.update(annotation['boundaries'])
        else:events.update({k:annotation[k] for k in ('grasp','place','transport') if k in annotation})
    for name,key in (signal_keys or {}).items():
        if name in events:continue
        for start in range(0,reader.frames,2048):
            signal=reader.read(key,slice(start,start+2048)).reshape(-1)
            found=np.flatnonzero(signal>0)
            if len(found):events[name]=int(start+found[0]);break
    for name,value in events.items():
        if type(value) is not int or not 0<=value<reader.frames:raise ValueError('Invalid recorded event boundary: '+name)
    return events


def _scan(row):
    reader=EpisodeReader(row);events=row['events'];pick=row['pick_object_id']
    total_move=before=after=0.;initial_height=lift_max=None;last_ee=last_q=last_v=last_obj=last_t=None
    duration=stop=cmdstop=samples=0;accsum=np.zeros(6);accn=0;distances={};dist_sum=dist_n=0;time_start=None
    for start in range(0,reader.frames,2048):
        end=min(reader.frames,start+2048);sl=slice(start,end);t=reader.timestamps(sl)
        reader.action_timestamp(sl)
        ee=reader.ee(sl);q=reader.read(row['action_key'],sl)
        if len(ee)!=len(t) or q.shape!=(len(t),6) or not np.isfinite(t).all() or not np.isfinite(ee).all() or not np.isfinite(q).all():raise ValueError('Invalid aligned trajectory: '+row['name'])
        if time_start is None:time_start=float(t[0])
        objects={n:reader.pose(n,sl) for n in reader.objects}
        for n,poses in objects.items():
            if poses is None:continue
            dist=np.linalg.norm(poses[:,:3,3]-ee,axis=1);distances[n]=min(distances.get(n,float('inf')),float(dist.min()))
            if n==pick:dist_sum+=float(dist.sum());dist_n+=len(dist)
        obj=objects.get(pick)
        for i,(stamp,p,command) in enumerate(zip(t,ee,q)):
            if last_t is not None:
                dt=stamp-last_t
                if dt<=0:raise ValueError('Non-increasing observation timestamps')
                stop+=np.linalg.norm(p-last_ee)/dt<.002;cmdstop+=np.max(np.abs(command-last_q))/dt<np.deg2rad(1);samples+=1
                v=(command-last_q)/dt
                if last_v is not None:accsum+=((v-last_v)/dt)**2;accn+=1
                last_v=v
            last_t,last_ee,last_q=stamp,p,command
            if obj is not None:
                point=obj[i,:3,3];height=float(point@np.asarray(row['up_axis']))
                if initial_height is None:initial_height=height
                lift_max=height if lift_max is None else max(lift_max,height)
                if last_obj is not None:
                    move=float(np.linalg.norm(point-last_obj));total_move+=move
                    if 'grasp' in events:
                        if start+i<=events['grasp']:before+=move
                        else:after+=move
                last_obj=point
    initial=reader.pose(pick,0)
    if initial is None:initial=reader.initial_pose(pick)
    initial_xyz=initial[0,:3,3] if initial is not None and initial.ndim==3 else initial[:3,3] if initial is not None else None
    row.update(duration=float(last_t-time_start+row['dt']),stop_fraction=float(stop/max(samples,1)),command_stop_fraction=float(cmdstop/max(samples,1)),
        jitter=float(np.sqrt(accsum[:5].mean()/max(accn,1))),gripper_jitter=float(np.sqrt(accsum[5]/max(accn,1))),
        target_movement_m=total_move if last_obj is not None else None,lift_height_m=max(0.,lift_max-initial_height) if initial_height is not None else None,
        target_ee_distance_m={'min':distances.get(pick),'mean':dist_sum/dist_n if dist_n else None},
        movement_before_grasp_m=before if 'grasp' in events and last_obj is not None else None,
        movement_after_grasp_m=after if 'grasp' in events and last_obj is not None else None,
        object_approach={n:{'min_distance_m':distances.get(n),'within_40mm':distances[n]<.04 if n in distances else None} for n in reader.objects})
    # Scatter remains workspace XY, independent from the base-frame animation.
    desc=row.get('object_telemetry',{}).get(pick,{})
    if initial_xyz is not None and row.get('base_to_workspace') is not None:
        t=np.asarray(row['base_to_workspace']);row['xy']=(t[:3,:3]@initial_xyz+t[:3,3])[:2].tolist()
    else:row.setdefault('xy',None)
    return row


def analyze(path):
    path=Path(path).resolve();canonical=path.is_dir();rows=[]
    if canonical:
        manifest=json.loads((path/'manifest.json').read_text());prov=json.loads((path/'project_provenance.json').read_text())
        entries=manifest['episodes'];args=prov;scene=prov.get('scene',{});dt=1/float(manifest['control_hz']);w=_workspace(args)
    else:
        import h5py
        from ..so101_dataset_contract import require_joint_replay
        require_joint_replay(path)
        with h5py.File(path,'r') as h:
            args=_json(h['data'].attrs.get('env_args'),{});entries=[n for n in h['data'] if n.startswith('demo_')]
        scene={};w=_workspace(args);sim=args.get('sim_args',{});dt=float(args.get('control_dt') or float(sim.get('dt',0))*int(sim.get('decimation',1)))
        if dt<=0:raise ValueError('Dataset control_dt or sim dt/decimation is required')
    display,base_transforms,base_up=_context(w,scene,canonical)
    base_to_ws=np.linalg.inv(np.asarray(base_transforms['workspace'])) if 'workspace' in base_transforms else None
    specs=w['profile']['objects'] if w else {}
    from ..real2sim.precision import uses_urdf_frame
    contract=prov.get('coordinate_contract',{}) if canonical else args.get('so101_coordinates',{})
    urdf=w['root']/'assets/SO101/urdf/so101_isaaclab.urdf' if w else None
    if w and contract.get('simulation_model'):
        from ..real2sim.precision import simulation_configuration
        urdf=w['root']/simulation_configuration(w,coordinates=contract)['robot_model']['urdf']
    elif w and not uses_urdf_frame(contract) and (w['root']/'assets/SO101/urdf/so101_sim.urdf').is_file():urdf=w['root']/'assets/SO101/urdf/so101_sim.urdf'
    for entry in entries:
        transforms=dict(base_transforms);up=base_up.copy()
        if canonical:
            name=entry['id'];folder=(path/entry['path']).resolve()
            if not folder.is_relative_to(path):raise ValueError('Episode outside dataset')
            task=entry.get('task_definition') or scene.get('task_definition');composition=entry.get('scene_configuration') or prov.get('scene_configuration')
            frames=len(np.load(folder/'raw_action.npy',mmap_mode='r'));telemetry=dict(entry.get('object_telemetry',{}))
            pick=task['pick_object_id'] if task else (composition or {}).get('pick_object_id')
            if not telemetry and (folder/'object_pose.npy').exists() and pick:telemetry[pick]={'file':'object_pose.npy','frame':'workspace','pose_format':'matrix4'}
            cameras={p.stem:{'path':p.name,'timestamp_key':p.stem+'_timestamp.npy' if (folder/(p.stem+'_timestamp.npy')).exists() else None} for p in folder.iterdir() if p.suffix in ('.npy','.mp4') and p.stem in manifest.get('cameras',['side','wrist'])}
            outcome=json.loads((folder/'outcome.json').read_text()) if (folder/'outcome.json').exists() else {}
            annotation=json.loads((folder/'annotation.json').read_text()) if (folder/'annotation.json').exists() else None
            qkey='raw_state.npy' if (folder/'raw_state.npy').exists() else None
            row=dict(name=name,data_ref={'path':str(path),'episode':entry['path']},frames=frames,dt=dt,
                action_key='raw_action.npy',action_timestamp_key='action_timestamp.npy' if (folder/'action_timestamp.npy').is_file() else None,
                q_key=qkey,ee_key='eef_pose.npy',ee_frame='workspace',timestamp_key='timestamp.npy',
                recorded_success=outcome.get('success') if outcome.get('success_known',True) else None,
                success_source=outcome.get('success_source','unknown'),validation_only=outcome.get('validation_only',False),
                source_group=entry.get('source_group',entry.get('group_id','UNKNOWN')),cameras=cameras,object_telemetry=telemetry,
                source_partition=entry.get('source_partition') or prov.get('source_partition'),source_lineage=entry.get('source_lineage'))
            signal_keys={}
        else:
            name=entry
            with h5py.File(path,'r') as h:
                g=h['data'][name];task=episode_definition(g,args);composition=_json(g.attrs.get('scene_configuration'),args.get('scene_configuration'))
                policy='obs/policy' if 'obs/policy' in g else 'obs';frames=len(g['joint_targets']);pick=task['pick_object_id'] if task else args.get('grasp_object')
                telemetry=dict(_json(g.attrs.get('object_telemetry'),args.get('object_telemetry',{})))
                active_ids=(composition or {}).get('active_object_ids',list(telemetry) or None)
                if not active_ids and w and 'initial_state/rigid_object' in g:
                    active_ids=[n for n in g['initial_state/rigid_object'] if n in w['task']['dynamic']]
                ids=active_ids or ([pick] if pick else list(telemetry))
                # Isaac legacy states may be environment-relative, not world coordinates.
                # Their recorded robot root and object roots share that frame, so use it directly.
                legacy_frame='world';root_key='initial_state/articulation/robot/root_pose'
                if root_key in g and args.get('root_quaternion_order') in ('wxyz','xyzw'):
                    base=pose_matrix(np.asarray(g[root_key]).reshape(-1,7)[0],args['root_quaternion_order'])
                    transforms['recorded_scene']=np.linalg.inv(base);legacy_frame='recorded_scene'
                    up=base[:3,:3].T@np.array([0,0,1.])
                for n in ids:
                    if n in telemetry and telemetry[n].get('path') in g:continue
                    old=next((key for key in ('states/rigid_object/'+n+'/root_pose','workspace_states/'+n) if key in g),None)
                    initial='initial_state/rigid_object/'+n+'/root_pose'
                    if old and initial in g and len(g[old])==frames:
                        telemetry[n]={'path':old,'frame':legacy_frame,'pose_format':'xyz_'+args.get('root_quaternion_order','wxyz'),
                                      'post_step':True,'initial_pose':np.asarray(g[initial]).reshape(-1,7)[0].tolist()}
                    else:
                        telemetry[n]={'unavailable':True,'frame':legacy_frame,'pose_format':'xyz_'+args.get('root_quaternion_order','wxyz')}
                        if initial in g:telemetry[n]['initial_pose']=np.asarray(g[initial]).reshape(-1,7)[0].tolist()
                cameras={key[:-4]:{'path':policy+'/'+key,'timestamp_key':args.get('observation_timestamp_path')} for key in g[policy] if key.endswith('_cam')}
                for key,spec in _json(g.attrs.get('external_videos'),{}).items():
                    cameras[key.removesuffix('_cam')]={**spec,'timestamp_key':args.get('observation_timestamp_path')}
                if 'obs/camera' in g:
                    for key,value in g['obs/camera'].items():
                        if len(value.shape)==4 and value.shape[-1]>=3:
                            if value.shape[0]!=frames:raise ValueError('Legacy RGB/observation frame count mismatch')
                            cameras.setdefault(key.removesuffix('_cam'),{'path':'obs/camera/'+key,'timestamp_key':args.get('observation_timestamp_path')})
                signal_keys={label:key for label,key in [('grasp','obs/events/grasp'),('place','obs/events/place'),('grasp','obs/datagen_info/subtask_term_signals/grasp')] if key in g}
                annotation=_json(g.attrs.get('annotation'))
                row=dict(name=name,data_ref={'path':str(path),'episode':name},frames=frames,dt=dt,
                    action_key='joint_targets',action_timestamp_key=args.get('observation_timestamp_path'),
                    q_key=policy+'/joint_pos' if policy+'/joint_pos' in g else None,ee_key=policy+'/ee_state',
                    ee_frame=args.get('ee_frame','robot_base'),timestamp_key=args.get('observation_timestamp_path'),
                    recorded_success=bool(g.attrs['success']) if 'success' in g.attrs else None,
                    success_source=g.attrs.get('success_source',args.get('success_source','unknown')),validation_only=args.get('validation_only',False),
                    source_group=_group_id(path,name,g.attrs,args),cameras=cameras,object_telemetry=telemetry,
                    source_partition=args.get('source_partition'),source_lineage=_json(g.attrs.get('source_lineage')))
                row['legacy_pose_alignment']='initial[0] + post_state[:-1]' if any(d.get('post_step') for d in telemetry.values()) else None
        from .episodes import validate_partition,validate_partition_episode
        partition=validate_partition(row['source_partition']) if row.get('source_partition') else None
        if partition:validate_partition_episode(partition,row.get('source_lineage'),args.get('source_partition_input'))
        row.update(split=partition['split'] if partition else None,lineage_status='Verified Source Partition / exact subtask lineage UNKNOWN' if partition and (row.get('source_lineage') or {}).get('kind')!='source_episode' else 'Verified Source Partition' if partition else 'Not partitioned',leakage_status='checked at training' if partition else 'UNKNOWN')
        if row['frames']<3:raise ValueError('At least three frames required for motion quality metrics')
        ids=(composition or {}).get('active_object_ids',scene.get('dynamic_objects') if canonical else active_ids)
        geometry_ids=ids or list(telemetry)
        row.update(task_id=task['task_id'] if task else 'UNKNOWN',task_definition=task,pick_object_id=pick,scene_configuration=composition,
            active_object_ids=ids or None,recorded_object_ids=list(telemetry),active_object_count=len(ids) if ids else None,object_specs={n:specs[n] for n in geometry_ids if n in specs},
            robot_urdf=str(urdf) if urdf else None,display_frame=display,pose_transforms=transforms,up_axis=up.tolist(),
            base_to_workspace=base_to_ws.tolist() if base_to_ws is not None else None,flags=[])
        if w and composition:
            from .tasks import recorded_scene
            recorded_scene({'scene_configuration':composition},w)
            if task and composition['pick_object_id']!=pick:raise ValueError('Episode target conflicts with scene configuration')
        reader=EpisodeReader(row);row['events']=_events(reader,annotation,signal_keys)
        row['annotation_status']='annotated' if annotation is not None or 'obs/datagen_info/subtask_term_signals/grasp' in signal_keys.values() else 'signals_only' if signal_keys else 'UNAVAILABLE'
        trusted=row['success_source'] in ('simulation_place_condition','simulation_stable_grasp_and_place','simulation_success_term')
        row['physical_success']=row['recorded_success'] if trusted and not row['validation_only'] else None
        rows.append(_scan(row))
    return flag_outliers(rows)


def analyze_canonical(path):return analyze(path)


def load_episode(row,max_points=1500):
    """Selected episode plot samples only; playback still reads each exact original frame."""
    reader=EpisodeReader(row);indices=np.unique(np.linspace(0,reader.frames-1,min(max_points,reader.frames)).astype(int))
    return {**row,'sample_indices':indices,'ee':reader.ee(indices),'actions':reader.read(row['action_key'],indices),
        'robot_q':reader.read(row['q_key'],indices) if row.get('q_key') else None,
        'object_paths':{n:reader.pose(n,indices) for n in reader.objects},'timestamps':reader.timestamps()}


def flag_outliers(rows):
    if not rows:raise ValueError('에피소드가 없습니다.')
    keys={(r['task_id'],tuple(r.get('active_object_ids') or [])) for r in rows}
    for task,objects in keys:
        group=[r for r in rows if r['task_id']==task and tuple(r.get('active_object_ids') or [])==objects]
        for key,low_label,high_label in [('duration','짧은 시간','긴 시간'),('stop_fraction',None,'EE 정지 비율 높음'),('command_stop_fraction',None,'정지 명령 많음'),('jitter',None,'명령 jitter 높음')]:
            values=np.array([r[key] for r in group]);median=np.median(values);spread=max(1.4826*np.median(abs(values-median)),abs(median)*.1,1e-6)
            for r in group:
                if len(group)>=5 and r[key]>median+3*spread:r['flags'].append(high_label)
                if len(group)>=5 and low_label and r[key]<median-3*spread:r['flags'].append(low_label)
    return rows


def select_distribution(rows,count):
    """Task/composition × XY strata, then source-group round robin; never assign split labels."""
    if not 1<=count<=len(rows):raise ValueError('선별 개수는 사용 가능한 episode 수 이하여야 합니다.')
    known=[r['xy'] for r in rows if r.get('xy') is not None];xy=np.array(known or [[0,0]]);lo=xy.min(0);span=np.maximum(xy.max(0)-lo,1e-9)
    groups={}
    for row in rows:
        cell=tuple(np.minimum(((np.array(row['xy'])-lo)/span*4).astype(int),3)) if row.get('xy') is not None else ('UNKNOWN','UNKNOWN')
        key=(row.get('split'),row['task_id'],tuple(row.get('active_object_ids') or []),*cell)
        groups.setdefault(key,[]).append(row)
    keys=sorted(groups,key=repr);quotas={k:count*len(groups[k])/len(rows) for k in keys};amounts={k:int(quotas[k]) for k in keys}
    for k in sorted(keys,key=lambda k:(-(quotas[k]-amounts[k]),repr(k)))[:count-sum(amounts.values())]:amounts[k]+=1
    chosen=[];used={}
    for k in keys:
        candidates=sorted(groups[k],key=lambda r:(len(r['flags']),r['jitter'],r['name']))
        for _ in range(amounts[k]):
            best=min(candidates,key=lambda r:(used.get(r.get('source_group',r['name']),0),len(r['flags']),r['jitter'],r['name']))
            group=best.get('source_group',best['name']);used[group]=used.get(group,0)+1
            chosen.append(best['name']);candidates.remove(best)
    return chosen


def export_subset(source, destination, names, *, partition=None, source_records=None):
    import h5py, os, uuid
    from ..so101_dataset_contract import read_contract, write_contract
    from .artifacts import fingerprint
    source=Path(source); destination=Path(destination)
    from .episodes import validate_partition,validate_partition_episode,source_catalog
    if partition:validate_partition(partition)
    if source_records is None:
        source_records,inherited=source_catalog(source,allow_unknown=True)
        if inherited and source_records:
            ids=sorted({source_records[n]['source_id'] for n in names})
            if ids!=inherited['source_ids']:
                from ..real2sim.precision import fingerprint as contract_hash
                payload={k:v for k,v in inherited.items() if k!='sha256'}
                payload.update(source_ids=ids,parent_partition_sha256=inherited['sha256'])
                partition=validate_partition({**payload,'sha256':contract_hash(payload)})
    if partition:
        for name in names:validate_partition_episode(partition,source_records.get(name))
    if destination.exists(): raise ValueError('기존 파일은 덮어쓰지 않습니다.')
    if not names or len(set(names))!=len(names): raise ValueError('선택 목록이 비었거나 중복입니다.')
    if source.is_dir():
        import shutil
        from ..real2sim.storage import atomic
        manifest = json.loads((source / 'manifest.json').read_text())
        entries = {e['id']: e for e in manifest['episodes']}
        if not set(names).issubset(entries): raise ValueError('알 수 없는 episode입니다.')
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            folder = (source / entries[name]['path']).resolve()
            if not folder.is_relative_to(source.resolve()): raise ValueError('Episode outside dataset')
            shutil.copytree(folder, destination / entries[name]['path'])
        selected=[]
        for name in names:
            entry=dict(entries[name])
            if source_records.get(name):entry['source_lineage']=source_records[name]
            if partition:entry.update(split=partition['split'],source_partition=partition)
            selected.append(entry)
        atomic(destination / 'manifest.json', {**manifest, 'episodes': selected})
        provenance = json.loads((source / 'project_provenance.json').read_text())
        scenes=[entries[n].get('scene_configuration') or provenance.get('scene_configuration') for n in names]
        scenes=list({s['sha256']:s for s in scenes if s}.values())
        if partition:provenance['source_partition']=partition
        atomic(destination / 'project_provenance.json', {**provenance, 'scene_configurations':scenes, 'curation': {
            'source': str(source.resolve()), 'source_sha256': fingerprint(source), 'selected_episodes': names}})
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with h5py.File(source,'r') as src,h5py.File(temp,'w') as out:
            for k,v in src.attrs.items():out.attrs[k]=v
            data=out.create_group('data')
            for k,v in src['data'].attrs.items():data.attrs[k]=v
            total=0
            args=_json(src['data'].attrs.get('env_args'),{})
            if partition:
                args['source_partition']=partition
                data.attrs['env_args']=json.dumps(args)
            for i,name in enumerate(names):
                src.copy(src['data'][name],data,name=f'demo_{i}')
                from ..utils.async_video_writer import copy_hdf5_videos
                copy_hdf5_videos(src['data'][name],data[f'demo_{i}'],destination)
                data[f'demo_{i}'].attrs['curation_source_episode']=name
                data[f'demo_{i}'].attrs['source_group']=_group_id(source,name,src['data'][name].attrs,args)
                if source_records.get(name):data[f'demo_{i}'].attrs['source_lineage']=json.dumps(source_records[name])
                if partition:data[f'demo_{i}'].attrs['split']=partition['split']
                total+=len(data[f'demo_{i}/actions'])
            data.attrs['total']=total
            data.attrs['curation']=json.dumps({'source':str(source.resolve()),'source_sha256':fingerprint(source),
                'selected_episodes':names,'selection':'user-reviewed subset; quality flags are diagnostics'})
            out.flush()
        # Exclusive publication: never replace an existing destination.
        os.link(temp,destination)
        write_contract(destination,read_contract(source))
    finally:temp.unlink(missing_ok=True)
    return destination
