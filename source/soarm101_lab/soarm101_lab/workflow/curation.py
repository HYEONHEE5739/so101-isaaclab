"""Read-only, bounded-memory episode inspection and non-destructive selection."""
import json
from pathlib import Path
import numpy as np
from .tasks import episode_definition
from ..data import episodes as episode_io
from ..utils.transforms import pose_matrix


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


def placement_region(workspace, composition, pick, base_to_workspace):
    """The same target-center envelope used by WorkspaceRandomizer.sample, in workspace XY."""
    if not workspace or base_to_workspace is None:return None
    reset=(composition or {}).get('reset') or workspace['task']['reset']
    specs=workspace['profile']['objects'];support=specs.get(reset.get('support_object'));target=specs.get(pick)
    if not support or not target:return None
    center=np.asarray(support['T_workspace'])[:2,3];half=np.asarray(support['dimensions_m'])[:2]/2
    limit=half-np.asarray(target['dimensions_m'])[:2]/2-reset['margin_m']
    if np.any(limit<=0):return None
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    return {'support':(center+corners*half).tolist(),'allowed':(center+corners*limit).tolist(),
            'robot_xy':np.asarray(base_to_workspace)[:2,3].tolist(),'mode':reset.get('mode','uniform')}


def _events(reader,annotation=None,signal_keys=None):
    events={}
    if annotation:
        if annotation.get('events'):events.update(annotation['events'])
        elif annotation.get('boundaries'):events.update(annotation['boundaries'])
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


def _scan(row,reader):
    events=row['events'];pick=row['pick_object_id']
    total_move=before=after=0.;initial_height=lift_max=None;last_ee=last_q=last_v=last_obj=last_t=None
    duration=stop=cmdstop=samples=0;accsum=np.zeros(6);accn=0;distances={};dist_sum=dist_n=0;time_start=None
    for start in range(0,reader.frames,2048):
        end=min(reader.frames,start+2048);sl=slice(start,end);t=reader.timestamps(sl)
        reader.action_timestamp(sl,observation_timestamps=t)
        ee=reader.ee(sl);q=reader.read(row['action_key'],sl)
        if len(ee)!=len(t) or q.shape!=(len(t),6) or not np.isfinite(t).all() or not np.isfinite(ee).all() or not np.isfinite(q).all():raise ValueError('Invalid aligned trajectory: '+row['name'])
        if time_start is None:time_start=float(t[0])
        objects={n:reader.pose(n,sl,observation_timestamps=t) for n in reader.objects}
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


def evaluation_inputs(path):
    """Read a run index, or discover legacy per-episode files without rewriting them."""
    path=Path(path).resolve()
    if not path.is_dir() or (path/'manifest.json').is_file():return None
    marker=path/'evaluation_datasets.json'
    if marker.is_file():
        report=json.loads(marker.read_text())
        if report.get('schema')!='so101.sim-evaluation/1':raise ValueError('Unknown evaluation dataset schema')
        files=[(path/item['path']).resolve() for item in report['datasets']]
    else:
        import h5py
        files=[]
        for candidate in sorted([*path.glob('*.hdf5'),*path.glob('*.h5'),*path.glob('result/*.hdf5')]):
            with h5py.File(candidate,'r') as h:
                args=_json(h['data'].attrs.get('env_args'),{}) if 'data' in h else {}
            if args.get('evaluation_schema'):files.append(candidate.resolve())
        if not files:return None
    if any(not p.is_relative_to(path) or p.suffix not in ('.hdf5','.h5') for p in files):
        raise ValueError('Evaluation dataset outside run folder')
    return files


def analyze(path):
    path=Path(path).resolve();canonical=path.is_dir();rows=[]
    if any(p.name and p.with_name('.'+p.name+'.workflow-lock').exists() for p in (path,*path.parents)):
        raise ValueError('Dataset 저장 중에는 Curation을 열 수 없습니다.')
    inputs=evaluation_inputs(path) if canonical else None
    if inputs is not None:
        if not inputs:raise ValueError('평가에 저장된 완료 episode가 없습니다.')
        for child in inputs:
            for row in analyze(child):
                prefix=child.parent.relative_to(path) if (path/'evaluation_datasets.json').is_file() else child.relative_to(path).with_suffix('')
                row['name']=str(prefix)+'/'+row['name']
                row['evaluation_run']=str(path);row['flags']=[];row['flag_details']=[]
                rows.append(row)
        return flag_outliers(rows)
    if canonical and not (path/'manifest.json').is_file():raise ValueError('Curation 편집 지원: Common HDF5 / Legacy NPY·MP4. LeRobot Dataset 편집은 지원하지 않습니다.')
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
                signal_keys={name:prefix+'/'+name for prefix in ('obs/events','obs/datagen_info/subtask_term_signals') if prefix in g for name in g[prefix]}
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
        from .episodes import validate_partition
        partition=validate_partition(row['source_partition']) if row.get('source_partition') else None
        kind=(row.get('source_lineage') or {}).get('kind')
        lineage_status=('Historical Source Partition / exact subtask Source' if kind=='mimic_segments' else
            'Historical Source Partition / exact subtask lineage UNKNOWN' if kind!='source_episode' else 'Historical Source Partition') if partition else 'Not partitioned'
        row.update(split=None,historical_split=partition['split'] if partition else None,lineage_status=lineage_status,leakage_status='historical evidence only')
        if row['frames']<3 and not args.get('evaluation_schema'):raise ValueError('At least three frames required for motion quality metrics')
        ids=(composition or {}).get('active_object_ids',scene.get('dynamic_objects') if canonical else active_ids)
        geometry_ids=ids or list(telemetry)
        row.update(task_id=task['task_id'] if task else 'UNKNOWN',task_definition=task,pick_object_id=pick,scene_configuration=composition,
            active_object_ids=ids or None,recorded_object_ids=list(telemetry),active_object_count=len(ids) if ids else None,object_specs={n:specs[n] for n in geometry_ids if n in specs},
            robot_urdf=str(urdf) if urdf else None,display_frame=display,pose_transforms=transforms,up_axis=up.tolist(),
            base_to_workspace=base_to_ws.tolist() if base_to_ws is not None else None,
            placement_region=placement_region(w,composition,pick,base_to_ws),flags=[])
        if w and composition:
            from .tasks import recorded_scene
            recorded_scene({'scene_configuration':composition},w)
            if task and composition['pick_object_id']!=pick:raise ValueError('Episode target conflicts with scene configuration')
        if not canonical:
            with h5py.File(path,'r') as h:
                g=h['data'][name];evaluation=_json(g.attrs.get('evaluation'),{})
                if evaluation or args.get('evaluation_schema'):
                    row.update(evaluation=evaluation,evaluation_outcome='success' if row['recorded_success'] else 'failure',
                        failure_reason=g.attrs.get('failure_reason',''),env_id=evaluation.get('env_id'))
        row['annotation_status']='annotated' if annotation is not None or 'obs/datagen_info/subtask_term_signals/grasp' in signal_keys.values() else 'signals_only' if signal_keys else 'UNAVAILABLE'
        trusted=row['success_source'] in ('simulation_place_condition','simulation_stable_grasp_and_place','simulation_success_term')
        row['physical_success']=row['recorded_success'] if trusted and not row['validation_only'] else None
        with episode_io.EpisodeReader(row) as reader:
            row['events']=_events(reader,annotation,signal_keys)
            rows.append(_scan(row,reader))
    return flag_outliers(rows)


def analyze_canonical(path):return analyze(path)


def load_episode(row,max_points=1500):
    """Selected episode plot samples only; playback still reads each exact original frame."""
    with episode_io.EpisodeReader(row) as reader:
        indices=np.unique(np.linspace(0,reader.frames-1,min(max_points,reader.frames)).astype(int))
        timestamps=reader.timestamps()
        return {**row,'sample_indices':indices,'ee':reader.ee(indices),'actions':reader.read(row['action_key'],indices),
            'robot_q':reader.read(row['q_key'],indices) if row.get('q_key') else None,
            'object_paths':{n:reader.pose(n,indices,observation_timestamps=timestamps[indices]) for n in reader.objects},'timestamps':timestamps}


def flag_outliers(rows):
    if not rows:raise ValueError('에피소드가 없습니다.')
    keys={(r['task_id'],tuple(r.get('active_object_ids') or [])) for r in rows}
    for task,objects in keys:
        group=[r for r in rows if r['task_id']==task and tuple(r.get('active_object_ids') or [])==objects]
        for key,low_label,high_label in [('duration','짧은 시간','긴 시간'),('stop_fraction',None,'EE 정지 비율 높음'),('command_stop_fraction',None,'정지 명령 많음'),('jitter',None,'명령 jitter 높음')]:
            values=np.array([r[key] for r in group]);median=np.median(values);spread=max(1.4826*np.median(abs(values-median)),abs(median)*.1,1e-6)
            for r in group:
                label=high_label if len(group)>=5 and r[key]>median+3*spread else low_label if len(group)>=5 and low_label and r[key]<median-3*spread else None
                if label:
                    r['flags'].append(label)
                    r.setdefault('flag_details',[]).append(dict(label=label,metric=key,value=float(r[key]),
                        median=float(median),threshold=float(median+3*spread if label==high_label else median-3*spread),
                        direction='high' if label==high_label else 'low',group_size=len(group)))
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


def _selection(source, names=None):
    """Read episode headers in stored order; historical proofs remain authoritative history."""
    import h5py
    from .episodes import validate_partition
    source=Path(source).resolve()
    if any(p.name and p.with_name('.'+p.name+'.workflow-lock').exists() for p in (source,*source.parents)):
        raise ValueError('Dataset 저장 중에는 Curation을 실행할 수 없습니다.')
    if source.is_dir():
        if not (source/'manifest.json').is_file():
            raise ValueError('Curation 저장은 Common HDF5 / Legacy NPY·MP4만 지원합니다. 평가 묶음은 개별 HDF5를 선택하세요. LeRobot 편집은 지원하지 않습니다.')
        manifest=json.loads((source/'manifest.json').read_text())
        args=json.loads((source/'project_provenance.json').read_text())
        entries=[(e['id'],e) for e in manifest['episodes']]
    else:
        with h5py.File(source,'r') as h:
            args=_json(h['data'].attrs.get('env_args'),{})
            entries=[(n,dict(h['data'][n].attrs)) for n in sorted((n for n in h['data'] if n.startswith('demo_')),key=lambda n:int(n[5:]))]
    if names is not None and (not names or len(set(names))!=len(names) or not set(names).issubset({n for n,_ in entries})):
        raise ValueError('선택 목록이 비었거나 중복/알 수 없는 Episode입니다.')
    selected=[]
    for name,entry in entries:
        partition=_json(entry.get('source_partition'),None) or args.get('source_partition')
        lineage=_json(entry.get('source_lineage'),None)
        if partition:validate_partition(partition)
        if names is None or name in names:
            selected.append((name,lineage or {'source_id':entry.get('curation_origin')},entry.get('source_group',entry.get('group_id')) or args.get('source_groups',{}).get(name)))
    return selected


def split_dataset(source,train_fraction=.8,seed=0,assignments=None,names=None):
    """Explicit episode allocation, independent of historical training roles."""
    if not 0<float(train_fraction)<1 or type(seed) is not int or seed<0:raise ValueError('Invalid split ratio/seed')
    rows=_selection(source,names);names=[r[0] for r in rows]
    if len(names)<2:raise ValueError('Train/Val에는 각각 최소 한 Episode가 필요합니다.')
    if assignments is None:
        count=max(1,min(len(names)-1,int(len(names)*train_fraction+.5)))
        train=set(np.random.default_rng(seed).choice(names,count,replace=False))
        assignments={n:'train' if n in train else 'val' for n in names}
    if set(assignments)!=set(names) or set(assignments.values())!={'train','val'}:raise ValueError('Assignments must cover selected episodes with train and val')
    def origins(value):
        if not isinstance(value,dict):return set()
        keys={str(value[k]) for k in ('source_id','partition_sha256') if value.get(k)}
        for item in value.values():
            if isinstance(item,dict):keys|=origins(item)
            elif isinstance(item,list):
                for child in item:keys|=origins(child)
        return keys
    pools={'train':set(),'val':set()}
    for name,lineage,group in rows:
        keys=origins(lineage)
        if group and not str(group).startswith('UNKNOWN'):keys.add('group:'+str(group))
        pools[assignments[name]]|=keys
    shared=sorted(pools['train']&pools['val'])
    return dict(schema='so101.curation-split/1',source=str(Path(source).resolve()),seed=seed,train_fraction=float(train_fraction),
        assignments={n:assignments[n] for n in names},counts={s:sum(v==s for v in assignments.values()) for s in ('train','val')},
        warnings=['Train/Val에 같은 Group 또는 알려진 원본 계보가 포함됩니다. 누수 위험을 확인하세요.'] if shared else [],
        shared_origins=shared,lineage_coverage='Known metadata only; absence of a warning does not prove independence')


def _check_cancel(cancel):
    if cancel is not None and cancel():raise InterruptedError('Curation 취소됨 · 결과를 게시하지 않았습니다.')


def export_splits(source,destination,train_fraction=.8,seed=0,assignments=None,names=None,*,cancel=None):
    import os,shutil,tempfile
    from ..real2sim.storage import atomic
    plan=split_dataset(source,train_fraction,seed,assignments,names);destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(Path(source).resolve()):raise ValueError('기존 입력/출력은 덮어쓰지 않습니다.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    lock=destination.with_name('.'+destination.name+'.workflow-lock')
    with lock.open('x'):pass
    temp=None
    try:
        temp=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.',dir=destination.parent))
        outputs={}
        for role in ('train','val'):
            _check_cancel(cancel)
            relative=Path(role) if Path(source).is_dir() else Path(role)/'dataset.hdf5'
            chosen=[n for n,v in plan['assignments'].items() if v==role]
            export_subset(source,temp/relative,chosen,cancel=cancel,role=role)
            outputs[role]=str(destination/relative)
        atomic(temp/'curation_split.json',plan)
        _check_cancel(cancel)
        if destination.exists():raise FileExistsError(destination)
        destination.mkdir(exist_ok=False)
        try:os.rename(temp,destination)
        except BaseException:destination.rmdir();raise
        return {**plan,'outputs':outputs,'folder':str(destination)}
    finally:
        if temp is not None:shutil.rmtree(temp,ignore_errors=True)
        lock.unlink(missing_ok=True)


def export_subset(source,destination,names,*,cancel=None,role=None):
    """Copy selected episodes, publish payload last, roll back only owned outputs."""
    import h5py,os,shutil,tempfile
    from ..real2sim.storage import atomic
    from ..so101_dataset_contract import read_contract,write_contract,sidecar
    from ..utils.artifact_identity import fingerprint
    from ..utils.async_video_writer import copy_hdf5_videos
    source=Path(source).resolve();destination=Path(destination).resolve()
    names=[r[0] for r in _selection(source,names)]
    if not names:raise ValueError('선택 목록이 비었습니다.')
    if destination.exists() or destination.is_relative_to(source):raise ValueError('기존 입력/출력은 덮어쓰지 않습니다.')
    _check_cancel(cancel)
    destination.parent.mkdir(parents=True,exist_ok=True)
    lock=destination.with_name('.'+destination.name+'.workflow-lock')
    with lock.open('x'):pass
    temp=None;published=[]
    try:
        temp=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.',dir=destination.parent))
        record={'source':str(source),'source_sha256':fingerprint(source),'selected_episodes':names,'role':role,
                'selection':'explicit episode selection; source_partition is historical, not this allocation'}
        _check_cancel(cancel)
        if source.is_dir():
            manifest=json.loads((source/'manifest.json').read_text());entries={e['id']:e for e in manifest['episodes']}
            for name in names:
                _check_cancel(cancel)
                folder=(source/entries[name]['path']).resolve();target=(temp/entries[name]['path']).resolve()
                if not folder.is_relative_to(source) or not target.is_relative_to(temp):raise ValueError('Episode outside dataset')
                shutil.copytree(folder,target)
            provenance=json.loads((source/'project_provenance.json').read_text())
            if provenance.get('curation'):record['previous']=provenance['curation']
            atomic(temp/'project_provenance.json',{**provenance,'curation':record})
            if sidecar(source).is_file():shutil.copyfile(sidecar(source),temp/sidecar(source).name)
            atomic(temp/'manifest.json',{**manifest,'episodes':[{**entries[n],'curation_origin':entries[n].get('curation_origin',str(source)+'#'+n)} for n in names]})
            _check_cancel(cancel)
            if destination.exists():raise FileExistsError(destination)
            destination.mkdir(exist_ok=False)
            try:os.rename(temp,destination)
            except BaseException:destination.rmdir();raise
        else:
            payload=temp/destination.name
            with h5py.File(source,'r') as src,h5py.File(payload,'w') as out:
                for k,v in src.attrs.items():out.attrs[k]=v
                data=out.create_group('data')
                for k,v in src['data'].attrs.items():data.attrs[k]=v
                if 'curation' in data.attrs:record['previous']=_json(data.attrs['curation'])
                total=0
                for i,name in enumerate(names):
                    _check_cancel(cancel)
                    src.copy(src['data'][name],data,name=f'demo_{i}')
                    copy_hdf5_videos(src['data'][name],data[f'demo_{i}'],payload)
                    if 'curation_source_episode' not in data[f'demo_{i}'].attrs:data[f'demo_{i}'].attrs['curation_source_episode']=name
                    if 'curation_origin' not in data[f'demo_{i}'].attrs:data[f'demo_{i}'].attrs['curation_origin']=str(source)+'#'+name
                    total+=len(data[f'demo_{i}/actions'])
                data.attrs['total']=total;data.attrs['curation']=json.dumps(record);out.flush()
            write_contract(payload,read_contract(source))
            _check_cancel(cancel)
            # The lock hides the final path until sidecar/videos and HDF5 are complete.
            for item in (temp/(destination.stem+'.videos'),sidecar(payload),payload):
                if not item.exists():continue
                target=destination.parent/item.name
                if target.exists():raise FileExistsError(target)
                if item.is_dir():
                    target.mkdir(exist_ok=False)
                    try:os.rename(item,target)
                    except BaseException:target.rmdir();raise
                else:os.link(item,target)
                published.append(target)
    except BaseException:
        for item in reversed(published):
            if item.is_dir():shutil.rmtree(item)
            else:item.unlink(missing_ok=True)
        raise
    finally:
        if temp is not None:shutil.rmtree(temp,ignore_errors=True)
        lock.unlink(missing_ok=True)
    return destination
