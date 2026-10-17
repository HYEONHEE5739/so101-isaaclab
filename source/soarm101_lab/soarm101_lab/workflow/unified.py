"""Common episode stages; DataGen starts without an automatic Source replay."""
import copy
import json
import signal
from pathlib import Path
import threading
import time

import numpy as np

from .dataset import EpisodeDataset,DatasetWriter,migrate,SCHEMA
from ..real2sim.storage import atomic


def _native_isaac(request,reader):
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensorCfg
    from ..real2sim.workspaces.environment import make_config
    from ..real2sim.workspaces.package import load
    from .tasks import configured
    scene=reader.args['scene_configuration'];opts=request['options']
    w=configured(load(request['workspace']),scene['task_definition'],scene['active_object_ids'],backend='isaac',reset=scene['reset'])
    cfg=make_config(w,opts.get('device','cuda:0'),opts.get('num_envs',1),coordinates=reader.contract)
    cfg.sim.dt = 1 / (reader.hz * cfg.decimation)
    cfg.events.reset_episode.params['randomization_seed']=opts.get('generation_seed',1)
    for name,link in [('grasp_fixed_contact','gripper'),('grasp_moving_contact','jaw')]:
        setattr(cfg.scene,name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+link,update_period=0.,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/'+w['task']['pick']]))
    native=ManagerBasedEnv(cfg=cfg)
    try:
        # Construction starts PhysX, but task reset events have not initialized
        # the vector slots yet. Replay subsequently restores only its own slot.
        native.reset()
    except BaseException:
        native.close();raise
    return native


def replay(adapter,reader,output,bridge=None,writer=None):
    """Replay actual recorded model targets; schema compatibility never sets physical success."""
    rows=[];output=Path(output)
    from .dataset import validate_episode
    from .episodes import source_catalog
    source_records=source_catalog(reader.path,allow_unknown=True)[0] if writer else {}
    # A vector scene steps idle robots too. Keep their hold targets fixed rather
    # than feeding each gravity-displaced measurement back as the next target.
    isaac_actions=None
    if not hasattr(adapter,'backend'):
        isaac_actions=adapter.native.scene['robot'].data.joint_pos[:,adapter.ids].clone()
    for name in reader.names:
        boundary=getattr(adapter,'interrupt_boundary',None)
        if boundary:boundary()
        values,attrs,videos=reader.episode(name)
        validating=bridge is not None and bridge.details.get('workflow_phase')=='source_validation'
        if validating:bridge.source_progress(len(rows),len(reader.names),name,0,len(values['actions']))
        if not attrs.get('source_lineage') and name in source_records:
            attrs['source_lineage']=json.dumps(source_records[name])
        validate_episode(values,attrs,videos,reader.hz,video_root=reader.path.parent)
        adapter.restore(values,attrs);joint_errors=[];object_errors=[];reason='';category=''
        marks=[];skipped=[];position=[0]
        adapter.annotation_marks=marks;adapter.annotation_position=position;adapter.annotation_skipped=skipped
        if bridge and writer:
            bridge.callbacks['mark']=lambda:marks.append(position[0])
            bridge.callbacks['skip']=lambda:skipped.append(True)
        first=adapter.state();expected=values['initial_state']['articulation']['robot']['joint_position'][0]
        from .robot_preview import pose_matrix
        if not np.allclose(adapter.base_world,pose_matrix(values['initial_state']['articulation']['robot']['root_pose'][0]),atol=1e-5):
            raise ValueError('Initial fixed robot base differs from recorded frame')
        if not np.allclose(first['joint_pos'],expected,atol=1e-5):raise ValueError('Initial joint restoration differs')
        for obj in adapter.workspace['task']['dynamic']:
            expected_obj=values['obs']['datagen_info']['object_pose'][obj][0]
            if not np.allclose(first['object_poses'][obj],expected_obj,atol=1e-5):raise ValueError('Initial object transform differs: '+obj)
        try:
            for i,command in enumerate(values['joint_targets']):
                if boundary:boundary()
                position[0]=i
                if validating:bridge.source_progress(len(rows),len(reader.names),name,i+1,len(values['actions']))
                if bridge:bridge.before_step()
                while getattr(adapter,'annotation_paused',False) and not skipped:
                    if boundary:boundary()
                    adapter.native.sim.render();time.sleep(.01)
                if skipped:reason='User skipped annotation episode';break
                before=adapter.state();joint_errors.append(float(np.linalg.norm(before['joint_pos']-values['obs']['policy']['joint_pos'][i])))
                for obj in adapter.workspace['task']['dynamic']:
                    object_errors.append(float(np.linalg.norm(before['object_poses'][obj][:3,3]-values['obs']['datagen_info']['object_pose'][obj][i,:3,3])))
                if writer:adapter.buffer.append(command,values['obs']['datagen_info']['target_eef_pose']['tool0'][i],before)
                if hasattr(adapter,'backend'):adapter.backend.step(command)
                else:
                    import torch
                    isaac_actions[adapter.index]=torch.as_tensor(command,device=adapter.native.device,dtype=isaac_actions.dtype)
                    adapter.native.step(isaac_actions)
                adapter.update_grasp()
                if bridge:
                    bridge.steps=i+1;bridge.details.update(episode=name,episode_index=reader.names.index(name)+1,
                        episode_total=len(reader.names),task_signals={'grasp':adapter.tracker.completed,'place':adapter.place_success()});bridge.status()
                if request_realtime(adapter):time.sleep(1/reader.hz)
            passed=adapter.check_success() and not skipped;category='success' if passed else adapter.failure_category()
            if not passed and not reason:
                reason='Stable grasp was not observed' if category=='grasp_failure' else 'Stable grasp observed; final placement condition not met'
        except (ValueError,RuntimeError) as exc:passed=False;reason=str(exc);category='simulation_failure'
        row={'episode':name,'physical_replay_passed':bool(passed),'category':category,'reason':reason,
            **({'annotation_accepted':False} if writer else {}),
            'grasp_confirmed':adapter.tracker.completed,'place_confirmed':bool(adapter.place_success()),
            'steps':adapter.step_index,'joint_l2_max_rad':max(joint_errors,default=0.),
            'object_position_error_max_m':max(object_errors,default=0.),'source_backend':reader.backend,
            'execution_backend':'mujoco' if hasattr(adapter,'backend') else 'isaac'}
        if writer and passed:
            exported_attrs={**attrs,'success':True,'success_source':'simulation_stable_grasp_and_place',
                'simulation_backend':row['execution_backend'],
                'annotation_method':'cross_backend_stable_contact_replay','source_group':attrs.get('source_group','UNKNOWN'),
                'physical_replay':json.dumps(row)}
            prepared=adapter.buffer.finish(exported_attrs)
            if not getattr(adapter,'annotation_auto',True):
                if len(marks)!=1 or not 0<marks[0]<len(values['actions'])-1:
                    row.update(physical_replay_passed=True,annotation_accepted=False,reason='Mark one grasp boundary before placement')
                    rows.append(row);adapter.buffer.abort();continue
                signal=(np.arange(len(values['actions']))>=marks[0]).astype(np.uint8)[:,None]
                prepared[0]['obs']['events']['grasp']=signal
                prepared[0]['obs']['datagen_info']['subtask_term_signals']['grasp']=signal
                annotation=json.loads(prepared[1]['annotation']);annotation.update(grasp=marks[0],method='manual_grasp_and_measured_place')
                prepared[1]['annotation']=json.dumps(annotation)
            row['exported_episode']=writer.commit(*prepared,video_staging=adapter.buffer.video);row['annotation_accepted']=True
        rows.append(row);print('[REPLAY] '+json.dumps(row),flush=True);adapter.buffer.abort()
        if validating:bridge.source_progress(len(rows),len(reader.names),name,len(values['actions']),len(values['actions']))
        atomic(output/'replay_progress.json',rows)
    physical=sum(r['physical_replay_passed'] for r in rows)
    accepted=sum(r.get('annotation_accepted',False) for r in rows) if writer else physical
    result={'passed':bool(rows) and physical==len(rows),'episodes':rows,
        'accepted':accepted,'rejected':len(rows)-accepted,'physical_accepted':physical,
        'schema_compatible':True,'hardware_tested':False}
    atomic(output/'cross_replay.json',result);return result


def request_realtime(adapter):return getattr(adapter,'realtime',False)


def replay_cache_identity(reader,request,adapter):
    """Conservative physical proof identity; no schema-only or cross-version reuse."""
    import hashlib
    from importlib.metadata import version,PackageNotFoundError
    from ..real2sim.precision import fingerprint
    package=Path(__file__).resolve().parents[1];code=hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        code.update(str(path.relative_to(package)).encode());code.update(path.read_bytes())
    versions={}
    for name in ('isaacsim','mujoco','torch','numpy','scipy','pin'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]='unavailable'
    if hasattr(adapter,'backend'):
        backend=adapter.backend;model=backend.model
        physics={'configuration':backend.metadata['simulation_configuration'],'model_sha256':backend.metadata['model_sha256'],
            'dt':float(model.opt.timestep),'decimation':backend.decimation,'integrator':int(model.opt.integrator),
            'solver':int(model.opt.solver),'gravity':model.opt.gravity.tolist()}
    else:
        def serial(value):
            if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [serial(v) for v in value]
            if callable(value):return value.__module__+'.'+value.__qualname__
            if isinstance(value,(str,int,float,bool)) or value is None:return value
            if hasattr(value,'tolist'):return serial(value.tolist())
            raise TypeError('Unserializable physics cache field: '+str(type(value)))
        cfg=adapter.native.cfg
        physics=serial({'sim':cfg.sim.to_dict(),'actions':cfg.actions.to_dict(),
            'configuration':cfg.simulation_configuration,'decimation':cfg.decimation,'num_envs':adapter.native.num_envs})
    identity={'source_sha256':request['options']['source_pool_sha256'],'episode_ids':list(reader.names),
        'workspace':reader.args['workspace'],'scene':reader.args['scene_configuration'],'coordinates':reader.contract,
        'target_backend':request['backend'],'physics':physics,'engine_versions':versions,
        'code_sha256':code.hexdigest(),'success_criteria':'stable_grasp_and_backend_place/1','control_hz':reader.hz}
    return fingerprint(identity),identity


def cached_replay(reader,request,adapter,output,bridge=None):
    """Developer verification helper, never a DataGen prerequisite. Cache positive proofs only."""
    from ..real2sim.workspaces.package import ROOT
    from ..real2sim.precision import fingerprint as proof_hash
    opts=request['options'];cache_path=None;identity=None;key=None
    if opts.get('replay_cache',True):
        try:
            key,identity=replay_cache_identity(reader,request,adapter)
            cache_path=Path(opts.get('replay_cache_dir') or ROOT/'outputs/workflow/replay_cache')/(key+'.json')
            if cache_path.is_file():
                cached=json.loads(cache_path.read_text());proof=cached['proof']
                if (cached['identity']==identity and cached['key']==key and proof_hash(proof)==cached['proof_sha256'] and proof['passed']
                        and [row['episode'] for row in proof['episodes']]==reader.names
                        and all(row['physical_replay_passed'] for row in proof['episodes'])):
                    result={**proof,'cache':{'hit':True,'key':key}}
                    atomic(output/'cross_replay.json',result);return result
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            cache_path=None
    from .artifacts import fingerprint
    before=fingerprint(reader.path)
    proof=replay(adapter,reader,output,bridge)
    if fingerprint(reader.path)!=before:raise ValueError('Source changed during physical replay; proof cannot be cached')
    proof['cache']={'hit':False,'key':key}
    if proof['passed'] and cache_path is not None:
        atomic(cache_path,{'key':key,'identity':identity,'proof':proof,'proof_sha256':proof_hash(proof)})
    atomic(output/'cross_replay.json',proof)
    return proof


def execute(request):
    if request['stage']=='datagen':
        from ..simulation.ik import validate_solver,validate_failure_policy
        request={**request,'options':{**request['options'],'ik_solver':validate_solver(request['options'].get('ik_solver','scipy_ls')),'ik_failure_policy':validate_failure_policy(request['options'].get('ik_failure_policy','strict'))}}
    output=Path(request['output']).resolve()
    if output.exists():raise FileExistsError('Choose a new workflow output directory')
    output.mkdir(parents=True,exist_ok=False);original=Path(request['input']).resolve();reader=EpisodeDataset(original)
    if reader.legacy or not reader.common:reader=EpisodeDataset(migrate(original,output/'imported.hdf5'))
    if request.get('include'):
        indexes=set()
        for token in request['include']:
            bounds=[int(v) for v in token.split('-')]
            if len(bounds)>2:raise ValueError('Invalid episode range: '+token)
            indexes.update(range(bounds[0],bounds[-1]+1))
        if not indexes or min(indexes)<0 or max(indexes)>=len(reader.names):raise ValueError('Episode selection outside dataset')
        reader.names=[name for index,name in enumerate(reader.names) if index in indexes]
    selected=request['options'].get('episode')
    if selected:
        matches=[name for name in reader.names if name==selected or reader.episode_attributes(name).get('episode_id')==selected]
        if len(matches)!=1:raise ValueError('Selected episode missing or ambiguous: '+selected)
        reader.names=matches
    from .dataset import validate_dataset
    validate_dataset(reader,require_annotations=request['stage']=='datagen')
    from ..real2sim.workspaces.package import load,metadata
    if reader.args['workspace']!=metadata(load(request['workspace'])):raise ValueError('Input Workspace identity differs')
    opts=copy.deepcopy(request['options']);backend=request['backend'];stage=request['stage']
    from .artifacts import fingerprint
    opts['source_pool_sha256']=fingerprint(original)
    from ..real2sim.workspaces.reset import datagen_seed
    if opts.get('generation_seed') is None:
        previous=EpisodeDataset(opts['resume_data']) if opts.get('resume_data') else None
        opts['generation_seed']=previous.args['generation_seed'] if previous else datagen_seed(load(request['workspace'])['task']['reset']['seed'])
    opts.setdefault('num_envs',1);opts.setdefault('max_attempts',10000);opts.setdefault('num_successful_demos',10)
    if opts['num_envs'] not in (1,2,4,8):raise ValueError('num_envs must be 1,2,4 or 8')
    scene=reader.args['scene_configuration'];opts.update(task_definition=scene['task_definition'],active_object_ids=scene['active_object_ids'],
        width=640,height=480,control_hz=reader.hz)
    request={**request,'input':str(reader.path),'options':opts}
    from .presentation import current
    app=None;native=None;adapter=None;bridge=current();writer=None;keyboard=None
    if stage=='datagen' and bridge:
        bridge.details['ik_solver']=opts['ik_solver'];bridge.details['ik_failure_policy']=opts['ik_failure_policy']
        bridge.start_datagen(opts['num_successful_demos'],opts['num_envs'])
    previous_interrupt=signal.getsignal(signal.SIGINT)
    from .data_resume import deferred_interrupt
    try:
        if backend=='isaac':
            from isaaclab.app import AppLauncher
            app=AppLauncher(headless=opts.get('mode','operator')=='operator',enable_cameras=True,device=opts.get('device','cuda:0')).app
            boundary=deferred_interrupt()
            native=_native_isaac(request,reader)
            from .adapters import IsaacAdapter
            adapter=IsaacAdapter(native,0,output,opts['generation_seed'],threading.Event())
            from .presentation import attach
            bridge=attach(native)
        elif backend=='mujoco':
            boundary=deferred_interrupt()
            from .mujoco import _environment
            from .adapters import MujocoAdapter
            native=_environment(request,output,hz=reader.hz)
            adapter=MujocoAdapter(native.backend,reader.hz,output,opts['generation_seed'],threading.Event())
            from .presentation import current
            bridge=current()
        else:raise ValueError('Unknown execution backend')
        adapter.realtime=opts.get('realtime',False)
        adapter.interrupt_boundary=boundary
        if bridge:bridge.interrupt_boundary=boundary
        adapter.annotation_auto=opts.get('auto',True)
        if stage=='annotate' and not adapter.annotation_auto and bridge is None:
            if backend!='isaac' or opts.get('mode','operator')!='developer':
                raise ValueError('Manual annotation requires Workflow controls or the Isaac developer window')
            from .presentation import annotation_keyboard
            keyboard=annotation_keyboard(adapter)
        # Physical Source replay is an explicit operation, not a generation preflight.
        if stage in ('replay','annotate'):
            if stage=='annotate':
                args=copy.deepcopy(reader.args);args.update(backend=backend,source_backend=reader.backend,annotated_from=str(original))
                writer=DatasetWriter(request.get('dataset_output') or output/'dataset.hdf5',args,reader.contract)
            proof=replay(adapter,reader,output,bridge,writer)
            if writer:
                writer.close();result={**proof,'passed':writer.count>0,'dataset':str(writer.path)}
            elif stage=='replay':result=proof
        if stage=='datagen':
            from .datagen import run_isaac,run_mujoco
            if backend=='isaac':result=run_isaac(native,request,reader,output,bridge,boundary)
            else:
                adapter.buffer.abort();adapter=None
                result=run_mujoco(request,reader,output,bridge,parent=native,boundary=boundary);native=None
        elif stage not in ('replay','annotate'):raise ValueError('Unsupported common episode stage')
        result.update(backend=backend,stage=stage,episode_schema=SCHEMA)
        atomic(output/'report.json',result)
        if not result['passed'] and not result.get('stopped'):raise RuntimeError('Common episode stage validation failed')
        return result
    except KeyboardInterrupt:
        if stage=='datagen':
            from ..simulation.ik import solver_configuration
            result={'ik_failure_policy':opts['ik_failure_policy'],'ik_solver':opts['ik_solver'],'ik_configuration':solver_configuration(opts['ik_solver'],opts['ik_failure_policy']),
                'passed':False,'stopped':True,'successes':0,'requested':opts['num_successful_demos'],
                'dataset':None,'backend':backend,'stage':stage,'episode_schema':SCHEMA}
            atomic(output/'report.json',result)
            return result
        if stage=='annotate' and writer and writer.count:
            writer.close()
            rows=json.loads((output/'replay_progress.json').read_text()) if (output/'replay_progress.json').exists() else []
            result={'passed':True,'stopped':True,'dataset':str(writer.path),'accepted':writer.count,
                'episodes':rows,'backend':backend,'stage':stage,'episode_schema':SCHEMA}
            atomic(output/'report.json',result);return result
        raise
    except BaseException as exc:
        atomic(output/'failure.json',{'passed':False,'stage':stage,'error':str(exc),'schema_compatible':reader.common})
        import traceback
        traceback.print_exc()
        raise
    finally:
        keyboard=None
        if writer:writer.close(False)
        if adapter:adapter.close()
        if native:native.close()
        if bridge:bridge.close()
        signal.signal(signal.SIGINT,previous_interrupt)
        if app:app.close()
