"""Common episode stages; DataGen starts without an automatic Source replay."""
import copy
import json
import signal
from pathlib import Path
import threading
import time

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


def replay(adapter,reader,output,bridge=None,writer=None,*,evaluation_playback=False,num_envs=1,input_validated=False):
    """Recorded targets and common acceptance; Annotation owns parallel execution."""
    from .annotation import replay_episode,run
    if writer:
        return run(adapter,reader,Path(output),bridge,writer,num_envs,input_validated=input_validated)
    from .dataset import validate_episode
    rows=[];output=Path(output)
    actions=None
    if not hasattr(adapter,'backend'):
        actions=adapter.native.scene['robot'].data.joint_pos[:,adapter.ids].clone()
    for name in reader.names:
        values,attrs,videos=reader.episode(name)
        validate_episode(values,attrs,videos,reader.hz,video_root=reader.path.parent)
        validating=bridge is not None and bridge.details.get('workflow_phase')=='source_validation'
        if validating:bridge.source_progress(len(rows),len(reader.names),name,0,len(values['actions']))
        episode=replay_episode(adapter,reader,name,values,attrs,bridge,evaluation_playback=evaluation_playback)
        try:
            command=next(episode)
            while True:
                if command is None:
                    command=next(episode);continue
                try:
                    if hasattr(adapter,'backend'):adapter.backend.step(command)
                    else:
                        import torch
                        actions[adapter.index]=torch.as_tensor(command,device=adapter.native.device,dtype=actions.dtype)
                        adapter.native.step(actions)
                except (ValueError,RuntimeError) as exc:command=episode.throw(exc)
                else:command=next(episode)
        except StopIteration as completed:row,_=completed.value
        finally:episode.close();adapter.buffer.abort()
        rows.append(row);print('[REPLAY] '+json.dumps(row),flush=True)
        if validating:bridge.source_progress(len(rows),len(reader.names),name,len(values['actions']),len(values['actions']))
        atomic(output/'replay_progress.json',rows)
    physical=sum(r['physical_replay_passed'] for r in rows)
    result={'passed':bool(rows) and physical==len(rows),'episodes':rows,
        'accepted':physical,'rejected':len(rows)-physical,'physical_accepted':physical,
        'schema_compatible':True,'hardware_tested':False}
    atomic(output/'cross_replay.json',result);return result


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
    from ..utils.artifact_identity import fingerprint
    before=fingerprint(reader.path)
    proof=replay(adapter,reader,output,bridge)
    if fingerprint(reader.path)!=before:raise ValueError('Source changed during physical replay; proof cannot be cached')
    proof['cache']={'hit':False,'key':key}
    if proof['passed'] and cache_path is not None:
        atomic(cache_path,{'key':key,'identity':identity,'proof':proof,'proof_sha256':proof_hash(proof)})
    atomic(output/'cross_replay.json',proof)
    return proof


def execute(request):
    if request['stage']=='annotate':
        count=request['options'].get('num_envs',1)
        if type(count) is not int or count not in (1,2,4,8):raise ValueError('Annotation num_envs must be 1,2,4 or 8')
        if not request['options'].get('auto',True) and count!=1:raise ValueError('Manual Annotation requires one environment')
    if request['stage']=='datagen':
        from ..simulation.ik import validate_solver,validate_failure_policy
        request={**request,'options':{**request['options'],'ik_solver':validate_solver(request['options'].get('ik_solver','scipy_ls')),'ik_failure_policy':validate_failure_policy(request['options'].get('ik_failure_policy','strict'))}}
        if request['options'].get('resume_data'):
            from .data_resume import inspect_resume_input
            inspect_resume_input('datagen',request['options']['resume_data'])
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
    validation_started=time.perf_counter()
    validate_dataset(reader,require_annotations=request['stage']=='datagen')
    input_validation_s=time.perf_counter()-validation_started
    from ..real2sim.workspaces.package import load,metadata
    if reader.args['workspace']!=metadata(load(request['workspace'])):raise ValueError('Input Workspace identity differs')
    opts=copy.deepcopy(request['options']);backend=request['backend'];stage=request['stage']
    from ..utils.artifact_identity import fingerprint
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
            if stage=='annotate':
                from .presentation import current
                bridge=current()
                if bridge:
                    # Manual controls can keep the developer window responsive
                    # while paused. Annotation sends status only, never images.
                    bridge.idle_render=native.sim.render if not opts.get('auto',True) and opts.get('mode')=='developer' else None
            else:bridge=attach(native)
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
            proof=replay(adapter,reader,output,bridge,writer,evaluation_playback=stage=='replay',
                         **({'num_envs':opts['num_envs'],'input_validated':True} if stage=='annotate' else {}))
            if writer:
                writer.close();result={**proof,'passed':writer.count>0,'dataset':str(writer.path)}
                result['annotation_profile']['input_validation_s']=input_validation_s
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
