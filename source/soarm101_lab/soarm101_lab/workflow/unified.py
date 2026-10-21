"""Common episode stages; DataGen starts without an automatic Source replay."""
import copy
import json
import signal
from pathlib import Path
import threading
import time

from .dataset import EpisodeDataset, migrate
from ..data.episodes import DatasetWriter, SCHEMA
from ..real2sim.storage import atomic



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
    original=Path(request['input']).resolve()
    path_stage=request['stage'] in ('replay','annotate','datagen')
    context=None
    if path_stage:
        from .worker_request import episode_context
        context=episode_context(original,request['workspace'],request['stage'],request['options'])
        if output.is_relative_to(context['workspace']['root']) or (original.is_dir() and output.is_relative_to(original)):
            raise ValueError('Output must be outside input Dataset and immutable Workspace')
    reader=context['reader'] if context else EpisodeDataset(original)
    output.mkdir(parents=True,exist_ok=False)
    if reader.legacy or not reader.common:reader=EpisodeDataset(migrate(original,output/('.input/imported.hdf5' if path_stage else 'imported.hdf5')))
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
    workspace=context['workspace'] if context else load(request['workspace'])
    if reader.args['workspace']!=metadata(workspace):raise ValueError('Input Workspace identity differs')
    opts=copy.deepcopy(request['options']);backend=request['backend'];stage=request['stage']
    if context:
        # Bind freshly read native evidence, never a GUI/Registry snapshot.
        opts.update(tasks=context['tasks'],task_definition=context['scene']['task_definition'],
                    instruction=context['scene']['task_definition']['language_instruction'],active_object_ids=context['scene']['active_object_ids'])
    from ..utils.artifact_identity import fingerprint
    opts['source_pool_sha256']=fingerprint(original)
    from ..real2sim.workspaces.reset import datagen_seed
    if opts.get('generation_seed') is None:
        previous=EpisodeDataset(opts['resume_data']) if opts.get('resume_data') else None
        opts['generation_seed']=previous.args['generation_seed'] if previous else datagen_seed(workspace['task']['reset']['seed'])
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
            from ..simulation.environment import create_isaac
            native=create_isaac(workspace,scene,reader.contract,control_hz=reader.hz,
                device=opts.get('device','cuda:0'),num_envs=opts['num_envs'],
                randomization_seed=opts['generation_seed'],overview_camera=bridge is not None and stage=='datagen')
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
            from .worker_request import prepare_mujoco
            from ..simulation.environment import create_mujoco
            from .adapters import MujocoAdapter
            native=create_mujoco(**prepare_mujoco(request,hz=reader.hz,workspace=workspace))
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
                from .annotation import run
                proof=run(adapter,reader,output,bridge,writer,opts['num_envs'],input_validated=True)
            else:
                from .replay import run
                proof=run(adapter,reader,output,bridge,evaluation_playback=True)
            if writer:
                writer.close();result={**proof,'passed':writer.count>0,'dataset':str(writer.path)}
                result['annotation_profile']['input_validation_s']=input_validation_s
            elif stage=='replay':result=proof
        if stage=='datagen':
            from .datagen import run_isaac,run_mujoco
            if backend=='isaac':result=run_isaac(native,request,reader,output,bridge,boundary)
            else:
                adapter.buffer.abort();adapter=None
                parent=native;native=None
                result=run_mujoco(request,reader,output,bridge,parent=parent,boundary=boundary)
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
