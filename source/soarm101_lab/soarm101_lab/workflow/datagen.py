"""One reference Mimic runner with threaded MuJoCo and vector Isaac execution."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import queue
import threading
import time

import numpy as np
import torch

from .dataset import EpisodeDataset
from ..data.episodes import DatasetWriter
from ..simulation.control import GenerationStopped
from .mimic import GenerationScheduler,MimicGenerator,source_pool,execute_trial
from .adapters import MujocoAdapter,IsaacAdapter
from ..real2sim.storage import atomic
from ..simulation.ik import solver_configuration,validate_solver,validate_failure_policy,policy_configuration


def _seed(seed,index):return int(np.random.SeedSequence([seed,index]).generate_state(1)[0])


def _lineage(reader,generator):
    from .episodes import source_catalog
    if not hasattr(reader,'source_records'):reader.source_records=source_catalog(reader.path,allow_unknown=True)[0]
    segments=[]
    for segment in generator.selection_trace:
        name=reader.mimic_names[segment['source_index']];attrs=reader.episode_attributes(name)
        original=json.loads(attrs.get('source_lineage','null')) or reader.source_records.get(name)
        segments.append({**segment,'episode':name,'source':original or {'kind':'UNKNOWN'}})
    partition=reader.args.get('source_partition')
    if partition and any(s['source'].get('kind')!='source_episode' for s in segments):raise ValueError('Cannot verify exact generated subtask Source')
    return {'kind':'mimic_segments','partition_sha256':partition['sha256'] if partition else None,'segments':segments}


def _output_args(reader,backend,opts):
    from ..utils.artifact_identity import fingerprint
    args=copy.deepcopy(reader.args);args.update(backend=backend,generation_method='isaaclab_mimic/1.0.15',
        source_dataset=str(reader.path),source_backend=reader.backend,
        generation_provenance={'algorithm':'Isaac Lab Mimic 1.0.15 portable reference',
            'source_sha256':opts.get('source_pool_sha256') or fingerprint(reader.path)})
    args['generation_configuration']={key:opts[key] for key in ('num_envs','generation_seed','num_successful_demos','max_attempts','max_episode_steps') if key in opts}
    args['generation_configuration'].update(ik_solver=validate_solver(opts.get('ik_solver','scipy_ls')),
        ik_configuration=solver_configuration(opts.get('ik_solver','scipy_ls'),opts.get('ik_failure_policy','strict')),
        ik_failure_policy=validate_failure_policy(opts.get('ik_failure_policy','strict')),
        ik_policy_configuration=policy_configuration(opts.get('ik_failure_policy','strict')))
    if opts.get('task_followup'):args['task_followup']=opts['task_followup']
    return args


def _resume(writer,opts):
    source=opts.get('resume_data')
    if not source:return 0,[]
    previous=EpisodeDataset(source)
    if not previous.common or previous.backend!=writer.args['backend']:raise ValueError('Resume requires the same common execution backend')
    if previous.args.get('generation_provenance',{}).get('source_sha256')!=writer.args['generation_provenance']['source_sha256']:
        raise ValueError('Resume Source pool changed')
    if previous.contract!=writer.contract:raise ValueError('Resume coordinates/model differ')
    if previous.args.get('task_followup')!=writer.args.get('task_followup'):raise ValueError('Resume followup contract differs')
    if previous.args.get('generation_seed')!=opts['generation_seed']:raise ValueError('Resume generation seed differs')
    old=previous.args.get('generation_configuration',{})
    new=writer.args.get('generation_configuration',{})
    if old.get('ik_solver','scipy_ls')!=new.get('ik_solver','scipy_ls'):
        raise ValueError('Resume IK solver differs; start a new run')
    old_policy=old.get('ik_failure_policy','strict');new_policy=new.get('ik_failure_policy','strict')
    if old_policy!=new_policy or old.get('ik_policy_configuration',policy_configuration(old_policy))!=new.get('ik_policy_configuration',policy_configuration(new_policy)):
        raise ValueError('Resume IK failure policy differs; start a new run')
    if old.get('ik_configuration',solver_configuration(old.get('ik_solver','scipy_ls')))!=new.get('ik_configuration',solver_configuration(new.get('ik_solver','scipy_ls'))):
        raise ValueError('Resume IK configuration differs; start a new run')
    for name in previous.names:
        values,attrs,videos=previous.episode(name)
        videos={k:{**v,'file':str(previous.path.parent/v['file'])} for k,v in videos.items()}
        writer.commit(values,attrs,videos)
    import h5py
    with h5py.File(previous.path,'r') as dataset:
        history=json.loads(bytes(dataset['generation/attempts_json'][:]).decode()) if 'generation/attempts_json' in dataset else None
    if history is None and previous.args.get('generation_attempt_history'):
        raise ValueError('Resume attempt history missing; choose the complete original DataGen output')
    if history is None:
        log=previous.path.parent/'attempts.json';history=json.loads(log.read_text()) if log.is_file() else []
    if history and sum(row['category']=='success' for row in history)!=len(previous.names):
        raise ValueError('Resume attempt history does not match committed success count')
    return len(previous.names),history


def _save_trial(writer,reader,adapter,generator,seed):
    lineage=_lineage(reader,generator);partition=reader.args.get('source_partition')
    attrs={'success':True,'success_source':'simulation_stable_grasp_and_place','source_lineage':json.dumps(lineage),
        'source_group':'mimic:'+','.join(sorted({s['source'].get('source_id','UNKNOWN') for s in lineage['segments']})),
        'mimic_source_segments':json.dumps(lineage['segments']),'seed':seed,
        'task_definition':json.dumps(reader.args['task_definitions'][0] if reader.args['task_definitions'] else None),'scene_configuration':json.dumps(reader.args['scene_configuration']),
        'split':partition['split'] if partition else '',
        'failure_reason':'','simulation_backend':writer.args['backend'],'episode_metrics':json.dumps(dict(adapter.buffer.metrics))}
    attrs.update(ik_failure_policy=adapter.ik_failure_policy,ik_policy_configuration=json.dumps(policy_configuration(adapter.ik_failure_policy)),ik_solver=adapter.ik_solver,ik_configuration=json.dumps(solver_configuration(adapter.ik_solver,adapter.ik_failure_policy)),
        ik_summary=json.dumps(dict(adapter.ik_attempt)))
    if getattr(adapter,'followup_acceptance',None):attrs['success_source']='simulation_stable_grasp_and_place_and_followup_term'
    if getattr(adapter,'processing_runtime',None):

        attrs['success_source']='script_mimic_terms'
        adapter.buffer.final_event_values={key:adapter.term_boolean(adapter.processing_events[key]) for key in adapter.buffer.final_events}
    values,attrs,videos=adapter.buffer.finish(attrs,success=adapter.check_success())
    # Save exactly the pre-IK generated target trajectory and subtask transitions.
    values['mimic_plan']={str(i):{'target_eef_pose':t['poses'],'gripper_action':t['gripper'],'action_noise':t['noise']} for i,t in enumerate(generator.trace)}
    return writer.commit(values,attrs,videos,video_staging=adapter.buffer.video)


def _attempt_details(adapter,generator,started):
    return {'ik_failure_policy':adapter.ik_failure_policy,'ik_policy_configuration':policy_configuration(adapter.ik_failure_policy),'ik_solver':adapter.ik_solver,'ik_configuration':solver_configuration(adapter.ik_solver,adapter.ik_failure_policy),
        'ik_summary':dict(adapter.ik_attempt),'ik_motion_evidence':adapter.ik_motion_evidence,'steps':adapter.step_index,'reset_count':adapter.reset_count,'duration_s':time.perf_counter()-started,
        'failure_diagnostics':adapter.failure_diagnostics(),
        'initial_configuration':getattr(adapter,'initial_configuration',None),
        'selected_sources':generator.selection_trace,'ik':getattr(adapter,'last_ik',None) or getattr(getattr(adapter,'backend',None),'last_ik',None)}


def _run_one(adapter,reader,writer,scheduler,index,attempt,seed,pool):
    generator=MimicGenerator(adapter,pool,seed);adapter.random=generator.random
    started=time.perf_counter();category='simulation_failure';reason=''
    try:
        scheduler.set_state(index,'running')
        result=asyncio.run(execute_trial(adapter,generator))
        category='success' if result['success'] and adapter.check_success() else adapter.failure_category()
        if category=='success':
            try:
                if not adapter.accept_followup():category='task_condition_not_met'
            except (ValueError,RuntimeError) as exc:category='task_contract_error';reason=str(exc)
        if category=='task_condition_not_met':reason='Explicit Annotation acceptance term was not satisfied'
    except GenerationStopped:category='interrupted'
    except TimeoutError as exc:category='timeout';reason=str(exc)
    except ValueError as exc:
        category='ik_failure' if hasattr(exc,'details') else 'planning_failure';reason=str(exc)
    except RuntimeError as exc:category='simulation_failure';reason=str(exc)
    details=_attempt_details(adapter,generator,started);details['reason']=reason;details['seed']=seed
    scheduler.finish(index,attempt,category,details,lambda:_save_trial(writer,reader,adapter,generator,seed))


def run_mujoco(request,reader,output,bridge=None,parent=None,boundary=None):
    from .worker_request import prepare_mujoco
    from ..simulation.environment import create_mujoco
    from ..simulation.mujoco_backend import MujocoBackend
    writer=None
    try:
        if (reader.args.get('task_followup',{}).get('python') or request['options'].get('followup_python')) and request.get('_followup_runtime') is None:
            raise ValueError('Explicit followup Python must be validated/loaded by the Annotation/Mimic Worker')
        opts=request['options'];validate_solver(opts.get('ik_solver','scipy_ls'));num_envs=opts.get('num_envs',1);seed=opts['generation_seed']
        if parent is None:parent=create_mujoco(**prepare_mujoco(request,hz=reader.hz))
        args=_output_args(reader,'mujoco',opts);args['generation_seed']=seed;writer=DatasetWriter(request.get('dataset_output') or output/'dataset.hdf5',args,reader.contract)
        writer.args['execution_physics']={'backend':'mujoco','physics_dt':float(parent.backend.model.opt.timestep),
            'control_decimation':parent.backend.decimation,'model_sha256':parent.backend.metadata['model_sha256'],
            'integrator':int(parent.backend.model.opt.integrator),'solver':int(parent.backend.model.opt.solver),
            'gravity_m_s2':parent.backend.model.opt.gravity.tolist()}
        resumed,history=_resume(writer,opts)
        scheduler=GenerationScheduler(opts['num_successful_demos'],opts.get('max_attempts',10000),num_envs,resumed,history)
        if bridge:bridge.bind_generation(scheduler)
    except BaseException:
        try:
            if writer:writer.close(False)
        finally:
            if parent:parent.close()
        raise
    metrics={};model_ids=[];data_ids=[];state_lock=threading.Lock();previews=queue.Queue(maxsize=1)
    def worker(index):
        backend=MujocoBackend(render_backend=parent.backend.render_backend,shared_backend=parent.backend)
        adapter=None
        try:
            adapter=MujocoAdapter(backend,reader.hz,output,_seed(seed,index),scheduler.stop)

            adapter.bind_followup(request.get('_followup_runtime'))
            adapter.configure_ik(opts.get('ik_solver','scipy_ls'),opts.get('ik_failure_policy','strict'))
            adapter.control=scheduler.control;adapter.worker_index=index
            adapter.max_episode_steps=opts.get('max_episode_steps',2000)
            last_preview=[0.]
            def preview(a):
                scheduler.set_state(index,'running',steps=a.step_index)
                if a.step_index%50==0:print(f'[MIMIC] Env {index} step {a.step_index}',flush=True)
                if bridge and index==0 and not previews.full() and time.monotonic()-last_preview[0]>.1:
                    last_preview[0]=time.monotonic()
                    # Render on the owning worker thread; recording keeps only policy cameras.
                    images=dict(a.last_images)
                    if 'overview' in backend.metadata.get('cameras',{}):
                        images['overview']=backend.get_rgb('overview')
                    try:previews.put_nowait((a.step_index,images))
                    except queue.Full:pass
            adapter.preview=preview
            with state_lock:model_ids.append(id(backend.model));data_ids.append(id(backend.data))
        except BaseException:
            if adapter:adapter.close()
            else:backend.close()
            raise
        try:
            # Upstream generate() reads this pool; boundaries and RNG belong to each generator.
            started=time.perf_counter();pool=source_pool(adapter,reader.path,reader.mimic_names)
            adapter.metrics.update(source_pool_load_count=1,source_pool_load_s=time.perf_counter()-started)
            while (attempt:=scheduler.claim(index)) is not None:
                _run_one(adapter,reader,writer,scheduler,index,attempt,_seed(seed,attempt),pool)
        finally:
            adapter.close()
            metrics[index]={'ik_failure_policy':adapter.ik_failure_policy,'ik_solver':adapter.ik_solver,**dict(adapter.metrics),**dict(adapter.buffer.metrics),**backend.render_metrics,'video_queue_peak':adapter.buffer.queue_peak,
                'render_context':copy.deepcopy(backend.metadata.get('render_context')),
                'reset_count':adapter.reset_count,'physics_steps':sum(r['steps'] for r in scheduler.records if r['env_id']==index)}
    executor=ThreadPoolExecutor(max_workers=num_envs,thread_name_prefix='mimic-mujoco')
    futures=[executor.submit(worker,i) for i in range(num_envs)];stopped=False
    try:
        while not all(f.done() for f in futures):
            if boundary:boundary()
            for future in futures:
                if future.done() and future.exception() is not None:
                    scheduler.stop.set()
                    future.result()
            if bridge:
                bridge.poll();scheduler.control.update(bridge.paused,bridge.single_step);bridge.single_step=False
                try:
                    steps,images=previews.get_nowait();bridge.steps=steps;bridge.publish({'sim_'+k:v for k,v in images.items()})
                except queue.Empty:pass
                bridge.status('PAUSED' if bridge.paused else 'RUNNING')
            time.sleep(.05)
        for future in futures:future.result()
    except KeyboardInterrupt:stopped=True;scheduler.stop.set()
    finally:
        scheduler.stop.set();executor.shutdown(wait=True)
        # An error from a worker must not publish unverified/incomplete data.
        try:
            writer.record_attempts(scheduler.records);writer.close(publish=True)
        finally:parent.close()
        atomic(output/'attempts.json',scheduler.records)
    result={'ik_failure_policy':opts.get('ik_failure_policy','strict'),'ik_policy_configuration':policy_configuration(opts.get('ik_failure_policy','strict')),'ik_solver':opts.get('ik_solver','scipy_ls'),'ik_configuration':solver_configuration(opts.get('ik_solver','scipy_ls'),opts.get('ik_failure_policy','strict')),**scheduler.progress(),'stopped':stopped,'dataset':str(writer.path),'requested':scheduler.target,
        'passed':scheduler.successes==scheduler.target or (stopped and scheduler.successes>0),
        'shared_model_count':len(set(model_ids)),'independent_data_count':len(set(data_ids)),'num_envs':num_envs,
        'worker_metrics':metrics,'method':'isaaclab_mimic/1.0.15','is_isaac_mimicgen':True}
    return result


class _ActionMailbox:
    """Per-env acknowledgements; one env does not join another episode's queue."""
    def __init__(self,index,queue):self.index=index;self.queue=queue;self.pending=None
    async def put(self,item):
        self.pending=asyncio.get_running_loop().create_future()
        await self.queue.put((self.index,item[1],self.pending))
    async def join(self):await self.pending


class _ResetMailbox:
    def __init__(self,index,queue):self.index=index;self.queue=queue;self.pending=None
    async def put(self,item):
        self.pending=asyncio.get_running_loop().create_future()
        await self.queue.put((self.index,self.pending))
    async def join(self):await self.pending


def _isaac_loop(native,adapters,reader,writer,scheduler,seed,bridge,boundary=None):
    from .mimic_reference.config import TerminationTermCfg
    queue=asyncio.Queue();reset_queue=asyncio.Queue();loop=asyncio.new_event_loop()
    async def worker(index):
        adapter=adapters[index]
        started=time.perf_counter();pool=source_pool(adapter,reader.path,reader.mimic_names)
        adapter.metrics.update(source_pool_load_count=1,source_pool_load_s=time.perf_counter()-started)
        while (attempt:=scheduler.claim(index)) is not None:
            trial_seed=_seed(seed,attempt);generator=MimicGenerator(adapter,pool,trial_seed);adapter.random=generator.random
            started=time.perf_counter();category='simulation_failure';reason=''
            try:
                result=await generator.generate(0,TerminationTermCfg(func=lambda env:torch.tensor([env.check_success()])),
                    env_reset_queue=_ResetMailbox(index,reset_queue),env_action_queue=_ActionMailbox(index,queue),export_demo=False)
                category='success' if result['success'] and adapter.check_success() else adapter.failure_category()
            except (GenerationStopped,asyncio.CancelledError):category='interrupted'
            except TimeoutError as exc:category='timeout';reason=str(exc)
            except ValueError as exc:category='ik_failure' if hasattr(exc,'details') else 'planning_failure';reason=str(exc)
            except RuntimeError as exc:category='simulation_failure';reason=str(exc)
            details=_attempt_details(adapter,generator,started);details.update(reason=reason,seed=trial_seed)
            scheduler.finish(index,attempt,category,details,lambda:_save_trial(writer,reader,adapter,generator,trial_seed))
    tasks=[loop.create_task(worker(i)) for i in range(len(adapters))]
    robot=native.scene['robot'];ids=adapters[0].ids
    actions=robot.data.joint_pos[:,ids].clone()
    try:
        while not all(t.done() for t in tasks):
            if boundary:boundary()
            if bridge:bridge.before_step();bridge.status()
            loop.run_until_complete(asyncio.sleep(0))
            # Kit updates its own asyncio loop during reset/step. Physics must run
            # outside a running generator event loop, as in upstream env_loop.
            while not reset_queue.empty():
                index,future=reset_queue.get_nowait()
                try:adapters[index].reset()
                except Exception as exc:future.set_exception(exc)
                else:
                    scheduler.set_state(index,'running');future.set_result(True)
            active=sum(not t.done() for t in tasks)
            # Per-physics-step synchronization only; completed envs may reset and rejoin immediately.
            if queue.qsize()<active:continue
            batch=[]
            while not queue.empty():batch.append(queue.get_nowait())
            if not batch:continue
            for index,command,future in batch:
                actions[index]=command.to(device=native.device,dtype=actions.dtype)
                adapters[index].before_batch_step(command.cpu().numpy())
            t=time.perf_counter();native.step(actions);elapsed=time.perf_counter()-t
            for index,command,future in batch:
                adapters[index].metrics['batch_step_wait_s']+=elapsed
                adapters[index].after_batch_step(command.cpu().numpy());future.set_result(True)
                scheduler.set_state(index,'running',steps=adapters[index].step_index)
            loop.run_until_complete(asyncio.sleep(0))
        for task in tasks:task.result()
    finally:
        scheduler.stop.set()
        for task in tasks:
            if not task.done():task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks,return_exceptions=True));loop.close()


def run_isaac(native,request,reader,output,bridge=None,boundary=None):
    if reader.args.get('task_followup',{}).get('python') or request['options'].get('followup_python'):
        raise ValueError('Explicit followup Python currently requires the MuJoCo Worker')
    opts=request['options'];validate_solver(opts.get('ik_solver','scipy_ls'));seed=opts['generation_seed'];count=opts['num_envs']
    args=_output_args(reader,'isaac',opts);args['generation_seed']=seed;writer=DatasetWriter(request.get('dataset_output') or output/'dataset.hdf5',args,reader.contract)
    writer.args['execution_physics']={'backend':'isaac','physics_dt':float(native.physics_dt),'control_decimation':native.cfg.decimation,
        'device':str(native.device),'num_envs':count,'simulation_configuration':native.cfg.simulation_configuration}
    try:
        resumed,history=_resume(writer,opts)
        scheduler=GenerationScheduler(opts['num_successful_demos'],opts.get('max_attempts',10000),count,resumed,history)
    except BaseException:
        writer.close(False);raise
    if bridge:bridge.bind_generation(scheduler)
    adapters=[IsaacAdapter(native,i,output,_seed(seed,i),scheduler.stop) for i in range(count)];stopped=False
    for adapter in adapters:
        adapter.configure_ik(opts.get('ik_solver','scipy_ls'),opts.get('ik_failure_policy','strict'))
        adapter.max_episode_steps=opts.get('max_episode_steps',2000)
    profile={'physics_step_s':0.,'render_s':0.,'physics_steps':0,'render_calls':0}
    step=native.sim.step;render=native.sim.render
    def profiled_render(*args,**kwargs):
        started=time.perf_counter()
        try:return render(*args,**kwargs)
        finally:
            profile['render_s']+=time.perf_counter()-started;profile['render_calls']+=1
    def profiled_step(*args,**kwargs):
        started=time.perf_counter();render_before=profile['render_s']
        try:return step(*args,**kwargs)
        finally:
            profile['physics_step_s']+=time.perf_counter()-started-(profile['render_s']-render_before)
            profile['physics_steps']+=1
    native.sim.step=profiled_step;native.sim.render=profiled_render
    try:_isaac_loop(native,adapters,reader,writer,scheduler,seed,bridge,boundary)
    except KeyboardInterrupt:stopped=True;scheduler.stop.set()
    finally:
        native.sim.step=step;native.sim.render=render
        for adapter in adapters:adapter.close()
        writer.record_attempts(scheduler.records)
        writer.close()
        atomic(output/'attempts.json',scheduler.records)
    return {'ik_failure_policy':opts.get('ik_failure_policy','strict'),'ik_policy_configuration':policy_configuration(opts.get('ik_failure_policy','strict')),'ik_solver':opts.get('ik_solver','scipy_ls'),'ik_configuration':solver_configuration(opts.get('ik_solver','scipy_ls'),opts.get('ik_failure_policy','strict')),**scheduler.progress(),'passed':scheduler.successes==scheduler.target or (stopped and scheduler.successes>0),
        'stopped':stopped,'dataset':str(writer.path),'requested':scheduler.target,'num_envs':count,
        'method':'isaaclab_mimic/1.0.15','is_isaac_mimicgen':True,'sim_profile':profile,
        'timing_semantics':'CPU wall time of SimulationContext.step excluding nested render; render calls include graphics synchronization; per-env batch wait overlaps',
        'worker_metrics':{i:{'ik_failure_policy':a.ik_failure_policy,'ik_solver':a.ik_solver,**dict(a.metrics),**dict(a.buffer.metrics),'reset_count':a.reset_count,'video_queue_peak':a.buffer.queue_peak} for i,a in enumerate(adapters)}}
