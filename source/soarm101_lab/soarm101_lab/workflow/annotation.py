"""Annotation execution: independent MuJoCo slots or one Isaac vector clock.

Annotation owns signals/manual controls and scheduling. Restoration, measured
errors and the two-phase recorded command stream belong to Simulation Playback.
"""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import threading
import time

import numpy as np

from ..real2sim.storage import atomic
from ..simulation.control import PhysicsControl,GenerationStopped
from ..simulation.playback import step_targets


def annotate_episode(adapter,reader,name,values,attrs,bridge=None):
    """Annotation signals/manual controls/output over borrowed physics playback."""
    from ..simulation.playback import restore_episode,replay_steps
    boundary=getattr(adapter,'interrupt_boundary',None)
    restore_episode(adapter,values,attrs)
    reason='';marks=[];skipped=[];position=[0]
    adapter.annotation_marks=marks;adapter.annotation_position=position;adapter.annotation_skipped=skipped
    if bridge:
        bridge.callbacks['mark']=lambda:marks.append(position[0])
        bridge.callbacks['skip']=lambda:skipped.append(True)
    def before_step(i):
        position[0]=i
        if bridge:
            if bridge.details.get('workflow_phase')=='source_validation':
                bridge.source_progress(reader.names.index(name),len(reader.names),name,i+1,len(values['actions']))
            bridge.before_step()
        while getattr(adapter,'annotation_paused',False) and not skipped:
            if boundary:boundary()
            adapter.native.sim.render();time.sleep(.01)
        return not skipped
    def record(i,command,before):
        adapter.record_frame(command,values['obs']['datagen_info']['target_eef_pose']['tool0'][i],before)
    metrics=yield from replay_steps(adapter,values,before_step=before_step,on_observation=record,after_step=adapter.update_grasp)
    passed=False;category='simulation_failure'
    if metrics['simulation_error'] is not None:reason=metrics['simulation_error']
    else:
        if skipped:reason='User skipped annotation episode'
        try:
            task_passed=adapter.check_success() and not skipped
            passed=task_passed;category='success' if task_passed else adapter.failure_category()
            if not passed and not reason:
                reason='Stable grasp was not observed' if category=='grasp_failure' else 'Stable grasp observed; final placement condition not met'
        except (ValueError,RuntimeError) as exc:passed=False;reason=str(exc);category='simulation_failure'
    row={'episode':name,'physical_replay_passed':bool(passed),'category':category,'reason':reason,
        'annotation_accepted':False,'grasp_confirmed':bool(adapter.tracker and adapter.tracker.completed),'place_confirmed':bool(adapter.place_success()),
        'steps':adapter.step_index,'joint_l2_max_rad':metrics['joint_l2_max_rad'],
        'object_position_error_max_m':metrics['object_position_error_max_m'],'source_backend':reader.backend,
        'execution_backend':'mujoco' if hasattr(adapter,'backend') else 'isaac'}
    processing=getattr(adapter,'processing_runtime',None)
    if processing:
        row.update(acceptance_source='script_terms',playback_completed=metrics['simulation_error'] is None and not skipped,task_accepted=bool(passed),events=dict(adapter.event_boundaries),
                   native_acceptance=bool(adapter.tracker and adapter.tracker.completed and adapter.place_success()))
        if category=='task_condition_not_met':row['reason']='Declared Annotation acceptance was not satisfied'
        elif getattr(adapter,'processing_failure',None):row['reason']='Declared failure term: '+adapter.processing_failure
    prepared=None
    if passed and processing:
        spec=processing['contract']['declaration']['annotation']
        manual=spec.get('manual_event') if not getattr(adapter,'annotation_auto',True) and len(marks)==1 and 0<marks[0]<len(values['actions'])-1 else None
        missing=[key for key in getattr(adapter.buffer,'signal_names',[]) if key!=manual and adapter.event_boundaries.get(key,0)<=0]
        if missing:
            row.update(category='missing_event_boundary',reason='Required signals need a 0→1 boundary: '+', '.join(missing))
            return row,None
    if passed:
        try:
            if not adapter.accept_followup():
                row.update(category='task_condition_not_met',reason='Explicit Annotation acceptance term was not satisfied')
                return row,None
        except (ValueError,RuntimeError) as exc:
            row.update(category='task_contract_error',reason=str(exc));return row,None
        exported_attrs={**attrs,'success':True,'success_source':'simulation_stable_grasp_and_place',
            'simulation_backend':row['execution_backend'],
            'annotation_method':'cross_backend_stable_contact_replay','source_group':attrs.get('source_group','UNKNOWN'),
            'physical_replay':json.dumps(row)}
        if getattr(adapter,'followup_acceptance',None):
            exported_attrs['success_source']='simulation_stable_grasp_and_place_and_followup_term'
        if processing:
            exported_attrs['success_source']='script_annotation_terms'
            adapter.buffer.final_event_values={key:adapter.term_boolean(adapter.processing_events[key]) for key in adapter.buffer.final_events}
        prepared=adapter.buffer.finish(exported_attrs,success=adapter.check_success())
        if not getattr(adapter,'annotation_auto',True):
            event=processing['contract']['declaration']['annotation'].get('manual_event') if processing else 'grasp'
            if event is None or len(marks)!=1 or not 0<marks[0]<len(values['actions'])-1:
                row.update(annotation_accepted=False,reason='Mark one grasp boundary before placement')
                prepared=None
            else:
                signal=(np.arange(len(values['actions']))>=marks[0]).astype(np.uint8)[:,None]
                prepared[0]['obs']['events'][event]=signal
                prepared[0]['obs']['datagen_info']['subtask_term_signals'][event]=signal
                annotation=json.loads(prepared[1]['annotation'])
                if annotation.get('schema','so101.common-subtasks/1')=='so101.common-subtasks/1':annotation.update(grasp=marks[0],method='manual_grasp_and_measured_place')
                else:annotation['events'][event]=marks[0];annotation['method']='manual_event'
                prepared[1]['annotation']=json.dumps(annotation)
                if processing:row.update(manual_event=event,manual_boundary=marks[0])
    return row,prepared


def run(primary,reader,output,bridge,writer,num_envs,*,input_validated=False):
    from .episodes import source_catalog
    auto=getattr(primary,'annotation_auto',True)
    if type(num_envs) is not int or num_envs not in (1,2,4,8):raise ValueError('Annotation num_envs must be 1,2,4 or 8')
    if not auto and num_envs!=1:raise ValueError('Manual Annotation requires one environment')
    started=time.perf_counter();sources=source_catalog(reader.path,allow_unknown=True)[0]
    stop=threading.Event();control=PhysicsControl(num_envs);lock=threading.RLock()
    completed=queue.Queue();pending={};rows=[];cursor=0;finished={}
    timings=defaultdict(float)
    workers={i:dict(episode=None,steps=0,total_steps=0,state='waiting',completed=0,successes=0,failures=0) for i in range(num_envs)}
    assignments=[];adapters=[];active_episodes={};last_report=0.
    job_started=bridge.config.get('started_at',time.time()) if bridge else time.time()
    finished_at=None;phase='running';error=''
    previous_provider=bridge.progress_provider if bridge else None
    boundary=getattr(primary,'interrupt_boundary',None)

    def claim(index):
        nonlocal cursor
        with lock:
            if stop.is_set() or cursor>=len(reader.names):return None
            ordinal=cursor;cursor+=1
            workers[index].update(episode=reader.names[ordinal],steps=0,state='running')
            assignments.append(dict(env_id=index,episode=reader.names[ordinal],ordinal=ordinal,
                                    start_s=time.perf_counter()-started))
            return ordinal

    def begin(a,ordinal):
        if a is not primary:a.bind_followup(getattr(primary,'followup_runtime',None))
        t=time.perf_counter();name=reader.names[ordinal]
        values,attrs,videos=reader.episode(name)
        with lock:
            timings['input_read_s']+=time.perf_counter()-t
            index=next(r['env_id'] for r in assignments if r['ordinal']==ordinal)
            workers[index]['total_steps']=len(values['actions'])
        if not input_validated:
            from ..data.episodes import validate_episode
            t=time.perf_counter()
            validate_episode(values,attrs,videos,reader.hz,video_root=reader.path.parent)
            with lock:timings['input_validation_s']+=time.perf_counter()-t
        if not attrs.get('source_lineage') and name in sources:attrs['source_lineage']=json.dumps(sources[name])
        # execute() has already validated the entire input, including video PTS/SHA.
        episode=annotate_episode(a,reader,name,values,attrs,bridge if not auto else None)
        active_episodes[a]=episode
        return episode

    def finish(a,index,ordinal,result):
        row,prepared=result;video=a.buffer.video if prepared else None
        if video is not None:
            with lock:timings['video_encode_s']+=sum(w.encoding_seconds for w in video.writers.values())
            a.buffer.video=None  # completed staging belongs to the ordered writer now
        a.buffer.abort()
        with lock:
            assignment=next(r for r in assignments if r['ordinal']==ordinal)
            assignment.update(end_s=time.perf_counter()-started,steps=row['steps'])
            workers[index].update(state='saving' if prepared else 'failed',steps=row['steps'])
            workers[index]['completed']+=1;workers[index]['failures']+=int(prepared is None)
            category=row.get('category','simulation_failure')
            if prepared is None and row['physical_replay_passed'] and category=='success':category='manual_boundary_missing'
            finished[ordinal]=dict(env_id=index,ready=prepared is not None,category=category,reason=row.get('reason',''))
        active_episodes.pop(a,None)
        completed.put((ordinal,row,prepared,video))

    def progress_snapshot():
        with lock:
            saved=sum(w['successes'] for w in workers.values())
            ready=sum(r['ready'] for r in finished.values())
            return {'annotation_progress':dict(phase='paused' if phase=='running' and bridge and bridge.paused else phase,
                total_episodes=len(reader.names),completed_episodes=len(finished),saved_episodes=saved,
                failed_episodes=sum(w['failures'] for w in workers.values()),
                pending_save_episodes=ready-saved if phase!='failed' else 0,
                failure_breakdown=dict(Counter(r['category'] for r in finished.values() if not r['ready'])),
                env_status={i:dict(w) for i,w in workers.items()},
                started_at=job_started,finished_at=finished_at,
                elapsed_s=max(0,(finished_at or time.time())-job_started),manual=not auto,
                marks=list(getattr(primary,'annotation_marks',[])) if not auto else [],error=error)}

    def report_progress(force=False):
        nonlocal last_report
        if boundary and not force:boundary()
        if not bridge:return
        if not force and hasattr(primary,'backend'):
            bridge.poll();control.update(bridge.paused,bridge.single_step);bridge.single_step=False
        if not force and time.monotonic()-last_report<.1:return
        last_report=time.monotonic()
        bridge.status('PAUSED' if bridge.paused and phase=='running' else 'RUNNING' if phase=='running' else 'FINISHED')

    def save_ready(final=False):
        while True:
            try:
                item=completed.get_nowait();pending[item[0]]=item
            except queue.Empty:break
        while pending and (final or len(rows) in pending):
            ordinal,row,prepared,video=pending.pop(min(pending) if final else len(rows))
            try:
                if prepared:
                    t=time.perf_counter()
                    row['exported_episode']=writer.commit(*prepared,video_staging=video)
                    timings['save_s']+=time.perf_counter()-t;row['annotation_accepted']=True
                    with lock:
                        worker=workers[finished[ordinal]['env_id']];worker['successes']+=1
                        if worker['state']=='saving':worker['state']='completed'
            finally:
                if video:video.abort()
            rows.append(row);print('[ANNOTATION] '+json.dumps(row),flush=True)
            atomic(output/'replay_progress.json',rows)

    executor=None;native=None;render=None;sim_step=None;stopped=False
    try:
        if bridge:bridge.progress_provider=progress_snapshot
        if not hasattr(primary,'backend'):
            native=primary.native;render=native.sim.render
            def measured_render(*args,**kwargs):
                t=time.perf_counter()
                try:return render(*args,**kwargs)
                finally:timings['native_render_s']+=time.perf_counter()-t
            native.sim.render=measured_render
            sim_step=native.sim.step
            def measured_physics_step(*args,**kwargs):
                t=time.perf_counter();render_before=timings['native_render_s']
                try:return sim_step(*args,**kwargs)
                finally:
                    timings['physics_step_s']+=time.perf_counter()-t-(timings['native_render_s']-render_before)
                    timings['physics_substeps']+=1
            native.sim.step=measured_physics_step
        if num_envs==1:
            adapters=[primary]
            actions=None
            if not hasattr(primary,'backend'):
                import torch
                actions=primary.native.scene['robot'].data.joint_pos[:,primary.ids].clone()
            while (ordinal:=claim(0)) is not None:
                episode=begin(primary,ordinal)
                try:
                    command=next(episode)
                    while True:
                        if command is None:command=next(episode);continue
                        if bridge and auto:bridge.before_step()
                        report_progress()
                        t=time.perf_counter();render_before=timings['native_render_s']
                        try:
                            step_targets(primary,command,actions)
                        except (ValueError,RuntimeError) as exc:command=episode.throw(exc)
                        else:
                            timings['physics_s']+=time.perf_counter()-t-(timings['native_render_s']-render_before)
                            command=next(episode)
                        with lock:workers[0]['steps']=primary.step_index
                except StopIteration as done:finish(primary,0,ordinal,done.value)
                save_ready();report_progress()
        elif hasattr(primary,'backend'):
            from ..simulation.mujoco_backend import MujocoBackend
            from .adapters import MujocoAdapter
            def worker(index):
                backend=MujocoBackend(render_backend=primary.backend.render_backend,shared_backend=primary.backend)
                a=MujocoAdapter(backend,reader.hz,output,primary.seed,stop)
                a.annotation_auto=auto
                with lock:adapters.append(a)
                try:
                    while (ordinal:=claim(index)) is not None:
                        episode=begin(a,ordinal)
                        try:
                            command=next(episode)
                            while True:
                                if command is None:command=next(episode);continue
                                control.before_step(index,stop)
                                if stop.is_set():raise GenerationStopped()
                                try:
                                    t=time.perf_counter();step_targets(a,command)
                                    a.metrics['physics_s']+=time.perf_counter()-t
                                except (ValueError,RuntimeError) as exc:command=episode.throw(exc)
                                else:command=next(episode)
                                with lock:workers[index]['steps']=a.step_index
                        except StopIteration as done:finish(a,index,ordinal,done.value)
                except GenerationStopped:pass
                finally:
                    episode=active_episodes.pop(a,None)
                    if episode:episode.close()
                    a.close()
            executor=ThreadPoolExecutor(max_workers=num_envs,thread_name_prefix='annotation')
            futures=[executor.submit(worker,i) for i in range(num_envs)]
            while not all(f.done() for f in futures):
                report_progress();save_ready()
                for future in futures:
                    if future.done():future.result()
                time.sleep(.01)  # only status polling; physics workers do not wait here
            for future in futures:future.result()
        else:
            import torch
            from .adapters import IsaacAdapter
            adapters=[primary]+[IsaacAdapter(native,i,output,primary.seed,stop) for i in range(1,num_envs)]
            actions=native.scene['robot'].data.joint_pos[:,primary.ids].clone()
            slots={}
            def advance(index,exception=None):
                a=adapters[index]
                while True:
                    if index not in slots:
                        ordinal=claim(index)
                        if ordinal is None:return
                        a.annotation_auto=auto;slots[index]=(ordinal,begin(a,ordinal),None)
                    ordinal,episode,_=slots[index]
                    try:
                        command=episode.throw(exception) if exception else next(episode)
                    except StopIteration as done:
                        finish(a,index,ordinal,done.value);del slots[index]
                        actions[index]=torch.as_tensor(a.state()['joint_pos'],device=native.device,dtype=actions.dtype)
                        exception=None
                        continue  # refill this slot before the next physics tick
                    actions[index]=torch.as_tensor(command,device=native.device,dtype=actions.dtype)
                    slots[index]=(ordinal,episode,command)
                    return
            for index in range(num_envs):advance(index)
            while slots:
                if bridge:bridge.before_step()
                report_progress();save_ready()
                t=time.perf_counter();render_before=timings['native_render_s']
                try:native.step(actions)
                except (ValueError,RuntimeError) as exc:
                    for index in list(slots):advance(index,exc)
                else:
                    timings['physics_s']+=time.perf_counter()-t-(timings['native_render_s']-render_before)
                    timings['batch_steps']+=1
                    # Complete all post-step observations before any slot restores
                    # its next episode (restoration also updates scene sensors).
                    tick_slots=list(slots)
                    for index in tick_slots:
                        ordinal,episode,_=slots[index]
                        try:
                            if next(episode) is not None:raise RuntimeError('Annotation post-step boundary missing')
                        except StopIteration as done:
                            finish(adapters[index],index,ordinal,done.value);del slots[index]
                            actions[index]=torch.as_tensor(adapters[index].state()['joint_pos'],device=native.device,dtype=actions.dtype)
                    for index in tick_slots:
                        advance(index)
                        with lock:workers[index]['steps']=adapters[index].step_index
            save_ready()
        save_ready()
    except KeyboardInterrupt:
        stopped=True;phase='stopped';stop.set()
    except BaseException as exc:
        phase='failed';error=str(exc);raise
    finally:
        stop.set();control.update(False)
        if executor:executor.shutdown(wait=True)
        for episode in list(active_episodes.values()):episode.close()
        if native and render:native.sim.render=render
        if native and sim_step:native.sim.step=sim_step
        for a in adapters:
            for key,value in a.buffer.metrics.items():timings[key]+=value
            for key,value in a.metrics.items():timings[key]+=value
            if hasattr(a,'backend'):
                for key,value in a.backend.render_metrics.items():timings[key]+=value
            # The caller owns primary; MuJoCo worker contexts close on their own
            # rendering thread. Only extra Isaac buffers are owned here.
            if a is primary:a.buffer.abort()
            elif not hasattr(a,'backend'):a.close()
        timings['video_encode_s']+=timings.get('encoding_s',0.)
        # Persist completed episodes on SIGINT; unfinished private videos are discarded.
        if stopped:save_ready(final=True)
        while pending:
            _,_,_,video=pending.pop(next(iter(pending)))
            if video:video.abort()
        while not completed.empty():
            _,_,_,video=completed.get_nowait()
            if video:video.abort()
        if phase=='running':phase='completed'
        finished_at=time.time()
        if bridge:
            bridge.details.update(progress_snapshot());report_progress(force=True)
            bridge.progress_provider=previous_provider
            bridge.callbacks.pop('mark',None);bridge.callbacks.pop('skip',None)
    physical=sum(r['physical_replay_passed'] for r in rows);accepted=sum(r['annotation_accepted'] for r in rows)
    elapsed=time.perf_counter()-started;timings['processing_s']=elapsed
    timings.update(input_video_decode_s=0.,input_video_reencode_s=0.)
    result=dict(passed=bool(rows) and physical==len(rows),episodes=rows,accepted=accepted,rejected=len(rows)-accepted,
                physical_accepted=physical,schema_compatible=True,hardware_tested=False,stopped=stopped,num_envs=num_envs,
                **progress_snapshot(),annotation_profile=dict(timings),assignments=sorted(assignments,key=lambda r:r['ordinal']),
                episodes_per_minute=len(rows)*60/elapsed,
                timing_semantics='CPU wall seconds; worker sums/encoding overlap; save includes validation; MuJoCo physics_s is backend.step; Isaac physics_s is native.step minus render including managers, physics_step_s is SimulationContext.step minus render; native_render includes reset/forward',
                input_video_decode_frames=0,input_video_reencode_frames=0)
    atomic(output/'cross_replay.json',result)
    return result
