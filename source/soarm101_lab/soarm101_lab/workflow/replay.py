"""Explicit Replay progress and task result reporting over Simulation Playback."""
import json
from pathlib import Path
import time

from ..data.episodes import validate_episode
from ..real2sim.storage import atomic
from ..simulation.playback import restore_episode,replay_steps,step_targets


def run(adapter,reader,output,bridge=None,*,evaluation_playback=False):
    """Read episodes and replay on a borrowed worker-owned adapter/environment.

    No annotation writer or scheduling is dispatched here. The caller owns the
    engine and close; this run owns the command tensor. Each episode finally
    aborts its adapter buffer.
    Restore/input errors propagate. Per-step ValueError/RuntimeError become the
    existing failed row; Interrupt/Cancel propagate after iterator cleanup.
    """
    rows=[];output=Path(output)
    actions=None
    if not hasattr(adapter,'backend'):
        actions=adapter.native.scene['robot'].data.joint_pos[:,adapter.ids].clone()
    for name in reader.names:
        values,attrs,videos=reader.episode(name)
        validate_episode(values,attrs,videos,reader.hz,video_root=reader.path.parent)
        validating=bridge is not None and bridge.details.get('workflow_phase')=='source_validation'
        if validating:bridge.source_progress(len(rows),len(reader.names),name,0,len(values['actions']))
        def before_step(i):
            if bridge:
                if bridge.details.get('workflow_phase')=='source_validation':
                    bridge.source_progress(reader.names.index(name),len(reader.names),name,i+1,len(values['actions']))
                bridge.before_step()
        episode=None
        try:
            restore_episode(adapter,values,attrs)
            episode=replay_steps(adapter,values,before_step=before_step,after_step=adapter.update_grasp)
            try:
                command=next(episode);frame=0
                while True:
                    if command is None:
                        try:
                            if bridge:
                                bridge.steps=frame+1;bridge.details.update(episode=name,episode_index=reader.names.index(name)+1,
                                    episode_total=len(reader.names),task_signals={'grasp':adapter.tracker.completed,'place':adapter.place_success()});bridge.status()
                            if getattr(adapter,'realtime',False):time.sleep(1/reader.hz)
                        except (ValueError,RuntimeError) as exc:command=episode.throw(exc)
                        else:frame+=1;command=next(episode)
                        continue
                    try:step_targets(adapter,command,actions)
                    except (ValueError,RuntimeError) as exc:command=episode.throw(exc)
                    else:command=next(episode)
            except StopIteration as completed:metrics=completed.value
            reason=metrics['simulation_error'];category='simulation_failure';passed=False
            if reason is None:
                reason=''
                try:
                    task_passed=adapter.check_success()
                    evaluation_replay=evaluation_playback and bool(reader.args.get('evaluation_schema'))
                    passed=True if evaluation_replay else task_passed
                    category='success' if task_passed else adapter.failure_category()
                    if not passed:reason='Stable grasp was not observed' if category=='grasp_failure' else 'Stable grasp observed; final placement condition not met'
                except (ValueError,RuntimeError) as exc:passed=False;reason=str(exc);category='simulation_failure'
            row={'episode':name,'physical_replay_passed':bool(passed),'category':category,'reason':reason,
                'grasp_confirmed':adapter.tracker.completed,'place_confirmed':bool(adapter.place_success()),
                'steps':adapter.step_index,'joint_l2_max_rad':metrics['joint_l2_max_rad'],
                'object_position_error_max_m':metrics['object_position_error_max_m'],'source_backend':reader.backend,
                'execution_backend':'mujoco' if hasattr(adapter,'backend') else 'isaac'}
            if evaluation_playback and reader.args.get('evaluation_schema'):
                row.update(recorded_evaluation_success=bool(attrs.get('success',False)),
                    replay_task_success=bool(adapter.check_success()),
                    replay_scope='recorded evaluation playback; passed is execution, task outcome may fail')
        finally:
            if episode:episode.close()
            adapter.buffer.abort()
        rows.append(row);print('[REPLAY] '+json.dumps(row),flush=True)
        if validating:bridge.source_progress(len(rows),len(reader.names),name,len(values['actions']),len(values['actions']))
        atomic(output/'replay_progress.json',rows)
    physical=sum(r['physical_replay_passed'] for r in rows)
    result={'passed':bool(rows) and physical==len(rows),'episodes':rows,
        'accepted':physical,'rejected':len(rows)-physical,'physical_accepted':physical,
        'schema_compatible':True,'hardware_tested':False}
    atomic(output/'cross_replay.json',result);return result
