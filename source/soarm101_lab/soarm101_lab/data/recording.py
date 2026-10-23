"""Streaming numeric/video frame buffers; callers own observation and physics.

No engine, Workspace, Stage or task-success callback is accepted here.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from ..utils.transforms import as_pose7,cartesian_actions


class SourceEpisode:
    """Caller-owned Source rows and bounded videos; no env/FK/format writer.

    begin copies the small initial-state tree once per episode. append freezes
    recorded numeric leaves, and VideoEpisode's existing queue copies RGB once.
    Call abort after save/discard/cancel/error; no full RGB history is retained.
    """
    def __init__(self,output,hz,camera_names,object_names,pick,video_crf=None):
        import os
        self.output=Path(output);self.hz=hz;self.camera_names=tuple(camera_names)
        self.object_names=tuple(object_names);self.pick=pick
        self.crf=int(os.environ.get('SO101_VIDEO_CRF','18')) if video_crf is None else video_crf
        self.rows=[];self.initial=None;self.video=None

    def begin(self,initial_state:dict)->None:
        import copy
        self.abort();self.initial=copy.deepcopy(initial_state)

    def append(self,sample:dict,images:dict)->None:
        """Freeze numeric sample leaves and enqueue RGB before caller's step.

        sample has six-axis state/action/raw_state/raw_action/requested_action,
        timestamp seconds, 4x4 eef_pose and named object_poses. images maps the
        declared cameras to uint8 RGB. Producer buffers may change on return.
        Camera/encoder errors propagate after private staging cleanup.
        """
        from ..utils.async_video_writer import VideoEpisode
        try:
            if set(images)!=set(self.camera_names):raise ValueError('Recorded camera set differs')
            if self.video is None:self.video=VideoEpisode(self.output,self.hz,self.crf)
            self.video.append(images)
            row={k:np.array(sample[k],copy=True) for k in ('state','action','raw_state','raw_action','requested_action','eef_pose')}
            row['timestamp']=float(sample['timestamp'])
            row['object_poses']={name:np.array(sample['object_poses'][name],copy=True) for name in self.object_names}
            if self.pick is not None:row['object_pose']=row['object_poses'][self.pick]
            self.rows.append(row)
        except BaseException:self.abort();raise

    def finish(self)->dict:
        if not self.rows or self.video is None:raise ValueError('Cannot save an empty episode')
        try:return self.video.finish(len(self.rows))
        except BaseException:self.abort();raise

    def abort(self)->None:
        if self.video is not None:self.video.abort();self.video=None
        self.rows=[];self.initial=None


def source_episode_values(rows,joint_targets,geometry):
    """Pure Common formatting of Source snapshots and precomputed geometry."""
    b=geometry['base_workspace'];world_base=geometry['world_base'];target=joint_targets
    q=np.array([r['raw_state'] for r in rows]);n=len(rows)
    ee=b@np.array([r['eef_pose'] for r in rows]);goals=b@geometry['target_poses']
    objects={name:b@np.array([r['object_poses'][name] for r in rows]) for name in rows[0]['object_poses']}
    objects.update({name:np.repeat((b@pose)[None],n,axis=0) for name,pose in geometry['static_poses'].items()})
    world={name:as_pose7(world_base@value) for name,value in objects.items()};ts=np.array([r['timestamp'] for r in rows])
    return {'actions':cartesian_actions(goals,ee[0],target[:,5]),'joint_targets':target,'action_timestamp':ts,
        'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':q,'ee_state':as_pose7(ee)},'object_poses':world,
            'datagen_info':{'eef_pose':{'tool0':ee},'target_eef_pose':{'tool0':goals},'object_pose':objects,'subtask_term_signals':{}},
            'camera_timestamps':{c:ts for c in geometry['camera_names']}},
        'initial_state':{'articulation':{'robot':{'root_pose':as_pose7(world_base)[None],'root_velocity':np.zeros((1,6)),
            'joint_position':q[0:1],'joint_velocity':np.zeros((1,6))}},
            'rigid_object':{name:{'root_pose':v[0:1],'root_velocity':np.zeros((1,6))} for name,v in world.items()}},
        'collection':{key:np.array([r[key] for r in rows]) for key in ('state','action','requested_action')}}


class EpisodeBuffer:
    """One caller owns a pre-step frame buffer and private bounded MP4 streams.

    begin(initial_state) borrows a frozen native-state snapshot. append receives
    detached observed state/images and measured task events at the same tick as
    the following absolute command. No physics or rendering occurs here.
    The caller must not mutate borrowed snapshots before finish/abort.
    finish returns (values, attrs, videos); it finalizes staging, not a dataset.
    Commit is serialized by DatasetWriter; abort releases staging/rows on every
    success, cancellation or error path. Repeated abort is allowed.
    """
    def __init__(self,output,hz:float,base_world:np.ndarray,camera_names):
        self.output=Path(output);self.hz=hz;self.base_world=np.array(base_world,copy=True);self.camera_names=tuple(camera_names)
        self.rows=[];self.video=None;self.initial=None
        self.metrics=defaultdict(float);self.queue_peak=0

    def abort(self):
        if self.video is not None:
            self.video.abort()
            self.metrics['encoding_s']+=sum(w.encoding_seconds for w in self.video.writers.values())
            self.queue_peak=max(self.queue_peak,max((w.queue_peak for w in self.video.writers.values()),default=0))
            self.video=None
        self.rows=[]

    def begin(self,initial_state:dict):self.abort();self.initial=initial_state

    def append(self,joints,target,state:dict,images:dict,*,grasp:bool,place:bool,ik=None,events=None)->None:
        from ..utils.async_video_writer import VideoEpisode
        if self.video is None:self.video=VideoEpisode(self.output,self.hz,18)
        t=time.perf_counter();self.video.append({k+'_cam':v for k,v in images.items()})
        self.metrics['video_enqueue_s']+=time.perf_counter()-t
        self.queue_peak=max(self.queue_peak,max((v._queue.qsize() for v in self.video.writers.values()),default=0))
        self.rows.append({'state':state,'joint_target':np.array(joints,copy=True),'eef_target':np.array(target,copy=True),
            'grasp':bool(grasp),'place':bool(place),'ik':ik,**({'events':dict(events)} if events is not None else {})})

    def finish(self,attrs:dict,*,success:bool)->tuple[dict,dict,dict]:
        if not self.rows or self.video is None:raise ValueError('No completed episode frames')
        n=len(self.rows);states=[r['state'] for r in self.rows]
        q=np.array([s['joint_pos'] for s in states]);targets=np.array([r['joint_target'] for r in self.rows])
        ee=np.array([s['eef_pose'] for s in states]);goals=np.array([r['eef_target'] for r in self.rows])
        ts=np.arange(n,dtype=np.float64)/self.hz
        objects={name:np.array([s['object_poses'][name] for s in states]) for name in states[0]['object_poses']}
        b=self.base_world;world={name:as_pose7(b@value) for name,value in objects.items()}
        values={'actions':cartesian_actions(goals,ee[0],targets[:,5]),'joint_targets':targets,'action_timestamp':ts,
            'obs':{'timestamp':ts[:,None],'policy':{'joint_pos':q,'ee_state':as_pose7(ee)},
                'object_poses':world,'datagen_info':{'eef_pose':{'tool0':ee},'target_eef_pose':{'tool0':goals},
                    'object_pose':objects,'subtask_term_signals':{}},
                'camera_timestamps':{name:ts for name in self.camera_names}},'initial_state':self.initial}
        if all(r.get('ik') is not None for r in self.rows):
            values['ik']={key:np.asarray([r['ik'][key] for r in self.rows]) for key in
                ('ik_converged','ik_candidate_applied','ik_held','ik_invalid_candidate','ik_invalid_command','ik_numerical_failure',
                 'ik_position_error_mm','ik_orientation_error_deg','ik_success','ik_duration_ms','ik_iterations_or_evaluations','position_error_mm',
                 'orientation_error_deg','joint_limit_violation','common_pose_passed','command_applied')}
            values['ik']['candidate_joint_rad']=np.asarray([r['ik']['fitted_joint_rad'] for r in self.rows])
            values['ik']['failure_reason']=np.asarray([r['ik']['failure_reason'].encode() for r in self.rows],dtype='S64')
        custom='events' in self.rows[0]
        keys=self.rows[0]['events'] if custom else ('grasp','place')
        events={key:np.array([(r['events'] if custom else r)[key] for r in self.rows],dtype=np.uint8)[:,None] for key in keys}
        if success:
            for key in (self.final_events if custom else ('place',)):
                if not custom or self.final_event_values.get(key,False):events[key][-1]=1
        signals=self.signal_names if custom else (['grasp'] if np.any(events['grasp']) else [])
        values['obs']['events']=events
        values['obs']['datagen_info']['subtask_term_signals']={key:events[key] for key in signals}
        indices={key:int(np.flatnonzero(signal)[0]) for key,signal in events.items() if np.any(signal)}
        attrs={k:v for k,v in attrs.items() if k!='annotation'}
        if custom:
            attrs['annotation']=json.dumps({'schema':'so101.event-annotation/1','frame_count':n,'events':indices,
                'signals':list(signals),'timing':'first pre-step event; explicitly declared final events mapped to last action'})
        # Native recipe and legacy recording retain identical schema/JSON ordering.
        if set(events)=={'grasp','place'} and list(signals)==['grasp'] and 'grasp' in indices and 'place' in indices:
            attrs['annotation']=json.dumps({'schema':'so101.common-subtasks/1','frame_count':n,**indices,
                'method':attrs.get('annotation_method','stable_contact_replay'),
                'timing':'first pre-step observation with event; final post-step placement mapped to last action'})
        t=time.perf_counter();specs=self.video.finish(n);self.metrics['video_finalize_s']+=time.perf_counter()-t
        videos={name:{**spec,'file':str((self.video.folder/spec['file']).resolve())} for name,spec in specs.items()}
        return values,attrs,videos
