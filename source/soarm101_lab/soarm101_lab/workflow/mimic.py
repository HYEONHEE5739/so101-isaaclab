"""Common SO101 Mimic configuration, reference generator and exact-budget scheduler."""
import asyncio
import copy
from collections import Counter
from contextlib import contextmanager
import json
import threading
import time

import numpy as np
import torch

from .mimic_reference.config import MimicEnvCfg,SubTaskConfig
from .mimic_reference.data_generator import DataGenerator
from .mimic_reference.datagen_info_pool import DataGenInfoPool


def configuration(pick,place,seed=1):
    cfg=MimicEnvCfg();cfg.datagen_config.seed=seed
    cfg.subtask_configs={'tool0':[
        SubTaskConfig(object_ref=pick,subtask_term_signal='grasp',subtask_term_offset_range=(0,10),
            selection_strategy='nearest_neighbor_object',selection_strategy_kwargs={'nn_k':3},
            action_noise=0.,num_interpolation_steps=5,num_fixed_steps=0,apply_noise_during_interpolation=False),
        SubTaskConfig(object_ref=place,subtask_term_signal=None,subtask_term_offset_range=(0,0),
            selection_strategy='nearest_neighbor_object',selection_strategy_kwargs={'nn_k':3},
            action_noise=0.,num_interpolation_steps=5,num_fixed_steps=0,apply_noise_during_interpolation=False)]}
    return cfg


_RNG_LOCK=threading.RLock()


class RandomStream:
    """Reference code uses global NumPy/Torch RNGs; isolate only its random operations."""
    def __init__(self,seed):
        self.numpy=np.random.RandomState(seed).get_state()
        self.torch=torch.Generator().manual_seed(seed).get_state();self.depth=0

    @contextmanager
    def active(self):
        with _RNG_LOCK:
            if self.depth:
                self.depth+=1
                try:yield
                finally:self.depth-=1
                return
            previous_np=np.random.get_state();previous_torch=torch.random.get_rng_state()
            np.random.set_state(self.numpy);torch.random.set_rng_state(self.torch);self.depth=1
            try:yield
            finally:
                self.numpy=np.random.get_state();self.torch=torch.random.get_rng_state();self.depth=0
                np.random.set_state(previous_np);torch.random.set_rng_state(previous_torch)


class MimicGenerator(DataGenerator):
    """Trace and RNG ownership around unchanged official methods; no alternate retargeting."""
    def __init__(self,env,pool,seed):
        self.random=RandomStream(seed);self.trace=[];self.selection_trace=[];self.boundaries=None
        super().__init__(env,src_demo_datagen_info_pool=pool)

    def randomize_subtask_boundaries(self):
        with self.random.active():result=super().randomize_subtask_boundaries()
        self.boundaries={k:v.tolist() for k,v in result.items()};return result

    def select_source_demo(self,*args,**kwargs):
        with self.random.active():return super().select_source_demo(*args,**kwargs)

    def generate_eef_subtask_trajectory(self,env_id,eef_name,subtask_ind,boundaries,constraints,selected):
        with self.random.active():result=super().generate_eef_subtask_trajectory(env_id,eef_name,subtask_ind,boundaries,constraints,selected)
        self.selection_trace.append({'eef':eef_name,'subtask':subtask_ind,'source_index':int(selected[eef_name]),
            'boundary':boundaries[eef_name][int(selected[eef_name]),subtask_ind].tolist()})
        return result

    def merge_eef_subtask_trajectory(self,*args,**kwargs):
        with self.random.active():result=super().merge_eef_subtask_trajectory(*args,**kwargs)
        self.trace.append({'poses':torch.stack([w.pose for w in result]).detach().cpu().numpy(),
            'gripper':torch.stack([w.gripper_action for w in result]).detach().cpu().numpy(),
            'noise':np.array([float(w.noise) for w in result])})
        return result


def source_pool(env,source):
    pool=DataGenInfoPool(env,env.cfg,env.device,asyncio_lock=asyncio.Lock())
    pool.load_from_dataset_file(str(source))
    if pool.num_datagen_infos==0:raise ValueError('No annotated Source in Mimic pool')
    return pool


class GenerationStopped(Exception):pass


class PhysicsControl:
    def __init__(self,num_envs):
        self.condition=threading.Condition();self.paused=False;self.tokens={i:0 for i in range(num_envs)}
    def update(self,paused,single_step=False):
        with self.condition:
            self.paused=paused
            if single_step:self.tokens={i:1 for i in self.tokens}
            self.condition.notify_all()
    def before_step(self,index,stop):
        with self.condition:
            while self.paused and self.tokens[index]==0:
                if stop.is_set():raise GenerationStopped()
                self.condition.wait(.03)
            if self.paused:self.tokens[index]-=1


class GenerationScheduler:
    """Global attempts and successful commits. Commit budget is held across writer calls."""
    def __init__(self,target,max_attempts,num_envs,resumed=0,history=()):
        if type(target) is not int or target<=resumed:raise ValueError('Target must exceed resumed count')
        if type(max_attempts) is not int or max_attempts<1:raise ValueError('Positive maximum attempts required')
        self.target=target;self.max_attempts=max_attempts;self.successes=resumed;self.started=max([r['attempt'] for r in history],default=resumed);self.completed=len(history) if history else resumed
        self.failures=Counter(r['category'] for r in history if r['category'] not in ('success','interrupted','target_already_reached'))
        self.workers={i:{'state':'idle','attempts':0,'successes':0,'failures':{}} for i in range(num_envs)}
        self.records=copy.deepcopy(list(history));self.stop=threading.Event();self.lock=threading.RLock();self.start_time=time.perf_counter();self.resumed=resumed
        self.history_complete=not resumed or bool(history)
        self.control=PhysicsControl(num_envs)

    def claim(self,worker):
        with self.lock:
            if self.stop.is_set() or self.successes>=self.target or self.started>=self.max_attempts:return None
            self.started+=1;self.workers[worker]['state']='resetting';self.workers[worker]['attempts']+=1
            return self.started

    def set_state(self, worker, state):
        with self.lock:self.workers[worker]['state']=state

    def finish(self,worker,attempt,category,details,commit=None):
        with self.lock:
            row={'attempt':attempt,'env_id':worker,'category':category,**details}
            if category=='success' and self.successes<self.target:
                row['episode']=commit();self.successes+=1;self.workers[worker]['successes']+=1
                if self.successes>=self.target:self.stop.set()
            elif category=='success':row['category']='target_already_reached'
            elif category!='interrupted':
                self.failures[category]+=1
                f=self.workers[worker]['failures'];f[category]=f.get(category,0)+1
            self.completed+=1;self.workers[worker]['state']=row['category'];self.records.append(row)
            print('[MIMIC] '+json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
            return row

    def progress(self):
        with self.lock:
            latest=self.records[-1] if self.records else {};ik=latest.get('ik') or {};reason=latest.get('reason','')
            if latest.get('category')=='ik_failure' and ik:
                reason=f"위치 {ik['position_error_m']*1000:.2f}/{ik['position_tolerance_m']*1000:g} mm · 방향 {ik['rotation_error_rad']:.3f}/{ik['rotation_tolerance_rad']:g} rad"
            outcome={'success':'성공','ik_failure':'IK 계산 탈락','planning_failure':'계획 실패','simulation_failure':'시뮬레이션 실패',
                'grasp_failure':'집기 실패','place_failure':'배치 실패','timeout':'시간 초과','interrupted':'중단'}.get(latest.get('category'),'생성 중')
            if self.successes==self.target:outcome='목표 달성';reason=''
            decided=self.successes+sum(self.failures.values())
            return dict(target_successes=self.target,successes=self.successes,attempts=self.started,completed_attempts=self.completed,
                active_attempts=self.started-self.completed,failures=sum(self.failures.values()),failure_breakdown=dict(self.failures),
                interrupted_attempts=sum(r['category'] in ('interrupted','target_already_reached') for r in self.records),
                success_rate=self.successes/decided if decided and self.history_complete else None,
                success_rate_definition='successful commits / (successful commits + classified failures); excludes interrupted/in-flight attempts',
                env_status=copy.deepcopy(self.workers),
                wall_seconds=time.perf_counter()-self.start_time,episode_outcome=outcome,outcome_reason=reason,resumed_successes=self.resumed)


async def execute_trial(adapter,generator):
    """Official generate() with synchronous reset in this environment's worker."""
    class ResetMailbox:
        async def put(self,env_id):adapter.reset(env_id)
        async def join(self):pass
    from .mimic_reference.config import TerminationTermCfg
    return await generator.generate(0,TerminationTermCfg(func=lambda env:torch.tensor([env.check_success()],device=env.device)),
        env_reset_queue=ResetMailbox(),export_demo=False)
