"""Paused-simulation evaluation: no false real-time claim, no stale asynchronous actions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .policy import MiniVLA, ActionGuard
from ...simulation.mujoco_backend import MujocoSO101
from .evaluate import summarize_episode, aggregate
from .train import seed_all

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--mapping',required=True); p.add_argument('--cases',required=True); p.add_argument('--out',required=True)
    p.add_argument('--max-steps',type=int,default=900); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--policy',choices=['vla','act'],default='vla')
    a=p.parse_args(); seed_all(a.seed)
    if a.policy=='vla': policy=MiniVLA(a.checkpoint)
    else:
        from .act_baseline import ACTInference
        policy=ACTInference(a.checkpoint)
    env=MujocoSO101(a.mapping,policy.cfg.cameras,policy.control_hz)
    guard=ActionGuard(env.cfg['lower'],env.cfg['upper'],env.cfg['max_velocity'],policy.control_hz)
    cases=json.loads(Path(a.cases).read_text()); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); results=[]
    try:
        for i,case in enumerate(cases):
            case_seed=int(case.get('seed',a.seed+i)); seed_all(case_seed); obs=env.reset(case); guard.reset(); trace=[]; start=time.perf_counter()
            # Initial measurement establishes distance reduction from t=0.
            initial=env.measure(); remaining=[]; model_metrics={}
            for step in range(a.max_steps):
                replan=not remaining
                if replan:
                    obs=env.observe(); stamp=obs['timestamp']
                    chunk,model_metrics=policy.predict(obs['images'],obs['state'],case['instruction'])
                    if env.data.time!=stamp: raise RuntimeError('Simulation advanced during inference; discard stale chunk')
                    remaining=list(chunk[:policy.cfg.replan_steps])
                before_v,before_r=guard.violations,guard.rate_limited
                command=guard.apply(remaining.pop(0),env.state()); row=env.step(command)
                row.update(action=command.tolist(),replan=replan,
                    joint_limit_violation=guard.violations-before_v,rate_limited=guard.rate_limited-before_r)
                if replan: row.update(model_metrics)
                trace.append(row)
                if row['place']: break
            metric_trace=[dict(initial,action=case['robot_state'],replan=False)]+trace
            metrics=summarize_episode(metric_trace,hz=policy.control_hz,
                approach_radius=env.cfg.get('approach_radius_m',.04))
            metrics['completion_sim_seconds']=len(trace)/policy.control_hz
            metrics.update(case=case['id'],split=case['split'],seed=case_seed,wall_seconds=time.perf_counter()-start)
            results.append(metrics)
            # Convert absent distractor infinity to null for strict JSON traces.
            for row in trace:
                if not np.isfinite(row['distractor_distance']): row['distractor_distance']=None
            (out/f'episode_{i:04d}.json').write_text(json.dumps({'case':case,'metrics':metrics,'trace':trace},indent=2,allow_nan=False))
            print(json.dumps(metrics),flush=True)
    finally: env.close()
    (out/'summary.json').write_text(json.dumps(aggregate(results),indent=2))

if __name__=='__main__': main()
