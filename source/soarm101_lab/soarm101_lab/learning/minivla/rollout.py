"""Paused-simulation evaluation: no false real-time claim, no stale asynchronous actions."""
import argparse
import json
import hashlib
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
            case_seed=int(case.get('seed',a.seed+i)); seed_all(case_seed); obs=env.reset(case); guard.reset(); trace=[]; first_chunk=None; observation_hash=None; start=time.perf_counter()
            # Initial measurement establishes distance reduction from t=0.
            initial=env.measure(); remaining=[]; model_metrics={}
            for step in range(a.max_steps):
                replan=not remaining
                if replan:
                    obs=env.observe(); stamp=obs['timestamp']
                    if a.policy=='vla':
                        import torch
                        noise=torch.randn(1,policy.cfg.horizon,6,device='cuda',generator=torch.Generator(device='cuda').manual_seed(case_seed+step))
                        chunk,model_metrics=policy.predict(obs['images'],obs['state'],case['instruction'],noise=noise)
                    else:
                        chunk,model_metrics=policy.predict(obs['images'],obs['state'],case['instruction'])
                    if first_chunk is None:
                        first_chunk=np.asarray(chunk).tolist()
                        payload=b''.join(np.asarray(im).tobytes() for im in obs['images'])+np.asarray(obs['state']).tobytes()
                        observation_hash=hashlib.sha256(payload).hexdigest()
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
            metrics.update(case=case['id'],split=case['split'],seed=case_seed,wall_seconds=time.perf_counter()-start,
                pair_id=case.get('pair_id'),target_body=case['target_body'],initial_chunk=first_chunk,initial_observation_sha256=observation_hash)
            results.append(metrics)
            # Convert absent distractor infinity to null for strict JSON traces.
            for row in trace:
                if not np.isfinite(row['distractor_distance']): row['distractor_distance']=None
            (out/f'episode_{i:04d}.json').write_text(json.dumps({'case':case,'metrics':metrics,'trace':trace},indent=2,allow_nan=False))
            print(json.dumps(metrics),flush=True)
    finally: env.close()
    (out/'summary.json').write_text(json.dumps(aggregate(results),indent=2))
    from .evaluate import paired_action_differences
    (out/'paired_comparison.json').write_text(json.dumps(paired_action_differences(results),indent=2))

if __name__=='__main__': main()
