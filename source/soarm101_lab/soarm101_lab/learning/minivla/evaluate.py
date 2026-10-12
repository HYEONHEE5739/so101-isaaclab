"""Stage metrics keep semantic selection separate from grasp/transport/place."""
import argparse
import json
from pathlib import Path
import numpy as np

def summarize_episode(trace, *, hz, approach_radius=.04, progress_threshold=.02):
    if not trace: raise ValueError('Empty rollout')
    targets=np.array([r['target_distance'] for r in trace],float)
    distractors=np.array([r['distractor_distance'] for r in trace],float)
    # Grounding is judged before manipulation, avoiding carry-induced distance confounds.
    stop=next((i for i,r in enumerate(trace) if r['grasp']),len(trace)-1)
    td=targets[:stop+1]; dd=distractors[:stop+1]
    candidates=np.where(np.minimum(td,dd)<approach_radius)[0]
    selected=None
    if len(candidates):
        i=int(candidates[0]); selected=bool(td[i]<dd[i])
    preferred=bool(td[-1]<dd[-1]) if np.isfinite(dd).any() else None
    reduction=float(td[0]-td.min())
    grounding=selected if selected is not None else bool(preferred and reduction>=progress_threshold)
    if not np.isfinite(dd).any(): grounding=None  # single-object trial cannot test target discrimination
    grasp=any(r['grasp'] for r in trace); transport=any(r['transport'] for r in trace)
    success=any(r['place'] for r in trace); approach=bool(td.min()<approach_radius)
    failure=('SUCCESS' if success else 'TARGET_SELECTION_FAIL' if grounding is False else
             'APPROACH_FAIL' if not approach else 'GRASP_FAIL' if not grasp else
             'TRANSPORT_FAIL' if not transport else 'PLACE_FAIL')
    actions=np.asarray([r['action'] for r in trace],float)
    acceleration=np.diff(actions,n=2,axis=0)*hz**2
    calls=[r for r in trace if r.get('replan')]
    return {'success':success,'grounding_success':grounding,'target_selection_correct':selected,
        'wrong_object_approach':None if selected is None else not selected,
        'approach':approach,'grasp':grasp,'transport':transport,'failure':failure,
        'target_distance_reduction_m':reduction,'target_preference':preferred,
        'final_approach_distance_m':float(td[-1]), 'final_approach_eef':trace[stop]['eef'],
        'completion_sim_seconds':len(trace)/hz,'replans':len(calls),
        'command_acceleration_rms_per_joint':np.sqrt((acceleration**2).mean(0)).tolist() if len(acceleration) else [0.]*6,
        'joint_limit_violations':sum(r.get('joint_limit_violation',0) for r in trace),
        'rate_limited_steps':sum(r.get('rate_limited',0) for r in trace),
        'inference_mean_s':float(np.mean([r['latency_s'] for r in calls])) if calls else None,
        'inference_p95_s':float(np.quantile([r['latency_s'] for r in calls],.95)) if calls else None,
        'peak_allocated_gib':max(r.get('peak_allocated_gib',0) for r in trace)}

def aggregate(records):
    grouped={}
    for row in records: grouped.setdefault(row.get('split','unknown'),[]).append(row)
    result={}
    for split, rows in grouped.items():
        summary={'episodes':len(rows)}
        for key in ('success','grounding_success','grasp','transport','target_selection_correct','wrong_object_approach'):
            vals=[float(r[key]) for r in rows if r.get(key) is not None]
            summary[key+'_rate']=float(np.mean(vals)) if vals else None
            summary[key+'_count']=len(vals)
        summary['failure_counts']={k:sum(r['failure']==k for r in rows) for k in sorted({r['failure'] for r in rows})}
        result[split]=summary
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); a=p.parse_args()
    records=[json.loads(Path(f).read_text())['metrics'] for f in a.files]
    print(json.dumps(aggregate(records),indent=2))
