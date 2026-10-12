"""Matched-noise instruction swap. Sensitivity is necessary, NOT semantic accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .policy import MiniVLA
from .data import load_manifest, episode_arrays

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--episode',required=True); p.add_argument('--frame',type=int,default=0)
    p.add_argument('--instruction-a',required=True); p.add_argument('--instruction-b',required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--out',required=True); args=p.parse_args()
    policy=MiniVLA(args.checkpoint); m=load_manifest(args.manifest)
    e=next(e for e in m['episodes'] if e['id']==args.episode)
    a=episode_arrays(e,policy.cfg.cameras,m['control_hz'])
    images=[np.array(a[c][args.frame]) for c in policy.cfg.cameras]; state=a['state'][args.frame]
    torch.manual_seed(args.seed); noise=torch.randn(1,policy.cfg.horizon,6,device='cuda')
    ca,ma=policy.predict(images,state,args.instruction_a,noise=noise)
    cb,mb=policy.predict(images,state,args.instruction_b,noise=noise)
    scale=np.array(policy.pre.stats['action_std'])
    result={'instruction_a':args.instruction_a,'instruction_b':args.instruction_b,
        'action_a':ca.tolist(),'action_b':cb.tolist(),
        'normalized_action_difference_rms':float(np.sqrt((((ca-cb)/scale)**2).mean())),
        'grounding_a':ma['grounding_uv'],'grounding_b':mb['grounding_uv'],
        'interpretation':'Nonzero difference is not proof of correct target selection. Run paired MuJoCo cases.'}
    Path(args.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
