"""Matched-noise instruction swap. Sensitivity is necessary, NOT semantic accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .policy import MiniVLA
from .data import load_manifest, episode_arrays

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint'); p.add_argument('--vlm-question'); p.add_argument('--expected-answer'); p.add_argument('--manifest',required=True)
    p.add_argument('--episode',required=True); p.add_argument('--frame',type=int,default=0)
    p.add_argument('--instruction-a'); p.add_argument('--instruction-b')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--out',required=True); args=p.parse_args()
    m=load_manifest(args.manifest)
    e=next(e for e in m['episodes'] if e['id']==args.episode)
    if args.vlm_question:
        from .config import MiniVLAConfig
        from .backbone import QwenBackbone, letterbox
        cfg=MiniVLAConfig(); arrays=episode_arrays(e,cfg.cameras,m['control_hz'])
        processor,model=QwenBackbone.load_pretrained(cfg,'cuda:0')
        image=letterbox(np.array(arrays['side'][args.frame]))
        text=processor.apply_chat_template([{'role':'user','content':[
            {'type':'image'},{'type':'text','text':args.vlm_question}]}],tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda:0')
        model.eval()
        with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=32,do_sample=False)
        answer=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        Path(args.out).write_text(json.dumps({'question':args.vlm_question,'answer':answer,
            'expected_answer':args.expected_answer, 'exact_match':answer.strip().lower()==args.expected_answer.strip().lower() if args.expected_answer else None,
            'scope':'Qwen visual-language QA diagnostic only; not action accuracy'},indent=2))
        return
    if not args.checkpoint or not args.instruction_a or not args.instruction_b:
        p.error('Action swap requires checkpoint and both instructions')
    policy=MiniVLA(args.checkpoint)
    a=episode_arrays(e,policy.cfg.cameras,m['control_hz'])
    images=[np.array(a[c][args.frame]) for c in policy.cfg.cameras]; state=a['state'][args.frame]
    torch.manual_seed(args.seed); noise=torch.randn(1,policy.cfg.horizon,6,device='cuda')
    ca,ma=policy.predict(images,state,args.instruction_a,noise=noise)
    cb,mb=policy.predict(images,state,args.instruction_b,noise=noise)
    scale=np.array(policy.pre.stats['action_std'])
    result={'instruction_a':args.instruction_a,'instruction_b':args.instruction_b,
        'action_a':ca.tolist(),'action_b':cb.tolist(),
        'normalized_action_difference_rms':float(np.sqrt((((ca-cb)/scale)**2).mean())),
        'metrics_a':ma,'metrics_b':mb,
        'interpretation':'Nonzero difference is not proof of correct target selection. Run paired MuJoCo cases.'}
    Path(args.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
