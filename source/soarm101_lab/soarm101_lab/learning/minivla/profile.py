"""Run on the actual 8GB laptop BEFORE a long training job."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import MiniVLAConfig
from .head import ActionHead
from lerobot.utils.random_utils import set_seed as seed_all

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--adaptation',choices=['frozen','qlora']); p.add_argument('--out'); a=p.parse_args()
    cfg=MiniVLAConfig.load(a.config)
    if a.adaptation: cfg.adaptation=a.adaptation
    seed_all(cfg.seed)
    from .backbone import QwenBackbone
    backbone=QwenBackbone(cfg); head=ActionHead(cfg).cuda()
    params=list(head.parameters())+[p for p in backbone.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4)
    images=[[np.zeros((480,640,3),np.uint8) for _ in cfg.cameras]]
    instruction=['pick up the red cube and place it in the bin']
    s=torch.zeros(1,6,device='cuda'); actions=torch.zeros(1,cfg.horizon,6,device='cuda')
    mask=torch.ones(1,cfg.horizon,dtype=torch.bool,device='cuda')
    report={'gpu':torch.cuda.get_device_name(),'config':vars(cfg),'resolved_revision':backbone.resolved_revision,
        'head_parameters':sum(p.numel() for p in head.parameters()),'lora_targets':backbone.targets,
        'after_load_allocated_gib':torch.cuda.memory_allocated()/2**30}
    backbone.train(); head.train(); torch.cuda.reset_peak_memory_stats()
    train_seconds=[]; lora_gradient_verified=False
    for iteration in range(2):
        torch.cuda.synchronize(); begin=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=backbone(images,instruction); loss,_=head.loss(f,s,actions,mask)
        loss.backward()
        if cfg.adaptation=='qlora' and iteration>0 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in backbone.parameters() if p.requires_grad):
            raise RuntimeError('No LoRA action gradient')
        if cfg.adaptation=='qlora' and iteration>0: lora_gradient_verified=True
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True); opt.step()
        torch.cuda.synchronize(); train_seconds.append(time.perf_counter()-begin)
    report['train_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    report['train_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    report['vl_tokens']=f[0].shape[1]
    report.update(feature_shape=list(f[0].shape), preprocessing=backbone.last_preprocessing,
        selected_decoder_index=cfg.feature_layer-1, train_step_seconds=train_seconds,
        lora_gradient_verified=lora_gradient_verified, frozen_backbone_gradient_free=all(p.grad is None for p in backbone.parameters()) if cfg.adaptation=='frozen' else None)
    # Drop optimizer/grads/graphs before measuring deployment footprint.
    opt.zero_grad(set_to_none=True); del opt,loss,f; torch.cuda.empty_cache()
    backbone.eval(); head.eval(); times=[]; torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            f=backbone(images,instruction); result=head.sample(f,s)
        torch.cuda.synchronize(); times.append(time.perf_counter()-start)
    report.update(inference_median_s=float(np.median(times[1:])),inference_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        output_shape=list(result.shape),replan_budget_s=cfg.replan_steps/30)
    report['within_replan_budget']=report['inference_median_s']<=report['replan_budget_s']
    if a.out: Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
