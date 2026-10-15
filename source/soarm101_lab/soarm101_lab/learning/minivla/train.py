import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
import torch
from .config import MiniVLAConfig, ARCHITECTURE_SCHEMA
from .data import Episodes, collate
from .head import ActionHead

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_batch(batch, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

def compute_loss(backbone, head, batch, augment=False):
    features=backbone(batch['images'], batch['instruction'], augment=augment)
    return head.loss(features,batch['state'],batch['actions'],batch['mask'])

def adapter_state(backbone):
    if backbone.cfg.adaptation=='frozen': return None
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu() for k,v in get_peft_model_state_dict(backbone.model).items()}

def restore_adapter(backbone, state):
    if state is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(backbone.model,state)

def require_architecture(checkpoint):
    if checkpoint.get('architecture_schema') != ARCHITECTURE_SCHEMA:
        raise ValueError('Legacy/incompatible MiniVLA checkpoint: expected '+ARCHITECTURE_SCHEMA+'; retraining is required')


def validate_resume(checkpoint, cfg, stats, control_hz):
    """Allow extending total steps while retaining model, dataset and optimizer contracts."""
    require_architecture(checkpoint)
    saved = dict(checkpoint['config']); current = asdict(cfg)
    saved.pop('steps'); current.pop('steps')
    if saved != current or checkpoint['statistics'] != stats:
        raise ValueError('Resume config/stats mismatch')
    if checkpoint['control_hz'] != control_hz:
        raise ValueError('Resume control frequency mismatch')
    if cfg.steps <= checkpoint['step']:
        raise ValueError('Total steps must exceed the saved training step')

def save_checkpoint(path, cfg, head, backbone, stats, optimizer, step, best, stale, hz):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save({'architecture_schema':ARCHITECTURE_SCHEMA,'config':asdict(cfg),'head':head.state_dict(),'adapter':adapter_state(backbone),
                'statistics':stats,'optimizer':optimizer.state_dict(),'step':step,'best':best,'stale':stale,
                'control_hz':hz},tmp)
    tmp.replace(path)

def validate(backbone, head, dataset, device):
    backbone.eval(); head.eval(); per_episode=[]
    # Fixed, evenly spaced temporal samples; do not select checkpoints using any OOD/test split.
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(12345)
        offset=0
        for array in dataset.arrays:
            n=len(array['state']); values=[]
            for t in np.unique(np.linspace(0,n-1,min(n,8),dtype=int)):
                batch=device_batch(collate([dataset[offset+int(t)]]),device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,metrics=compute_loss(backbone,head,batch)
                values.append(metrics['action_loss'])
            offset+=n; per_episode.append(float(np.mean(values)))
    return float(np.mean(per_episode))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--stats',required=True); p.add_argument('--out',required=True); p.add_argument('--resume')
    args=p.parse_args(); cfg=MiniVLAConfig.load(args.config)
    stats=json.loads(Path(args.stats).read_text())
    train=Episodes(args.manifest,stats,cfg,'train'); val=Episodes(args.manifest,stats,cfg,'val')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume: raise FileExistsError('Run exists: use --resume or a fresh --out')
    if args.resume:
        require_architecture(torch.load(args.resume,map_location='cpu',weights_only=True))
    seed_all(cfg.seed)
    from .backbone import QwenBackbone
    backbone=QwenBackbone(cfg)
    # Pin subsequent loads to the downloaded revision, not mutable main.
    if backbone.resolved_revision: cfg.revision=backbone.resolved_revision
    head=ActionHead(cfg).to('cuda:0')
    groups=[{'params':head.parameters(),'lr':cfg.head_lr,'base_lr':cfg.head_lr}]
    lora=[v for v in backbone.parameters() if v.requires_grad]
    if lora: groups.append({'params':lora,'lr':cfg.lora_lr,'base_lr':cfg.lora_lr})
    opt=torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)
    params=[v for g in opt.param_groups for v in g['params']]
    start=0; best=float('inf'); stale=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        validate_resume(ck, cfg, stats, train.manifest['control_hz'])
        head.load_state_dict(ck['head']); restore_adapter(backbone,ck['adapter']); opt.load_state_dict(ck['optimizer'])
        start=ck['step']; best=ck['best']; stale=ck['stale']
        previous_best=Path(args.resume).parent/'best.pt'
        if not previous_best.is_file(): raise ValueError('Resume requires the saved best.pt alongside last.pt')
        if previous_best.resolve()!=(out/'best.pt').resolve(): shutil.copy2(previous_best,out/'best.pt')
        save_checkpoint(out/'last.pt',cfg,head,backbone,stats,opt,start,best,stale,train.manifest['control_hz'])
    cfg.save(out/'config.json')
    Path(out/'statistics.json').write_text(json.dumps(stats,indent=2))
    Path(out/'manifest.json').write_text(Path(args.manifest).read_text())
    import importlib.metadata
    runtime={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','bitsandbytes')}
    runtime.update(python=platform.python_version(),gpu=torch.cuda.get_device_name(),lora_targets=backbone.targets,
                   head_parameters=sum(v.numel() for v in head.parameters()),lora_parameters=sum(v.numel() for v in lora))
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
    for step in range(start,cfg.steps):
        seed_all(cfg.seed+step); backbone.train(); head.train(); opt.zero_grad(set_to_none=True)
        warmup=max(1,int(cfg.steps*.05))
        scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,cfg.steps-warmup)))
        for group in opt.param_groups: group['lr']=group['base_lr']*scale
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter(); losses=[]
        indices=torch.multinomial(train.weights,cfg.accumulation,replacement=True)
        for idx in indices:
            batch=device_batch(collate([train[int(idx)]]),'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=compute_loss(backbone,head,batch,augment=True)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            (loss/cfg.accumulation).backward(); losses.append(float(loss.detach()))
        if lora and step==start+1 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in lora):
            raise RuntimeError('Action loss does not reach LoRA')
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        log={'step':step+1,'loss':float(np.mean(losses)),'grad_norm':float(norm),
             'seconds':time.perf_counter()-begin,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        if (step+1)%cfg.val_every==0 or step+1==cfg.steps:
            score=validate(backbone,head,val,'cuda:0'); log['validation_action_loss']=score
            improved=score<best
            if improved: best=score; stale=0
            else: stale+=1
            for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                save_checkpoint(out/name,cfg,head,backbone,stats,opt,step+1,best,stale,train.manifest['control_hz'])
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
        print(json.dumps(log),flush=True)
        if stale>=cfg.patience: break

if __name__=='__main__': main()
