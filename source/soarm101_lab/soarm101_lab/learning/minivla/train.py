import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import shutil
import signal
import time
import numpy as np
import torch
from .config import MiniVLAConfig, ARCHITECTURE_SCHEMA
from .data import Episodes, collate, dataset_identity
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


def validate_resume(checkpoint, cfg, stats, control_hz, identity=None):
    """Allow extending total steps while retaining model, dataset and optimizer contracts."""
    require_architecture(checkpoint)
    if identity is not None and checkpoint.get('dataset_identity') is not None and checkpoint['dataset_identity']!=identity:
        raise ValueError('Resume dataset byte identity mismatch')
    saved = asdict(MiniVLAConfig.from_dict(checkpoint['config'])); current = asdict(cfg)
    saved.pop('steps'); current.pop('steps')
    for key in ('save_freq','val_every','early_stopping','patience'):
        saved.pop(key,None);current.pop(key,None)  # Saving cadence does not change model/optimizer contracts.
    if saved != current or checkpoint['statistics'] != stats:
        raise ValueError('Resume config/stats mismatch')
    if checkpoint['control_hz'] != control_hz:
        raise ValueError('Resume control frequency mismatch')
    if cfg.steps <= checkpoint['step']:
        raise ValueError('Total steps must exceed the saved training step')

def save_checkpoint(path, cfg, head, backbone, stats, optimizer, step, best, stale, hz, identity=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    torch.save({'architecture_schema':ARCHITECTURE_SCHEMA,'config':asdict(cfg),'head':head.state_dict(),'adapter':adapter_state(backbone),
                'statistics':stats,'optimizer':optimizer.state_dict(),'step':step,'best':best,'stale':stale,
                'control_hz':hz,'dataset_identity':identity},tmp)
    tmp.replace(path)
    print(f'[WORKFLOW] Checkpoint saved · step {step:,} · {path}',flush=True)

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
    if not args.resume:cfg.accumulation=1  # New training: one batch per optimizer update.
    stats=json.loads(Path(args.stats).read_text())
    train=Episodes(args.manifest,stats,cfg,'train'); val=Episodes(args.manifest,stats,cfg,'val')
    identity=dataset_identity(train.manifest)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('*.pt')) or list((out/'checkpoints').glob('*.pt')):
        raise FileExistsError('Checkpoints already exist: choose a fresh --out; --resume reads the original checkpoint')
    if args.resume:
        previous=torch.load(args.resume,map_location='cpu',weights_only=True);require_architecture(previous)
        if cfg.revision=='main':cfg.revision=previous['config']['revision']
        if previous.get('dataset_identity') is None and not train.manifest.get('storage'):
            # Historical canonical checkpoints need their original full Workflow artifact hash.
            resume_root=Path(args.resume).parent
            if resume_root.name=='checkpoints':resume_root=resume_root.parent
            provenance=resume_root/'project_provenance.json'
            from ...workflow.artifacts import fingerprint
            proof=json.loads(provenance.read_text()) if provenance.is_file() else {}
            if not proof.get('dataset_sha256') or fingerprint(Path(args.manifest).parent)!=proof['dataset_sha256']:
                raise ValueError('Legacy Resume dataset bytes cannot be verified; inference remains supported, do not switch loaders')
        validate_resume(previous,cfg,stats,train.manifest['control_hz'],identity)
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
        validate_resume(ck, cfg, stats, train.manifest['control_hz'],identity)
        head.load_state_dict(ck['head']); restore_adapter(backbone,ck['adapter']); opt.load_state_dict(ck['optimizer'])
        start=ck['step']; best=ck['best']; stale=ck['stale']
        previous_root=Path(args.resume).parent
        if previous_root.name=='checkpoints':previous_root=previous_root.parent
        previous_best=previous_root/'best.pt'
        if previous_best.is_file():
            best_state=torch.load(previous_best,map_location='cpu',weights_only=True)
            if best_state['step']<=start:
                if previous_best.resolve()!=(out/'best.pt').resolve():shutil.copy2(previous_best,out/'best.pt')
            else:
                best=float('inf');stale=0  # A later run's best is not evidence for this older checkpoint.
        else:best=float('inf');stale=0
        save_checkpoint(out/'last.pt',cfg,head,backbone,stats,opt,start,best,stale,train.manifest['control_hz'],identity)
    cfg.save(out/'config.json')
    Path(out/'statistics.json').write_text(json.dumps(stats,indent=2))
    Path(out/'manifest.json').write_text(Path(args.manifest).read_text())
    import importlib.metadata
    runtime={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','bitsandbytes')}
    runtime.update(python=platform.python_version(),gpu=torch.cuda.get_device_name(),lora_targets=backbone.targets,
                   head_parameters=sum(v.numel() for v in head.parameters()),lora_parameters=sum(v.numel() for v in lora))
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
    validation_protocol=dict(method='episode_mean_of_fixed_temporal_samples',max_samples_per_episode=8,episodes=len(val.arrays),samples=sum(min(len(a['state']),8) for a in val.arrays))
    (out/'validation_protocol.json').write_text(json.dumps(validation_protocol,indent=2))
    print(f'[WORKFLOW] MiniVLA batch_size={cfg.batch_size} · accumulation={cfg.accumulation} · effective_batch_size={cfg.batch_size*cfg.accumulation}',flush=True)
    completed_step=start;stopped=False
    stop_requested=False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested=True
    previous_sigint=signal.signal(signal.SIGINT,request_stop)
    try:
        for step in range(start,cfg.steps):
            if stop_requested: raise KeyboardInterrupt
            seed_all(cfg.seed+step); backbone.train(); head.train(); opt.zero_grad(set_to_none=True)
            warmup=max(1,int(cfg.steps*.05))
            scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,cfg.steps-warmup)))
            for group in opt.param_groups: group['lr']=group['base_lr']*scale
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter(); losses=[]
            indices=torch.multinomial(train.weights,cfg.accumulation*cfg.batch_size,replacement=True)
            dataloading_s=preprocessing_s=0.
            for offset in range(0,len(indices),cfg.batch_size):
                loaded=time.perf_counter();batch=collate([train[int(idx)] for idx in indices[offset:offset+cfg.batch_size]])
                dataloading_s+=time.perf_counter()-loaded
                prepared=time.perf_counter();batch=device_batch(batch,'cuda:0')
                preprocessing_s+=time.perf_counter()-prepared
                if step==start and offset==0:
                    print(f"[WORKFLOW] MiniVLA first batch · state={tuple(batch['state'].shape)} · actions={tuple(batch['actions'].shape)} · mask={tuple(batch['mask'].shape)} · images={len(batch['images'])} · instructions={len(batch['instruction'])}",flush=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=compute_loss(backbone,head,batch,augment=True)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/cfg.accumulation).backward(); losses.append(float(loss.detach()))
            if lora and step==start+1 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in lora):
                raise RuntimeError('Action loss does not reach LoRA')
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step(); completed_step=step+1; torch.cuda.synchronize()
            log={'step':step+1,'loss':float(np.mean(losses)),'grad_norm':float(norm),
                 'seconds':time.perf_counter()-begin,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                 'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
            if (step+1)%cfg.val_every==0 or step+1==cfg.steps:
                print(f"[WORKFLOW] Validation · step {step+1:,} · {validation_protocol['episodes']} episodes / {validation_protocol['samples']} samples",flush=True)
                score=validate(backbone,head,val,'cuda:0')
                if not math.isfinite(score):raise FloatingPointError('Nonfinite validation loss')
                log['validation_action_loss']=score
                improved=score<best
                if improved: best=score; stale=0
                else: stale+=1
                for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                    save_checkpoint(out/name,cfg,head,backbone,stats,opt,step+1,best,stale,train.manifest['control_hz'],identity)
            if (step+1)%cfg.save_freq==0 or step+1==cfg.steps or (cfg.early_stopping and stale>=cfg.patience):
                save_checkpoint(out/'checkpoints'/f'step_{step+1:06d}.pt',cfg,head,backbone,stats,opt,step+1,best,stale,train.manifest['control_hz'],identity)
                save_checkpoint(out/'last.pt',cfg,head,backbone,stats,opt,step+1,best,stale,train.manifest['control_hz'],identity)
            log.update(training_loss=log['loss'],learning_rate=[g['lr'] for g in opt.param_groups],step_duration=time.perf_counter()-begin,gpu_memory={'allocated_gib':log['peak_allocated_gib'],'reserved_gib':log['peak_reserved_gib']})
            log.update(target_steps=cfg.steps,samples=(step+1)*cfg.accumulation*cfg.batch_size,
                batch_size=int(batch['state'].shape[0]),accumulation=cfg.accumulation,effective_batch_size=len(indices),forward_backward_batches=len(losses),
                dataloading_s=dataloading_s,preprocessing_s=preprocessing_s,
                update_s=max(0.,log['seconds']-dataloading_s-preprocessing_s))
            if math.isfinite(best):log['best_validation_loss']=best
            if 'validation_action_loss' in log:log['validation_loss']=log['validation_action_loss']
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
            from ...workflow.presentation import training_status_text
            print(training_status_text(log),flush=True)
            if stop_requested: raise KeyboardInterrupt
            if cfg.early_stopping and stale>=cfg.patience: break
    except KeyboardInterrupt:
        stopped=True
        # SIGINT is deferred until the complete optimizer/metrics step finishes.
        save_checkpoint(out/'last.pt',cfg,head,backbone,stats,opt,completed_step,best,stale,train.manifest['control_hz'],identity)
        save_checkpoint(out/'checkpoints'/f'step_{completed_step:06d}.pt',cfg,head,backbone,stats,opt,completed_step,best,stale,train.manifest['control_hz'],identity)
        print(json.dumps({'stopped':True,'step':completed_step}),flush=True)
    finally:
        signal.signal(signal.SIGINT,previous_sigint)
    (out/'training_status.json').write_text(json.dumps({'stopped':stopped,'step':completed_step,'target_steps':cfg.steps,'early_stopped':not stopped and completed_step<cfg.steps}))

if __name__=='__main__': main()
