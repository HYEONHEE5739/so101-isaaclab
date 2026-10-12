"""Official LeRobot 0.4.1 ACT architecture, same data/splits/action normalization.
ImageNet-pretrained ResNet18, CVAE, no language input. Separate environment recommended.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.nn import functional as F
from .config import MiniVLAConfig
from .data import Episodes, collate, RobotPreprocessor
from .train import seed_all

def build(cfg, pretrained=True):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature
    inputs={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,))}
    inputs.update({f'observation.images.{c}':PolicyFeature(type=FeatureType.VISUAL,
        shape=(3,cfg.image_height,cfg.image_width)) for c in cfg.cameras})
    ac=ACTConfig(input_features=inputs,output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))},
        chunk_size=cfg.horizon,n_action_steps=cfg.replan_steps,
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)
    return ACTPolicy(ac)

def prepare(batch,cfg,device,augment=False):
    result={'observation.state':batch['state'].to(device)}
    mean=torch.tensor([.485,.456,.406],device=device)[:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[:,None,None]
    for i,c in enumerate(cfg.cameras):
        images=[]
        for views in batch['images']:
            im=Image.fromarray(views[i])
            if augment:
                im=ImageEnhance.Brightness(im).enhance(float(np.random.uniform(.9,1.1)))
                im=ImageEnhance.Contrast(im).enhance(float(np.random.uniform(.9,1.1)))
            im=ImageOps.pad(im,(cfg.image_width,cfg.image_height),method=Image.Resampling.LANCZOS)
            x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).to(device).float()/255
            images.append((x-mean)/std)
        result[f'observation.images.{c}']=torch.stack(images)
    if 'actions' in batch:
        result['action']=batch['actions'].to(device); result['action_is_pad']=~batch['mask'].to(device)
    return result

def objective(policy,batch,cfg):
    # Call official architecture. Normalize loss over valid elements, so episode padding
    # does not dilute reconstruction supervision; record this deviation from stock ACT loss.
    b=dict(batch); b['observation.images']=[b[f'observation.images.{c}'] for c in cfg.cameras]
    pred,(mu,logvar)=policy.model(b)
    mask=~b['action_is_pad']; weights=torch.ones(6,device=pred.device); weights[-1]=cfg.gripper_weight
    w=mask[...,None]*weights
    l1=(F.l1_loss(pred,b['action'],reduction='none')*w).sum()/w.sum()
    kl=torch.zeros((),device=pred.device) if mu is None else (-.5*(1+logvar-mu.square()-logvar.exp())).sum(-1).mean()
    return l1+10*kl,float(l1.detach())

class ACTInference:
    def __init__(self,checkpoint):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.cfg=MiniVLAConfig(**ck['config']); self.control_hz=ck['control_hz']
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model=build(self.cfg,pretrained=False).to(self.device); self.model.load_state_dict(ck['act']); self.model.eval()
        self.pre=RobotPreprocessor(ck['statistics'])
    @torch.no_grad()
    def predict(self,images,state,instruction,noise=None):
        batch={'images':[images],'state':torch.from_numpy(self.pre.normalize(state,'state')).unsqueeze(0)}
        if self.device!='cpu': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter(); b=prepare(batch,self.cfg,self.device)
        chunk=self.model.predict_action_chunk(b)[0].cpu().numpy()
        if self.device!='cpu': torch.cuda.synchronize()
        return self.pre.denormalize(chunk),{'latency_s':time.perf_counter()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if self.device!='cpu' else 0}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--stats',required=True); p.add_argument('--out',required=True); p.add_argument('--resume')
    a=p.parse_args(); cfg=MiniVLAConfig.load(a.config); seed_all(cfg.seed)
    stats=json.loads(Path(a.stats).read_text()); train=Episodes(a.manifest,stats,cfg,'train'); val=Episodes(a.manifest,stats,cfg,'val')
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=build(cfg,pretrained=not bool(a.resume)).to(device)
    opt=torch.optim.AdamW(model.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume: raise FileExistsError('Existing run: --resume or new output')
    start=0; best=float('inf'); stale=0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['config']!=asdict(cfg) or ck['statistics']!=stats: raise ValueError('Resume config/stats mismatch')
        model.load_state_dict(ck['act']); opt.load_state_dict(ck['optimizer']); start=ck['step']; best=ck['best']; stale=ck['stale']
    for step in range(start,cfg.steps):
        seed_all(cfg.seed+step); model.train(); opt.zero_grad(set_to_none=True); total=0
        for idx in torch.multinomial(train.weights,cfg.accumulation,replacement=True):
            batch=prepare(collate([train[int(idx)]]),cfg,device,True)
            loss,_=objective(model,batch,cfg)
            (loss/cfg.accumulation).backward(); total+=float(loss.detach())/cfg.accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        log={'step':step+1,'loss':total}
        if (step+1)%cfg.val_every==0 or step+1==cfg.steps:
            model.eval(); values=[]; offset=0
            with torch.no_grad():
                for array in val.arrays:
                    vals=[]; n=len(array['state'])
                    for t in np.unique(np.linspace(0,n-1,min(n,8),dtype=int)):
                        _,l1=objective(model,prepare(collate([val[offset+int(t)]]),cfg,device),cfg); vals.append(l1)
                    values.append(np.mean(vals)); offset+=n
            score=float(np.mean(values)); improved=score<best
            if improved: best=score; stale=0
            else: stale+=1
            ck=dict(config=asdict(cfg),statistics=stats,act=model.state_dict(),optimizer=opt.state_dict(),
                step=step+1,best=best,stale=stale,control_hz=train.manifest['control_hz'])
            for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                tmp=out/(name+'.tmp'); torch.save(ck,tmp); tmp.replace(out/name)
            log['validation_l1']=score
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
        print(json.dumps(log),flush=True)
        if stale>=cfg.patience: break

if __name__=='__main__': main()
