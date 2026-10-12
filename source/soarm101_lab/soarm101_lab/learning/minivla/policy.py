import time
import numpy as np
import torch
from .config import MiniVLAConfig
from .data import RobotPreprocessor
from .head import ActionHead
from .train import restore_adapter

class MiniVLA:
    """Public RGB + instruction + canonical state -> absolute action chunk API."""
    def __init__(self, checkpoint):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.cfg=MiniVLAConfig(**ck['config']); self.control_hz=ck['control_hz']
        from .backbone import QwenBackbone
        self.backbone=QwenBackbone(self.cfg); restore_adapter(self.backbone,ck['adapter'])
        self.head=ActionHead(self.cfg).to('cuda:0'); self.head.load_state_dict(ck['head'])
        self.pre=RobotPreprocessor(ck['statistics'])
        self.backbone.eval(); self.head.eval()
    @torch.no_grad()
    def predict(self, images, state, instruction, noise=None):
        q=np.asarray(state,np.float32)
        if q.shape!=(6,) or not np.isfinite(q).all(): raise ValueError('Expected finite canonical 6D state')
        s=torch.from_numpy(self.pre.normalize(q,'state')).unsqueeze(0).cuda()
        torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=self.backbone([images],[instruction])
            chunk=self.head.sample(f,s,noise=noise)
            ctx,mask=self.head.context(f); grounding=self.head.grounding(ctx,mask)
        torch.cuda.synchronize()
        action=self.pre.denormalize(chunk[0].float().cpu().numpy())
        if not np.isfinite(action).all(): raise FloatingPointError('Policy produced nonfinite action')
        return action,{'latency_s':time.perf_counter()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'grounding_uv':grounding[0].float().cpu().tolist()}

class ActionGuard:
    def __init__(self, lower, upper, max_velocity, hz):
        self.low=np.asarray(lower,float); self.high=np.asarray(upper,float)
        self.speed=np.asarray(max_velocity,float); self.hz=float(hz)
        if any(x.shape!=(6,) or not np.isfinite(x).all() for x in (self.low,self.high,self.speed)):
            raise ValueError('Guard limits must be finite 6D')
        if np.any(self.low>=self.high) or np.any(self.speed<=0) or self.hz<=0: raise ValueError('Invalid guard bounds')
        self.reset()
    def reset(self): self.last=None; self.violations=0; self.rate_limited=0
    def apply(self, requested, observed):
        a=np.asarray(requested,float); q=np.asarray(observed,float)
        if a.shape!=(6,) or q.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(q).all():
            raise ValueError('Nonfinite or malformed command/state')
        if np.any(q<self.low-.05) or np.any(q>self.high+.05): raise ValueError('Observed state outside configured bounds')
        self.violations+=int(np.any((a<self.low)|(a>self.high)))
        target=np.clip(a,self.low,self.high); base=q if self.last is None else self.last
        delta=self.speed/self.hz
        clipped=np.clip(target,base-delta,base+delta)
        # Also bound tracking error relative to actual state to avoid accumulating a distant command.
        clipped=np.clip(clipped,q-delta,q+delta)
        self.rate_limited+=int(np.any(np.abs(clipped-target)>1e-8))
        self.last=np.clip(clipped,self.low,self.high)
        return self.last.copy()
