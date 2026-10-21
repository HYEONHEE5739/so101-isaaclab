import time
import numpy as np
import torch
from .config import MiniVLAConfig
from .head import ActionHead
from .checkpoint import restore_adapter, require_architecture

class RobotPreprocessor:
    def __init__(self, stats):
        self.stats = stats
        for key in ('state', 'action'):
            mean=np.asarray(stats[key+'_mean']); std=np.asarray(stats[key+'_std'])
            if mean.shape!=(6,) or std.shape!=(6,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
                raise ValueError('Invalid statistics')
    def normalize(self, x, key):
        return (np.asarray(x, np.float32)-np.asarray(self.stats[key+'_mean'], np.float32))/np.asarray(self.stats[key+'_std'], np.float32)
    def denormalize(self, x, key='action'):
        return np.asarray(x)*np.asarray(self.stats[key+'_std'])+np.asarray(self.stats[key+'_mean'])

class MiniVLA:
    """Public RGB + instruction + canonical state -> absolute action chunk API."""
    def __init__(self, checkpoint):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require_architecture(ck)
        self.checkpoint_path=str(checkpoint);self.checkpoint_step=ck['step']
        self.cfg=MiniVLAConfig.from_dict(ck['config']); self.control_hz=ck['control_hz']
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
        torch.cuda.synchronize()
        action=self.pre.denormalize(chunk[0].float().cpu().numpy())
        if not np.isfinite(action).all(): raise FloatingPointError('Policy produced nonfinite action')
        return action,{'latency_s':time.perf_counter()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
