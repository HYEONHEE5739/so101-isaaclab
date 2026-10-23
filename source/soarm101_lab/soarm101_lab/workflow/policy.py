"""Shared ACT/SmolVLA inference, preserving LeRobot saved normalization processors."""
import json
from pathlib import Path

import numpy as np
from ..representation import require_policy


def checkpoint_metadata(path) -> dict:
    """Small native configuration for a selected path, never load weights/state.

    UI callers must run this in their existing worker. This is display/setup
    metadata; resume_checkpoint/require_policy still own execution validation.
    """
    def read(p):
        with p.open('rb') as stream:raw=stream.read(1024*1024+1)
        if len(raw)>1024*1024:raise ValueError('Checkpoint configuration exceeds display limit: '+str(p))
        value=json.loads(raw)
        if not isinstance(value,dict):raise ValueError('Checkpoint configuration must be an object')
        return value
    path=Path(path).expanduser().resolve()
    if path.suffix=='.pt':
        folder=minivla_root(path);config=folder/'config.json'
        saved=read(config) if config.is_file() else {}
        return dict(path=str(path),architecture='minivla',steps=saved.get('steps'),
                    batch_size=saved.get('batch_size',1),model_config=str(config) if config.is_file() else None)
    model=path/'pretrained_model' if (path/'pretrained_model').is_dir() else path
    config=read(model/'config.json')
    train=read(model/'train_config.json') if (model/'train_config.json').is_file() else {}
    architecture=config.get('type') or train.get('policy',{}).get('type')
    from ..learning.datasets import installed_policies
    if architecture not in installed_policies():raise ValueError('Policy type absent from installed LeRobot')
    return dict(path=str(model),architecture=architecture,steps=train.get('steps'))


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


def minivla_root(path):
    path=Path(path).expanduser().resolve()
    root=path.parent if path.suffix=='.pt' else path
    return root.parent if root.name=='checkpoints' else root


def selected_minivla_checkpoint(path, selected=None):
    root=minivla_root(path)
    candidate=Path(selected).expanduser() if selected else (Path(path) if Path(path).suffix=='.pt' else root/('best.pt' if (root/'best.pt').is_file() else 'last.pt'))
    if not candidate.is_absolute():candidate=root/candidate if selected else candidate
    candidate=candidate.resolve()
    if not candidate.is_relative_to(root) or candidate.suffix!='.pt' or not candidate.is_file():
        raise ValueError('선택한 MiniVLA checkpoint가 해당 학습 폴더에 없습니다.')
    return candidate


def load_policy(path, expected, device):
    if Path(path).suffix=='.pt' or (Path(path) / 'best.pt').is_file() or (Path(path)/'last.pt').is_file():
        require_policy(minivla_root(path), expected)
        return MiniVLAJointPolicy(path, device), lambda batch: batch, lambda action: action
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    require_policy(path, expected)
    cfg = PreTrainedConfig.from_pretrained(path)
    if cfg.type not in ('act','smolvla','diffusion'):
        raise ValueError('SO-101 evaluation currently supports ACT/SmolVLA/Diffusion; other native policies remain available for Training')
    from ..learning.datasets import validate_sim_features
    validate_sim_features(cfg,path)
    cfg.device=device
    policy = get_policy_class(cfg.type).from_pretrained(path,config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={'device_processor': {'device': device}})
    return policy, pre, post


def observation(q, side, wrist, task, device, *, images=None):
    import torch
    out = {'observation.state': torch.as_tensor(q, dtype=torch.float32, device=device), 'task': task}
    images = images if images is not None else {'side':side,'wrist':wrist}
    for name, image in images.items():
        if image is None: continue
        key = name + '_cam'
        value = torch.as_tensor(image, device=device)[..., :3].to(torch.float32) / 255.
        out['observation.images.' + key] = value.permute(2, 0, 1)
    return out


def batch_actions(policy, pre, post, states, images, task, device):
    """Keep the official policy's queue batched, with one column per environment."""
    import torch
    rows=[observation(q,None,None,task,device,images=rgb) for q,rgb in zip(states,images)]
    batch={key:torch.stack([r[key] for r in rows]) for key in rows[0] if key!='task'}
    batch['task']=[task]*len(rows)
    with torch.inference_mode():
        actions=post(policy.select_action(pre(batch)))
    actions=torch.as_tensor(actions).detach().cpu().numpy()
    import numpy as np
    if actions.shape!=(len(rows),6) or not np.isfinite(actions).all():
        raise ValueError('Policy must output finite [num_envs,6] model-radian actions')
    return actions



def inspect_checkpoint(path):
    """Read native Checkpoint contracts without writing or borrowing a registry.

    Discovery is separate: missing coordinates/provenance is an actionable
    compatibility failure here, never repaired from another checkpoint or run.
    """
    from ..so101_dataset_contract import read_contract, JOINT_SPACE
    from ..representation import describe
    path=Path(path).expanduser().resolve()
    mini=path.suffix=='.pt' or (path/'best.pt').is_file() or (path/'last.pt').is_file()
    if mini:
        root=minivla_root(path)
        selected=selected_minivla_checkpoint(path)
        provenance_path=root/'project_provenance.json'
        config_path=root/'config.json'
    else:
        root=path/'pretrained_model' if (path/'pretrained_model').is_dir() else path
        selected=None;provenance_path=root/'workflow_provenance.json';config_path=root/'config.json'
        for name in ('model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
            if not (root/name).is_file():raise ValueError('Incomplete policy Checkpoint: '+name)
    if not provenance_path.is_file():
        raise ValueError('Checkpoint provenance missing: '+str(provenance_path)+' · 발견 가능하지만 평가 계약은 확인할 수 없습니다.')
    try:coordinates=read_contract(root)
    except (OSError,ValueError) as exc:raise ValueError('Checkpoint Coordinate contract missing/invalid: '+str(root)) from exc
    if not coordinates or coordinates.get('action_space')!=JOINT_SPACE:
        raise ValueError('Policy requires an explicit joint-radian Coordinate contract: '+str(root))
    provenance=json.loads(provenance_path.read_text());config=json.loads(config_path.read_text())
    result=dict(path=str(root),type='policy',format='so101.minivla-policy/1' if mini else config,
                coordinates=coordinates,representation=describe(coordinates),provenance=provenance,
                workspace=provenance.get('workspace'),tasks=provenance.get('task_definitions',[]),
                backend=provenance.get('backend'))
    if mini:result.update(model_config=config,checkpoint_file=str(selected))
    else:
        result['processors']={name:json.loads((root/(name+'.json')).read_text()) for name in ('policy_preprocessor','policy_postprocessor')}
    return result


class MiniVLAJointPolicy:
    """Expose MiniVLA's canonical actions through the workflow's model-radian API."""
    def __init__(self, path, device):
        import numpy as np
        from ..learning.minivla.policy import MiniVLA
        from ..so101_joint_mapping import RobotCoordinates
        self.policy = MiniVLA(selected_minivla_checkpoint(path))
        self.device = device
        provenance = json.loads((minivla_root(path) / 'project_provenance.json').read_text())
        coords = provenance['coordinates']
        self.coordinates = RobotCoordinates(coords['gripper_closed_rad'], coords['gripper_open_rad'],
            measured_gripper_limits=coords.get('measured_gripper_limits_rad'))
        limits = np.asarray(provenance.get('joint_limits_rad') or provenance.get('scene', {}).get('joint_limits_rad'), dtype=float)
        if limits.shape != (6, 2): raise ValueError('MiniVLA checkpoint is missing model joint limits')
        self.guard = ActionGuard(np.r_[limits[:5, 0], 0.], np.r_[limits[:5, 1], 1.], [2.] * 5 + [1.], self.policy.control_hz)
        self.reset()

    def reset(self):
        self.remaining = []
        self.guard.reset()
        self._batch_slots = []

    def select_action(self, batch):
        import numpy as np
        import torch
        if batch['observation.state'].ndim==2:
            # MiniVLA's existing predict() is single-observation. Share its model,
            # while each slot owns its action chunk and guard (no weight copies).
            import copy
            count=len(batch['observation.state'])
            if len(self._batch_slots)!=count:
                self._batch_slots=[]
                for _ in range(count):
                    slot=copy.copy(self);slot.guard=copy.deepcopy(self.guard);slot.reset()
                    self._batch_slots.append(slot)
            actions=[];metrics=[]
            for i,slot in enumerate(self._batch_slots):
                row={k:(v[i] if isinstance(v,torch.Tensor) or isinstance(v,list) else v) for k,v in batch.items()}
                actions.append(slot.select_action(row));metrics.append(slot.last_inference_metrics)
            self.last_inference_metrics={'per_env':metrics}
            return torch.stack(actions)
        q = batch['observation.state'].detach().cpu().numpy().reshape(6)
        self.last_inference_metrics = None
        if not self.remaining:
            images = []
            for camera in self.policy.cfg.cameras:
                value = batch['observation.images.' + camera + '_cam'].detach().cpu().numpy()
                images.append(np.clip(value.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            chunk, self.last_inference_metrics = self.policy.predict(images, self.coordinates.to_policy(q, measured=True), batch['task'])
            chunk = np.asarray(chunk)
            if chunk.shape != (self.policy.cfg.horizon, 6) or not np.isfinite(chunk).all():
                raise ValueError('MiniVLA must output a finite [H,6] action chunk')
            self.remaining = list(chunk[:self.policy.cfg.replan_steps])
        action = self.guard.apply(self.remaining.pop(0), self.coordinates.to_policy(q, measured=True))
        return torch.as_tensor(self.coordinates.to_model(action), device=self.device, dtype=torch.float32)
