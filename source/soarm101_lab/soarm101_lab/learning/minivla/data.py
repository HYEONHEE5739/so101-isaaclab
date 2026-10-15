"""Canonical, explicitly aligned per-episode NPY data. Never infer motor units."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from ...so101_joint_mapping import JOINTS as MODEL_JOINTS, POLICY_UNITS

JOINTS = list(MODEL_JOINTS)
UNITS = list(POLICY_UNITS)

def load_manifest(path):
    path = Path(path)
    m = json.loads(path.read_text())
    if m.get('schema_version') != 1: raise ValueError('schema_version must be 1')
    if m.get('joint_names') != JOINTS or m.get('units') != UNITS:
        raise ValueError('Explicit joint order/units required: five radians + gripper fraction')
    if m.get('action_semantics') != 'absolute_target_after_observation':
        raise ValueError('Action must be the command applied AFTER observation, not next observed qpos')
    if m.get('control_hz', 0) <= 0: raise ValueError('Invalid control_hz')
    ids, groups = set(), {}
    for e in m['episodes']:
        if e['id'] in ids: raise ValueError('Duplicate episode id')
        ids.add(e['id'])
        if e['split'] not in ('train', 'val', 'test', 'attribute_ood', 'composition_ood', 'object_ood', 'spatial_ood'):
            raise ValueError('Unknown split')
        # Every variant of a source demo / counterfactual scene stays together.
        group = e['group_id']
        if group in groups and groups[group] != e['split']: raise ValueError('Group leakage between splits')
        groups[group] = e['split']
        if not e['instruction'].strip(): raise ValueError('Empty instruction')
        e['_path'] = str(path.parent / e['path'])
    return m

def episode_arrays(e, cameras, hz):
    p = Path(e['_path'])
    arrays = {k: np.load(p / (k+'.npy'), mmap_mode='r', allow_pickle=False)
              for k in ('state', 'action', 'timestamp', 'action_timestamp')}
    t = len(arrays['state'])
    for k in ('state', 'action'):
        a = arrays[k]
        if a.shape != (t, 6) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'{p}/{k}: finite float32 [T,6] required')
        if np.any((a[:, -1] < 0) | (a[:, -1] > 1)): raise ValueError('Gripper must be calibrated fraction [0,1]')
    if t == 0: raise ValueError('Empty episode')
    ts, ats = arrays['timestamp'], arrays['action_timestamp']
    if ts.shape != (t,) or ats.shape != (t,) or not np.isfinite(ts).all() or not np.isfinite(ats).all():
        raise ValueError('Bad timestamps')
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(ats) <= 0): raise ValueError('Nonmonotonic timestamps')
    if t > 1 and not np.allclose(np.diff(ts), 1/hz, atol=.2/hz, rtol=0):
        raise ValueError('Resample observations to the declared control frequency first')
    if np.any(ats < ts - 1e-6) or np.any(ats - ts >= 1/hz):
        raise ValueError('Action must follow its observation within one control period')
    for camera in cameras:
        a = np.load(p / f'{camera}.npy', mmap_mode='r', allow_pickle=False)
        if a.ndim != 4 or a.shape[0] != t or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError('Camera: uint8 [T,H,W,3] required')
        cts = np.load(p / f'{camera}_timestamp.npy', allow_pickle=False)
        if cts.shape != (t,) or not np.isfinite(cts).all() or np.any(np.diff(cts) < 0):
            raise ValueError('Invalid camera timestamps')
        if np.any(cts > ts+1e-6) or np.any(ts-cts > 2/hz): raise ValueError('Future/stale camera frame')
        arrays[camera] = a
    for key in ('grounding', 'grounding_mask'):
        if (p / f'{key}.npy').exists(): arrays[key] = np.load(p / f'{key}.npy', allow_pickle=False)
    if 'grounding' in arrays:
        a = arrays['grounding']; mask = arrays.get('grounding_mask')
        if a.shape != (t, 2, 2) or mask is None or mask.shape != (t, 2): raise ValueError('Bad grounding labels')
        if mask.dtype != np.bool_ or not np.isfinite(a).all() or np.any((a[mask] < 0) | (a[mask] > 1)):
            raise ValueError('Grounding must be finite normalized UV, with bool mask')
    return arrays

def training_fingerprint(m):
    payload = {'joints': m['joint_names'], 'units': m['units'], 'hz': m['control_hz'],
               'episodes': [{k:v for k,v in e.items() if k != '_path'} for e in m['episodes'] if e['split']=='train']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def fit_statistics(manifest, output, cameras=('side',)):
    m = load_manifest(manifest)
    moments = {k: [0, np.zeros(6), np.zeros(6)] for k in ('state', 'action')}
    train = [e for e in m['episodes'] if e['split']=='train']
    if not train: raise ValueError('No training episodes')
    for e in train:
        arrays = episode_arrays(e, cameras, m['control_hz'])
        for key, (n, mean, m2) in moments.items():
            a = arrays[key].astype(np.float64); count=len(a); delta=a.mean(0)-mean
            total=n+count
            moments[key] = [total, mean+delta*count/total,
                            m2+((a-a.mean(0))**2).sum(0)+delta**2*n*count/total]
    stats = {'fingerprint': training_fingerprint(m), 'joint_names': JOINTS, 'units': UNITS}
    for key, (n, mean, m2) in moments.items():
        stats[key+'_mean'] = mean.tolist()
        stats[key+'_std'] = np.maximum(np.sqrt(m2/n), .01).tolist()
    Path(output).write_text(json.dumps(stats, indent=2))
    return stats

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

class Episodes(Dataset):
    def __init__(self, manifest, stats, cfg, split):
        self.manifest = load_manifest(manifest)
        self.cfg = cfg
        if stats['fingerprint'] != training_fingerprint(self.manifest): raise ValueError('Stats/manifest train mismatch')
        self.pre = RobotPreprocessor(stats)
        self.episodes = [e for e in self.manifest['episodes'] if e['split']==split]
        if not self.episodes: raise ValueError(f'No episodes in split {split}')
        self.arrays = [episode_arrays(e, cfg.cameras, self.manifest['control_hz']) for e in self.episodes]
        self.index = [(i,t) for i,a in enumerate(self.arrays) for t in range(len(a['state']))]
        self.weights = torch.tensor([1/len(self.arrays[i]['state']) for i,t in self.index], dtype=torch.double)
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        i,t = self.index[idx]; a=self.arrays[i]; h=self.cfg.horizon
        count=min(h,len(a['action'])-t)
        actions=np.zeros((h,6), np.float32)
        actions[:count]=self.pre.normalize(a['action'][t:t+count], 'action')
        return {'images': [np.array(a[c][t]) for c in self.cfg.cameras],
                'instruction': self.episodes[i]['instruction'],
                'state': torch.from_numpy(self.pre.normalize(a['state'][t], 'state')),
                'actions': torch.from_numpy(actions), 'mask': torch.arange(h)<count,
                'episode': self.episodes[i]['id'], 'frame':t}

def collate(samples):
    out={k:torch.stack([s[k] for s in samples]) for k in ('state','actions','mask')}
    out.update({k:[s[k] for s in samples] for k in ('images','instruction','episode','frame')})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('output'); p.add_argument('--cameras', nargs='+',default=['side'])
    a=p.parse_args(); fit_statistics(a.manifest,a.output,a.cameras)
