"""Historical Canonical/LeRobot manifest readers; no training role policy."""
import json
from pathlib import Path
import numpy as np
from ..so101_joint_mapping import JOINTS as MODEL_JOINTS, POLICY_UNITS
JOINTS=list(MODEL_JOINTS)
UNITS=list(POLICY_UNITS)

def lerobot_storages(manifest):
    storage=manifest.get('storage',{})
    return storage.get('sources',[storage]) if storage else []

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
        if not e['instruction'].strip(): raise ValueError('Empty instruction')
        if m.get('storage',{}).get('format') == 'lerobot':
            e['_storage']=lerobot_storages(m)[e.get('source_index',0)]
        else:e['_path'] = str(path.parent / e['path'])
    if m.get('storage'):
        if m['storage'].get('format')!='lerobot':raise ValueError('Unknown episode storage format')
        from ..utils.artifact_identity import fingerprint
        for storage in lerobot_storages(m):
            if fingerprint(storage['root'])!=storage['sha256']:raise ValueError('LeRobot dataset bytes changed')
    return m

def lerobot_reader(storage):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ..representation import describe
    from ..so101_dataset_contract import read_contract,JOINT_SPACE
    contract=read_contract(storage['root']);describe(contract)
    if contract!=storage['coordinate_contract'] or contract['action_space']!=JOINT_SPACE:
        raise ValueError('LeRobot requires the recorded model-radian coordinate contract')
    if storage.get('action_source')!='joint_targets':raise ValueError('Direct loader requires absolute joint_targets, not next observed qpos')
    reader=LeRobotDataset(storage['repo_id'],root=storage['root'],download_videos=False,video_backend='torchcodec')
    for key in ('observation.state','action'):
        feature=reader.meta.info['features'][key]
        if tuple(feature['shape'])!=(6,) or (feature.get('names') is not None and feature['names']!=JOINTS):
            raise ValueError('LeRobot state/action joint names or dimension conflict: '+key)
    return reader


def episode_arrays(e, cameras, hz, reader=None):
    if '_storage' in e:
        from ..so101_joint_mapping import RobotCoordinates
        reader=reader or lerobot_reader(e['_storage'])
        start,end=e['from_index'],e['to_index']
        keys=['observation.state','action','timestamp','episode_index','frame_index','task_index']
        if 'action_timestamp' in reader.hf_dataset.column_names:keys.append('action_timestamp')
        rows=reader.hf_dataset.select_columns(keys)[start:end]
        values={k:np.asarray(v) for k,v in rows.items()}
        ts=values['timestamp'].reshape(-1).astype(np.float64)
        if (len(ts)!=e['frames'] or np.any(values['episode_index']!=e['dataset_episode_index'])
                or not np.array_equal(values['frame_index'].reshape(-1),np.arange(len(ts)))):
            raise ValueError('LeRobot episode/frame index mismatch')
        instructions={str(reader.meta.tasks.iloc[int(i)].name) for i in values['task_index'].reshape(-1)}
        if instructions!={e['instruction']}:raise ValueError('LeRobot Task instruction differs from recorded episode')
        endpoints=e['_storage']['canonical_coordinates']
        coords=RobotCoordinates(endpoints['gripper_closed_rad'],endpoints['gripper_open_rad'],
            measured_gripper_limits=endpoints.get('measured_gripper_limits_rad'))
        converted={}
        for key,measured in (('observation.state',True),('action',False)):
            try:converted[key]=coords.to_policy(values[key],measured=measured)
            except ValueError as exc:raise ValueError(f"{e['id']} ({e['_storage']['root']}) {key}: {exc}") from exc
        arrays={'state':converted['observation.state'],
                'action':converted['action'],'timestamp':ts,
                'action_timestamp':values.get('action_timestamp',ts).reshape(-1).astype(np.float64)}
        t=len(ts)
    else:
        p = Path(e['_path'])
        arrays = {k: np.load(p / (k+'.npy'), mmap_mode='r', allow_pickle=False)
                  for k in ('state', 'action', 'timestamp', 'action_timestamp')}
        t = len(arrays['state'])
    for k in ('state', 'action'):
        a = arrays[k]
        if a.shape != (t, 6) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'{e["id"]}/{k}: finite float32 [T,6] required')
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
    if '_storage' in e:return arrays # Numeric Parquet only: statistics never decode RGB.
    for camera in cameras:
        from ..utils.async_video_writer import VideoFrames
        a = VideoFrames(p / f'{camera}.mp4',t,hz) if (p / f'{camera}.mp4').exists() else np.load(p / f'{camera}.npy', mmap_mode='r', allow_pickle=False)
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

