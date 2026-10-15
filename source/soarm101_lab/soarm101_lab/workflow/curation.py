"""Read-only trajectory analysis and non-destructive episode selection."""
import json
from pathlib import Path
import numpy as np
from .tasks import episode_definition


def analyze(path):
    if Path(path).is_dir():
        return analyze_canonical(path)
    import h5py
    from ..so101_dataset_contract import require_joint_replay
    require_joint_replay(path)
    rows = []
    with h5py.File(path, 'r') as f:
        args = json.loads(f['data'].attrs.get('env_args', '{}'))
        sim = args.get('sim_args', {})
        dt = float(sim.get('dt', 0)) * int(sim.get('decimation', 1))
        if dt <= 0:
            raise ValueError('Dataset에 dt/decimation이 없어 시간 지표를 계산할 수 없습니다.')
        workspace_path = args.get('workspace_path')
        if not workspace_path:
            raise ValueError('Workspace 경로가 없는 dataset입니다.')
        from ..real2sim.workspaces.package import load
        workspace = load(workspace_path)
        transform = np.linalg.inv(np.asarray(workspace['profile']['workspace']['T_world']))
        for name in sorted(f['data'], key=lambda x: int(x.split('_')[-1]) if x.startswith('demo_') else -1):
            if not name.startswith('demo_'): continue
            g = f['data'][name]; task = episode_definition(g, args)
            if task is None: raise ValueError(f'{name}: task 정보 없음')
            pick = task['pick_object_id']
            pose = np.asarray(g[f'initial_state/rigid_object/{pick}/root_pose']).reshape(-1, 7)[0, :3]
            xy = (transform @ np.r_[pose, 1])[:2]
            obs = g['obs/policy'] if 'obs/policy' in g else g['obs']
            if 'ee_state' not in obs or 'joint_targets' not in g:
                raise ValueError(f'{name}: EE state/joint_targets 없음')
            ee = np.asarray(obs['ee_state'])[:, :3]
            q = np.asarray(g['joint_targets'])
            if len(q) < 3 or len(ee) != len(q) or not np.isfinite(q).all() or not np.isfinite(ee).all():
                raise ValueError(f'{name}: 길이 또는 비정상 trajectory 값')
            speed = np.linalg.norm(np.diff(ee, axis=0), axis=1)/dt
            command_stop = float(np.mean(np.max(np.abs(np.diff(q, axis=0)), axis=1)/dt < np.deg2rad(1)))
            # Joint command acceleration; arm and gripper kept separate.
            acc = np.diff(q, n=2, axis=0)/dt**2
            rows.append(dict(name=name, task_id=task['task_id'], xy=xy.tolist(),
                duration=len(q)*dt, frames=len(q), command_stop_fraction=command_stop, stop_fraction=float(np.mean(speed < .002)),
                jitter=float(np.sqrt(np.mean(acc[:, :5]**2))),
                gripper_jitter=float(np.sqrt(np.mean(acc[:, 5]**2))),
                ee=ee.tolist(), actions=q.tolist(), dt=dt, flags=[],
                robot_q=np.asarray(obs['joint_pos']).tolist() if 'joint_pos' in obs else None,
                robot_urdf=str(workspace['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf')))
    return flag_outliers(rows)


def flag_outliers(rows):
    if not rows: raise ValueError('에피소드가 없습니다.')
    for task in {r['task_id'] for r in rows}:
        group = [r for r in rows if r['task_id'] == task]
        for key, low_label, high_label in [('duration','짧은 시간','긴 시간'),
                                          ('stop_fraction',None,'EE 정지 비율 높음'), ('command_stop_fraction',None,'정지 명령 많음'), ('jitter',None,'명령 jitter 높음')]:
            values = np.array([r[key] for r in group]); median = np.median(values)
            spread = max(1.4826*np.median(abs(values-median)), abs(median)*.1, 1e-6)
            for r in group:
                if len(group) >= 5 and r[key] > median+3*spread: r['flags'].append(high_label)
                if len(group) >= 5 and low_label and r[key] < median-3*spread: r['flags'].append(low_label)
    return rows


def analyze_canonical(path):
    """Use recorded workspace-frame poses; never launch a simulator for inspection."""
    from .artifacts import inspect_canonical
    from ..real2sim.workspaces.package import ROOT
    path = Path(path)
    details = inspect_canonical('source', path)
    manifest = json.loads((path / 'manifest.json').read_text())
    definition = details['provenance']['scene']['task_definition']
    dt = 1 / float(manifest['control_hz'])
    rows = []
    for entry in manifest['episodes']:
        task=entry.get('task_definition') or definition
        folder = path / entry['path']
        ee = np.load(folder / 'eef_pose.npy', allow_pickle=False)[:, :3, 3]
        obj = np.load(folder / 'object_pose.npy', mmap_mode='r', allow_pickle=False)
        q = np.load(folder / 'raw_action.npy', allow_pickle=False)
        if len(q) < 3 or len(q) != len(ee) or not np.isfinite(q).all() or not np.isfinite(ee).all():
            raise ValueError(f'{entry["id"]}: 길이 또는 비정상 trajectory 값')
        speed = np.linalg.norm(np.diff(ee, axis=0), axis=1) / dt
        acc = np.diff(q, n=2, axis=0) / dt**2
        rows.append(dict(name=entry['id'], task_id=task['task_id'], xy=obj[0, :2, 3].tolist(),
            duration=len(q)*dt, frames=len(q), command_stop_fraction=float(np.mean(np.max(np.abs(np.diff(q, axis=0)), axis=1)/dt < np.deg2rad(1))),
            stop_fraction=float(np.mean(speed < .002)), jitter=float(np.sqrt(np.mean(acc[:, :5]**2))),
            gripper_jitter=float(np.sqrt(np.mean(acc[:, 5]**2))), ee=ee.tolist(), actions=q.tolist(), dt=dt, flags=[],
            robot_q=np.load(folder / 'raw_state.npy', allow_pickle=False).tolist(),
            robot_urdf=str(ROOT / 'assets/SO101/urdf/so101_isaaclab.urdf')))
    return flag_outliers(rows)


def select_distribution(rows, count):
    """Preserve eligible task × 4x4 spatial histogram, deterministic exact count."""
    if not 1 <= count <= len(rows): raise ValueError('선별 개수는 사용 가능한 episode 수 이하여야 합니다.')
    xy = np.array([r['xy'] for r in rows]); lo=xy.min(0); span=np.maximum(xy.max(0)-lo,1e-9)
    bins=np.minimum(((xy-lo)/span*4).astype(int),3)
    groups={}
    for row, cell in zip(rows,bins): groups.setdefault((row['task_id'],*cell),[]).append(row)
    keys=sorted(groups); quotas={k:count*len(groups[k])/len(rows) for k in keys}
    amounts={k:int(quotas[k]) for k in keys}
    for k in sorted(keys,key=lambda k: (-(quotas[k]-amounts[k]),k))[:count-sum(amounts.values())]: amounts[k]+=1
    chosen=[]
    for k in keys:
        candidates=sorted(groups[k],key=lambda r:(len(r['flags']),r['jitter'],r['name']))
        chosen += [r['name'] for r in candidates[:amounts[k]]]
    return chosen


def export_subset(source, destination, names):
    import h5py, os, uuid
    from ..so101_dataset_contract import read_contract, write_contract
    from .artifacts import fingerprint
    source=Path(source); destination=Path(destination)
    if destination.exists(): raise ValueError('기존 파일은 덮어쓰지 않습니다.')
    if not names or len(set(names))!=len(names): raise ValueError('선택 목록이 비었거나 중복입니다.')
    if source.is_dir():
        import shutil
        from ..real2sim.storage import atomic
        manifest = json.loads((source / 'manifest.json').read_text())
        entries = {e['id']: e for e in manifest['episodes']}
        if not set(names).issubset(entries): raise ValueError('알 수 없는 episode입니다.')
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            folder = (source / entries[name]['path']).resolve()
            if not folder.is_relative_to(source.resolve()): raise ValueError('Episode outside dataset')
            shutil.copytree(folder, destination / entries[name]['path'])
        atomic(destination / 'manifest.json', {**manifest, 'episodes': [entries[n] for n in names]})
        provenance = json.loads((source / 'project_provenance.json').read_text())
        scenes=[entries[n].get('scene_configuration') or provenance.get('scene_configuration') for n in names]
        scenes=list({s['sha256']:s for s in scenes if s}.values())
        atomic(destination / 'project_provenance.json', {**provenance, 'scene_configurations':scenes, 'curation': {
            'source': str(source.resolve()), 'source_sha256': fingerprint(source), 'selected_episodes': names}})
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with h5py.File(source,'r') as src,h5py.File(temp,'w') as out:
            for k,v in src.attrs.items():out.attrs[k]=v
            data=out.create_group('data')
            for k,v in src['data'].attrs.items():data.attrs[k]=v
            total=0
            for i,name in enumerate(names):
                src.copy(src['data'][name],data,name=f'demo_{i}')
                data[f'demo_{i}'].attrs['curation_source_episode']=name
                total+=len(data[f'demo_{i}/actions'])
            data.attrs['total']=total
            data.attrs['curation']=json.dumps({'source':str(source.resolve()),'source_sha256':fingerprint(source),
                'selected_episodes':names,'selection':'user-reviewed subset; quality flags are diagnostics'})
            out.flush()
        # Exclusive publication: never replace an existing destination.
        os.link(temp,destination)
        write_contract(destination,read_contract(source))
    finally:temp.unlink(missing_ok=True)
    return destination
