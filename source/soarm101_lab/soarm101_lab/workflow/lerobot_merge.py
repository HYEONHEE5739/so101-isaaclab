"""Merge local converted LeRobot data while retaining SO-101 episode evidence.

LeRobot owns Parquet/video/statistics aggregation. This module owns compatibility,
original Source partitions and episode provenance; it never loads a simulator.
"""
import copy
import json
import shutil
import time
import uuid
from pathlib import Path

from ..real2sim.precision import fingerprint as evidence_hash
from ..real2sim.storage import atomic
from ..representation import require_compatible
from ..so101_dataset_contract import JOINT_SPACE, write_contract
from ..utils.artifact_identity import fingerprint
from .artifacts import inspect_artifact
from .episodes import partition_evidence, validate_partition, validate_partition_episode


def combine_partitions(partitions):
    """Union compatible Source inventories; keep each episode's original proof."""
    if any(p is None for p in partitions):
        if not all(p is None for p in partitions):
            raise ValueError('미분할 데이터와 Train/Val 데이터를 함께 병합할 수 없습니다. 같은 분할 상태끼리 선택하세요.')
        return None
    partitions = [validate_partition(p) for p in partitions]
    if len({p['split'] for p in partitions}) != 1:
        raise ValueError('Train과 Val은 함께 병합할 수 없습니다. Train끼리, Val끼리 선택하세요.')
    if all(p == partitions[0] for p in partitions):
        return copy.deepcopy(partitions[0])
    inventory, assignments = {}, {}
    for partition in partitions:
        for source, record in partition['inventory'].items():
            if source in inventory and (inventory[source] != record or assignments[source] != partition['assignments'][source]):
                raise ValueError('Source Partition 충돌: 같은 원본의 계보 또는 Train/Val 배정이 다릅니다.')
            inventory[source] = record
            assignments[source] = partition['assignments'][source]
    payload = dict(schema='so101.source-partition/1', split=partitions[0]['split'],
                   inventory=inventory, assignments=assignments,
                   source_ids=sorted({s for p in partitions for s in p['source_ids']}),
                   merged_partitions=sorted({p['sha256'] for p in partitions}))
    return validate_partition({**payload, 'sha256': evidence_hash(payload)})


def merge_coordinates(records):
    """Compare physical contracts separately from per-episode Task selection."""
    contracts, signatures, scenes = [], [], []
    for record in records:
        contract = copy.deepcopy({k: v for k, v in record['coordinates'].items()
                                  if k not in ('source_dataset', 'source_datasets')})
        signature = copy.deepcopy(contract)
        model = signature.get('simulation_model')
        if isinstance(model, dict) and model.get('schema') == 'so101.simulation-model/1':
            if model.get('sha256') != evidence_hash({k: v for k, v in model.items() if k != 'sha256'}):
                raise ValueError('기록된 시뮬레이션 모델 계약의 hash가 다릅니다.')
            model.pop('sha256')
            model.pop('scene_configuration', None)
        prov = record.get('provenance', {})
        candidates = [(contract.get('simulation_model') or {}).get('scene_configuration'),
                      prov.get('scene_configuration'), *prov.get('scene_configurations', []),
                      *prov.get('episode_scene_configurations', {}).values()]
        for scene in candidates:
            if scene is None:
                continue
            if scene.get('sha256') != evidence_hash({k: v for k, v in scene.items() if k != 'sha256'}):
                raise ValueError('기록된 Scene 설정의 hash가 다릅니다.')
            scenes.append({k: v for k, v in scene.items()
                           if k not in ('sha256', 'task_definition', 'pick_object_id', 'place_target_id')})
        contracts.append(contract)
        signatures.append(signature)
    if any(s != signatures[0] for s in signatures[1:]) or any(s != scenes[0] for s in scenes[1:]):
        raise ValueError('관절·단위·Gripper·로봇/충돌 모델·Scene 물리 설정이 다릅니다.')
    result = contracts[0]
    if any(c != result for c in contracts[1:]):
        # A multi-Task dataset has no single Task selection. Full original
        # contracts and per-episode scenes remain in workflow_provenance.json.
        model = result['simulation_model']
        model['scene_configuration'] = None
        model['sha256'] = evidence_hash({k: v for k, v in model.items() if k != 'sha256'})
    return result


def unpartitioned_sources(lineage):
    """Require recorded Source identities without inventing a Train/Val role."""
    if not isinstance(lineage, dict):
        raise ValueError('미분할 데이터의 원본 Source 계보가 없습니다.')
    if lineage.get('kind') == 'mimic_segments' and lineage.get('partition_sha256') is None:
        sources = [s.get('source') for s in lineage.get('segments', [])]
    else:
        sources = [lineage]
    if not sources:
        raise ValueError('미분할 데이터의 Mimic Source 계보가 비어 있습니다.')
    for source in sources:
        if not isinstance(source, dict) or source.get('kind') != 'source_episode' or not source.get('source_id'):
            raise ValueError('미분할 데이터의 원본 Source 계보를 확인할 수 없습니다.')
        identity = source.get('source') or {}
        if (not identity.get('dataset_sha256') or not identity.get('episode_id')
                or not isinstance(identity.get('groups'), list) or not identity['groups']):
            raise ValueError('미분할 데이터의 원본 Source 식별 정보가 불완전합니다.')
        if identity.get('payload_sha256') and source['source_id'] != 'source:' + identity['payload_sha256']:
            raise ValueError('원본 Source payload identity가 다릅니다.')
    return sources


def episode_origins(record):
    """Flatten prior merges so merging A with merge(A,B) cannot duplicate A."""
    prov = record['provenance']
    if 'merge_episode_origins' in prov:
        return prov['merge_episode_origins']
    episodes = {str(e['episode_index']): e for e in prov['episodes']}
    return {str(i): {'dataset_sha256': record.get('sha256') or str(Path(record['path']).resolve()),
                     'episode_index': i, 'source_dataset': prov.get('source_dataset'),
                     'source_episode': episodes[str(i)].get('source_episode')}
            for i in range(record['format']['total_episodes'])}


def validate_inputs(records):
    """Metadata-only validation, also used by the selection dialog. No Hub calls."""
    if len(records) < 2:
        raise ValueError('변환된 LeRobot 데이터셋을 2개 이상 선택하세요.')
    paths = [Path(r['path']).resolve() for r in records]
    if len(set(paths)) != len(paths) or len({r['id'] for r in records}) != len(records):
        raise ValueError('같은 데이터셋을 중복 선택할 수 없습니다.')
    base = records[0]
    seen, tasks, partitions, source_identities = set(), {}, [], {}
    for record in records:
        info, prov = record.get('format'), record.get('provenance') or {}
        if record.get('type') != 'dataset' or not isinstance(info, dict) or not str(info.get('codebase_version', '')).startswith('v3.'):
            raise ValueError('LeRobot v3로 변환된 데이터셋 폴더를 선택하세요 (meta/info.json).')
        if record.get('status', 'SUCCEEDED') != 'SUCCEEDED' or info.get('total_episodes', 0) < 1:
            raise ValueError('완료된 비어 있지 않은 데이터셋만 병합할 수 있습니다.')
        require_compatible(record['coordinates'], base['coordinates'], JOINT_SPACE)
        if prov.get('canonical_coordinates') != base.get('provenance', {}).get('canonical_coordinates'):
            raise ValueError('관절·단위·Gripper·로봇/충돌 모델 계약이 다릅니다.')
        for key in ('fps', 'robot_type', 'features'):
            if info.get(key) != base['format'].get(key):
                raise ValueError('LeRobot 데이터 계약이 다릅니다: ' + key)
        if not prov.get('workspace') or prov['workspace'] != base['provenance'].get('workspace') or record.get('backend') != base.get('backend'):
            raise ValueError('Workspace 또는 시뮬레이션 엔진이 다릅니다.')
        p = prov.get('source_partition')
        if p is not None:
            validate_partition(p)
        partitions.append(p)
        count = info['total_episodes']
        expected = {str(i) for i in range(count)}
        for key in ('source_groups', 'episode_source_lineage'):
            if set(prov.get(key, {})) != expected:
                raise ValueError('Episode 계보 누락: ' + key)
        if {str(e['episode_index']) for e in prov.get('episodes', [])} != expected:
            raise ValueError('Episode Task 계보가 완전하지 않습니다.')
        for task in prov.get('task_definitions', []):
            if task['task_id'] in tasks and tasks[task['task_id']] != task:
                raise ValueError('같은 Task ID의 정의가 다릅니다.')
            tasks[task['task_id']] = task
        origins = episode_origins(record)
        if set(origins) != expected:
            raise ValueError('병합 원본 episode 목록이 완전하지 않습니다.')
        for i in range(count):
            part, proof = partition_evidence(prov, i)
            combine_partitions([p, part])
            lineage = prov['episode_source_lineage'][str(i)]
            if p is not None:
                validate_partition_episode(part, lineage, proof)
                if part['split'] != p['split'] or not set(part['source_ids']).issubset(p['source_ids']):
                    raise ValueError('Episode의 Source Partition이 데이터셋과 다릅니다.')
            else:
                if proof is not None or not prov['source_groups'][str(i)] or str(prov['source_groups'][str(i)]).startswith('UNKNOWN'):
                    raise ValueError('미분할 데이터의 원본 Group 또는 분할 증거가 일치하지 않습니다.')
                for source in unpartitioned_sources(lineage):
                    key, identity = source['source_id'], source['source']
                    if key in source_identities and source_identities[key] != identity:
                        raise ValueError('같은 원본 Source의 식별 정보가 다릅니다.')
                    source_identities[key] = identity
            origin = origins[str(i)]
            keys = [('dataset', origin['dataset_sha256'], origin['episode_index'])]
            if origin.get('source_dataset') and origin.get('source_episode'):
                keys.append(('episode', origin['source_dataset'], origin['source_episode']))
            if any(key in seen for key in keys):
                raise ValueError('중복 episode가 포함되어 있습니다. 원본과 그 병합본을 함께 선택하지 마세요.')
            seen.update(keys)
    partition = combine_partitions(partitions)
    coordinates = merge_coordinates(records)
    if not tasks:
        raise ValueError('Task 정의가 없는 데이터셋은 병합할 수 없습니다.')
    return {'episodes': sum(r['format']['total_episodes'] for r in records),
            'frames': sum(r['format']['total_frames'] for r in records),
            'partition': partition, 'coordinates': coordinates, 'tasks': list(tasks.values())}


def merged_provenance(records, summary):
    base = records[0]['provenance']
    result = {k: copy.deepcopy(base.get(k)) for k in ('workspace', 'backend', 'canonical_coordinates', 'action_source', 'representation')}
    result.update(source_partition=summary['partition'], task_definitions=summary['tasks'], episodes=[],
                  source_env_args={'validation_only': any(r['provenance'].get('source_env_args', {}).get('validation_only', False) for r in records)},
                  merge_inputs=[dict(id=r['id'], path=r['path'], sha256=r['sha256'], coordinates=r['coordinates'],
                                     provenance=r['provenance']) for r in records])
    result['workspace_path'] = base.get('workspace_path') or base.get('source_env_args', {}).get('workspace_path')
    maps = ('source_groups', 'episode_source_lineage', 'episode_review', 'episode_scene_configurations')
    for key in (*maps, 'episode_source_partitions', 'episode_source_partition_inputs', 'merge_episode_origins'):
        result[key] = {}
    offset, scenes = 0, {}
    for record in records:
        prov = record['provenance']
        origins = episode_origins(record)
        episodes = {e['episode_index']: e for e in prov['episodes']}
        for i in range(record['format']['total_episodes']):
            key = str(offset + i)
            for name in maps:
                value = prov.get(name, {}).get(str(i))
                if name == 'episode_scene_configurations' and value is None:
                    value = prov.get('scene_configuration')
                if value is not None:
                    result[name][key] = copy.deepcopy(value)
            part, proof = partition_evidence(prov, i)
            result['episode_source_partitions'][key] = copy.deepcopy(part)
            result['episode_source_partition_inputs'][key] = copy.deepcopy(proof)
            result['merge_episode_origins'][key] = copy.deepcopy(origins[str(i)])
            result['episodes'].append({**copy.deepcopy(episodes[i]), 'episode_index': offset + i})
        for scene in prov.get('scene_configurations', []):
            scenes[scene['sha256']] = scene
        offset += record['format']['total_episodes']
    result['scene_configurations'] = list(scenes.values())
    if len(scenes) == 1:
        result['scene_configuration'] = next(iter(scenes.values()))
    return result


def merge(records, output, repo_id='local/so101_merged'):
    """Worker-only local aggregation. Publish only after contracts/counts pass."""
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    started = time.monotonic()
    output = Path(output).expanduser().resolve()
    verified = []
    for record in records:
        actual = inspect_artifact('dataset', record['path'])
        identity = fingerprint(record['path'])
        if identity != record['sha256']:
            raise ValueError('등록 후 변경된 입력 데이터셋입니다: ' + record['path'])
        verified.append({**record, **actual, 'sha256': identity})
        if output.is_relative_to(Path(record['path']).resolve()):
            raise ValueError('병합 결과는 원본 데이터셋 밖에 저장하세요.')
    summary = validate_inputs(verified)
    if output.exists():
        raise FileExistsError('병합 결과를 덮어쓸 수 없습니다: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.' + output.name + '.merge-' + uuid.uuid4().hex)
    try:
        print(f"[WORKFLOW] LeRobot 병합 시작 · 데이터셋 {len(records)}개 · {summary['episodes']} episodes", flush=True)
        aggregate_datasets(repo_ids=[repo_id] * len(verified), roots=[Path(r['path']) for r in verified],
                           aggr_repo_id=repo_id, aggr_root=staging)
        meta = LeRobotDatasetMetadata(repo_id, root=staging)
        if meta.total_episodes != summary['episodes'] or meta.total_frames != summary['frames']:
            raise ValueError('병합 결과 episode/frame 수가 입력 합계와 다릅니다.')
        atomic(staging / 'workflow_provenance.json', merged_provenance(verified, summary))
        write_contract(staging, summary['coordinates'])
        for record in verified:
            if fingerprint(record['path']) != record['sha256']:
                raise ValueError('병합 중 원본이 변경되었습니다: ' + record['path'])
        atomic(staging / 'merge_report.json', dict(schema='so101.lerobot-merge/1', passed=True,
               repo_id=repo_id, episodes=summary['episodes'], frames=summary['frames'], split=(summary['partition'] or {}).get('split'),
               seconds=time.monotonic() - started, inputs=[{k: r[k] for k in ('id', 'path', 'sha256')} for r in verified]))
        if output.exists():
            raise FileExistsError(str(output))
        staging.rename(output)
        print(f"[WORKFLOW] LeRobot 병합 완료 · {summary['episodes']} episodes · {output}", flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output
