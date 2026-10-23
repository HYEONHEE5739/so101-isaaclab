"""Converted dataset merge contracts and real local LeRobot video aggregation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from soarm101_lab.real2sim.precision import fingerprint as evidence_hash
from soarm101_lab.real2sim.storage import atomic
from soarm101_lab.so101_dataset_contract import JOINT_SPACE, current_contract, write_contract
from soarm101_lab.so101_joint_mapping import JOINTS, RobotCoordinates
from soarm101_lab.workflow.paths import output_location
from soarm101_lab.workflow.worker_request import stage_input
from soarm101_lab.utils.artifact_identity import fingerprint
from soarm101_lab.workflow.lerobot_merge import merged_provenance, validate_inputs, merge


def converted_record(root, name, split='train'):
    inventory = {f'{name}_{s}': dict(dataset_sha256=name, episode_id=s, groups=[f'{name}/{s}']) for s in ('train', 'val')}
    p = dict(schema='so101.source-partition/1', split=split, inventory=inventory,
             assignments={key: key.rsplit('_', 1)[1] for key in inventory}, source_ids=[name + '_' + split])
    p['sha256'] = evidence_hash(p)
    task = dict(task_id='pick', language_instruction='Pick up the cube.')
    features = {key: dict(dtype='float32', shape=[6], names=list(JOINTS)) for key in ('observation.state', 'action')}
    features.update({f'observation.images.{cam}_cam': dict(dtype='video', shape=[64, 64, 3], names=['height', 'width', 'channel']) for cam in ('side', 'wrist')})
    info = dict(codebase_version='v3.0', fps=30, robot_type='so101', features=features, total_episodes=1, total_frames=5)
    prov = dict(workspace={'fixture': 'workspace'}, backend='mujoco', source_partition=p, source_dataset=str(root / (name + '.hdf5')),
                canonical_coordinates=RobotCoordinates().metadata(), action_source='joint_targets', task_definitions=[task],
                source_groups={'0':name}, episode_source_lineage={'0':dict(kind='source_episode', source_id=p['source_ids'][0], source=inventory[p['source_ids'][0]])},
                episodes=[dict(episode_index=0, source_episode='demo_0', task_definition=task)],
                episode_review={'0':dict(recorded_success=True, annotation={'grasp':2, 'place':4})})
    return dict(id=name, path=str(root / name), type='dataset', status='SUCCEEDED', format=info, provenance=prov,
                workspace=prov['workspace'], tasks=[task], backend='mujoco', coordinates={**current_contract(JOINT_SPACE), 'action_source':'joint_targets'})


def unpartitioned_record(root, name):
    import hashlib
    from soarm101_lab.real2sim.workspaces.package import ROOT
    record = converted_record(root, name)
    prov = record['provenance']
    prov['source_partition'] = None
    prov['source_partition_input'] = None
    prov['episode_source_lineage']['0'] = dict(kind='mimic_segments', partition_sha256=None,
        segments=[dict(eef='tool0', subtask=0, source=prov['episode_source_lineage']['0'])])
    task = dict(task_id='pick_' + name, language_instruction='Pick up the ' + name + ' cube.')
    prov['task_definitions'] = [task]
    prov['episodes'][0]['task_definition'] = task
    scene = dict(workspace=prov['workspace'], active_object_ids=['a', 'b', 'c'], backend='mujoco',
                 reset={'seed':0}, pick_object_id=name, place_target_id='cup', task_definition=task)
    scene['sha256'] = evidence_hash(scene)
    prov['scene_configuration'] = scene
    prov['scene_configurations'] = [scene]
    prov['episode_scene_configurations'] = {'0':scene}
    urdf = 'assets/SO101/urdf/so101_isaaclab.urdf'
    prov['workspace_path'] = str(ROOT)
    model = dict(schema='so101.simulation-model/1',
                 robot_model={'collision':'same', 'urdf':urdf, 'urdf_sha256':hashlib.sha256((ROOT/urdf).read_bytes()).hexdigest()},
                 scene_configuration=scene)
    model['sha256'] = evidence_hash(model)
    record['coordinates']['simulation_model'] = model
    record['tasks'] = [task]
    return record


def test_unpartitioned_multitask_merge_preserves_absent_roles_and_original_evidence(tmp_path):
    a, b, c = [unpartitioned_record(tmp_path, n) for n in ('a', 'b', 'c')]
    for r in (a, b, c):r['sha256'] = r['id'] * 64
    before = copy.deepcopy([a, b, c])
    summary = validate_inputs([a, b])
    assert summary['partition'] is None and len(summary['tasks']) == 2
    model = summary['coordinates']['simulation_model']
    assert model['scene_configuration'] is None
    assert model['sha256'] == evidence_hash({k:v for k,v in model.items() if k!='sha256'})
    prov = merged_provenance([a, b], summary)
    assert prov['source_partition'] is None and prov['episode_source_partitions'] == {'0':None,'1':None}
    assert prov['episode_source_lineage']['1'] == b['provenance']['episode_source_lineage']['0']
    assert prov['merge_inputs'][1]['coordinates'] == b['coordinates']
    assert len(prov['scene_configurations']) == 2 and 'scene_configuration' not in prov
    merged = {**a, 'id':'merged', 'path':str(tmp_path/'merged'), 'provenance':prov,
              'coordinates':summary['coordinates'], 'format':{**a['format'],'total_episodes':2,'total_frames':10}, 'sha256':'merged'}
    assert validate_inputs([merged, c])['episodes'] == 3
    with pytest.raises(ValueError, match='중복 episode'):validate_inputs([a, merged])
    assert [a, b, c] == before


@pytest.mark.parametrize('mismatch', ['payload','model_hash','physics','scene_hash','scene_physics'])
def test_unpartitioned_merge_rejects_invalid_evidence_and_physical_changes(tmp_path, mismatch):
    a, b = [unpartitioned_record(tmp_path, n) for n in ('a', 'b')]
    if mismatch == 'payload':b['provenance']['episode_source_lineage']['0']['segments'][0]['source']['source']['payload_sha256'] = 'different'
    elif mismatch == 'model_hash':b['coordinates']['simulation_model']['sha256'] = 'broken'
    elif mismatch == 'physics':
        model = b['coordinates']['simulation_model'];model['robot_model']['collision'] = 'changed'
        model['sha256'] = evidence_hash({k:v for k,v in model.items() if k!='sha256'})
    elif mismatch == 'scene_hash':b['provenance']['scene_configuration']['sha256'] = 'broken'
    elif mismatch == 'scene_physics':
        scene = b['provenance']['scene_configuration'];scene['reset']['seed'] = 99
        scene['sha256'] = evidence_hash({k:v for k,v in scene.items() if k!='sha256'})
        model = b['coordinates']['simulation_model'];model['sha256'] = evidence_hash({k:v for k,v in model.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_inputs([a, b])


def test_partition_union_keeps_original_per_episode_proofs_and_overlap_detection(tmp_path):
    a, b = [converted_record(tmp_path, n) for n in ('a', 'b')]
    for r in (a, b):r['sha256'] = r['id'] * 64
    summary = validate_inputs([a, b])
    assert summary['episodes'] == 2 and summary['partition'] is None
    p = merged_provenance([a, b], summary)
    assert p['episode_source_lineage']['1'] == b['provenance']['episode_source_lineage']['0']
    assert p['episode_source_partitions']['1'] == b['provenance']['source_partition']
    assert p['episode_review']['1'] == b['provenance']['episode_review']['0']
    merged = {**a, 'id':'merged', 'path':str(tmp_path / 'merged'), 'provenance':p,
              'format':{**a['format'], 'total_episodes':2, 'total_frames':10}, 'sha256':'merged'}
    with pytest.raises(ValueError, match='중복 episode'):validate_inputs([a, merged])
    c = converted_record(tmp_path, 'c'); c['sha256'] = 'c' * 64
    assert validate_inputs([merged, c])['episodes'] == 3


@pytest.mark.parametrize('mismatch', ['features','fps','workspace','model','canonical','partition_assignment','duplicate'])
def test_incompatible_or_duplicate_inputs_are_rejected(tmp_path, mismatch):
    a, b = [converted_record(tmp_path, n) for n in ('a', 'b')]
    if mismatch == 'features':b['format']['features']['action']['shape'] = [5]
    elif mismatch == 'fps':b['format']['fps'] = 60
    elif mismatch == 'workspace':b['provenance']['workspace'] = {'other':'workspace'}
    elif mismatch == 'model':b['coordinates']['simulation_model'] = {'different_collision':'hash'}
    elif mismatch == 'canonical':b['provenance']['canonical_coordinates']['gripper_open_rad'] *= 2
    elif mismatch == 'partition_assignment':
        b['provenance']['source_partition'] = copy.deepcopy(a['provenance']['source_partition'])
        b['provenance']['source_partition']['sha256'] = 'broken'
    elif mismatch == 'duplicate':b = copy.deepcopy(a)
    with pytest.raises(ValueError):validate_inputs([a, b])


def test_historical_roles_and_missing_lineage_do_not_gate_merge(tmp_path):
    a = converted_record(tmp_path, 'a')
    b = converted_record(tmp_path, 'b', 'val')
    assert validate_inputs([a,b])['partition'] is None
    b['provenance'].pop('source_partition')
    b['provenance']['episode_source_lineage'] = {}
    assert validate_inputs([a,b])['episodes'] == 2


def write_video_dataset(record, value):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(record['path'])
    features = copy.deepcopy(record['format']['features'])
    for feature in features.values():feature['shape'] = tuple(feature['shape'])
    ds = LeRobotDataset.create('local/' + record['id'], root=root, fps=30, robot_type='so101',
                              features=features, image_writer_threads=2)
    expected = []
    for i in range(5):
        q = np.full(6, value + i * .001, dtype=np.float32); q[-1] = .2
        expected.append(q)
        ds.add_frame({'observation.state':q, 'action':q + .01, 'task':record['provenance']['task_definitions'][0]['language_instruction'],
                      'observation.images.side_cam':np.full((64,64,3), value * 1000, np.uint8),
                      'observation.images.wrist_cam':np.full((64,64,3), value * 1000 + 20, np.uint8)})
    ds.save_episode(); ds.stop_image_writer(); ds.finalize()
    write_contract(root, record['coordinates']); atomic(root / 'workflow_provenance.json', record['provenance'])
    return np.asarray(expected)


def test_real_video_merge_worker_registration_and_training_reader(tmp_path):
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.data.legacy import episode_arrays, lerobot_reader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=tmp_path/'workflow'
    records, expected = [], []
    for i, name in enumerate(('a', 'b')):
        r = converted_record(tmp_path, name)
        expected.append(write_video_dataset(r, .03 + .06 * i))
        records.append(stage_input('train',r['path']))
    opts = dict(merge_datasets=[r['path'] for r in records], storage_name='test_merge')
    assert Operations(root=root).readiness('merge',records[0]['path'], options=opts) == ('READY', '')
    result = Operations(root=root).run('merge',records[0]['path'], options=opts)
    assert not (root/'artifacts').exists()
    assert Path(result['path']) == root / 'datasets/test_merge'
    for r in records:assert stage_input('train',r['path'])['format']['total_episodes']==1
    ds = LeRobotDataset('local/so101_merged', root=result['path'], video_backend='torchcodec')
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == 10
    np.testing.assert_allclose(ds.meta.stats['observation.state']['mean'], np.concatenate(expected).mean(axis=0), atol=1e-7)
    for i in range(2):
        meta = ds.meta.episodes[i]
        assert meta['dataset_from_index'] == 5 * i and meta['dataset_to_index'] == 5 * (i+1)
        rows = ds.hf_dataset.select_columns(['observation.state','action','timestamp','frame_index','episode_index'])[5*i:5*(i+1)]
        np.testing.assert_array_equal(np.asarray(rows['observation.state']), expected[i])
        np.testing.assert_array_equal(np.asarray(rows['action']), expected[i] + .01)
        np.testing.assert_allclose(rows['timestamp'], np.arange(5)/30, atol=1e-7)
        assert [int(x) for x in rows['frame_index']] == list(range(5)) and {int(x) for x in rows['episode_index']} == {i}
        frame = ds[5*i]
        assert abs(float(frame['observation.images.side_cam'].mean()) * 255 - (30 + 60*i)) < 3
        assert abs(float(frame['observation.images.wrist_cam'].mean()) * 255 - (50 + 60*i)) < 3
    from soarm101_lab.learning.datasets import inspect_dataset
    assert inspect_dataset(result['path'])['format']['total_episodes']==2
    with pytest.raises(FileExistsError):merge([r['path'] for r in records],result['path'])
    assert json.loads((Path(result['path']) / 'merge_report.json').read_text())['passed']


def test_failure_keeps_inputs_and_does_not_publish_partial_output(tmp_path, monkeypatch):
    from soarm101_lab.workflow import lerobot_merge
    from lerobot.datasets import aggregate
    root=tmp_path/'workflow'
    records = []
    for i, name in enumerate(('a','b')):
        r = converted_record(tmp_path, name)
        write_video_dataset(r, .03 + .06*i)
        records.append(stage_input('train',r['path']))
    def interrupted(**kwargs):
        kwargs['aggr_root'].mkdir()
        (kwargs['aggr_root'] / 'partial').write_text('partial')
        raise KeyboardInterrupt()
    monkeypatch.setattr(aggregate, 'aggregate_datasets', interrupted)
    with pytest.raises(KeyboardInterrupt):lerobot_merge.merge([r['path'] for r in records],tmp_path/'result')
    assert not (tmp_path / 'result').exists() and not list(tmp_path.glob('.*.merge-*'))
    for r in records:assert stage_input('train',r['path'])['format']['total_episodes']==1


def test_real_unpartitioned_multitask_merge_uses_shared_coordinates_and_task_indices(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=tmp_path/'workflow'
    records = []
    for i, name in enumerate(('a','b','c')):
        r = unpartitioned_record(tmp_path, name)
        write_video_dataset(r, .03 + .06*i)
        records.append(stage_input('train',r['path']))
    output=merge([r['path'] for r in records],tmp_path/'merged')
    result=stage_input('train',output)
    assert result['provenance']['source_partition'] is None and len(result['tasks']) == 3
    ds = LeRobotDataset('local/so101_merged',root=output,video_backend='torchcodec')
    assert ds.meta.total_episodes == 3 and ds.meta.total_frames == 15
    for i, record in enumerate(records):
        task_index = ds.hf_dataset[i*5]['task_index'].item()
        assert ds.meta.tasks.iloc[task_index].name == record['tasks'][0]['language_instruction']
    from soarm101_lab.learning.datasets import inspect_dataset
    assert inspect_dataset(output)['format']['total_episodes']==3
    for record in records:assert stage_input('train',record['path'])['format']['total_episodes']==1


def test_known_exact_source_episode_duplicate_rejected(tmp_path):
    a,b=[converted_record(tmp_path,n) for n in ('a','b')]
    b['provenance']['episode_source_lineage']=copy.deepcopy(a['provenance']['episode_source_lineage'])
    with pytest.raises(ValueError,match='중복 episode'):validate_inputs([a,b])
