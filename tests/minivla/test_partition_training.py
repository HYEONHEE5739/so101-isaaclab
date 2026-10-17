"""Two original MP4/Parquet roots, explicit Source partitions, no RGB materialization."""
import json,sys
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'real2sim'))
from test_source_partitions import source_fixture
from test_lerobot_direct import lerobot_fixture
from soarm101_lab.workflow.episodes import source_split
from soarm101_lab.learning.minivla.data import Episodes
from soarm101_lab.learning.minivla.config import MiniVLAConfig


@pytest.fixture
def separate_datasets(tmp_path,monkeypatch):
    from soarm101_lab.workflow import mujoco as worker
    src=tmp_path/'source.hdf5';source_fixture(src,4);plan=source_split(src,seed=2)
    artifacts=[]
    for split in ('train','val'):
        a=lerobot_fixture(tmp_path/split);p=plan['partitions'][split];proof={'verification':'source_only_input','sha256':'synthetic_recorded_input','source_ids':p['source_ids']}
        from soarm101_lab.real2sim.precision import fingerprint
        proof['evidence_sha256']=fingerprint(proof)
        a['provenance'].update(source_partition=p,source_partition_input=proof,
            source_groups={str(i):'UNKNOWN:actual_subtask_sources' for i in range(2)},
            episode_source_lineage={str(i):{'kind':'partition','partition_sha256':p['sha256'],'exact_source_ids':None} for i in range(2)})
        (Path(a['path'])/'workflow_provenance.json').write_text(json.dumps(a['provenance']))
        if split=='val':
            import pyarrow.parquet as pq,pyarrow as pa
            file=next((Path(a['path'])/'data').rglob('*.parquet'));table=pq.read_table(file)
            for key in ('observation.state','action'):
                value=np.array(table[key].to_pylist(),np.float32);value[:,:5]+=1
                table=table.set_column(table.schema.get_field_index(key),key,pa.array(value.tolist(),type=table[key].type))
            pq.write_table(table,file)
        artifacts.append(a)
    def trainer(request,output):
        model=output/'policy';model.mkdir();p=Path(request['input'])
        (model/'project_provenance.json').write_text((p/'project_provenance.json').read_text());return {'passed':True}
    monkeypatch.setattr(worker,'train',trainer)
    return artifacts


def test_separate_roots_train_statistics_rgb_chunk_and_lineage(separate_datasets,tmp_path):
    from soarm101_lab.workflow.mujoco import train_from_lerobot
    train,val=separate_datasets;out=tmp_path/'prepared'
    train_from_lerobot(train,{'workspace':'unused','validation_artifact':val},out)
    p=out/'dataset';stats=json.loads((p/'statistics.json').read_text());cfg=MiniVLAConfig()
    a=Episodes(p/'manifest.json',stats,cfg,'train');b=Episodes(p/'manifest.json',stats,cfg,'val')
    assert len(a)==len(b)==10 and all(e['source_index']==1 for e in b.episodes)
    assert np.max(np.abs(np.array(stats['state_mean'])[:5]-.17))<1e-6
    assert np.all(b.arrays[0]['state'][:,:5]>1) and np.all(a.arrays[0]['state'][:,:5]<1)
    assert a[4]['mask'].sum()==b[4]['mask'].sum()==1
    assert a[0]['images'][0].shape==b[0]['images'][0].shape==(24,32,3)
    assert a[0]['actions'].shape==(40,6) and not list(out.rglob('*.npy'))
    assert not (out/'canonical_source').exists()
    assert all(e['source_group'].startswith('UNKNOWN') for e in a.episodes+b.episodes)


def test_unverified_validation_artifact_rejected_before_trainer(separate_datasets,tmp_path):
    from soarm101_lab.workflow.mujoco import train_from_lerobot
    a,b=separate_datasets;b['provenance'].pop('source_partition_input')
    with pytest.raises(ValueError,match='unverifiable'):train_from_lerobot(a,{'workspace':'unused','validation_artifact':b},tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_canonical_validation_partition_does_not_fit_statistics_or_lose_source_group(tmp_path):
    from test_data_and_runtime import create_data
    from soarm101_lab.workflow.episodes import export_source_splits,prepare_dataset
    root=tmp_path/'canonical';root.mkdir();_,manifest=create_data(root)
    (root/'project_provenance.json').write_text(json.dumps({'workspace':None,'backend':'mujoco'}))
    for entry in manifest['episodes']:
        (root/entry['path']/'outcome.json').write_text(json.dumps({'success':True,'validation_only':False}))
    partition=export_source_splits(root,tmp_path/'split',seed=4)
    val=Path(partition['outputs']['val']);out=tmp_path/'prepared_val'
    prepare_dataset(val,out,require_validation=False)
    assert json.loads((out/'statistics.json').read_text())['status']=='not_fitted'
    e=json.loads((out/'manifest.json').read_text())['episodes'][0]
    assert e['split']=='val' and e['source_group'].startswith('g') and e['group_id'].startswith('partition:')


def test_preflight_preserves_partition_and_blocks_overrides(separate_datasets):
    from soarm101_lab.workflow.mujoco import validate_training_inputs
    a,b=separate_datasets
    assert validate_training_inputs(a,{'validation_artifact':b})
    with pytest.raises(ValueError,match='Source Group'):
        validate_training_inputs(a,{'validation_artifact':b,'source_groups':{'0':'fake'}})
    with pytest.raises(ValueError,match='both required'):
        validate_training_inputs(a,{})
    with pytest.raises(ValueError,match='Train/Val'):
        validate_training_inputs(a,{'validation_artifact':a})


@pytest.mark.parametrize('changed',['fps','camera','gripper','task','workspace'])
def test_preflight_rejects_incompatible_validation_contract(separate_datasets,changed):
    from soarm101_lab.workflow.episodes import validate_training_inputs
    a,b=separate_datasets
    if changed in ('fps','camera'):
        path=Path(b['path'])/'meta/info.json';info=json.loads(path.read_text())
        if changed=='fps':info['fps']=15
        else:info['features']['observation.images.side_cam']['shape'][0]+=1
        path.write_text(json.dumps(info))
    elif changed=='gripper':b['provenance']['canonical_coordinates']['gripper_open_rad']+=.1
    elif changed=='task':b['tasks']=[{'task_id':'different'}]
    else:b['workspace']={'workspace_id':'different'}
    with pytest.raises(ValueError,match='계약|순서'):validate_training_inputs(a,{'validation_artifact':b})


def test_measured_limits_require_the_recorded_urdf_hash(tmp_path):
    import hashlib
    from soarm101_lab.workflow.episodes import training_coordinates
    from soarm101_lab.so101_joint_mapping import RobotCoordinates
    urdf=tmp_path/'robot.urdf'
    urdf.write_text('<robot><joint name="gripper"><limit lower="-0.174533" upper="1.74533"/></joint></robot>')
    a={'provenance':{'canonical_coordinates':RobotCoordinates().metadata()},'coordinates':{'simulation_model':{
        'robot_model':{'urdf':'robot.urdf','urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest()}}}}
    coords=training_coordinates(a,tmp_path)
    assert coords.metadata()['measured_gripper_limits_rad']==[-.174533,1.74533]
    urdf.write_text(urdf.read_text().replace('-0.174533','-1.0'))
    with pytest.raises(ValueError,match='hash'):training_coordinates(a,tmp_path)
