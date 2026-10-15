"""Real Parquet/MP4 fixture, unchanged MiniVLA batch contract, no model download."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from soarm101_lab.learning.minivla.data import Episodes,fit_statistics,load_manifest,collate
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.so101_joint_mapping import JOINTS,RobotCoordinates


def lerobot_fixture(root,height=24,width=32,frames=5):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from soarm101_lab.so101_dataset_contract import current_contract,JOINT_SPACE,write_contract
    ds=LeRobotDataset.create('local/direct-fixture',root=root,robot_type='so101',fps=30,
        features={'observation.state':{'dtype':'float32','shape':(6,),'names':list(JOINTS)},
                  'action':{'dtype':'float32','shape':(6,),'names':list(JOINTS)},
                  'observation.images.side_cam':{'dtype':'video','shape':(height,width,3),'names':['height','width','channel']}},
        use_videos=True,image_writer_threads=1,image_writer_processes=0)
    for ep in range(2):
        for t in range(frames):
            q=np.full(6,(ep+1)*.1+t*.01,np.float32);q[-1]=np.deg2rad(20+t)
            rgb=np.zeros((height,width,3),np.uint8);rgb[...,0]=30+ep*50+t*10;rgb[...,1]=110;rgb[...,2]=220-t*10
            ds.add_frame({'observation.state':q,'action':q+.02,'observation.images.side_cam':rgb,'task':f'Pick target {ep}'})
        ds.save_episode()
    ds.finalize()
    contract={**current_contract(JOINT_SPACE),'action_source':'joint_targets'};write_contract(root,contract)
    provenance={'source_groups':{'0':'source_a','1':'source_b'},'action_source':'joint_targets',
        'canonical_coordinates':RobotCoordinates().metadata(),
        'episode_scene_configurations':{'0':{'active_object_ids':['target'],'sha256':'single'},'1':{'active_object_ids':['target','other'],'sha256':'double'}},
        'episodes':[{'episode_index':i,'task_definition':{'task_id':f'task{i}','language_instruction':f'Pick target {i}'}} for i in range(2)]}
    (root/'workflow_provenance.json').write_text(json.dumps(provenance))
    return {'path':str(root),'coordinates':contract,'workspace':None,'provenance':provenance,'tasks':[]}


@pytest.fixture
def direct_data(tmp_path,monkeypatch):
    from soarm101_lab.workflow import mujoco as worker
    artifact=lerobot_fixture(tmp_path/'lerobot');out=tmp_path/'training'
    def prepared(request,output):
        p=Path(request['input']);model=output/'policy';model.mkdir()
        provenance=json.loads((p/'project_provenance.json').read_text());provenance['source_dataset']=str(p)
        (model/'project_provenance.json').write_text(json.dumps(provenance))
        return {'passed':True}
    monkeypatch.setattr(worker,'train',prepared)
    worker.train_from_lerobot(artifact,{'workspace':'unused'},out)
    return artifact,out


def test_direct_numeric_chunk_rgb_and_train_only_stats(direct_data):
    artifact,out=direct_data;p=out/'dataset';cfg=MiniVLAConfig()
    stats=json.loads((p/'statistics.json').read_text());ds=Episodes(p/'manifest.json',stats,cfg,'train')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    reference=LeRobotDataset('local/direct-fixture',root=artifact['path'])
    coords=RobotCoordinates();raw=np.stack([reference.hf_dataset[i]['observation.state'].numpy() for i in range(5)])
    np.testing.assert_allclose(stats['state_mean'],coords.to_policy(raw,measured=True).mean(0),rtol=1e-6)
    for t in range(5):
        old=reference[t];sample=ds[t]
        np.testing.assert_array_equal(ds.arrays[0]['state'][t],coords.to_policy(old['observation.state']))
        np.testing.assert_array_equal(ds.arrays[0]['action'][t],coords.to_policy(old['action']))
        rgb=old['observation.images.side_cam'].permute(1,2,0).numpy()*255
        np.testing.assert_allclose(sample['images'][0],rgb,atol=1e-4)
        assert sample['images'][0].dtype==np.uint8 and sample['images'][0].shape==(24,32,3)
        assert sample['instruction']=='Pick target 0' and sample['frame']==t
        assert sample['mask'].sum()==5-t and torch.all(sample['actions'][5-t:]==0)
        np.testing.assert_allclose(ds.pre.denormalize(sample['actions'][:5-t].numpy()),ds.arrays[0]['action'][t:],atol=1e-6)
    batch=collate([ds[4]]);assert batch['actions'].shape==(1,40,6) and batch['mask'].shape==(1,40)
    m=load_manifest(p/'manifest.json');assert [e['group_id'] for e in m['episodes']]==['source_a','source_b']
    assert m['episodes'][1]['scene_configuration']['active_object_ids']==['target','other']
    assert not list(out.rglob('*.npy')) and not (out/'canonical_source').exists()
    assert len(ds.decoders)<=2 and len(ds.frame_cache)<=8 and ds.cache_bytes<=16*2**20


def test_statistics_never_decode_and_direct_bytes_guard(direct_data,monkeypatch):
    artifact,out=direct_data;p=out/'dataset'
    import torchcodec.decoders
    monkeypatch.setattr(torchcodec.decoders,'VideoDecoder',lambda *a,**k:pytest.fail('Statistics decoded video'))
    stats=fit_statistics(p/'manifest.json',out/'check_stats.json')
    assert stats==json.loads((p/'statistics.json').read_text())
    video=next(Path(artifact['path']).rglob('*.mp4'));video.write_bytes(video.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='bytes changed'):Episodes(p/'manifest.json',stats,MiniVLAConfig(),'train')


def test_direct_decode_errors_are_not_skipped(direct_data,monkeypatch):
    _,out=direct_data;p=out/'dataset';stats=json.loads((p/'statistics.json').read_text())
    ds=Episodes(p/'manifest.json',stats,MiniVLAConfig(),'train')
    import torchcodec.decoders
    def broken(*args,**kwargs):raise RuntimeError('fixture decoder failure')
    monkeypatch.setattr(torchcodec.decoders,'VideoDecoder',broken)
    with pytest.raises(ValueError,match='episode 0, frame 0'):ds[0]


def test_direct_frame_and_action_timestamp_mismatch(direct_data):
    _,out=direct_data;p=out/'dataset';m=json.loads((p/'manifest.json').read_text());m['episodes'][0]['dataset_episode_index']=1
    (p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='index mismatch'):fit_statistics(p/'manifest.json',out/'invalid.json')


def test_direct_group_leakage_rejected(direct_data):
    _,out=direct_data;p=out/'dataset';m=json.loads((p/'manifest.json').read_text());m['episodes'][1]['group_id']='source_a'
    (p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='leakage'):load_manifest(p/'manifest.json')


def test_direct_action_timestamp_and_gripper_contract(direct_data):
    from types import SimpleNamespace
    from soarm101_lab.learning.minivla.data import episode_arrays
    _,out=direct_data;p=out/'dataset';m=load_manifest(p/'manifest.json')
    stats=json.loads((p/'statistics.json').read_text());ds=Episodes(p/'manifest.json',stats,MiniVLAConfig(),'train')
    hf=ds.reader.hf_dataset
    bad=hf.add_column('action_timestamp',(np.asarray(hf['timestamp']).reshape(-1)-.001).tolist())
    reader=SimpleNamespace(hf_dataset=bad,meta=ds.reader.meta)
    with pytest.raises(ValueError,match='follow'):episode_arrays(m['episodes'][0],['side'],30,reader)
    raw=np.asarray(hf.select_columns(['action'])[:]['action'],dtype=np.float32);raw[:,5]=10.
    bad=hf.remove_columns('action').add_column('action',raw.tolist())
    with pytest.raises(ValueError,match='Gripper outside'):episode_arrays(m['episodes'][0],['side'],30,SimpleNamespace(hf_dataset=bad,meta=ds.reader.meta))


@pytest.mark.parametrize('groups',[{}, {'0':'UNKNOWN:source','1':'UNKNOWN:source'}])
def test_unverified_source_group_lineage_rejected(direct_data,groups):
    from soarm101_lab.workflow import mujoco as worker
    artifact,out=direct_data;artifact['provenance']['source_groups']=groups
    with pytest.raises(ValueError,match='verified source_groups'):
        worker.train_from_lerobot(artifact,{'workspace':'unused'},out.parent/'unverified')
