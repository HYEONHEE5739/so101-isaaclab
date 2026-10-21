"""Real Parquet/MP4 fixture, unchanged MiniVLA batch contract, no model download."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
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


def test_native_learning_reader_without_partition(tmp_path):
    from soarm101_lab.learning.datasets import inspect_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    data=lerobot_fixture(tmp_path/'data')
    checked=inspect_dataset(data['path'])
    reader=LeRobotDataset('local/direct-fixture',root=data['path'])
    assert len(reader)==10 and checked['format']['total_episodes']==2
    assert reader[0]['observation.images.side_cam'].shape==(3,24,32)
