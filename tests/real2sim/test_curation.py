import json
import numpy as np
import h5py
import pytest
from soarm101_lab.workflow.curation import select_distribution, export_subset


def test_exact_deterministic_stratification():
    rows=[dict(name=f'demo_{i}',task_id='a' if i<8 else 'b',xy=[i%2,i%3],flags=[],jitter=i) for i in range(12)]
    chosen=select_distribution(rows,6)
    assert len(set(chosen))==6
    assert chosen==select_distribution(rows,6)
    assert set(select_distribution(rows,12))=={r['name'] for r in rows}
    with pytest.raises(ValueError):select_distribution(rows,13)
    with pytest.raises(ValueError):select_distribution([],1)


def test_export_preserves_source_and_full_episode(tmp_path):
    src=tmp_path/'input.hdf5';out=tmp_path/'curated.hdf5'
    with h5py.File(src,'w') as f:
        d=f.create_group('data');d.attrs['env_args']='{"task":"test"}';d.attrs['total']=6
        for i in range(3):
            g=d.create_group(f'demo_{i}');g.attrs['success']=True
            g.create_dataset('actions',data=np.ones((2,6))*i)
            g.create_dataset('obs/camera',data=np.zeros((2,4,4,3),dtype=np.uint8))
    from soarm101_lab.so101_dataset_contract import write_contract,read_contract
    write_contract(src,{'schema':'so101.coordinate-contract/1','action_space':'test'})
    before=src.read_bytes()
    export_subset(src,out,['demo_2','demo_0'])
    assert src.read_bytes()==before
    assert read_contract(src)==read_contract(out)
    with h5py.File(out) as f:
        assert len(f['data'])==2 and f['data'].attrs['total']==4
        assert f['data/demo_0/actions'][0,0]==2
        assert f['data/demo_0'].attrs['curation_source_episode']=='demo_2'
        assert 'obs/camera' in f['data/demo_0']
    with pytest.raises(ValueError):export_subset(src,out,['demo_0'])


def test_ui_select_exclude_and_restore(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    d=CurationDialog(Registry(tmp_path))
    d.rows=[dict(name='demo_0',task_id='a',xy=[0,0],flags=['長'],duration=1,stop_fraction=.2,jitter=1,gripper_jitter=1,
                 ee=[[0,0,0],[.01,0,.01]],actions=[[0]*6,[1]*6])]
    d.scatter.rows=d.rows;d.select(0);d.toggle()
    assert d.excluded=={'demo_0'}
    d.restore_all();assert not d.excluded
    d.close()


def test_robot_urdf_fk_and_visuals():
    from pathlib import Path
    from soarm101_lab.workflow.robot_preview import RobotPreview
    root=Path(__file__).resolve().parents[2]
    robot=RobotPreview(root/'assets/SO101/urdf/so101_isaaclab.urdf')
    q=np.zeros(6);a=robot.transforms(q)
    q[0]=.5;b=robot.transforms(q)
    assert np.allclose(a['base'],b['base'])
    assert not np.allclose(a['tool0'],b['tool0'])
    tri,colors=robot.triangles(q)
    assert tri.shape[1:]==(3,3) and len(tri)==len(colors)
    assert np.isfinite(tri).all()
