import h5py
import numpy as np
from soarm101_lab.workflow.data_resume import combine


def test_datagen_resume_rejects_video_folder_before_worker_output(tmp_path):
    import pytest
    from soarm101_lab.workflow.data_resume import inspect_resume_input
    from soarm101_lab.workflow.unified import execute
    video=tmp_path/'dataset.videos';video.mkdir()
    with pytest.raises(ValueError,match='영상 보관용'):inspect_resume_input('datagen',video)
    output=tmp_path/'must_not_be_created'
    with pytest.raises(ValueError,match='영상 보관용'):
        execute({'stage':'datagen','options':{'resume_data':str(video)},'output':str(output),'input':'unused'})
    assert not output.exists()


def test_datagen_resume_requires_actual_generated_result(tmp_path,monkeypatch):
    import pytest
    from soarm101_lab.workflow import data_resume
    from soarm101_lab.workflow.dataset import SCHEMA
    path=tmp_path/'data.hdf5';path.touch()
    item={'format':SCHEMA,'env_args':{}}
    monkeypatch.setattr(data_resume,'inspect_artifact',lambda *args:item)
    with pytest.raises(ValueError,match='이전 DataGen 결과'):data_resume.inspect_resume_input('datagen',path)
    item['env_args'].update(generation_method='isaaclab_mimic/1.0.15',generation_seed=1047)
    assert data_resume.inspect_resume_input('datagen',path) is item


def test_merge_rejects_different_simulation_calibration_before_modifying_output(tmp_path):
    import pytest,json
    from soarm101_lab.so101_dataset_contract import write_contract
    paths=[tmp_path/'old.hdf5',tmp_path/'new.hdf5']
    for path,mode in zip(paths,['urdf_identity','precision_bound_residual']):
        with h5py.File(path,'w') as h:h.create_group('data').create_group('demo_0').create_dataset('actions',data=np.zeros((3,6)))
        write_contract(path,{'schema':'so101.coordinate-contract/1','simulation_model':{'model_mode':mode}})
    before=[p.read_bytes() for p in paths]
    with pytest.raises(ValueError,match='calibration/robot/coordinate'):combine(*paths)
    assert [p.read_bytes() for p in paths]==before


def test_merge_rejects_source_partition_mixture(tmp_path):
    import pytest,json
    paths=[tmp_path/'train.hdf5',tmp_path/'val.hdf5']
    for path,split in zip(paths,['train','val']):
        with h5py.File(path,'w') as h:
            data=h.create_group('data');data.attrs['env_args']=json.dumps({'source_partition':{'split':split}})
            data.create_group('demo_0').create_dataset('actions',data=np.zeros((3,6)))
    before=[p.read_bytes() for p in paths]
    with pytest.raises(ValueError,match='Source Partition'):combine(*paths)
    assert [p.read_bytes() for p in paths]==before


def test_merge_preserves_previous_and_orders_completed_episodes(tmp_path):
    old = tmp_path / 'old.hdf5'
    new = tmp_path / 'new.hdf5'
    for path, values in ((old, [1, 2]), (new, [3])):
        with h5py.File(path, 'w') as f:
            data = f.create_group('data')
            data.attrs['env_args'] = '{}'
            for i, value in enumerate(values):
                ep = data.create_group(f'demo_{i}')
                ep.create_dataset('actions', data=np.full((2, 6), value))
                ep.attrs['success'] = True
    before = old.read_bytes()
    combine(old, new)
    assert old.read_bytes() == before
    with h5py.File(new) as f:
        assert len(f['data']) == 3
        assert f['data'].attrs['total'] == 6
        assert f['data/demo_2/actions'][0, 0] == 3
        assert f['data/demo_0'].attrs['success']


def test_resume_rejects_wrong_workspace_task_and_datagen_input(monkeypatch):
    import pytest
    from soarm101_lab.workflow import data_resume
    monkeypatch.setattr(data_resume, 'inspect_artifact', lambda *args: {
        'workspace': {'id': 'a'}, 'tasks': [{'task_id': 'pick'}],
        'episode_count': 20, 'env_args': {'source_dataset': '/tmp/input.hdf5'}})
    assert data_resume.validate('datagen', 'unused', {'id': 'a'}, [{'task_id': 'pick'}],
                                '/tmp/input.hdf5') == 20
    for workspace, tasks, source in [
        ({'id': 'b'}, [{'task_id': 'pick'}], '/tmp/input.hdf5'),
        ({'id': 'a'}, [], '/tmp/input.hdf5'),
        ({'id': 'a'}, [{'task_id': 'pick'}], '/tmp/other.hdf5'),
    ]:
        with pytest.raises(ValueError):
            data_resume.validate('datagen', 'unused', workspace, tasks, source)


def test_interrupt_is_deferred_until_boundary(monkeypatch):
    import signal
    import pytest
    from soarm101_lab.workflow.data_resume import deferred_interrupt
    handlers = {}
    monkeypatch.setattr(signal, 'signal', lambda number, callback: handlers.update({number: callback}))
    boundary = deferred_interrupt()
    boundary()
    handlers[signal.SIGINT](signal.SIGINT, None)
    with pytest.raises(KeyboardInterrupt):
        boundary()


def test_lossless_compression_keeps_dtype_attrs_and_hard_links(tmp_path):
    from soarm101_lab.workflow.data_resume import compress
    path = tmp_path / 'frames.hdf5'
    rgb = np.full((8, 240, 320, 3), 127, dtype=np.uint8)
    state = np.arange(48, dtype=np.float32).reshape(8, 6)
    with h5py.File(path, 'w') as f:
        f.attrs['contract'] = 'preserved'
        g = f.create_group('data/demo_0'); g.attrs['success'] = True
        d = g.create_dataset('rgb', data=rgb); d.attrs['encoding'] = 'RGB'
        g['same_rgb'] = d
        g.create_dataset('joint_pos', data=state)
    result = compress(path)
    assert result['saved_bytes'] > rgb.nbytes * .9
    with h5py.File(path) as f:
        g = f['data/demo_0']
        assert f.attrs['contract'] == 'preserved' and g.attrs['success']
        assert g['rgb'].dtype == np.uint8 and g['rgb'].compression == 'gzip'
        assert g['rgb'].attrs['encoding'] == 'RGB' and g['same_rgb'].id == g['rgb'].id
        np.testing.assert_array_equal(g['rgb'], rgb)
        np.testing.assert_array_equal(g['joint_pos'], state)
    assert compress(path)['saved_bytes'] == 0
