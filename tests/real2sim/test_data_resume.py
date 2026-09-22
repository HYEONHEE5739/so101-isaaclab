import h5py
import numpy as np
from soarm101_lab.workflow.data_resume import combine


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
