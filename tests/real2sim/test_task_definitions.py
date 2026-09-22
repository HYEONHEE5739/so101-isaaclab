import copy
import json
import numpy as np
import pytest
from test_workspaces import published
from soarm101_lab.real2sim.workspaces.package import load, metadata
from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
from soarm101_lab.workflow.tasks import catalog, validate, configured, stamp_file, episode_definition, evaluation_metrics
from soarm101_lab.workflow.artifacts import Registry
from soarm101_lab.workflow.operations import Operations


def test_six_explicit_tasks_language_never_parsed(published):
    w = load(published[3]); tasks = catalog(w)
    assert len(tasks) == 6
    assert {(t['pick_object_id'], t['target_object_id']) for t in tasks} == {
        (c, p) for c in ('cube_red', 'cube_blue', 'cube_green') for p in ('cup_a', 'cup_b')}
    definition = tasks[3]; definition['language_instruction'] = 'Pick the red cube into cup A, unrelated text.'
    configured_w = configured(w, definition)
    assert configured_w['task']['pick'] == 'cube_blue' and configured_w['task']['place'] == 'cup_b'
    assert w['task']['pick'] == 'cube_red' and metadata(w) == metadata(configured_w)
    for key, value in [('pick_object_id', 'cube_red'), ('randomization_spec', {}), ('success_condition', {})]:
        bad = copy.deepcopy(definition); bad[key] = value
        with pytest.raises(ValueError): validate(bad, w)


def make_source(tmp_path, w, definitions):
    import h5py
    file = tmp_path / 'source.hdf5'
    with h5py.File(file, 'w') as h:
        data = h.create_group('data'); data.attrs['env_args'] = json.dumps({'workspace': metadata(w), 'so101_coordinates': coordinate_contract(w)})
        for i, definition in enumerate(definitions):
            e = data.create_group('demo_' + str(i))
            e.create_dataset('actions', data=np.zeros((2, 7)))
            e.create_dataset('joint_targets', data=np.zeros((2, 6)))
            e.create_dataset('obs/policy/joint_pos', data=np.zeros((2, 6)))
            if definition:
                from soarm101_lab.workflow.tasks import stamp_group
                stamp_group(e, definition)
    return file


def test_episode_tasks_inherit_and_cannot_relabel(published, tmp_path):
    w = load(published[3]); definition = catalog(w)[3]; definition['language_instruction'] = 'My custom instruction'
    file = make_source(tmp_path, w, [definition]); r = Registry(tmp_path / 'registry')
    artifact = r.register('source', file)
    assert artifact['tasks'] == [definition]
    op = Operations(r)
    plan = op.plan('annotate', artifact, str(w['root']), {}, run_dir=tmp_path)
    assert plan['options']['tasks'] == [definition] and '--headless' in plan['argv']
    with pytest.raises(ValueError, match='relabel'):
        op.plan('annotate', artifact, str(w['root']), {'tasks': [catalog(w)[0]]}, run_dir=tmp_path)
    convert = op.plan('convert', artifact, str(w['root']), {'repo_id': 'test/task', 'task': 'ignored'}, run_dir=tmp_path)
    assert convert['options']['task'] == 'My custom instruction'


def test_no_silent_task_guess_and_mixed_datagen_rejected(published, tmp_path):
    w = load(published[3]); r = Registry(tmp_path / 'registry')
    file = make_source(tmp_path, w, [None]); a = r.register('source', file)
    assert Operations(r).readiness('annotate', a, str(w['root']))[0] == 'NOT_READY'
    file.unlink(); file = make_source(tmp_path, w, catalog(w)[:2]); a = r.register('source', file)
    assert Operations(r).readiness('annotate', a, str(w['root']))[0] == 'NOT_READY'


def test_generated_episode_metadata_and_metrics(published, tmp_path):
    import h5py
    w = load(published[3]); definitions = catalog(w)[:2]
    file = make_source(tmp_path, w, [None, None]); stamp_file(file, definitions[1])
    with h5py.File(file) as h:
        args = json.loads(h['data'].attrs['env_args'])
        for e in h['data'].values():
            assert episode_definition(e, args) == definitions[1]
            assert e.attrs['task_id'] == definitions[1]['task_id']
    rows = [{'task_id': definitions[0]['task_id'], 'steps': 10, 'success': True},
            {'task_id': definitions[0]['task_id'], 'steps': 20, 'success': False},
            {'task_id': definitions[1]['task_id'], 'steps': 30, 'success': False}]
    metrics = evaluation_metrics(definitions, rows, True)
    assert metrics[definitions[0]['task_id']]['success_rate'] == .5
    assert metrics[definitions[1]['task_id']]['success_rate'] == 0
    real = evaluation_metrics(definitions, [{'task_id': definitions[0]['task_id'], 'episode': 0}], False)
    assert real[definitions[0]['task_id']]['success_rate'] is None


def test_shared_selector_presets_editing_and_evaluation_scope(published):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from soarm101_lab.workflow.task_ui import TaskSelector
    app = QApplication.instance() or QApplication([])
    selector = TaskSelector(); selector.configure(load(published[3]), 'source')
    selector.pick.setCurrentText('cube_blue'); selector.target.setCurrentText('cup_b')
    assert selector.selection()[0]['task_id'] == 'cube_blue_to_cup_b'
    selector.edit_instruction('Custom language')
    assert selector.selection()[0]['language_instruction'] == 'Custom language'
    selector.configure(load(published[3]), 'sim_eval')
    selector.scope.setCurrentIndex(selector.scope.findData('all'))
    assert len(selector.selection(True)) == 6
    selector.scope.setCurrentIndex(selector.scope.findData('multiple'))
    for i in range(selector.multiple.count()):
        selector.multiple.item(i).setCheckState(Qt.CheckState.Checked if i < 2 else Qt.CheckState.Unchecked)
    assert len(selector.selection(True)) == 2
    definition = selector.selection(True)[0]
    selector.configure(load(published[3]), 'train', [definition])
    assert not selector.instruction.isEnabled() and selector.selection() == [definition]
    selector.close()


def test_training_checkpoint_keeps_task_metadata(published, tmp_path, monkeypatch):
    from soarm101_lab.workflow.policy import train
    from soarm101_lab.so101_dataset_contract import JOINT_SPACE
    w = load(published[3]); definitions = catalog(w)[:2]
    definitions[1]['language_instruction'] = 'User-edited training instruction'
    dataset = tmp_path / 'dataset'; (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/info.json').write_text('{"repo_id":"local/test"}')
    output = tmp_path / 'training'; checkpoint = output / 'checkpoints/last/pretrained_model'
    def fake_train(*args, **kwargs):
        assert '--policy.path=lerobot/smolvla_base' in args[0]
        assert '--policy.push_to_hub=false' in args[0]
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}')
        (checkpoint / 'model.safetensors').write_bytes(b'test-only')
    monkeypatch.setattr('subprocess.run', fake_train)
    artifact = {'path': str(dataset), 'coordinates': {**coordinate_contract(w), 'action_space': JOINT_SPACE},
                'workspace': metadata(w), 'tasks': definitions}
    train(artifact, {'architecture': 'smolvla', 'steps': 1}, output)
    provenance = json.loads((checkpoint / 'workflow_provenance.json').read_text())
    assert provenance['task_definitions'] == definitions
    assert provenance['training']['policy']['type'] == 'smolvla'
    assert provenance['training']['policy']['pretrained_path'] == 'lerobot/smolvla_base'
    assert provenance['training']['save_checkpoint'] is True
    assert provenance['training']['rename_map']['observation.images.side_cam'] == 'observation.images.camera1'
