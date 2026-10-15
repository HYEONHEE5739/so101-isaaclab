import json
import pytest
from soarm101_lab.workflow.policy import resume_checkpoint, train
from soarm101_lab.so101_dataset_contract import JOINT_SPACE


def checkpoint(tmp_path):
    root = tmp_path / '040000'
    model = root / 'pretrained_model'
    state = root / 'training_state'
    model.mkdir(parents=True)
    state.mkdir()
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/info.json').write_text('{}')
    config = {'dataset': {'root': str(dataset)}, 'policy': {'type': 'smolvla'},
              'steps': 100000, 'rename_map': {'side': 'camera1'}}
    (model / 'train_config.json').write_text(json.dumps(config))
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'test')
    for name in ('optimizer_param_groups.json', 'optimizer_state.safetensors',
                 'rng_state.safetensors', 'scheduler_state.json'):
        (state / name).write_text('{}')
    (state / 'training_step.json').write_text('{"step":40000}')
    return root, {'path': str(dataset), 'coordinates': {'action_space': JOINT_SPACE}}


def test_resume_validation(tmp_path):
    root, artifact = checkpoint(tmp_path)
    opts = {'architecture': 'smolvla', 'steps': 100000}
    assert resume_checkpoint(root, artifact, opts)[2] == 40000
    assert resume_checkpoint(root / 'pretrained_model', artifact, opts)[2] == 40000
    for bad in ({'steps': 40000}, {'architecture': 'act'}):
        with pytest.raises(ValueError):
            resume_checkpoint(root, artifact, {**opts, **bad})
    with pytest.raises(ValueError):
        resume_checkpoint(root, {**artifact, 'path': str(tmp_path)}, opts)
    (root / 'training_state/optimizer_state.safetensors').unlink()
    with pytest.raises(ValueError):
        resume_checkpoint(root, artifact, opts)


def test_resume_uses_original_state_and_new_output(tmp_path, monkeypatch):
    root, artifact = checkpoint(tmp_path)
    original = (root / 'pretrained_model/train_config.json').read_bytes()
    output = tmp_path / 'resumed'
    def run(argv, **kwargs):
        assert '--resume=true' in argv
        assert '--config_path=' + str(root / 'pretrained_model/train_config.json') in argv
        assert '--output_dir=' + str(output) in argv
        assert not any(a.startswith('--policy.path=') for a in argv)
        raise RuntimeError('stop before GPU')
    monkeypatch.setattr('subprocess.run', run)
    with pytest.raises(RuntimeError, match='stop before GPU'):
        train(artifact, {'resume_checkpoint': str(root), 'steps': 100000}, output)
    assert (root / 'pretrained_model/train_config.json').read_bytes() == original
    saved = json.loads((tmp_path / 'resumed.train_config.json').read_text())
    assert saved['resume'] is True
    assert saved['rename_map'] == {'side': 'camera1'}


def test_intermediate_checkpoint_import_uses_training_provenance(tmp_path):
    from soarm101_lab.workflow.policy import import_checkpoint
    root, artifact = checkpoint(tmp_path)
    run = tmp_path
    # Fixture checkpoint must be under the training output's checkpoints.
    import shutil
    dest = run / 'policy/checkpoints/040000'
    dest.parent.mkdir(parents=True)
    shutil.move(str(root), dest)
    artifact.update(id='dataset-id', tasks=[{'task_id': 'pick'}], workspace={'id': 'w'})
    model = dest / 'pretrained_model'
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        (model / name).write_text('{}')
    (run / 'request.json').write_text(json.dumps({
        'stage': 'train', 'input': artifact, 'output': str(run / 'policy')}))
    class Registry:
        def verify(self, item):
            assert item == artifact
        def register(self, kind, path, **kwargs):
            assert kind == 'policy'
            return {'path': str(path)}
    result = import_checkpoint(dest, Registry())
    assert result['path'] == str(model)
    assert json.loads((model / 'workflow_provenance.json').read_text())['workspace'] == {'id': 'w'}
    (model / 'workflow_provenance.json').unlink()
    (run / 'request.json').unlink()
    with pytest.raises(ValueError, match='provenance missing'):
        import_checkpoint(dest, Registry())


def test_resume_output_is_original_folder(tmp_path):
    import shutil
    from soarm101_lab.workflow.policy import resume_output
    root, artifact = checkpoint(tmp_path)
    dest = tmp_path / 'policy/checkpoints/040000'
    dest.parent.mkdir(parents=True)
    shutil.move(root, dest)
    assert resume_output(dest) == tmp_path / 'policy'
    assert resume_output(dest / 'pretrained_model') == tmp_path / 'policy'
    (tmp_path / 'mini').mkdir()
    for name in ('last.pt', 'best.pt'): (tmp_path / 'mini' / name).touch()
    assert resume_output(tmp_path / 'mini/last.pt', 'minivla') == tmp_path / 'mini'


def test_minivla_resume_can_extend_steps_but_not_change_model():
    from dataclasses import asdict
    from soarm101_lab.learning.minivla.config import MiniVLAConfig
    from soarm101_lab.learning.minivla.train import validate_resume
    cfg = MiniVLAConfig()
    saved = {'architecture_schema': cfg.architecture_schema, 'config': asdict(cfg), 'statistics': {'mean': [0.]}, 'control_hz': 30, 'step': cfg.steps}
    cfg.steps += 100
    validate_resume(saved, cfg, saved['statistics'], 30)
    cfg.horizon += 1
    with pytest.raises(ValueError, match='config/stats'): validate_resume(saved, cfg, saved['statistics'], 30)


@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
def test_workflow_resume_plan_reuses_training_output(tmp_path, monkeypatch, backend):
    import shutil
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.workflow.artifacts import Registry
    root, artifact = checkpoint(tmp_path)
    dest = tmp_path / 'policy/checkpoints/040000'
    dest.parent.mkdir(parents=True); shutil.move(root, dest)
    artifact.update(type='dataset', id='dataset', tasks=[{'task_id': 'pick', 'language_instruction': 'Pick the cube'}])
    registry = Registry(tmp_path / 'registry')
    monkeypatch.setattr(registry, 'verify', lambda *args: None)
    run = tmp_path / 'resume_run'; run.mkdir()
    plan = Operations(registry).plan('train', artifact, '',
        {'resume_checkpoint': str(dest), 'steps': 100000, 'backend': backend}, run_dir=run)
    assert plan['output'] == str(tmp_path / 'policy')
    assert json.loads((run / 'request.json').read_text())['output'] == str(tmp_path / 'policy')
