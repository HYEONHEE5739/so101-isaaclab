import json
import pytest
from soarm101_lab.learning.training import resume_checkpoint, train
from soarm101_lab.so101_dataset_contract import JOINT_SPACE


def checkpoint(tmp_path):
    root = tmp_path / '040000'
    model = root / 'pretrained_model'
    state = root / 'training_state'
    model.mkdir(parents=True)
    state.mkdir()
    dataset = tmp_path / 'dataset'
    from test_learning_interfaces import dataset as make_dataset
    make_dataset(dataset)
    config = {'dataset': {'root': str(dataset)}, 'policy': {'type': 'smolvla'},
              'steps': 100000, 'rename_map': {'side': 'camera1'}}
    (model / 'train_config.json').write_text(json.dumps(config))
    (model / 'config.json').write_text('{"type":"smolvla"}')
    from soarm101_lab.so101_dataset_contract import current_contract,write_contract
    write_contract(model,current_contract(JOINT_SPACE))
    (model/'workflow_provenance.json').write_text((dataset/'workflow_provenance.json').read_text())
    (model / 'model.safetensors').write_bytes(b'test')
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):(model/name).write_text('{"steps":[]}')
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
        train(artifact['path'], {'resume_checkpoint': str(root), 'steps': 100000}, output)
    assert (root / 'pretrained_model/train_config.json').read_bytes() == original
    saved = json.loads((tmp_path / 'resumed.train_config.json').read_text())
    assert saved['resume'] is True
    assert saved['rename_map'] == {'side': 'camera1'}




def test_resume_output_is_original_folder(tmp_path):
    import shutil
    from soarm101_lab.learning.training import resume_output
    root, artifact = checkpoint(tmp_path)
    dest = tmp_path / 'policy/checkpoints/040000'
    dest.parent.mkdir(parents=True)
    shutil.move(root, dest)
    assert resume_output(dest) == tmp_path / 'policy'
    assert resume_output(dest / 'pretrained_model') == tmp_path / 'policy'
    (tmp_path / 'mini').mkdir()
    for name in ('last.pt', 'best.pt'): (tmp_path / 'mini' / name).touch()
    (tmp_path/'mini/config.json').write_text('{}')
    with pytest.raises(ValueError,match='비활성화'):resume_output(tmp_path/'mini/last.pt','minivla')




@pytest.mark.parametrize('backend', ['isaac', 'mujoco'])
def test_workflow_resume_plan_reuses_training_output(tmp_path, monkeypatch, backend):
    import shutil
    from soarm101_lab.workflow.operations import Operations
    root, artifact = checkpoint(tmp_path)
    dest = tmp_path / 'policy/checkpoints/040000'
    dest.parent.mkdir(parents=True); shutil.move(root, dest)
    from soarm101_lab.learning.datasets import inspect_dataset
    artifact.update(inspect_dataset(artifact['path']))
    artifact.update(type='dataset', id='dataset', tasks=[{'task_id': 'pick', 'language_instruction': 'Pick the cube'}])
    registry=tmp_path/'workflow'
    monkeypatch.setattr('soarm101_lab.workflow.worker_request.stage_input',lambda *a,**kw:artifact)
    run = tmp_path / 'resume_run'; run.mkdir()
    plan = Operations(root=registry).plan('train',artifact['path'],'',
        {'resume_checkpoint': str(dest), 'steps': 100000, 'backend': backend}, run_dir=run)
    assert plan['output'] == str(tmp_path / 'policy')
    assert json.loads((run / 'request.json').read_text())['output'] == str(tmp_path / 'policy')




def test_periodic_minivla_checkpoint_is_not_replaced(tmp_path):
    from soarm101_lab.workflow.policy import selected_minivla_checkpoint
    from soarm101_lab.learning.training import resume_output
    root=tmp_path/'policy';(root/'checkpoints').mkdir(parents=True)
    for name in ('best.pt','last.pt','checkpoints/step_000002.pt'):(root/name).touch()
    (root/'config.json').write_text('{}')
    chosen=root/'checkpoints/step_000002.pt'
    assert selected_minivla_checkpoint(root,str(chosen))==chosen
    with pytest.raises(ValueError,match='비활성화'):resume_output(chosen,'minivla')
    with pytest.raises(ValueError):selected_minivla_checkpoint(root,str(tmp_path/'outside.pt'))
