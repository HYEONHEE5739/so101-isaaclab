"""Caller-visible request and byte identity contracts, characterized before extraction."""
import hashlib
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soarm101_lab.utils.artifact_identity import fingerprint
from soarm101_lab.workflow.operations import mujoco_request


@pytest.mark.parametrize('stage', ['convert', 'annotate', 'datagen'])
def test_worker_request_preserves_payload_and_option_ownership(tmp_path, stage):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'manifest.json').write_text(json.dumps({'cameras': ['side']}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'cameras': ['side']}))
    options = {'repo_id': 'local/example', 'seed': 1047, 'nested': {'value': 1}}
    before = json.loads(json.dumps(options))
    result = mujoco_request(stage, sys.executable, '', str(dataset), str(config), '', 'none', options)
    assert result == {'stage': stage, 'workspace': '', 'input': str(dataset), 'config': str(config),
                      'checkpoint': '', 'renderer': 'none', 'options': before}
    assert result['options'] is options
    assert options == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.json', 'dataset']


@pytest.mark.parametrize(('change', 'message'), [
    ({'stage': 'unknown'}, 'Unsupported workflow worker stage'),
    ({'options': []}, 'Options must be a JSON object'),
    ({'python': '/missing/python'}, 'Worker Python executable missing'),
    ({'renderer': 'unknown'}, 'Unknown renderer'),
    ({'stage': 'source', 'renderer': 'none'}, 'This stage requires RGB rendering'),
    ({}, 'Input must be a canonical MiniVLA dataset directory'),
])
def test_worker_request_validation_order_and_messages(change, message):
    args = dict(stage='convert', python=sys.executable, workspace='/missing/workspace',
                input_path='/missing/dataset', config='', checkpoint='', renderer='none', options={})
    args.update(change)
    with pytest.raises(ValueError) as error:
        mujoco_request(**args)
    assert str(error.value) == message


def test_custom_training_camera_contract_and_convert_id_are_still_required(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'cameras': ['side']}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'cameras': ['wrist']}))
    with pytest.raises(ValueError, match='Unsupported workflow worker stage'):
        mujoco_request('train', sys.executable, '', str(tmp_path), str(config), '', 'none', {})
    with pytest.raises(ValueError, match='Dataset repo_id required'):
        mujoco_request('convert', sys.executable, '', str(tmp_path), '', '', 'none', {})


def test_identity_is_location_independent_but_keeps_names_bytes_and_cache_rules(tmp_path):
    expected = hashlib.sha256(b'a.binABnested/b.binCD').hexdigest()
    roots = [tmp_path / 'one', tmp_path / 'two']
    for root in roots:
        (root / 'nested').mkdir(parents=True)
        (root / 'a.bin').write_bytes(b'AB')
        (root / 'nested/b.bin').write_bytes(b'CD')
        (root / '.cache').mkdir()
        (root / '.cache/ignored').write_bytes(b'cache')
        assert fingerprint(root) == expected
        assert fingerprint(root / 'a.bin') == hashlib.sha256(b'a.binAB').hexdigest()
    (roots[0] / '.cache/ignored').write_bytes(b'new cache')
    assert fingerprint(roots[0]) == expected
    (roots[0] / 'nested/b.bin').write_bytes(b'CE')
    assert fingerprint(roots[0]) != expected
    (roots[1] / 'a.bin').rename(roots[1] / 'renamed.bin')
    assert fingerprint(roots[1]) != expected
    with pytest.raises(ValueError, match='Empty artifact'):
        fingerprint(tmp_path / 'missing')


def test_hdf5_identity_includes_referenced_videos_once_and_rejects_escape(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'dataset.hdf5'
    video = tmp_path / 'videos/side.mp4'
    video.parent.mkdir()
    video.write_bytes(b'video fixture; no decoding in fingerprint')
    with h5py.File(path, 'w') as h:
        for index in range(2):
            g = h.create_group(f'data/demo_{index}')
            g.attrs['external_videos'] = json.dumps({'side': {'file': 'videos/side.mp4'}})
    expected = hashlib.sha256(b'dataset.hdf5' + path.read_bytes() + b'videos/side.mp4' + video.read_bytes())
    assert fingerprint(path) == expected.hexdigest()
    video.write_bytes(b'changed')
    assert fingerprint(path) != expected.hexdigest()
    video.unlink()
    with pytest.raises(FileNotFoundError):
        fingerprint(path)
    with h5py.File(path, 'a') as h:
        h['data/demo_0'].attrs['external_videos'] = json.dumps({'side': {'file': '../outside.mp4'}})
    with pytest.raises(ValueError, match='External video outside dataset parent'):
        fingerprint(path)


def test_existing_import_paths_are_the_same_interface():
    from soarm101_lab.utils.artifact_identity import fingerprint as identity
    from soarm101_lab.workflow import operations, worker_request
    assert fingerprint is identity
    assert operations.mujoco_request is worker_request.mujoco_request
    assert operations.MUJOCO_STAGES is worker_request.MUJOCO_STAGES


@pytest.mark.parametrize('module', [
    'soarm101_lab.workflow.worker_request', 'soarm101_lab.utils.artifact_identity',
])
def test_shared_interfaces_do_not_load_orchestrator_ui_or_engines(module):
    code = '''
import importlib,sys
importlib.import_module(sys.argv[1])
assert not any(n == 'soarm101_lab.workflow.operations' or
    n.split('.')[0] in {'PyQt6','torch','isaaclab','isaacsim','omni','pxr','mujoco'} for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, module], env=dict(os.environ), check=True,
                   capture_output=True, text=True)


def test_request_worker_and_identity_import_direction():
    """Architectural rule only: behavior is covered by the interface tests above."""
    package = Path(__file__).resolve().parents[2] / 'source/soarm101_lab/soarm101_lab'
    worker = ast.parse((package / 'workflow/mujoco.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'operations' for n in ast.walk(worker))
    for directory in (package / 'workflow', package / 'learning/minivla'):
        for path in directory.glob('*.py'):
            imports = (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom))
            assert not any((n.module or '').endswith('artifacts') and any(a.name == 'fingerprint' for a in n.names)
                           for n in imports), path


@pytest.mark.parametrize('module', [
    'soarm101_lab.simulation.control', 'soarm101_lab.workflow.policy',
    'soarm101_lab.real2sim.workspaces.tasks',
])
def test_shared_physics_and_policy_do_not_import_mimic_or_model_training(module):
    code = '''
import importlib,sys
importlib.import_module(sys.argv[1])
assert not any(n.startswith(('soarm101_lab.learning.minivla',
                           'soarm101_lab.workflow.mimic')) or
    n == 'soarm101_lab.workflow.tasks' or
    n.split('.')[0] in {'PyQt6','torch','isaaclab','isaacsim','omni','pxr','mujoco'}
    for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, module], env=dict(os.environ), check=True,
                   capture_output=True, text=True)


def test_minivla_inference_import_does_not_load_training_or_dataset_preparation():
    code = '''
import sys
from soarm101_lab.learning.minivla.policy import MiniVLA
assert 'soarm101_lab.learning.minivla.train' not in sys.modules
assert 'soarm101_lab.learning.minivla.data' not in sys.modules
assert not any(n.startswith('soarm101_lab.workflow') for n in sys.modules)
assert not any(n.split('.')[0] in {'isaaclab','isaacsim','omni','pxr','mujoco','PyQt6'}
               for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], env=dict(os.environ), check=True,
                   capture_output=True, text=True)
