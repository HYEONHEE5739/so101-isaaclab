"""Project adapters: optional imports, coordinate contract, data lineage and replay."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from soarm101_lab.workflow.operations import ROOT, mujoco_request
from soarm101_lab.so101_joint_mapping import RobotCoordinates
from soarm101_lab.simulation.mujoco_backend import PolicyEnvironment


def test_coordinates_roundtrip_and_checked_saturation():
    coordinates = RobotCoordinates()
    q = np.array([[.1, .2, .3, .4, .5, 0], [.2, .3, .4, .5, .6, .8]])
    np.testing.assert_allclose(coordinates.to_policy(coordinates.to_model(q)), q, atol=1e-7)
    raw = np.zeros(6); raw[5] = -.005
    assert coordinates.to_policy(raw, measured=True)[5] == 0
    assert coordinates.saturated_states == 1
    with pytest.raises(ValueError): coordinates.to_policy(raw)
    with pytest.raises(ValueError): coordinates.to_model([0] * 5 + [1.1])
    with pytest.raises(ValueError): RobotCoordinates(open_rad=0)


def test_source_uses_saved_precision_and_records_model_frame(tmp_path,env,monkeypatch):
    monkeypatch.setattr('soarm101_lab.utils.voice.log_say',lambda *a,**k:None)
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.real2sim import hardware, precision
    from soarm101_lab.real2sim.workspaces import demo
    from soarm101_lab.so101_dataset_contract import current_contract,MIMIC_SPACE
    from soarm101_lab.so101_joint_mapping import JOINTS
    import hashlib,math
    spec=precision.FakeCalibrationDevice('leader').motor_spec('shoulder_pan')
    refs=[dict(joint='shoulder_pan',raw=r,angle_rad=math.radians(q),split='train',calibration_hash='fake')
          for r,q in [(1500,0),(2500,90)]]
    entry=precision.review_joint(precision.fit_joint(refs,spec,'shoulder_pan',[-1.91986,1.91986]))
    profile=dict(schema=precision.SCHEMA,id='fixture_active',joints={'shoulder_pan':entry},role='leader',
        urdf_sha256=hashlib.sha256((ROOT/'assets/SO101/urdf/so101_isaaclab.urdf').read_bytes()).hexdigest())
    mapper=precision.PrecisionMapper(profile)
    class Leader:
        precision_mapper=mapper
        metadata={'fixture':True,'precision_mapping':mapper.metadata()}
        def __init__(self,*args,model_hash=None):
            assert model_hash == profile["urdf_sha256"]
        def connect(self):pass
        def activate(self):pass
        def close(self):pass
        def read(self):
            native={n:0. for n in JOINTS};native['shoulder_pan']=precision.native_from_raw(2600,spec)
            return native,{}
    class Bridge:
        pending=set()
        polls=0
        def poll(self):
            self.polls+=1
            if self.polls>1:self.pending.add('save')
    monkeypatch.setattr(precision,'approved_mappings',lambda workspace:{'leader':mapper.metadata()})
    monkeypatch.setattr(hardware,'Leader',Leader)
    monkeypatch.setattr(worker,'load',lambda path:{'root':ROOT})
    monkeypatch.setattr(demo,'coordinate_contract',lambda w:current_contract(MIMIC_SPACE))
    def environment(request,*args):
        assert request['options']['_collection_mappings']['leader']==mapper.metadata()
        return env
    monkeypatch.setattr(worker,'_environment',environment)
    request={'workspace':'fixture','options':{'port':'fixture','instruction':'pick red','episodes':1,'episode_time_s':2}}
    worker.source(request,tmp_path,lambda *args:None,Bridge())
    actions=np.load(tmp_path/'dataset/episode_0000/raw_action.npy')
    assert actions[0,0]==pytest.approx(math.radians(99))
    provenance=json.loads((tmp_path/'dataset/project_provenance.json').read_text())
    assert precision.uses_urdf_frame(provenance['coordinate_contract'])
    assert provenance['coordinate_contract']['hardware_mapping']==mapper.metadata()


@pytest.fixture
def env(tmp_path):
    from soarm101_lab.simulation.mujoco_backend import configure_rendering
    configure_rendering('none')
    pytest.importorskip('mujoco')
    profile = json.loads((ROOT / 'tests/real2sim/fixtures/scene_profile.json').read_text())
    backend = PolicyEnvironment(tmp_path / 'scene', {'profile': profile}, renderer='none', control_hz=30)
    backend.backend.get_rgb = lambda camera=None: np.zeros((240, 320, 3), np.uint8)
    yield backend
    backend.close()


def make_dataset(root, env, smoke=False):
    from soarm101_lab.workflow.episodes import EpisodeWriter
    writer = EpisodeWriter(root, env, 'pick the red cube')
    for i in range(3):
        env.reset(i)
        writer.begin()
        for _ in range(4):
            obs = env.observe()
            target = env.backend.home.copy()
            writer.append(obs, target)
            env.backend.step(target)
        writer.save(success=True, validation_only=smoke)  # fixture labels; not a manipulation benchmark
    return writer


def test_custom_policy_chunk_drives_real_cpu_environment(tmp_path, env, monkeypatch):
    from types import SimpleNamespace
    from soarm101_lab.workflow import mujoco as worker
    checkpoint = tmp_path / 'policy.pt'
    checkpoint.touch()
    (tmp_path / 'project_provenance.json').write_text(json.dumps({'coordinates': env.coordinates.metadata()}))
    monkeypatch.setattr(worker, '_verify_workspace', lambda *args: None)
    monkeypatch.setattr(worker, '_environment', lambda *args: env)
    class Policy:
        cfg = SimpleNamespace(cameras=('side',), horizon=16, replan_steps=4)
        control_hz = 30
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
        def predict(self, images, state, instruction):
            assert images[0].shape == (240, 320, 3)
            return np.tile(state, (16, 1)), {'fixture_policy': True}
    request = {'checkpoint': str(checkpoint), 'options': {'instruction': 'pick the cube', 'max_steps': 6}}
    result = worker.rollout(request, tmp_path, lambda *args: None, None, policy_factory=Policy)
    assert result['passed'] and result['episodes'][0]['steps'] == 6
    trace = json.loads((tmp_path / 'episode_0000.json').read_text())['trace']
    assert len(trace) == 6 and trace[-1]['timestamp_s'] > trace[0]['timestamp_s']
    assert all(len(row['canonical_action']) == 6 for row in trace)


def test_source_discard_save_quit_with_fake_leader(tmp_path, env, monkeypatch):
    spoken=[]
    monkeypatch.setattr('soarm101_lab.utils.voice.log_say',lambda text,**k:spoken.append(text))
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.real2sim import hardware
    from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER, JOINTS
    closed = []
    requested = env.backend.home.copy()
    requested[2] = env.backend.limits[2, 1] + .015
    requested[5] = -.02
    class Leader:
        def __init__(self, *args, model_hash=None): assert model_hash is None
        def connect(self): pass
        def activate(self): pass
        def close(self): closed.append(True)
        def read(self):
            return DEFAULT_MAPPER.urdf_to_calibrated(dict(zip(JOINTS, requested))), {}
    class Bridge:
        pending = set()
        commands = iter([None, 'discard', None, 'save', None, 'quit'])
        def poll(self):
            command = next(self.commands, 'quit')
            if command: self.pending.add(command)
    monkeypatch.setattr(hardware, 'Leader', Leader)
    monkeypatch.setattr(worker, '_environment', lambda *args: env)
    monkeypatch.setattr(worker.time, 'sleep', lambda *args: None)
    request = {'options': {'port': 'fixture', 'instruction': 'pick red', 'episodes': 2, 'episode_time_s': 10}}
    report = worker.source(request, tmp_path, lambda *args: None, Bridge())
    assert report['episode_count'] == 1 and closed == [True]
    assert 'resetting' in spoken and 'save episode 1' in spoken and spoken[-1]=='stop recording'
    manifest = json.loads((tmp_path / 'dataset/manifest.json').read_text())
    assert len(manifest['episodes']) == 1
    assert np.load(tmp_path / 'dataset/episode_0000/action.npy').shape == (1, 6)
    np.testing.assert_allclose(np.load(tmp_path / 'dataset/episode_0000/requested_action.npy')[0], requested)
    applied = np.load(tmp_path / 'dataset/episode_0000/raw_action.npy')[0]
    assert applied[2] == env.backend.limits[2, 1] and applied[5] == 0
    assert report['leader_target_limits']['clamped_samples_by_joint']['elbow_flex'] > 0


def test_backend_preserves_valid_targets_and_rejects_nonfinite(env):
    applied=env.backend._target(env.backend.home)
    np.testing.assert_array_equal(applied,env.backend.home)
    with pytest.raises(ValueError): env.backend._target([float('nan')]*6)


def test_capture_alignment_prepare_group_split_and_smoke_rejection(tmp_path, env):
    from soarm101_lab.learning.minivla.data import load_manifest, episode_arrays
    from soarm101_lab.workflow.episodes import prepare_dataset
    writer = make_dataset(tmp_path / 'source', env)
    manifest = load_manifest(writer.root / 'manifest.json')
    arrays = episode_arrays(manifest['episodes'][0], ['side'], 30)
    np.testing.assert_allclose(np.diff(arrays['timestamp']), 1 / 30)
    np.testing.assert_array_equal(arrays['timestamp'], arrays['action_timestamp'])
    prepare_dataset(writer.root, tmp_path / 'prepared')
    prepared = load_manifest(tmp_path / 'prepared/manifest.json')
    assert {e['split'] for e in prepared['episodes']} == {'train', 'val', 'test'}
    assert (tmp_path / 'prepared/statistics.json').is_file()
    with pytest.raises(ValueError, match='Source group leakage'):
        changed = json.loads((writer.root / 'manifest.json').read_text())
        changed['episodes'][1]['group_id'] = changed['episodes'][0]['group_id']
        (writer.root / 'manifest.json').write_text(json.dumps(changed))
        prepare_dataset(writer.root, tmp_path / 'leaky', {'episode_0000': 'train', 'episode_0001': 'val', 'episode_0002': 'test'})
    smoke = make_dataset(tmp_path / 'smoke', env, smoke=True)
    with pytest.raises(ValueError, match='Synthetic'):
        prepare_dataset(smoke.root, tmp_path / 'bad_train')




def test_recorded_state_actions_replay_exactly(tmp_path, env):
    writer = make_dataset(tmp_path / 'source', env)
    for entry in writer.manifest['episodes']:
        folder = writer.root / entry['path']
        env.backend.restore_state(json.loads((folder / 'replay.json').read_text()))
        raw = np.load(folder / 'raw_state.npy')
        actions = np.load(folder / 'raw_action.npy')
        for i, target in enumerate(actions):
            np.testing.assert_allclose(env.backend.get_state()['joint_pos'], raw[i], atol=1e-12)
            env.backend.step(target)
    assert env.backend.metadata['control_timestep_s'] == pytest.approx(1 / 30)


def test_service_request_and_dataset_errors(tmp_path):
    with pytest.raises(ValueError, match='canonical'):
        mujoco_request('train', sys.executable, '', str(tmp_path), '', '', 'egl', {})
    with pytest.raises(ValueError, match='canonical'):
        mujoco_request('datagen', sys.executable, '', '', '', '', 'egl', {})
    with pytest.raises(ValueError, match='Options'):
        mujoco_request('environment', sys.executable, '', '', '', '', 'egl', [])


def test_service_import_has_no_optional_engines_or_models():
    code = '''
import sys
from soarm101_lab.workflow.operations import mujoco_request
assert not any(n.split('.')[0] in ('torch','minivla','mujoco','isaaclab','omni','pxr') for n in sys.modules)
'''
    p = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/soarm101_lab')},
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr


def test_training_worker_uses_integrated_module_and_clean_source_path(tmp_path, env, monkeypatch):
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.workflow.episodes import prepare_dataset
    writer = make_dataset(tmp_path / 'source', env)
    prepared = tmp_path / 'prepared'
    prepare_dataset(writer.root, prepared)
    output = tmp_path / 'training'; output.mkdir()
    monkeypatch.setenv('PYTHONPATH', '/gui/venv/site-packages')
    def launch(argv, **kwargs):
        assert argv[:3] == [sys.executable, '-m', 'soarm101_lab.learning.minivla.train']
        assert kwargs['cwd'] == ROOT
        assert kwargs['env']['PYTHONPATH'] == str(ROOT / 'source/soarm101_lab')
        config = Path(argv[argv.index('--config') + 1])
        assert json.loads(config.read_text())['layers'] == 4
        assert json.loads(config.read_text())['save_freq'] == 7
        raise RuntimeError('validated launch before GPU training')
    monkeypatch.setattr(worker.subprocess, 'run', launch)
    with pytest.raises(RuntimeError, match='validated launch'):
        worker.train({'input': str(prepared), 'options': {'save_freq':7},
                      'config': str(worker.MODEL_ROOT / 'configs/default.json')}, output)


def test_leader_import_no_isaac_dependencies():
    code = '''
import sys
from soarm101_lab.devices import SO101Leader
assert not any(n.split('.')[0] in ('isaaclab','omni','pxr') for n in sys.modules)
'''
    p = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT / 'source/soarm101_lab')},
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr


def test_ik_is_bounded_and_does_not_change_physics(env):
    backend = env.backend
    before = backend._snapshot()
    goal = backend.target_pose(backend.home)
    result = backend.solve_pose(goal, backend.home)
    np.testing.assert_allclose(result, backend.home, atol=1e-6)
    assert backend._snapshot() == before
    goal[:3, 3] += 10
    with pytest.raises(ValueError, match='Unreachable'):
        backend.solve_pose(goal, backend.home)
    assert backend._snapshot() == before






def test_canonical_registry_and_resume_preserve_original(tmp_path, env):
    from soarm101_lab.workflow.artifacts import Registry, fingerprint
    from soarm101_lab.workflow.episodes import EpisodeWriter
    writer = make_dataset(tmp_path / 'source', env)
    registry = Registry(tmp_path / 'registry')
    artifact = registry.register('source', writer.root)
    assert artifact['backend'] == 'mujoco'
    before = fingerprint(writer.root)
    resumed = EpisodeWriter(tmp_path / 'resumed', env, 'pick the red cube')
    resumed.resume(writer.root)
    assert len(resumed.manifest['episodes']) == 3
    assert fingerprint(writer.root) == before
    registry.verify(artifact)


def test_canonical_curation_preserves_lineage_and_rejects_bad_annotation(tmp_path, env):
    from soarm101_lab.workflow.curation import analyze, export_subset
    from soarm101_lab.workflow.artifacts import inspect_artifact, fingerprint
    from soarm101_lab.workflow.episodes import prepare_dataset
    writer = make_dataset(tmp_path / 'source', env)
    original = fingerprint(writer.root)
    rows = analyze(writer.root)
    assert len(rows) == 3 and all(r['frames'] == 4 for r in rows)
    subset = export_subset(writer.root, tmp_path / 'subset', ['episode_0002'])
    assert fingerprint(writer.root) == original
    entry = json.loads((subset / 'manifest.json').read_text())['episodes'][0]
    assert entry['group_id'] == writer.manifest['episodes'][2]['group_id']
    assert inspect_artifact('source', subset)['episode_count'] == 1
    prepare_dataset(subset, tmp_path / 'single_prepared', require_validation=False)
    with pytest.raises(ValueError): prepare_dataset(subset, tmp_path / 'mini_prepared')
    (subset / entry['path'] / 'annotation.json').write_text(json.dumps({
        'schema': 'so101.mujoco-subtasks/1', 'frame_count': 4, 'grasp': 3, 'place': 1}))
    with pytest.raises(ValueError, match='ordered'): inspect_artifact('annotated', subset)


def test_datagen_rejects_unsuccessful_physical_attempt():
    from soarm101_lab.workflow.mimic import GenerationScheduler
    scheduler=GenerationScheduler(1,1,1);commits=[]
    attempt=scheduler.claim(0)
    scheduler.finish(0,attempt,'grasp_failure',{'steps':30},lambda:commits.append(True))
    assert scheduler.successes==0 and not commits
    assert scheduler.progress()['failure_breakdown']=={'grasp_failure':1}
    assert scheduler.claim(0) is None


def test_canonical_hub_upload_has_lerobot_at_repository_root(tmp_path):
    from types import SimpleNamespace
    from soarm101_lab.workflow.hub import publish
    root = tmp_path / 'dataset'; (root / 'lerobot/meta').mkdir(parents=True)
    (root / 'lerobot/meta/info.json').write_text('{}')
    uploads = []
    class API:
        def repo_info(self, **kwargs): return SimpleNamespace(private=True)
        def upload_folder(self, **kwargs):
            uploads.append(kwargs['folder_path'])
            return SimpleNamespace(oid='fixture-commit')
        def upload_file(self, **kwargs): return SimpleNamespace(oid='fixture-commit', commit_url='fixture')
    artifact = {'id': 'fixture', 'type': 'dataset', 'format': 'so101.canonical/1', 'path': str(root)}
    publish(artifact, {'repo_id': 'fixture/so101', 'private': True}, tmp_path / 'publication.json', api=API())
    assert uploads == [str(root / 'lerobot')]


@pytest.mark.parametrize('measured_limits',[None,[-.174533,1.74533]])
def test_common_policy_facade_maps_and_guards_minivla_actions(tmp_path, monkeypatch, measured_limits):
    from types import SimpleNamespace
    import torch
    import soarm101_lab.learning.minivla.policy as model_policy
    from soarm101_lab.workflow.policy import MiniVLAJointPolicy, observation
    coordinates = RobotCoordinates(measured_gripper_limits=measured_limits)
    (tmp_path / 'project_provenance.json').write_text(json.dumps({
        'coordinates': coordinates.metadata(), 'joint_limits_rad': [[-2, 2]] * 5 + [[0, 2]]}))
    (tmp_path/'best.pt').write_bytes(b'fixture; model loader replaced below')
    calls = []
    class Policy:
        cfg = SimpleNamespace(cameras=('side',), horizon=4, replan_steps=2)
        control_hz = 30
        def __init__(self, *args): pass
        def predict(self, images, state, instruction):
            calls.append(state.copy())
            return np.tile(np.r_[state[:5], 1.5], (4, 1)), {}
    monkeypatch.setattr(model_policy, 'MiniVLA', Policy)
    policy = MiniVLAJointPolicy(tmp_path, 'cpu')
    q = coordinates.to_model(np.r_[np.zeros(5), .5])
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    batch = observation(q, rgb, rgb, 'pick red', 'cpu')
    first = policy.select_action(batch); policy.select_action(batch)
    assert first.shape == (6,) and torch.isfinite(first).all() and len(calls) == 1
    assert .5 < coordinates.to_policy(first.numpy())[5] < .6
    assert policy.guard.violations == 2
    policy.reset(); policy.select_action(batch)
    assert len(calls) == 2
    if measured_limits:
        policy.reset();q[5]=-.04528482
        policy.select_action(observation(q,rgb,rgb,'pick red','cpu'))
        assert calls[-1][5]==0 and policy.coordinates.measured_gripper_limits==tuple(measured_limits)
        policy.reset();q[5]=-.3
        with pytest.raises(ValueError,match='recorded model limits'):
            policy.select_action(observation(q,rgb,rgb,'pick red','cpu'))


def test_front_camera_dataset_and_policy_contract(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_builder_workspace import fruit_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    from soarm101_lab.workflow.episodes import EpisodeWriter
    from soarm101_lab.workflow import mujoco as worker
    from soarm101_lab.learning.minivla.data import load_manifest,episode_arrays
    export=fruit_export.__wrapped__(tmp_path);workspace=import_scene(export,registry=tmp_path/'workspaces')
    env=PolicyEnvironment(tmp_path/'front_scene',workspace,'none',control_hz=30)
    env.backend.get_rgb=lambda camera=None:np.zeros((240,320,3),np.uint8)
    writer=EpisodeWriter(tmp_path/'front_dataset',env,'Place the apple in the basket.')
    writer.begin()
    for _ in range(4):
        obs=env.observe();writer.append(obs,env.backend.home);env.backend.step(env.backend.home)
    writer.save(False,validation_only=True)
    manifest=load_manifest(writer.root/'manifest.json')
    arrays=episode_arrays(manifest['episodes'][0],['front'],30)
    assert arrays['front'].shape==(4,240,320,3)
    assert not (writer.root/'episode_0000/side.npy').exists()
    checkpoint=tmp_path/'fixture_policy.pt';checkpoint.touch()
    (tmp_path/'project_provenance.json').write_text(json.dumps({'coordinates':env.coordinates.metadata()}))
    monkeypatch.setattr(worker,'_verify_workspace',lambda *args:None)
    monkeypatch.setattr(worker,'_environment',lambda *args:env)
    class Policy:
        cfg=SimpleNamespace(cameras=('front',),horizon=4,replan_steps=2)
        control_hz=30
        def __init__(self,*args):pass
        def predict(self,images,state,instruction):
            assert len(images)==1 and images[0].shape==(240,320,3)
            return np.tile(state,(4,1)),{'fixture_only':True}
    result=worker.rollout({'checkpoint':str(checkpoint),'options':{'instruction':'Place the apple in the basket.','max_steps':4}},tmp_path,lambda *args:None,None,policy_factory=Policy)
    assert result['passed'] and result['episodes'][0]['steps']==4
