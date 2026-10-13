from soarm101_lab.workflow.evaluation import episode_result, log_evaluation


def test_real_policy_execution_uses_simple_urdf_path_without_real_devices(tmp_path,monkeypatch):
    import json,time
    import numpy as np
    import torch
    from types import SimpleNamespace
    from pathlib import Path
    from soarm101_lab.workflow import evaluation,tasks,presentation
    from soarm101_lab.real2sim import hardware,cameras
    from soarm101_lab import so101_dataset_contract
    from soarm101_lab.so101_joint_mapping import JOINTS,DEFAULT_MAPPER
    from soarm101_lab.so101_robot_spec import joint_limits
    goals=[];closed=[]
    class FakeFollower:
        metadata={}
        def __init__(self,*args,**kw):
            self.limits={n:[0,100] if n=='gripper' else [-180,180] for n in JOINTS}
        def connect(self):pass
        def activate(self):pass
        def read(self):return dict.fromkeys(JOINTS,0.),{}
        def encode_goal(self,native):return native,native
        def send(self,native):goals.append(native);return native,{}
        def close(self):closed.append('follower')
    class FakeCamera:
        def __init__(self,*args):pass
        def start(self):pass
        def nearest(self):return {'read_end':{'monotonic_ns':time.monotonic_ns()},'rgb':np.zeros((8,8,3),np.uint8)}
        def close(self):closed.append('camera')
    root=Path(__file__).resolve().parents[2]
    w={'root':root,'profile':{'cameras':{'front':{'resolution':[8,8]}}},'task':{'observation_cameras':['front']}}
    action=torch.tensor([2.5,0,0,0,0,.5])
    policy=SimpleNamespace(reset=lambda:None,select_action=lambda batch:action)
    monkeypatch.setattr(evaluation,'load',lambda *args:w)
    monkeypatch.setattr(evaluation,'metadata',lambda *args:{'fixture':True})
    monkeypatch.setattr(evaluation,'coordinate_contract',lambda *args:{})
    monkeypatch.setattr(evaluation,'load_policy',lambda *args:(policy,lambda x:x,lambda x:x))
    monkeypatch.setattr(evaluation,'observation',lambda *args,**kw:{})
    monkeypatch.setattr(evaluation.time,'sleep',lambda *args:None)
    monkeypatch.setattr(presentation,'current',lambda:None)
    monkeypatch.setattr(tasks,'validate',lambda t,w:t)
    monkeypatch.setattr(so101_dataset_contract,'read_contract',lambda *args:{})
    monkeypatch.setattr(hardware,'Follower',FakeFollower)
    monkeypatch.setattr(cameras,'CameraReader',FakeCamera)
    output=tmp_path/'evaluation.json'
    request={'stage':'real_eval','workspace':'fixture','output':str(output),
        'input':{'id':'fixture','type':'policy','path':'fixture'},'options':{'enable_motion':True,'device':'cpu',
        'port':'/fake','follower_id':'fixture','camera_devices':{'front':'/fake/front'},'max_steps':1,
        'tasks':[{'task_id':'fixture','language_instruction':'fixture'}]}}
    evaluation.evaluate(request)
    result=json.loads(output.read_text())
    assert len(goals)==1 and set(closed)=={'follower','camera'}
    q=DEFAULT_MAPPER.calibrated_to_urdf(goals[0])
    assert abs(q['shoulder_pan']-joint_limits()['shoulder_pan'][1])<1e-6
    np.testing.assert_allclose(result['steps'][0]['model_action'],list(q.values()),atol=1e-12)
    assert result['steps'][0]['requested_model_action'][0]==2.5
    goals.clear();closed.clear();action[0]=float('nan')
    import pytest
    with pytest.raises(ValueError,match='Non-finite'):evaluation.evaluate(request)
    assert not goals and set(closed)=={'follower','camera'}


def test_failure_and_success_rates(capsys):
    task = {'task_id': 'red_a', 'language_instruction': 'pick'}
    bad = episode_result(task, 0, 300, False, {
        'height_ok': [True], 'wall_ok': [False], 'speed_ok': [False]})
    assert bad['failure_conditions'] == ['컵 안쪽 벽', '정지 속도']
    good = episode_result(task, 1, 100, True, {
        'height_ok': [True], 'wall_ok': [True], 'speed_ok': [True]})
    assert good['failure_conditions'] == []
    log_evaluation([bad], 'red_a')
    log_evaluation([bad, good], 'red_a')
    text = capsys.readouterr().out
    assert '미충족: 컵 안쪽 벽, 정지 속도' in text
    assert '1/2 (50.0%)' in text


def test_evaluation_seed_modes(monkeypatch):
    import pytest
    from soarm101_lab.workflow.evaluation import evaluation_seed
    assert evaluation_seed({}) == 0
    assert evaluation_seed({'eval_seed': 42}) == 42
    monkeypatch.setattr('secrets.randbelow', lambda n: 123)
    assert evaluation_seed({'eval_seed_mode': 'random'}) == 123
    for value in (-1, True, '42', 2**31):
        with pytest.raises(ValueError):
            evaluation_seed({'eval_seed': value})
