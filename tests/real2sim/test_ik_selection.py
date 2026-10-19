"""Solver dispatch, unchanged native behavior and physics-frame contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import torch
from soarm101_lab.simulation.ik import (solve_pose,solve_selected,IKFailure,DLS_PARAMETERS,
                                      validate_solver,solver_configuration)
from soarm101_lab.so101_joint_mapping import JOINTS

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def kin():
    from soarm101_lab.utils.so101_kinematics import SO101Kinematics
    return SO101Kinematics(str(ROOT/'assets/SO101/urdf/so101_isaaclab.urdf'),'tool0',list(JOINTS[:5]))

@pytest.fixture
def limits(kin):
    return np.column_stack((kin.model.lowerPositionLimit,kin.model.upperPositionLimit))

@pytest.mark.parametrize('bad',['dls','scipy','',None,7])
def test_invalid_solver_fails_before_output_or_engine(tmp_path,bad):
    from soarm101_lab.workflow.unified import execute
    with pytest.raises(ValueError,match='Unknown ik_solver'):
        execute({'stage':'datagen','options':{'ik_solver':bad},'output':str(tmp_path/'out')})
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('solver',['pinocchio_dls','scipy_ls'])
def test_same_pose_seed_native_parity_and_gripper(kin,limits,solver):
    seed=np.array([.1,-.8,.9,.6,.1,.25]);goal=kin.forward_kinematics(seed[:5]+.01)
    command,info=solve_selected(kin.forward_kinematics,goal,seed,limits,solver=solver,kinematics=kin)
    if solver=='pinocchio_dls':
        raw,ok,native=kin.inverse_kinematics(seed[:5],goal,**DLS_PARAMETERS)
        np.testing.assert_array_equal(command[:5],raw if ok else seed[:5])
        assert info['ik_success']==ok and info['native_result']['weighted_error']==native['weighted_error']
    else:
        original,details=solve_pose(kin.forward_kinematics,goal,seed,limits)
        np.testing.assert_array_equal(command,original)
        assert info['position_error_m']==details['position_error_m']
    assert command[5]==seed[5] and not info['joint_limit_violation']
    assert info['ik_solver']==solver and info['ik_duration_ms']>0


def test_world_to_base_matches_native_and_no_fallback(kin,limits,monkeypatch):
    from scipy.spatial.transform import Rotation
    base=np.eye(4);base[:3,3]=[1,.4,.2];base[:3,:3]=Rotation.from_euler('xyz',[.2,.1,.7]).as_matrix()
    seed=np.array([.1,-.8,.9,.6,.1,.25]);goal=kin.forward_kinematics(seed[:5]+.02)
    a,_=solve_selected(kin.forward_kinematics,goal,seed,limits,solver='pinocchio_dls',kinematics=kin)
    b,_=solve_selected(lambda arm:base@kin.forward_kinematics(arm),base@goal,seed,limits,
                       solver='pinocchio_dls',kinematics=kin,base_from_frame=np.linalg.inv(base),frame='world')
    np.testing.assert_allclose(a,b,atol=1e-12)
    def forbidden(*args,**kwargs):raise AssertionError('Hidden SciPy fallback')
    monkeypatch.setattr('soarm101_lab.simulation.ik.solve_pose',forbidden)
    goal[:3,3]=[5,4,3]
    held,info=solve_selected(kin.forward_kinematics,goal,seed,limits,solver='pinocchio_dls',kinematics=kin)
    np.testing.assert_array_equal(held,seed)
    assert not info['ik_success'] and not info['common_pose_passed']
    assert info['failure_reason']=='native_nonconvergence' and not info['command_applied']


def test_scipy_failure_retains_common_diagnostics(kin,limits):
    seed=np.array([.1,-.8,.9,.6,.1,.25]);goal=np.eye(4);goal[:3,3]=[5,4,3]
    with pytest.raises(IKFailure) as caught:
        solve_selected(kin.forward_kinematics,goal,seed,limits)
    info=caught.value.details
    assert info['ik_solver']=='scipy_ls' and info['position_error_mm']>1000
    assert not info['common_pose_passed'] and info['ik_iterations_or_evaluations']>0

@pytest.mark.parametrize('count',[1,2,4])
def test_worker_local_pinocchio_data_and_mujoco_frame(tmp_path,count):
    from test_mujoco_backend import scene_inputs
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    workspace=scene_inputs()
    config={'identity_sim':True,'robot_model':{'urdf':'assets/SO101/urdf/so101_isaaclab.urdf'}}
    with MujocoBackend(workspace,simulation_configuration=config) as parent:
        workers=[MujocoBackend(shared_backend=parent) for _ in range(count)]
        try:
            for i,worker in enumerate(workers):
                seed=worker.home.copy();seed[0]+=.03*i
                goal=worker.target_pose(seed)
                command=worker.solve_pose(goal,seed,solver='pinocchio_dls')
                expected=worker.workspace_from_world@np.asarray(workspace['profile']['robot_base']['T_world'])@worker.kinematics.forward_kinematics(seed[:5])
                np.testing.assert_allclose(expected,goal,atol=1e-9)
                np.testing.assert_allclose(command,seed,atol=1e-9)
            assert len({id(w.kinematics.data) for w in workers})==count
            assert len({id(w.kinematics.model) for w in workers})==count
            assert len({id(w.data) for w in workers})==count
            assert len({id(w.model) for w in workers})==1
        finally:
            for worker in workers:worker.close()

@pytest.mark.parametrize('solver',['scipy_ls','pinocchio_dls'])
def test_isaac_adapter_dispatch_and_seed_policy(kin,limits,solver,tmp_path):
    from soarm101_lab.workflow.adapters import IsaacAdapter
    from collections import defaultdict
    adapter=object.__new__(IsaacAdapter)
    measured=np.array([.1,-.8,.9,.6,.1,.25]);previous=measured+.01
    adapter.native=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor(measured[None])))})
    adapter.index=0;adapter.ids=list(range(6));adapter.stop=threading.Event();adapter.step_index=0
    adapter.previous=previous;adapter.kinematics=kin;adapter.limits=limits
    adapter.metrics=defaultdict(float);adapter.ik_attempt=defaultdict(float);adapter.configure_ik(solver)
    goal=kin.forward_kinematics(measured[:5]);grip=.2
    actual=adapter.target_eef_pose_to_action({'tool0':torch.tensor(goal)},{'tool0':torch.tensor([grip],dtype=torch.float64)})
    seed=measured if solver=='pinocchio_dls' else previous
    np.testing.assert_allclose(adapter.last_ik['seed_joint_rad'],np.r_[seed[:5],grip])
    assert actual[5]==grip and adapter.last_ik['ik_solver']==solver
    assert adapter.metrics['ik_calls']==1 and adapter.ik_attempt['ik_calls']==1


def test_isaac_failed_scipy_call_is_counted(kin,limits):
    from soarm101_lab.workflow.adapters import IsaacAdapter
    from collections import defaultdict
    adapter=object.__new__(IsaacAdapter)
    adapter.stop=threading.Event();adapter.step_index=0;adapter.previous=np.array([.1,-.8,.9,.6,.1,.25])
    adapter.native=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor(adapter.previous[None])))})
    adapter.index=0;adapter.ids=list(range(6))
    adapter.kinematics=kin;adapter.limits=limits;adapter.ik_solver='scipy_ls'
    adapter.metrics=defaultdict(float);adapter.ik_attempt=defaultdict(float)
    goal=np.eye(4);goal[:3,3]=[5,4,3]
    with pytest.raises(IKFailure):
        adapter.target_eef_pose_to_action({'tool0':torch.tensor(goal)},{'tool0':torch.tensor([.2])})
    assert adapter.metrics['ik_calls']==1 and adapter.metrics['ik_s']>0
    assert adapter.ik_attempt['ik_pose_failures']==1


def test_mujoco_dls_hold_outside_limits_is_ik_failure_not_planning(tmp_path,monkeypatch):
    from test_mujoco_backend import scene_inputs
    from soarm101_lab.simulation.mujoco_backend import MujocoBackend
    config={'identity_sim':True,'robot_model':{'urdf':'assets/SO101/urdf/so101_isaaclab.urdf'}}
    with MujocoBackend(scene_inputs(),simulation_configuration=config) as backend:
        seed=backend.home.copy();backend.solve_pose(backend.target_pose(seed),seed,solver='pinocchio_dls')
        seed[0]=backend.limits[0,1]+.001
        monkeypatch.setattr(backend.kinematics,'inverse_kinematics',lambda *a,**kw:(seed[:5],False,{'iterations':200,'weighted_error':1.}))
        goal=np.eye(4);goal[:3,3]=[5,4,3]
        with pytest.raises(IKFailure) as caught:backend.solve_pose(goal,seed,solver='pinocchio_dls')
        assert caught.value.details['failure_reason']=='held_measured_joint_limit'
        assert caught.value.details['joint_limit_violation'] and not caught.value.details['command_applied']


@pytest.mark.parametrize('solver',['pinocchio_dls','scipy_ls'])
@pytest.mark.parametrize('policy',['strict','hold','best_effort'])
def test_failure_policy_separates_candidate_from_convergence(kin,limits,monkeypatch,solver,policy):
    seed=np.array([.1,-.8,.9,.6,.1,.25]);measured=seed-.005;candidate=seed[:5]+.02
    goal=kin.forward_kinematics(seed[:5]);goal[0,3]+=.02
    if solver=='pinocchio_dls':
        monkeypatch.setattr(kin,'inverse_kinematics',lambda *a,**kw:(candidate,False,{'iterations':200,'weighted_error':.02}))
    else:
        def result(*args,**kwargs):
            assert kwargs['reject_on_error'] is False
            return np.r_[candidate,seed[5]],dict(position_error_m=.02,rotation_error_rad=.2,
                position_tolerance_m=.003,rotation_tolerance_rad=.12,solver_status=1,evaluations=80)
        monkeypatch.setattr('soarm101_lab.simulation.ik.solve_pose',result)
    call=lambda:solve_selected(kin.forward_kinematics,goal,seed,limits,solver=solver,kinematics=kin,
        failure_policy=policy,measured=measured,previous_command=seed)
    if policy=='strict' and solver=='scipy_ls':
        with pytest.raises(IKFailure):call()
        return
    command,info=call()
    assert info['ik_converged']==(solver=='scipy_ls')
    assert info['ik_numerical_failure']
    np.testing.assert_array_equal(command[:5],candidate if policy=='best_effort' else measured[:5])
    assert command[5]==seed[5] and info['ik_candidate_applied']==(policy=='best_effort')
    assert info['ik_held']==(policy!='best_effort')


@pytest.mark.parametrize('bad,reason',[(np.full(5,np.nan),'candidate_nonfinite'),(np.zeros(4),'candidate_shape'),
                                      (np.full(5,99.),'joint_limit_violation'),(np.array([.9,-.8,.9,.6,.1]),'candidate_jump')])
def test_best_effort_rejects_invalid_candidates_without_clipping(kin,limits,monkeypatch,bad,reason):
    seed=np.array([.1,-.8,.9,.6,.1,.25]);goal=kin.forward_kinematics(seed[:5])
    monkeypatch.setattr(kin,'inverse_kinematics',lambda *a,**kw:(bad,False,{'iterations':200,'weighted_error':1.}))
    with pytest.raises(IKFailure) as caught:
        solve_selected(kin.forward_kinematics,goal,seed,limits,solver='pinocchio_dls',kinematics=kin,
                       failure_policy='best_effort',measured=seed,previous_command=seed)
    info=caught.value.details
    assert info['ik_invalid_candidate'] and not info['ik_candidate_applied']
    assert info['failure_reason']==reason and info['applied_joint_rad'] is None


def test_unknown_failure_policy_rejected_before_engine(tmp_path):
    from soarm101_lab.workflow.unified import execute
    with pytest.raises(ValueError,match='Unknown ik_failure_policy'):
        execute({'stage':'datagen','options':{'ik_failure_policy':'fallback'},'output':str(tmp_path/'out')})
    assert not (tmp_path/'out').exists()
