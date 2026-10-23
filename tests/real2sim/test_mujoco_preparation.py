"""Worker input policy remains independent of engine construction."""
import copy
import json
from types import SimpleNamespace

import pytest

from soarm101_lab.workflow.worker_request import prepare_mujoco


@pytest.mark.parametrize('stage', ['environment','source','annotate','datagen','rollout'])
@pytest.mark.parametrize('resume', [False,True])
@pytest.mark.parametrize('recorded_scene', [None, {'reset':{'mode':'recorded'}}])
def test_preparation_preserves_contract_reset_and_profile(stage,resume,recorded_scene,tmp_path,monkeypatch):
    from soarm101_lab.workflow import worker_request,dataset
    from soarm101_lab.real2sim import precision
    from soarm101_lab.real2sim.workspaces import tasks,reset
    from soarm101_lab import so101_dataset_contract as contract
    original={'scene_configuration':{'reset':{'mode':'base'}}}
    calls=[]
    def configure(w,*args,**kw):
        calls.append(('configure',kw));return copy.deepcopy(w)
    def source(w,resume):
        calls.append(('source',resume));return w
    recorded={'simulation_model':{'scene_configuration':recorded_scene}}
    path=tmp_path/'source.hdf5';path.touch()
    monkeypatch.setattr(contract,'read_contract',lambda p:recorded)
    monkeypatch.setattr(tasks,'configured',configure)
    monkeypatch.setattr(reset,'source_workspace',source)
    monkeypatch.setattr(worker_request,'load',lambda p:pytest.fail('Validated Workspace reloaded'))
    monkeypatch.setattr(dataset,'recording_provenance',lambda p:{'scene':{'physics_profile':'legacy_urdf'}})
    def simulation(w,**kw):calls.append(('simulation',kw));return {'identity_sim':False}
    monkeypatch.setattr(precision,'simulation_configuration',simulation)
    monkeypatch.setattr(precision,'calibration_status',lambda p:'fixture')
    options={'physics_profile':'isaac_matched','control_hz':30,'_collection_mappings':{},'task_definition':{'task_id':'chosen'},'active_object_ids':['cube']}
    if resume:options['resume_data']=str(path)
    request={'stage':stage,'workspace':'fixture','input':str(path),'renderer':'none','options':options}
    before=copy.deepcopy(request)
    out=prepare_mujoco(request,coordinates={'gripper_closed_rad':.02,'gripper_open_rad':1.1,'measured_gripper_limits_rad':[-.1,1.2]},hz=25,workspace=original)
    assert out['workspace'] is original and request==before
    assert out['physics_profile']==('legacy_urdf' if resume else 'isaac_matched')
    assert out['closed_rad']==.02 and out['open_rad']==1.1 and out['measured_gripper_limits']==[-.1,1.2]
    assert out['control_hz']==25 and out['simulation_configuration']=={'identity_sim':False}
    assert out['task_definition'] is options['task_definition']
    if stage=='source':assert calls[1]==('source',str(path) if resume else None)
    elif recorded_scene is not None:assert calls[1][1]['reset']==recorded_scene['reset']
    else:assert [c[0] for c in calls]==['configure','simulation']
    assert calls[-1][1]['coordinates'] is recorded and calls[-1][1]['hardware_role']=='leader'
    assert original=={'scene_configuration':{'reset':{'mode':'base'}}}


@pytest.mark.parametrize('viewer', [False,True])
def test_post_constructor_failure_closes_backend(monkeypatch,viewer):
    import numpy as np
    from soarm101_lab.simulation import mujoco_backend
    mujoco_backend.configure_rendering('none')
    import mujoco.viewer
    from soarm101_lab.simulation.environment import create_mujoco
    events=[]
    class Backend:
        model=object();data=object()
        limits=np.array([[-3.,3.]]*5+[[0.,.01] if not viewer else [0.,2.]])
        def __init__(self,**kw):events.append('create')
        def close(self):events.append('close')
    monkeypatch.setattr(mujoco_backend,'MujocoBackend',Backend)
    def launch(*args,**kwargs):raise RuntimeError('viewer init failed')
    monkeypatch.setattr(mujoco.viewer,'launch_passive',launch)
    with pytest.raises(RuntimeError if viewer else ValueError):
        create_mujoco({},renderer='none',viewer=viewer)
    assert events==['create','close']


@pytest.mark.parametrize('provided_parent', [False,True])
def test_datagen_writer_initialization_failure_closes_owned_parent(tmp_path,monkeypatch,provided_parent):
    from soarm101_lab.workflow import datagen,worker_request
    from soarm101_lab.simulation import environment
    closed=[];created=[]
    parent=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(worker_request,'prepare_mujoco',lambda *a,**k:{})
    def create(**kw):created.append(True);return parent
    monkeypatch.setattr(environment,'create_mujoco',create)
    monkeypatch.setattr(datagen,'_output_args',lambda *a:{})
    def writer(*a,**kw):raise OSError('disk failure')
    monkeypatch.setattr(datagen,'DatasetWriter',writer)
    with pytest.raises(OSError,match='disk failure'):
        datagen.run_mujoco({'options':{'num_envs':2,'generation_seed':1}},SimpleNamespace(hz=30,contract={},args={}),tmp_path,parent=parent if provided_parent else None)
    assert closed==[True] and created==([] if provided_parent else [True])


def test_simulation_factory_has_no_workflow_or_dataset_imports():
    import ast,inspect
    from soarm101_lab.simulation.environment import create_mujoco
    signature=inspect.signature(create_mujoco)
    assert not set(signature.parameters)&{'request','stage','output','dataset_reader','artifact','registry','resume'}
    tree=ast.parse(inspect.getsource(create_mujoco))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert node.module=='mujoco_backend'


def test_import_mujoco_factory_is_inert():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import sys; import soarm101_lab.simulation.environment; assert not any(n.split('.')[0] in ('mujoco','isaaclab','pxr','omni','PyQt6','torch') for n in sys.modules)"],check=True)
