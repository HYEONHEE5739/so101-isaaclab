"""Preparation snapshots cross construction boundaries without reloading assets."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize('resume', [None, 'original.hdf5'])
@pytest.mark.parametrize('mapping_changed', [False, True])
def test_source_reuses_prepared_workspace_after_mapping_check(tmp_path, monkeypatch, resume, mapping_changed):
    from soarm101_lab.real2sim.workspaces import package, tasks, reset, environment
    from soarm101_lab.real2sim import precision
    from soarm101_lab.workflow import presentation
    from soarm101_lab import so101_dataset_contract, representation
    workspace={'task':{'pick':'cube','place':'cup'},'scene_configuration':{'reset':{}},'prepared':True}
    mapping={'sha256':'approved'};simulation={'recorded':True};calls=[]
    monkeypatch.setattr(package,'load',Mock(return_value=workspace))
    monkeypatch.setattr(tasks,'configured',Mock(side_effect=lambda w:w))
    monkeypatch.setattr(reset,'source_workspace',Mock(side_effect=lambda w,r:w))
    monkeypatch.setattr(precision,'approved_mappings',lambda w:{'leader':mapping})
    monkeypatch.setattr(precision,'simulation_configuration',Mock(return_value=simulation))
    monkeypatch.setattr(so101_dataset_contract,'read_contract',lambda p:{'hardware_mapping':mapping})
    monkeypatch.setattr(representation,'require_compatible',lambda *a:None)
    monkeypatch.setitem(sys.modules,'soarm101_lab.real2sim.workspaces.demo',NS(coordinate_contract=lambda w:{}))
    class Prepared(Exception):pass
    def make_config(w, device, count, **kw):
        assert calls==['connect','mapping'] and w is workspace
        assert kw=={'simulation':simulation,'overview_camera':False,'grasp_observations':False}
        raise Prepared
    make=Mock(side_effect=make_config);monkeypatch.setattr(environment,'make_config',make)
    class Leader:
        def __init__(self,*a):
            self.precision_mapper=NS(metadata=self.metadata)
        def connect(self):calls.append('connect')
        def metadata(self):
            calls.append('mapping');return {'sha256':'changed' if mapping_changed else 'approved'}
    monkeypatch.setitem(sys.modules,'isaaclab.app',NS(AppLauncher=Mock()))
    monkeypatch.setitem(sys.modules,'isaaclab.envs',NS(ManagerBasedEnv=Mock()))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.datasets',NS(EpisodeData=Mock(),HDF5DatasetFileHandler=Mock()))
    monkeypatch.setitem(sys.modules,'soarm101_lab.tasks.manager_based.soarm101_lab',NS(SO101TeleopEnvCfg=Mock()))
    monkeypatch.setitem(sys.modules,'soarm101_lab.devices',NS(SO101Leader=Leader,SO101LeaderCfg=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'soarm101_lab.utils.keyboard',NS(KeyboardControl=Mock()))
    monkeypatch.setitem(sys.modules,'soarm101_lab.utils.episode_randomizer',NS(EpisodeRandomizer=Mock()))
    monkeypatch.setitem(sys.modules,'soarm101_lab.utils.voice',NS(log_say=Mock()))
    monkeypatch.setattr(presentation,'current',lambda:None)
    root=Path(__file__).resolve().parents[2]
    script=runpy.run_path(str(root/'scripts/teleoperation/record_mimic_dataset.py'))
    main=script['main'];main.__globals__['args_cli']=NS(dataset_file=str(tmp_path/'new.hdf5'),workspace='scene',
        port='never-open',recalibrate=False,calibration_file_name='unused',resume_data=resume,
        identity_sim_for_validation=False,device='cuda:0',num_envs=1)
    with pytest.raises(ValueError if mapping_changed else Prepared):main()
    package.load.assert_called_once_with('scene')
    tasks.configured.assert_called_once_with(workspace)
    reset.source_workspace.assert_called_once_with(workspace,resume)
    assert make.call_count==(0 if mapping_changed else 1)


@pytest.mark.parametrize('prepared', [False, True])
def test_runtime_reuses_original_workspace_for_profile_application(tmp_path,monkeypatch,prepared):
    import torch
    from soarm101_lab.real2sim import runtime, scene
    from soarm101_lab.real2sim.workspaces import package, environment
    from soarm101_lab.real2sim.core import JOINTS
    workspace={'identity':'original'};calls=[];profile={'measured':True}
    usd=tmp_path/'robot.usda';usd.write_text('fixture')
    cfg=NS(scene=NS(robot=NS(spawn=NS(usd_path=str(usd)))),sim=NS(),decimation=8,
           actions=NS(joint_positions=NS()),workflow_workspace={'identity':'derived'})
    load=Mock(return_value=workspace);monkeypatch.setattr(package,'load',load)
    def make(w,*a,**kw):
        assert w is workspace
        assert kw=={'task_definition':{'selected':True},'simulation':{'approved':True},
                    'overview_camera':False,'grasp_observations':False}
        calls.append('config');return cfg
    monkeypatch.setattr(environment,'make_config',make)
    robot=NS(joint_names=list(JOINTS),data=NS(soft_joint_pos_limits=torch.zeros(1,6,2)))
    class Native:
        def __init__(self,**kw):
            calls.append('create');self.scene={'robot':robot};self.sim=NS(render=lambda:calls.append('render'))
            self.action_manager=NS(get_term=lambda n:NS(_joint_names=list(JOINTS)))
        def reset(self):calls.append('reset')
    monkeypatch.setitem(sys.modules,'isaaclab.envs',NS(ManagerBasedEnv=Native))
    monkeypatch.setitem(sys.modules,'isaaclab.sim',NS(get_current_stage=lambda:None))
    monkeypatch.setitem(sys.modules,'soarm101_lab.tasks.manager_based.soarm101_lab',NS(SO101TeleopEnvCfg=Mock()))
    def apply(env,p,workspace=None):
        assert p is profile and workspace is load.return_value
        calls.append('profile')
    monkeypatch.setattr(runtime,'apply_profile',apply)
    monkeypatch.setattr(scene,'apply_robot_appearance',lambda *a:None)
    r=runtime.Runtime.__new__(runtime.Runtime)
    r.backend='isaac';r.bus=NS(root=tmp_path);r.scene_outputs=[];r.args=NS(device='cpu')
    r.live_workspace='scene';r.task_definition={'selected':True};r.simulation_configuration={'approved':True}
    r.publish=lambda:None
    try:
        r.build_environment(profile,workspace=workspace if prepared else None)
        if prepared:load.assert_not_called()
        else:load.assert_called_once_with('scene')
        assert calls==['config','create','reset','profile']+['render']*5
        assert r.profile is profile and cfg.scene.num_envs==1 and cfg.sim.render_interval==8
        np.testing.assert_array_equal(r.limits,np.zeros((6,2)))
    finally:
        for folder in r.scene_outputs:folder.cleanup()
