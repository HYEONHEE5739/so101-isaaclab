"""Engine construction ownership; no Kit startup in contract fixtures."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from soarm101_lab.simulation.environment import create_isaac


def test_import_is_inert():
    subprocess.run([sys.executable, '-c',
        "import sys; import soarm101_lab.simulation.environment; import soarm101_lab.simulation.workspace_scene; import soarm101_lab.real2sim.workspaces.environment; "
        "assert not any(n.split('.')[0] in ('isaaclab','omni','pxr','torch','PyQt6','mujoco') for n in sys.modules); "
        "assert not any(n.startswith('soarm101_lab.workflow') for n in sys.modules)"], check=True)


@pytest.mark.parametrize('overview', [False, True])
@pytest.mark.parametrize('count', [1, 2, 4, 8])
@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_isaac_configuration_and_reset_ownership(monkeypatch, count, failure, overview):
    calls=[]
    workspace={'identity':'original'}
    scene={'task_definition':{'pick_object_id':'cube'},'active_object_ids':['cube'],'reset':{'seed':7}}
    coordinates={'simulation_model':{'recorded':True}}
    derived={'task':{'pick':'cube'}}
    cfg=SimpleNamespace(scene=SimpleNamespace(),sim=SimpleNamespace(),decimation=8,
        events=SimpleNamespace(reset_episode=SimpleNamespace(params={})))
    def configured(w, definition, active, **kw):
        assert w is workspace and definition is scene['task_definition'] and active is scene['active_object_ids']
        assert kw=={'backend':'isaac','reset':scene['reset']}
        calls.append('configure');return derived
    def make_config(w, device, num_envs, **kw):
        assert w is derived and device=='cuda:2' and num_envs==count
        assert kw['coordinates'] is coordinates
        assert kw['overview_camera'] is overview
        calls.append('config');return cfg
    class Native:
        def __init__(self, **kw):
            assert kw['cfg'] is cfg;calls.append('create')
        def reset(self):
            calls.append('reset')
            if failure:raise failure('reset failed')
        def close(self):calls.append('close')
    monkeypatch.setitem(sys.modules,'isaaclab.envs',SimpleNamespace(ManagerBasedEnv=Native))
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',SimpleNamespace(ContactSensorCfg=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'soarm101_lab.real2sim.workspaces.environment',SimpleNamespace(make_config=make_config))
    from soarm101_lab.real2sim.workspaces import tasks
    monkeypatch.setattr(tasks,'configured',configured)
    def run():return create_isaac(workspace,scene,coordinates,control_hz=25,device='cuda:2',num_envs=count,randomization_seed=17,overview_camera=overview)
    if failure:
        with pytest.raises(failure):run()
        assert calls==['configure','config','create','reset','close']
    else:
        native=run()
        assert calls==['configure','config','create','reset']
        native.close()
    assert cfg.sim.dt==1/200
    assert cfg.events.reset_episode.params=={'randomization_seed':17}
    assert cfg.scene.grasp_fixed_contact['prim_path']=='{ENV_REGEX_NS}/Robot/gripper'
    assert cfg.scene.grasp_moving_contact['prim_path']=='{ENV_REGEX_NS}/Robot/jaw'
    assert cfg.scene.grasp_fixed_contact['filter_prim_paths_expr']==['{ENV_REGEX_NS}/cube']
    assert workspace=={'identity':'original'}


def test_workspace_and_simulation_do_not_import_presentation():
    import ast
    from pathlib import Path
    from soarm101_lab.simulation import environment
    package = Path(environment.__file__).resolve().parents[1]
    for directory in (package / 'simulation', package / 'real2sim/workspaces'):
        for path in directory.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').endswith(('workflow.presentation', 'workflow.stages')), path
