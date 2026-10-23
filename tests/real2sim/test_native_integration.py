"""Native CLI boundaries; model math and Trainer remain in installed LeRobot."""
import json,copy,sys
from pathlib import Path
import pytest
from test_learning_interfaces import dataset
from soarm101_lab.learning.training import training_command,native_arguments
from soarm101_lab.learning.datasets import installed_policies
from soarm101_lab.learning.datasets import inspect_dataset,inspect_policy,validate_sim_features


def test_installed_factory_options_and_owned_outputs(tmp_path):
    data=dataset(tmp_path/'dataset');record=inspect_dataset(data)
    assert {'act','smolvla','diffusion'}<=set(installed_policies())
    args,config,path=training_command(record,dict(architecture='diffusion',training_mode='scratch',native_args=['--policy.horizon=16','--log_freq=3']),tmp_path/'out')
    assert args[-2:]==['--policy.horizon=16','--log_freq=3']
    assert not config['policy']['push_to_hub'] and not path.exists()
    for flags in [['--output_dir=/another'],['--policy.push_to_hub=true'],['--dataset.episodes=[0]'],['--token=private']]:
        with pytest.raises(ValueError):native_arguments({'native_args':flags})
    assert native_arguments({'native_args':'--policy.tokenizer_max_length=48'})==['--policy.tokenizer_max_length=48']
    with pytest.raises(ValueError):training_command(record,dict(architecture='act',training_mode='scratch',base_model='/model'),tmp_path/'out')
    from soarm101_lab.workflow.operations import Operations
    with pytest.raises(ValueError,match='outside'):
        Operations(root=tmp_path/'workflow').plan('train',str(data),'',{'architecture':'act','output':str(data/'bad')},dry=True)
    with pytest.raises(ValueError,match='credentials'):
        Operations(root=tmp_path/'workflow').plan('train',str(data),'',{'policy_options':{'token':'private'}},dry=True)


def test_owned_lock_does_not_open_foreign_policy(tmp_path):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{"type":"diffusion"}')
    (model/'model.safetensors').write_bytes(b'fixture')
    for name in ('policy_preprocessor','policy_postprocessor'):(model/(name+'.json')).write_text('{"steps":[]}')
    lock=tmp_path/'.model.workflow-lock';lock.write_text('/run')
    with pytest.raises(ValueError,match='still'):inspect_policy(model)
    with pytest.raises(ValueError,match='still'):inspect_policy(model,owned_lock=tmp_path/'other')
    assert inspect_policy(model,owned_lock=lock)['format']['type']=='diffusion'


def test_cache_fingerprint_uses_relative_cache_exclusion(tmp_path):
    from soarm101_lab.utils.artifact_identity import fingerprint
    inside=tmp_path/'.cache'/'dataset';inside.mkdir(parents=True);(inside/'payload').write_bytes(b'x')
    outside=tmp_path/'dataset';outside.mkdir();(outside/'payload').write_bytes(b'x')
    (inside/'.cache').mkdir();(inside/'.cache'/'session').write_bytes(b'not payload')
    assert fingerprint(inside)==fingerprint(outside)


def test_saved_processor_mapping_preserves_smolvla(tmp_path):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import PolicyFeature,FeatureType
    cfg=SmolVLAConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(6,)),
        'observation.images.camera1':PolicyFeature(FeatureType.VISUAL,(3,256,256))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(6,))},device='cpu')
    (tmp_path/'policy_preprocessor.json').write_text(json.dumps({'steps':[{'registry_name':'rename_observations_processor','config':{'rename_map':{'observation.images.side_cam':'observation.images.camera1'}}}]}))
    validate_sim_features(cfg,tmp_path)
    with pytest.raises(ValueError,match='No Policy camera'):validate_sim_features(cfg)


def test_ui_hub_input_mode_and_official_log_plot(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.ui import WorkflowPanel,TrainingLossPlot
    app=QApplication.instance() or QApplication([])
    panel=WorkflowPanel(lambda *a:None,root=tmp_path/'workflow')
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('train'))
        panel.architecture.setCurrentIndex(panel.architecture.findData('diffusion'))
        panel.dataset_source.setCurrentIndex(panel.dataset_source.findData('hub'));panel.dataset_repo.setText('lerobot/pusht@v3.0')
        panel.storage_name.setText('model');panel.native_args.setText('--policy.horizon=16')
        assert panel.selected_input()=='hf://lerobot/pusht@v3.0'
        opts=panel.stage_options();assert opts['training_mode']=='scratch' and opts['native_args']=='--policy.horizon=16'
        log=tmp_path/'process.log';log.write_text('INFO native step:2 smpl:2 ep:0 loss:0.345 grdn:0.5\n')
        plot=TrainingLossPlot(log);plot.reload();assert plot.rows==[{'step':2,'loss':.345}]
        plot.close()
    finally:panel.close()


def test_explicit_hub_script_reference_is_not_a_relative_path(tmp_path):
    from soarm101_lab.workflow.task_scripts import script_inputs
    from soarm101_lab.workflow.operations import default_python
    assert installed_policies(default_python())==installed_policies()
    path=tmp_path/'model.py'
    cfg=dict(stage='sim_eval',workspace='/workspace',checkpoint='hf://owner/model@commit',
        scene={'objects':['cube','cup']},options={'backend':'mujoco','storage_name':'test'},
        evaluations=[dict(id='new_task',instruction='Move',success='success',episodes=1)])
    path.write_text('SCRIPT = '+repr(cfg)+'\ndef success(ctx): return False\n')
    model,workspace,opts=script_inputs(path,'sim_eval')
    assert model=='hf://owner/model@commit' and workspace=='/workspace'
