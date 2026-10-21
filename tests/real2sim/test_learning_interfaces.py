"""Path learning boundaries; official model objectives are tested separately on local checkpoints."""
import json
from pathlib import Path
import pytest
from soarm101_lab.learning.datasets import inspect_dataset,validate_datasets,leakage
from soarm101_lab.learning.training import train
from soarm101_lab.workflow.operations import Operations
from soarm101_lab.so101_dataset_contract import current_contract,write_contract,JOINT_SPACE
from soarm101_lab.so101_joint_mapping import JOINTS


def dataset(root,**provenance):
    (root/'meta').mkdir(parents=True);(root/'data').mkdir()
    (root/'data/file.parquet').write_bytes(b'fixture; metadata tests only')
    info=dict(repo_id='local/test',total_episodes=2,total_frames=8,total_tasks=1,fps=30,
        features={k:dict(dtype='float32',shape=[6],names=list(JOINTS)) for k in ('action','observation.state')})
    info['features']['observation.images.side_cam']=dict(dtype='video',shape=[48,64,3])
    (root/'meta/info.json').write_text(json.dumps(info));write_contract(root,current_contract(JOINT_SPACE))
    provenance.setdefault('task_definitions',[dict(task_id='pick',language_instruction='Pick target')])
    (root/'workflow_provenance.json').write_text(json.dumps(provenance))
    return root


def test_independent_roles_and_schema(tmp_path):
    a=dataset(tmp_path/'val_name',source_partition={'split':'val'})
    b=dataset(tmp_path/'train_name',source_partition={'split':'train'})
    x,y,e=validate_datasets(a,b)
    assert x['path']==str(a) and y['path']==str(b) and not e['independence_proven']
    assert validate_datasets(a)[1] is None
    with pytest.raises(ValueError,match='different'):validate_datasets(a,a)
    info=json.loads((b/'meta/info.json').read_text());info['features']['action']['names']=list(reversed(JOINTS))
    (b/'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError,match='joint order'):validate_datasets(a,b)


def test_evidenced_leakage_warns_not_group_force(tmp_path):
    prov=dict(source_groups={'0':'same'},episode_source_lineage={'0':{'source_id':'original#0'}})
    a=inspect_dataset(dataset(tmp_path/'a',**prov));b=inspect_dataset(dataset(tmp_path/'b',**prov))
    e=leakage(a,b);assert e['shared_lineages']==1 and e['shared_groups']==['same'] and e['warnings']


@pytest.mark.parametrize('architecture',['act','smolvla'])
def test_official_trainer_configuration_entire_dataset(tmp_path,monkeypatch,architecture):
    data=dataset(tmp_path/'train');out=tmp_path/'model';calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        assert argv[1:3]==['-m','lerobot.scripts.lerobot_train']
        cfg=json.loads((tmp_path/'model.train_config.json').read_text())
        assert cfg['dataset']['root']==str(data) and 'episodes' not in cfg['dataset'] and cfg['eval_freq']==0
        assert cfg['policy']['type']==architecture
        ck=out/'checkpoints/last/pretrained_model';ck.mkdir(parents=True)
        (ck/'config.json').write_text('{}');(ck/'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr('subprocess.run',run)
    train(data,dict(architecture=architecture,steps=2,device='cpu'),out)
    assert len(calls)==1 and (out/'checkpoints/last/pretrained_model/workflow_provenance.json').is_file()


def test_disabled_minivla_and_learning_ready(tmp_path):
    data=dataset(tmp_path/'data');op=Operations(root=tmp_path/'outputs')
    for arch in ('act','smolvla'):
        assert op.readiness('train',str(data),options={'architecture':arch,'storage_name':'model'})[0]=='READY'
    state,reason=op.readiness('train',str(data),options={'architecture':'minivla','storage_name':'model'})
    assert state=='NOT_READY' and '비활성화' in reason
    with pytest.raises(ValueError,match='비활성화'):train(data,{'architecture':'minivla'},tmp_path/'out')


def test_validation_failure_report_and_collision(tmp_path):
    from soarm101_lab.learning.validation import evaluate_checkpoint
    out=tmp_path/'result'
    with pytest.raises((ValueError,FileNotFoundError)):evaluate_checkpoint(tmp_path/'missing',tmp_path/'data',out)
    assert json.loads((out/'report.json').read_text())['status']=='FAILED'
    with pytest.raises(FileExistsError):evaluate_checkpoint('missing','missing',out)


def test_minivla_inference_has_no_trainer_dependency():
    import subprocess,sys
    script="from soarm101_lab.learning.minivla import policy;import sys;assert 'soarm101_lab.learning.minivla.train' not in sys.modules;assert 'soarm101_lab.learning.minivla.data' not in sys.modules"
    subprocess.run([sys.executable,'-c',script],check=True)


def test_validation_interrupt_records_cancelled(tmp_path,monkeypatch):
    from soarm101_lab.learning import validation
    def interrupted(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(validation,'validate_inputs',interrupted)
    with pytest.raises(KeyboardInterrupt):validation.evaluate_checkpoint('missing','missing',tmp_path/'cancelled')
    assert json.loads((tmp_path/'cancelled/report.json').read_text())['status']=='CANCELLED'


def test_post_training_validation_failure_keeps_training_success(tmp_path,monkeypatch):
    from types import SimpleNamespace
    data=dataset(tmp_path/'train');val=dataset(tmp_path/'val');out=tmp_path/'model';calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if argv[2]=='soarm101_lab.learning.validation':return SimpleNamespace(returncode=1)
        ck=out/'checkpoints/last/pretrained_model';ck.mkdir(parents=True)
        (ck/'config.json').write_text('{}');(ck/'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr('subprocess.run',run)
    train(data,dict(architecture='act',validation_dataset=str(val),device='cpu'),out)
    state=json.loads((out/'offline_validation_result.json').read_text())
    assert state['training_status']=='SUCCEEDED' and state['validation_status']=='FAILED' and len(calls)==2
