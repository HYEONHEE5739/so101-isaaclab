"""Native Path storage boundaries; retired UUID/cache/promote APIs are removed."""
import json,sys
from pathlib import Path
import pytest
from soarm101_lab.workflow.paths import input_directory,output_location,validate_storage_name
from soarm101_lab.workflow.operations import Operations
from test_source_partitions import source_fixture
from soarm101_lab.real2sim.workspaces.package import load

@pytest.fixture
def storage_workspace(tmp_path):
    from test_builder_workspace import make_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    root=tmp_path/'workspace_fixture';root.mkdir()
    return import_scene(make_export(root),'storage_test',registry=root/'registry')


@pytest.mark.parametrize('name',['..','../foo','/tmp/foo','a/b','a\\b','a:b','a\nname','a..b',' a','a ','','CON','NUL','LPT1','x'*101])
def test_invalid_names(name):
    with pytest.raises(ValueError):validate_storage_name(name)


@pytest.mark.parametrize('backend',['isaac','mujoco'])
@pytest.mark.parametrize('stage,bucket',[('source','sources'),('annotate','annotations'),('datagen','datagen'),('convert','datasets'),('train','models'),('sim_eval','evaluations'),('real_eval','evaluations'),('hf_dataset','evaluations')])
def test_worker_path_contract(tmp_path,backend,stage,bucket):
    out,boundary=output_location(stage,{},dict(backend=backend,storage_name='학습_실험01'),tmp_path)
    assert boundary==tmp_path/bucket/'학습_실험01'
    if stage=='source':assert out==(boundary/'origin/source.hdf5' if backend=='isaac' else boundary/'origin')
    elif stage in ('real_eval','hf_dataset') or (stage=='sim_eval' and backend=='isaac'):assert out==boundary/'result.json'
    else:assert out==boundary
    assert not boundary.exists()
    custom=tmp_path/('b'+out.suffix if out!=boundary else 'b')
    with pytest.raises(ValueError,match='이름과 지정 출력'):
        output_location(stage,{},dict(backend=backend,storage_name='a',output=str(custom)),tmp_path)


@pytest.mark.parametrize('architecture',['act','smolvla','minivla'])
def test_model_storage_uses_name_without_source_split_or_architecture_groups(tmp_path,architecture):
    dataset={'type':'dataset','path':str(tmp_path/'datasets/generated_100/converted_100'),
             'collection':{'source':'original_source','variant':'split_1','partition':'train'}}
    output,_=output_location('train',dataset,{'architecture':architecture,'storage_name':'experiment'},tmp_path)
    assert output==tmp_path/'models/experiment'
    assert dataset['collection']['partition']=='train'


def test_flat_model_storage_still_rejects_unknown_architecture(tmp_path):
    with pytest.raises(ValueError,match='Unknown policy architecture'):
        output_location('train',{},dict(architecture='unknown',storage_name='name'),tmp_path)
    assert not (tmp_path/'models').exists()


def test_each_output_uses_only_stage_store_and_name(tmp_path):
    annotation={'type':'annotated','path':str(tmp_path/'annotations/source_family/split_2/train/annotation_run/dataset.hdf5'),
                'collection':{'source':'source_family','variant':'split_2','partition':'train'}}
    generated,_=output_location('datagen',annotation,{'storage_name':'generated'},tmp_path)
    assert generated==tmp_path/'datagen/generated'
    converted,_=output_location('convert',{'type':'datagen','path':str(generated/'dataset.hdf5')},
                                {'storage_name':'converted'},tmp_path)
    assert converted==tmp_path/'datasets/converted'
    source={'type':'source','path':str(tmp_path/'sources/source_family/split_2/train'),
            'collection':annotation['collection']}
    annotated,_=output_location('annotate',source,{'storage_name':'annotation_run'},tmp_path)
    assert annotated==tmp_path/'annotations/annotation_run'


@pytest.mark.parametrize('backend',['mujoco','isaac'])
def test_source_resume_target_and_output_lock(tmp_path,storage_workspace,backend,monkeypatch):
    from soarm101_lab.workflow import data_resume
    root=tmp_path/'workflow';service=Operations(root=root)
    monkeypatch.setattr(data_resume,'validate',lambda *a,**kw:4)
    options=dict(backend=backend,port='FAKE',storage_name='이름_자유',resume_data='fixture',episodes=4)
    from soarm101_lab.real2sim.workspaces import reset
    monkeypatch.setattr(reset,'source_workspace',lambda w,*a:w)
    state,reason=service.readiness('source',str(storage_workspace),'',options)
    assert state=='NOT_READY' and '4' in reason
    options['episodes']=5
    assert service.readiness('source',str(storage_workspace),'',options)[0]=='READY'
    boundary=root/'sources/이름_자유';boundary.mkdir(parents=True)
    assert service.readiness('source',str(storage_workspace),'',options)[0]=='NOT_READY'
    assert not (root/'artifacts').exists()


def test_storage_names_are_required_and_not_derived_from_inputs(tmp_path):
    for stage in ('source','annotate','datagen','convert','merge','train','sim_eval'):
        with pytest.raises(ValueError):output_location(stage,'/data/train_green',{},tmp_path)
        output,boundary=output_location(stage,'/data/train_green',{'storage_name':'val_green','backend':'mujoco'},tmp_path)
        assert boundary.name=='val_green'
        assert output.is_relative_to(tmp_path)


def test_storage_symlink_escape_is_rejected(tmp_path):
    root=tmp_path/'workflow';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'datasets').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'):output_location('convert','input',{'storage_name':'free'},root)
