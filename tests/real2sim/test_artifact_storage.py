"""Named storage contracts and real file/Registry integration; no hardware or GPU."""
import hashlib,json,subprocess,sys
from pathlib import Path
import h5py,pytest
from soarm101_lab.workflow.artifacts import (Registry,artifact_name,artifact_description,
    output_location,validate_storage_name,fingerprint)
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
    if stage=='source' and backend=='isaac':assert out==boundary/'source.hdf5'
    elif stage in ('real_eval','hf_dataset') or (stage=='sim_eval' and backend=='isaac'):assert out==boundary/'result.json'
    else:assert out==boundary
    assert not boundary.exists()
    custom=tmp_path/('b'+out.suffix if out!=boundary else 'b')
    with pytest.raises(ValueError,match='이름과 지정 출력'):
        output_location(stage,{},dict(backend=backend,storage_name='a',output=str(custom)),tmp_path)


@pytest.mark.parametrize('backend',['isaac','mujoco'])
@pytest.mark.parametrize('stage',['source','annotate','datagen'])
def test_named_run_files_and_legacy_hash(storage_workspace,tmp_path,monkeypatch,backend,stage):
    """A real subprocess publishes fixture HDF5 files at the planned worker destination."""
    registry=Registry(tmp_path/'workflow');workspace=storage_workspace
    original=tmp_path/'legacy_run/source.hdf5';original.parent.mkdir();source_fixture(original,20,load(workspace))
    with h5py.File(original,'a') as h:
        for g in h['data'].values():g.create_dataset('obs/datagen_info/subtask_term_signals/grasp',data=[0,0,1,1])
    before=fingerprint(original)
    source=registry.register('source',original,display_name='기존_Source20')
    annotated=registry.register('annotated',original,parents=[source['id']],display_name='기존_Annotation20')
    artifact=registry.register('workspace',workspace) if stage=='source' else source if stage=='annotate' else annotated
    popen=subprocess.Popen
    def fixture_worker(argv,**kwargs):
        if argv[0]=='git':return popen(argv,**kwargs)
        # Keep actual plan/request and Operations completion/registration; replace physics only.
        run=next((registry.root/'runs').glob('*/request.json'))
        request=json.loads(run.read_text());out=Path(request['output'])
        script='''import sys,json,shutil
from pathlib import Path
src,out=map(Path,sys.argv[1:3]);bundle=sys.argv[3]=='True'
dest=out/'dataset.hdf5' if bundle else out
dest.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(src,dest);shutil.copyfile(str(src)+'.so101.json',str(dest)+'.so101.json')
if bundle:(out/'report.json').write_text(json.dumps({'passed':True,'dataset':str(dest)}))
'''
        return popen([sys.executable,'-c',script,str(original),str(out),str(backend=='mujoco' or stage!='source')],**kwargs)
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.Popen',fixture_worker)
    op=Operations(registry);opts=dict(backend=backend,storage_name='사용자_결과',port='/dev/fixture',python=sys.executable)
    result=op.run(stage,artifact,str(workspace),opts)
    bucket={'source':'sources','annotate':'annotations','datagen':'datagen'}[stage]
    assert Path(result['path']).is_relative_to(registry.root/bucket/'사용자_결과')
    assert result['display_name']=='사용자_결과' and result['parents']==[artifact['id']]
    assert registry.verify(result) and fingerprint(original)==before
    assert registry.get(source['id'])['status']=='SUCCEEDED'  # Named annotation never promotes/deletes Source.
    run=Path(op.session).parent;state=json.loads((run/'state.json').read_text())
    assert state['plan']['output']==json.loads((run/'request.json').read_text())['output']
    assert not Path(state['plan']['output']).is_relative_to(registry.root/'runs')
    assert (run/'process.log').is_file() and state['status']=='SUCCEEDED'
    assert op.readiness(stage,artifact,str(workspace),opts)[0]=='NOT_READY'
    with pytest.raises(FileExistsError):op.run(stage,artifact,str(workspace),opts)
    assert fingerprint(original)==before
    assert not list((registry.root/bucket).glob('*.workflow-lock'))


def test_curation_named_filter_and_train14_val6(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.episodes import source_catalog,source_split
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'workflow');original=tmp_path/'original.hdf5';source_fixture(original,20)
    parent=registry.register('source',original);before=fingerprint(original)
    d=CurationDialog(registry);d.path=str(original);d.kind='source'
    from soarm101_lab.workflow.curation import analyze
    d.rows=analyze(original);monkeypatch.setattr(d,'run_job',lambda fn,done:done(fn()))
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(args[2],''))
    d.storage_name.setText('source20_checked');d.save()
    filtered=registry.latest({'source'});assert filtered['parents']==[parent['id']]
    assert filtered['path']==str(registry.root/'curation/filtered/source20_checked.hdf5')
    d.storage_name.setText('source20_train14_val6');d.train_ratio.setValue(70);d.split_plan=source_split(original,.7,2)
    d.save_source_split()
    records=registry.all()[-2:];assert {r['display_name'] for r in records}=={'source20_train14_val6_train','source20_train14_val6_val'}
    ids={}
    for r in records:
        catalog,partition=source_catalog(r['path']);ids[partition['split']]=set(partition['source_ids'])
        assert r['parents']==[parent['id']] and len(catalog)==(14 if partition['split']=='train' else 6)
        registry.verify(r)
    assert not ids['train']&ids['val'] and fingerprint(original)==before
    assert (registry.root/'curation/splits/source20_train14_val6/source_split.json').is_file()
    d.close()


def test_registry_legacy_name_fallback_and_content_identity(tmp_path):
    p=tmp_path/'runs/legacy_name/report.json';p.parent.mkdir(parents=True);p.write_text('{"passed":true}')
    r=Registry(tmp_path/'workflow');record=r.register('replay',p)
    index=r.root/'artifacts'/(record['id']+'.json');old=json.loads(index.read_text());old.pop('display_name');index.write_text(json.dumps(old))
    before=index.read_bytes();legacy=r.get(record['id'])
    assert artifact_name(legacy)=='legacy_name' and 'legacy_name' in artifact_description(legacy)
    assert r.verify(legacy) and index.read_bytes()==before
    named=r.register('replay',p,display_name='같은_표시이름')
    other=tmp_path/'other.json';other.write_bytes(p.read_bytes());second=r.register('replay',other,display_name='같은_표시이름')
    assert named['id']!=second['id'] and named['sha256']==fingerprint(p)


def test_ui_names_filters_preview_and_model_graph(tmp_path,storage_workspace,monkeypatch):
    from PyQt6.QtWidgets import QApplication,QComboBox
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.ui import TrainingLossPlot
    from soarm101_lab.real2sim.storage import atomic
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'workflow')
    source=tmp_path/'source.hdf5';source_fixture(source,2,load(storage_workspace))
    first=registry.register('source',source,display_name='한글_Source')
    second=registry.register('source',source,display_name='other_Source')
    registry.register('workspace',storage_workspace)
    model=tmp_path/'models/named_policy/policy';model.mkdir(parents=True)
    (model/'metrics.jsonl').write_text('{"step":1,"loss":0.2}\n')
    policy=dict(id='named-policy',type='policy',path=str(model),parents=[first['id']],status='SUCCEEDED',
                created_at=1,display_name='MiniVLA_실험01',format='so101.minivla-policy/1',workspace=first['workspace'])
    atomic(registry.root/'artifacts/named-policy.json',policy)
    monkeypatch.setattr(ui,'Registry',lambda:registry)
    panel=ui.WorkflowPanel(lambda *args:None);panel.workspace.setText(str(storage_workspace))
    panel.stage.setCurrentIndex(panel.stage.findData('replay'));panel.refresh()
    assert panel.inputs.findData(first['id'])>=0 and panel.inputs.findData(second['id'])>=0
    panel.artifact_search.setText('한글');assert panel.inputs.count()==1 and panel.inputs.currentData()==first['id']
    assert first['path'] in panel.artifact_details.text() and '한글_Source' in panel.inputs.currentText()
    panel.storage_name.setText('재생_결과');panel.update_ready()
    assert panel.stage_options()['storage_name']=='재생_결과' and '재생_결과' in panel.output_preview.text()
    panel.artifact_task.setText('no-such-task');assert panel.inputs.count()==0
    panel.artifact_task.clear();panel.artifact_search.clear();panel.inputs.setCurrentIndex(panel.inputs.findData(first['id']))
    assert panel.training_metrics_path()==model/'metrics.jsonl'
    panel.show_training_loss();dialog=panel._loss_dialog
    choices=dialog.findChild(QComboBox,'lossModelChoice');assert choices.findData(policy['id'])>=0
    choices.setCurrentIndex(choices.findData(policy['id']));choices.activated.emit(choices.currentIndex())
    chart=dialog.findChild(TrainingLossPlot,'trainingLossPlot');assert chart.path==model/'metrics.jsonl'
    dialog.close();panel.close()


def test_named_checkpoint_import_finds_detached_run(tmp_path):
    from test_training_resume import checkpoint
    from soarm101_lab.workflow.policy import import_checkpoint
    from types import SimpleNamespace
    import shutil
    source,artifact=checkpoint(tmp_path)
    dest=tmp_path/'workflow/models/act_experiment/checkpoints/040000';dest.parent.mkdir(parents=True);shutil.move(source,dest)
    model=dest/'pretrained_model'
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):(model/name).write_text('{}')
    request=tmp_path/'workflow/runs/unique_run/request.json';request.parent.mkdir(parents=True)
    artifact.update(id='dataset-id')
    request.write_text(json.dumps({'stage':'train','input':artifact,'output':str(dest.parent.parent),'options':{'storage_name':'act_experiment'}}))
    registry=SimpleNamespace(root=tmp_path/'workflow',verify=lambda a:None,register=lambda kind,path,**kwargs:dict(path=str(path),**kwargs))
    result=import_checkpoint(model,registry)
    assert result['display_name']=='act_experiment' and result['parents']==['dataset-id']


def test_output_reservation_and_existing_empty_bundle(storage_workspace,tmp_path):
    registry=Registry(tmp_path/'workflow');artifact=registry.register('workspace',storage_workspace);op=Operations(registry)
    opts={'backend':'isaac','port':'fixture','storage_name':'duplicate'}
    bundle=registry.root/'sources/duplicate';bundle.mkdir(parents=True)
    assert op.readiness('source',artifact,'',opts)[0]=='NOT_READY'
    bundle.rmdir()
    lock=bundle.parent/'.duplicate.workflow-lock';lock.write_text('other run')
    with pytest.raises(FileExistsError,match='다른 실행'):op.run('source',artifact,options=opts)
    assert lock.read_text()=='other run' and not bundle.exists()


def test_act_resume_name_and_explicit_canonical_output(storage_workspace,tmp_path,monkeypatch):
    from test_training_resume import checkpoint
    from soarm101_lab.real2sim.workspaces.package import metadata
    from soarm101_lab.workflow.tasks import catalog
    import shutil
    old,_=checkpoint(tmp_path);dest=tmp_path/'models/act_name/checkpoints/040000';dest.parent.mkdir(parents=True);shutil.move(old,dest)
    opts={'architecture':'act','resume_checkpoint':str(dest),'backend':'mujoco'}
    original=dest.parent.parent
    assert output_location('train',{},opts,tmp_path)[0]==original
    with pytest.raises(ValueError,match='새 이름'):
        output_location('train',{},dict(opts,storage_name='new_name'),tmp_path)
    fresh=tmp_path/'models/new_name'
    registry=Registry(tmp_path/'workflow');monkeypatch.setattr(registry,'verify',lambda *args:None)
    w=load(storage_workspace)
    artifact=dict(id='canonical-dataset',path=str(tmp_path/'canonical'),type='dataset',format='so101.canonical/1',workspace=metadata(w),tasks=[catalog(w)[0]])
    Path(artifact['path']).mkdir();(Path(artifact['path'])/'manifest.json').write_text('{}')
    run=tmp_path/'run';run.mkdir()
    plan=Operations(registry).plan('train',artifact,str(storage_workspace),dict(opts,output=str(fresh),storage_name='new_name'),run_dir=run)
    assert plan['output']==str(fresh) and plan['options']['resume_output']==str(fresh)
