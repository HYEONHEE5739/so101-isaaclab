"""Named storage contracts and real file/Registry integration; no hardware or GPU."""
import hashlib,json,subprocess,sys
from pathlib import Path
import h5py,pytest
from soarm101_lab.workflow.artifacts import (Registry,artifact_name,artifact_description,
    output_location,validate_storage_name,fingerprint,episode_input_kind)
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


def test_source_automatic_names_use_selected_task_and_old_labels_preserve_records(tmp_path):
    import copy
    from soarm101_lab.workflow.artifacts import artifact_name
    task={'task_id':'cube_green_to_cup_a'}
    artifact={'type':'workspace','path':str(tmp_path/'workspace')}
    for backend in ('mujoco','isaac'):
        output,boundary=output_location('source',artifact,{'backend':backend,'tasks':[task]},tmp_path)
        assert boundary.name.startswith('source_cube_green_to_cup_a_')
        custom,_=output_location('source',artifact,{'backend':backend,'tasks':[task],'storage_name':'초록블록_20개'},tmp_path)
        assert '초록블록_20개/origin' in str(custom)
    old={'type':'source','path':str(tmp_path/'source_20261019_012620_7aa4/origin/dataset.hdf5'),
         'display_name':'source_20261019_012620_7aa4','tasks':[task]}
    before=copy.deepcopy(old)
    assert artifact_name(old)=='cube_green_to_cup_a_20261019_012620' and old==before
    old['display_name']='초록블록_원본20'
    assert artifact_name(old)=='초록블록_원본20'
    legacy={'type':'source','path':str(tmp_path/'20261016_094817_source_cube_red_to_cup_a_656da738/origin/dataset'),
            'tasks':[{'task_id':'cube_red_to_cup_a'}]}
    assert artifact_name(legacy)=='cube_red_to_cup_a_20261016_094817'


@pytest.mark.parametrize('backend',['mujoco','isaac'])
def test_annotation_default_name_is_flat_and_preserves_source_identity(tmp_path,backend):
    import copy
    from soarm101_lab.workflow.artifacts import suggested_name
    artifact={'type':'source','path':str(tmp_path/'sources/테스트_1/origin/dataset.hdf5'),
              'display_name':'테스트_1','collection':{'source':'테스트_1','variant':'origin'}}
    before=copy.deepcopy(artifact)
    output,boundary=output_location('annotate',artifact,{'backend':backend},tmp_path)
    assert output==boundary==tmp_path/'annotations/annotation_테스트_1'
    assert artifact==before and not output.exists()
    artifact['display_name']='origin'
    assert suggested_name('annotate',artifact)=='annotation_테스트_1'
    artifact.update(display_name='선별본_2',path=str(tmp_path/'sources/테스트_1/선별본_2/dataset.hdf5'))
    assert output_location('annotate',artifact,{},tmp_path)[0]==tmp_path/'annotations/annotation_선별본_2'
    for partition in ('train','val'):
        artifact.update(display_name=partition,path=str(tmp_path/f'sources/테스트_1/split_2/{partition}/dataset.hdf5'),
                        source_partition={'split':partition})
        assert suggested_name('annotate',artifact)==f'annotation_테스트_1_split_2_{partition}'
    artifact.update(display_name='한'*90)
    assert len(suggested_name('annotate',artifact).encode())<=200
    custom,_=output_location('annotate',artifact,{'storage_name':'직접_주석'},tmp_path)
    assert custom==tmp_path/'annotations/직접_주석'


@pytest.mark.parametrize('backend',['mujoco','isaac'])
def test_source_resume_target_is_checked_before_worker_and_keeps_engine_count_contract(tmp_path,storage_workspace,backend):
    from soarm101_lab.real2sim.workspaces.reset import source_workspace
    registry=Registry(tmp_path/'workflow');artifact=registry.register('workspace',storage_workspace)
    original=tmp_path/'previous.hdf5';source_fixture(original,4,source_workspace(load(storage_workspace)))
    before=fingerprint(original)
    options=dict(backend=backend,resume_data=str(original),episodes=4,port='/dev/not_connected',
                 storage_name='수집_이어서',python=sys.executable)
    service=Operations(registry)
    status,reason=service.readiness('source',artifact,str(storage_workspace),options)
    assert status=='NOT_READY' and ('기존 4개' in reason or '기존 4' in reason)
    options['episodes']=5
    run=tmp_path/'run';run.mkdir()
    plan=service.plan('source',artifact,str(storage_workspace),options,run_dir=run)
    assert plan['options']['resume_completed']==4 and plan['options']['resume_target']==5
    assert plan['options']['episodes']==(5 if backend=='mujoco' else 1)
    assert not Path(plan['output']).exists() and fingerprint(original)==before


def test_episode_picker_classifies_recorded_evidence_without_modifying_input(tmp_path):
    path = tmp_path/'input.hdf5'
    with h5py.File(path,'w') as h:
        data=h.create_group('data');data.attrs['env_args']='{}'
        data.create_group('demo_0')
    before=path.read_bytes()
    assert episode_input_kind(path)=='source' and path.read_bytes()==before
    with h5py.File(path,'a') as h:
        h.create_dataset('data/demo_0/obs/datagen_info/subtask_term_signals/grasp',data=[0,1])
    assert episode_input_kind(path)=='annotated'
    with h5py.File(path,'a') as h:
        h['data'].attrs['env_args']=json.dumps({'generation_method':'isaaclab_mimic/1.0.15'})
    assert episode_input_kind(path)=='datagen'


def test_input_picker_registers_selected_source_and_binds_its_scene(tmp_path,storage_workspace,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'workflow');registry.register('workspace',storage_workspace)
    selected=tmp_path/'new_source.hdf5';source_fixture(selected,2,load(storage_workspace))
    before=selected.read_bytes()
    monkeypatch.setattr(ui,'Registry',lambda:registry)
    panel=ui.WorkflowPanel(lambda *args:pytest.fail('Selecting input must not start a worker'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
        monkeypatch.setattr(ui.QFileDialog,'getExistingDirectory',lambda *args:str(selected.parent))
        panel.browse_data()
        artifact=registry.get(panel.inputs.currentData())
        assert artifact['type']=='source' and artifact['path']==str(selected)
        assert panel.workspace_path==str(storage_workspace)
        assert selected.read_bytes()==before
        assert panel.stage_options()['storage_name'].startswith('annotation_')
        panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
        panel.browse_data()
        assert panel.inputs.count()==0 and not panel.run.isEnabled()
        assert selected.read_bytes()==before
    finally:panel.close()


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
    expected=output_location(stage,artifact,opts,registry.root)[1]
    assert Path(result['path']).is_relative_to(expected)
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


def test_policy_picker_starts_at_models_and_eval_output_ignores_legacy_groups(tmp_path):
    policy={'type':'policy','format':{'type':'act'},'path':str(tmp_path/'models/act/model_run/checkpoints/050000/pretrained_model'),
            'collection':{'source':'stale_source','variant':'split_1'}}
    root=Registry(tmp_path)
    assert root.input_directory('real_eval')==tmp_path/'models'
    for backend in ('mujoco','isaac'):
        output,boundary=output_location('sim_eval',policy,{'backend':backend,'storage_name':'evaluation'},tmp_path)
        assert boundary==tmp_path/'evaluations/evaluation'
        assert output==(boundary/'result.json' if backend=='isaac' else boundary)
    mini={'type':'policy','format':'so101.minivla-policy/1','path':str(tmp_path/'models/minivla/old_run/policy')}
    assert root.input_directory('sim_eval')==tmp_path/'models'


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
    filtered=registry.latest({'source'})
    family=registry.source_collection(original)
    assert filtered['path']==str(family/'source20_checked/dataset.hdf5')
    origin=next(a for a in registry.all() if a.get('execution',{}).get('origin_source_id')==parent['id'])
    assert filtered['parents']==[origin['id']]
    d.storage_name.setText('source20_train14_val6');d.train_ratio.setValue(70);d.split_plan=source_split(original,.7,2)
    d.save_source_split()
    records=registry.all()[-2:];assert {r['display_name'] for r in records}=={'train','val'}
    origin=next(a for a in registry.all() if a.get('execution',{}).get('origin_source_id')==parent['id'])
    ids={}
    for r in records:
        catalog,partition=source_catalog(r['path']);ids[partition['split']]=set(partition['source_ids'])
        assert r['parents']==[origin['id']] and len(catalog)==(14 if partition['split']=='train' else 6)
        registry.verify(r)
    assert not ids['train']&ids['val'] and fingerprint(original)==before
    assert (Path(d.saved_collection)/'source_split.json').is_file()
    assert Path(d.saved_collection)==family/'source20_train14_val6'
    assert source_catalog(origin['path'])[0]==source_catalog(original)[0]
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


def test_ui_input_names_and_model_graph(tmp_path,storage_workspace,monkeypatch):
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
    panel=ui.WorkflowPanel(lambda *args:None);panel.workspace_path = str(storage_workspace)
    panel.stage.setCurrentIndex(panel.stage.findData('replay'));panel.refresh()
    assert panel.inputs.findData(first['id'])>=0 and panel.inputs.findData(second['id'])>=0
    panel.inputs.setCurrentIndex(panel.inputs.findData(first['id']))
    assert first['path'] == panel.inputs.toolTip() and '한글_Source' in panel.inputs.currentText()
    panel.storage_name.setText('replay_custom')
    assert panel.stage_options()['storage_name']=='replay_custom'
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


def test_registry_display_cache_tracks_external_edits_and_preserves_full_records(tmp_path,monkeypatch):
    from soarm101_lab.real2sim.storage import atomic
    registry=Registry(tmp_path);path=tmp_path/'artifacts/example.json'
    record=dict(id='example',type='dataset',path=str(tmp_path/'dataset'),created_at=1,status='SUCCEEDED',
        tasks=[{'task_id':'red'}],report={'measurements':'large'*10000},execution={'options':{}},
        provenance={'source_partition':{'split':'train','source_ids':['a']},'episode_source_lineage':{'0':'proof'}})
    atomic(path,record);reads=[];read=Path.read_text
    def counted(p,*args,**kwargs):
        if p==path:reads.append(p)
        return read(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',counted)
    display=registry.get('example',metadata_only=True)
    display['tasks'][0]['task_id']='changed';display['provenance'].clear()
    again=registry.get('example',metadata_only=True)
    assert len(reads)==1 and again['tasks']==record['tasks']
    assert 'report' not in again and 'episode_source_lineage' not in again['provenance']
    assert registry.get('example')==record and len(reads)==2
    other=Registry(tmp_path);assert other.get('example')==record
    atomic(path,dict(record,status='SUPERSEDED'))
    assert registry.get('example',metadata_only=True)['status']=='SUPERSEDED'
    atomic(tmp_path/'artifacts/new.json',dict(record,id='new',created_at=2))
    assert [a['id'] for a in registry.all(metadata_only=True)]==['example','new']
    path.unlink()
    assert [a['id'] for a in registry.all(metadata_only=True)]==['new']
    with pytest.raises(FileNotFoundError):registry.get('example',metadata_only=True)


def test_duplicate_registration_returns_full_record_after_display_cache(tmp_path,monkeypatch):
    from soarm101_lab.workflow import artifacts
    path=tmp_path/'payload';path.write_text('unchanged data')
    monkeypatch.setattr(artifacts,'inspect_artifact',lambda *a:{'report':{'full':'proof'},
        'provenance':{'episode_source_lineage':{'0':'proof'}}})
    registry=Registry(tmp_path/'workflow');first=registry.register('source',path)
    assert 'report' not in registry.get(first['id'],metadata_only=True)
    assert registry.register('source',path)==first
