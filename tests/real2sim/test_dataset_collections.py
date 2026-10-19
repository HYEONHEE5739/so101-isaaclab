"""Versioned Source storage and folder-scoped inputs; no engine or hardware run."""
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest
from test_artifact_storage import storage_workspace
from test_source_partitions import source_fixture
from test_common_mimic import episode
from soarm101_lab.workflow.artifacts import Registry, fingerprint, output_location
from soarm101_lab.workflow.episodes import source_catalog, export_source_splits
from soarm101_lab.real2sim.workspaces.package import load


@pytest.fixture
def collection(tmp_path, storage_workspace, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow import ui
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.curation import analyze
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'workflow');registry.register('workspace',storage_workspace)
    original=tmp_path/'old_datasets/source_demo.hdf5';original.parent.mkdir()
    source_fixture(original,20,load(storage_workspace));registry.register('source',original)
    dialog=CurationDialog(registry);dialog.path=str(original);dialog.kind='source';dialog.rows=analyze(original)
    monkeypatch.setattr(dialog,'run_job',lambda fn,done:done(fn()))
    monkeypatch.setattr(ui,'Registry',lambda:registry)
    yield app,registry,original,dialog
    dialog.close()


def save_split(dialog,ratio,seed):
    dialog.storage_name.setText(f'분할_{ratio}_{seed}')
    dialog.train_ratio.setValue(ratio);dialog.split_seed.setValue(seed)
    dialog.preview_source_split();dialog.save_source_split()
    return Path(dialog.saved_collection)


def test_same_origin_has_independent_named_splits_and_origin_reopens(collection,monkeypatch):
    app,registry,original,dialog=collection
    before=fingerprint(original)
    first=save_split(dialog,70,1);first_bytes={str(p):p.read_bytes() for p in first.rglob('*') if p.is_file()}
    second=save_split(dialog,50,2)
    assert first.name=='분할_70_1' and second==first.with_name('분할_50_2')
    origin=first.parent/'origin'/original.name
    assert fingerprint(original)==before
    assert source_catalog(origin)[0]==source_catalog(original)[0]
    assert all(Path(p).read_bytes()==content for p,content in first_bytes.items())
    for folder,counts in ((first,(14,6)),(second,(10,10))):
        pools=[]
        for role,count in zip(('train','val'),counts):
            records,partition=source_catalog(folder/role/'dataset.hdf5')
            assert len(records)==count and partition['split']==role
            pools.append(set(partition['source_ids']))
        assert not pools[0]&pools[1]
        assert (folder/'source_split.json').is_file()
    assert registry.source_collection(origin)==first.parent
    opened=[];monkeypatch.setattr(dialog,'load_data',opened.append)
    dialog.open_latest();assert opened==[str(origin)]


def test_managed_source_named_subset_and_split_keep_family_and_original(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.curation import analyze
    app=QApplication.instance() or QApplication([])
    registry=Registry(tmp_path/'workflow');family=registry.root/'sources/테스트_1'
    original=family/'origin/dataset.hdf5';original.parent.mkdir(parents=True)
    source_fixture(original,6);parent=registry.register('source',original)
    before=fingerprint(original);errors=[]
    dialog=CurationDialog(registry);dialog.path=str(original);dialog.kind='source';dialog.rows=analyze(original)
    monkeypatch.setattr(dialog,'run_job',lambda fn,done:done(fn()))
    monkeypatch.setattr(dialog,'error',errors.append)
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:pytest.fail('Source uses its named folder directly'))
    try:
        for name in ('','origin','../outside'):
            dialog.storage_name.setText(name);dialog.save()
        assert len(errors)==3 and len(registry.all())==1 and not (registry.root/'curation').exists()
        errors.clear();dialog.storage_name.setText('선별_1');dialog.excluded={dialog.rows[-1]['name']}
        dialog.save();subset=registry.latest({'source'})
        assert not errors and subset['path']==str(family/'선별_1/dataset.hdf5')
        assert subset['parents']==[parent['id']] and subset['collection']=={'source':'테스트_1','variant':'선별_1'}
        assert len(source_catalog(subset['path'])[0])==5
        assert registry.ensure_source_origin(subset['path'])['id']==subset['id']
        subset_hash=fingerprint(subset['path']);dialog.save()
        assert len(errors)==1 and fingerprint(subset['path'])==subset_hash
        errors.clear();dialog.excluded=set();dialog.storage_name.setText('분할_1')
        dialog.preview_source_split();dialog.update_split_controls()
        assert not dialog.storage_name.isHidden()
        dialog.save();split=family/'분할_1'
        assert not errors and Path(dialog.saved_collection)==split
        assert all((split/role/'dataset.hdf5').is_file() for role in ('train','val'))
        assert len(source_catalog(split/'train/dataset.hdf5')[0])+len(source_catalog(split/'val/dataset.hdf5')[0])==6
        assert fingerprint(original)==before and fingerprint(subset['path'])==subset_hash
        out,_=output_location('annotate',subset,{'backend':'mujoco','storage_name':'annotation'},registry.root)
        assert out==registry.root/'annotations/annotation'
    finally:dialog.close()


def test_annotation_folder_shows_only_train_val_of_selected_split(collection):
    app,registry,original,dialog=collection
    first=save_split(dialog,70,1);second=save_split(dialog,50,2)
    from soarm101_lab.workflow.ui import WorkflowPanel
    panel=WorkflowPanel(lambda *args:pytest.fail('Input selection must not launch an engine'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'));panel.select_data(second)
        assert panel.inputs.count()==2
        selected=[registry.get(panel.inputs.itemData(i)) for i in range(panel.inputs.count())]
        assert {Path(a['path']) for a in selected}=={second/s/'dataset.hdf5' for s in ('train','val')}
        assert {panel.inputs.itemText(i) for i in range(2)}=={'train','val'}
        assert panel.data_select.text()=='데이터 선택'
        for i,artifact in enumerate(selected):
            panel.inputs.setCurrentIndex(i)
            assert panel.task_selector.isHidden()
            assert panel._ready_request[2]['id']==artifact['id']
            assert panel.task_selector.selection()==artifact['tasks']
        panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
        assert panel.inputs.count()==2 and panel.input_folders['annotate']==str(second)
        panel.select_data(first)
        assert all(Path(registry.get(panel.inputs.itemData(i))['path']).is_relative_to(first) for i in range(2))
        empty=first.parent/'empty';empty.mkdir();panel.select_data(empty)
        assert panel.inputs.count()==0 and not panel.run.isEnabled()
        assert panel.inputs.toolTip()==''
        saved=json.loads((registry.root/'ui.json').read_text())
        assert saved['input_folders']['annotate']==str(empty)
    finally:panel.close()


def test_source_picker_accepts_only_published_version_and_scopes_dropdown(collection,monkeypatch):
    from soarm101_lab.workflow import ui
    app,registry,original,dialog=collection
    workspace=registry.latest({'workspace'})
    version=Path(workspace['path'])
    panel=ui.WorkflowPanel(lambda *args:pytest.fail('Choosing a Scene must not start a worker'))
    try:
        old=panel.inputs.currentData()
        with pytest.raises(ValueError,match='Scene 버전'):
            panel.select_data(version.parent)
        assert panel.inputs.currentData()==old
        calls=[]
        monkeypatch.setattr(ui.QFileDialog,'getExistingDirectory',lambda *args:calls.append(args) or str(version))
        panel.data_select.click()
        assert calls[0][2]==str(registry.input_directory('source'))
        assert panel.inputs.count()==1 and panel.inputs.currentData()==workspace['id']
        assert panel.workspace_path==str(version) and panel.input_folders['source']==str(version)
    finally:panel.close()


def test_source_family_picker_lists_origin_and_every_named_split(collection):
    from soarm101_lab.workflow.ui import WorkflowPanel
    app,registry,original,dialog=collection
    first=save_split(dialog,70,1);second=save_split(dialog,50,2)
    before=fingerprint(original)
    panel=WorkflowPanel(lambda *args:pytest.fail('Choosing Source data must not run physics'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
        panel.select_data(first.parent)
        labels={panel.inputs.itemText(i) for i in range(panel.inputs.count())}
        assert labels=={'origin',*(f'{p.name}/{s}' for p in (first,second) for s in ('train','val'))}
        assert panel.inputs.currentText()=='origin'
        assert fingerprint(original)==before
    finally:panel.close()


def test_group_picker_starts_above_individual_input_in_managed_storage(collection,monkeypatch):
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3)
    from soarm101_lab.workflow import ui
    panel=ui.WorkflowPanel(lambda *args:pytest.fail('Browsing must not start a worker'))
    directories=[];files=[]
    monkeypatch.setattr(ui.QFileDialog,'getExistingDirectory',lambda *args:(directories.append(args) or ''))
    monkeypatch.setattr(ui.QFileDialog,'getOpenFileName',lambda *args:(files.append(args) or ('','')))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
        panel.browse_data()
        assert Path(directories[0][2])==registry.root/'sources'
        assert files==[]
        assert registry.input_directory('train')==registry.root/'datasets'
        assert registry.input_directory('datagen')==registry.root/'annotations'
    finally:panel.close()


def test_unregistered_flat_legacy_split_is_discovered_without_rewriting(collection,tmp_path):
    app,registry,original,dialog=collection
    folder=tmp_path/'external_split_2';result=export_source_splits(original,folder,.5,4)
    before={p:fingerprint(p) for p in result['outputs'].values()}
    from soarm101_lab.workflow.ui import WorkflowPanel
    panel=WorkflowPanel(lambda *args:None)
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'));panel.select_data(folder)
        assert panel.inputs.count()==2
        assert {registry.get(panel.inputs.itemData(i))['path'] for i in range(2)}==set(result['outputs'].values())
        assert all(fingerprint(path)==value for path,value in before.items())
    finally:panel.close()


def test_stage_folders_and_overlapping_inputs_are_independent_and_restored(collection,tmp_path,monkeypatch):
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3)
    from soarm101_lab.workflow.ui import WorkflowPanel
    panel=WorkflowPanel(lambda *args:pytest.fail('Selecting inputs must not start workers'))
    expected={}
    try:
        # Replay/Annotation/Convert all accept this Source, so a shared combo can leak IDs.
        for stage,role in (('replay','train'),('annotate','val'),('convert','train')):
            panel.stage.setCurrentIndex(panel.stage.findData(stage));panel.select_data(folder)
            artifact=next(a for a in registry.all() if Path(a['path'])==folder/role/'dataset.hdf5')
            panel.inputs.setCurrentIndex(panel.inputs.findData(artifact['id']));expected[stage]=artifact['id']
        unchanged=dict(panel.input_folders)
        for name in ('annotations_A','annotations_B'):
            other=tmp_path/name;other.mkdir()
            panel.stage.setCurrentIndex(panel.stage.findData('datagen'));panel.select_data(other)
            dialogs=[]
            monkeypatch.setattr('soarm101_lab.workflow.ui.QFileDialog.getExistingDirectory',lambda *args:(dialogs.append(args) or ''))
            panel.browse_data();assert Path(dialogs[-1][2])==registry.root/'annotations'
            assert all(panel.input_folders[k]==v for k,v in unchanged.items())
            for stage,identifier in expected.items():
                panel.stage.setCurrentIndex(panel.stage.findData(stage))
                assert panel.inputs.currentData()==identifier
                assert panel.input_folders[stage]==str(folder)
        saved=json.loads((registry.root/'ui.json').read_text())
        assert {s:saved['input_selections'][s]['artifact'] for s in expected}==expected
    finally:panel.close()
    reopened=WorkflowPanel(lambda *args:pytest.fail('Reopening must not start workers'))
    try:
        for stage,identifier in expected.items():
            reopened.stage.setCurrentIndex(reopened.stage.findData(stage))
            assert reopened.inputs.currentData()==identifier
            assert reopened.input_folders[stage]==str(folder)
    finally:reopened.close()


@pytest.mark.parametrize('stage,bucket',[('replay','sources'),('annotate','sources'),('datagen','annotations'),
    ('convert','datagen'),('train','datasets'),('hf_dataset','datasets'),('sim_eval','models'),('real_eval','models'),('hf_policy','models')])
def test_group_directory_uses_stage_store_even_when_previous_input_has_another_type(tmp_path,stage,bucket):
    registry=Registry(tmp_path/'workflow')
    wrong=registry.root/'annotations/family/split_1/train/run/dataset.hdf5'
    if bucket=='annotations':wrong=registry.root/'datagen/family/split_1/train/run/dataset.hdf5'
    artifact={'type':'annotated','path':str(wrong),'collection':{'source':'family','variant':'split_1','partition':'train'}}
    assert registry.input_directory(stage)==registry.root/bucket


def test_curation_does_not_change_other_stage_folders(collection,monkeypatch):
    app,registry,original,dialog=collection
    first=save_split(dialog,70,1);second=save_split(dialog,50,2)
    from soarm101_lab.workflow import ui,curation_ui
    class SavedDialog:
        saved_collection=str(second)
        def __init__(self,*args,**kwargs):pass
        def load_data(self,*args):pass
        def open_latest(self):pass
        def exec(self):app.processEvents()
    monkeypatch.setattr(curation_ui,'CurationDialog',SavedDialog)
    panel=ui.WorkflowPanel(lambda *args:None)
    try:
        panel.input_folders.update(annotate=str(first),replay=str(first),convert=str(first),datagen=str(first.parent))
        before=dict(panel.input_folders);panel.open_curation()
        assert panel.input_folders==before
        panel.stage.setCurrentIndex(panel.stage.findData('replay'));panel.open_curation()
        assert panel.input_folders['replay']==str(second)
        assert all(panel.input_folders[s]==p for s,p in before.items() if s!='replay')
    finally:panel.close()


@pytest.mark.parametrize('stage,bucket', [('annotate','annotations'),('datagen','datagen'),('convert','datasets'),
    ('train','models'),('sim_eval','evaluations'),('hf_dataset','evaluations'),('hf_policy','evaluations')])
def test_stage_output_uses_flat_annotation_and_other_stage_groups(collection,stage,bucket):
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3)
    train=next(a for a in registry.all() if Path(a['path'])==folder/'train/dataset.hdf5')
    expected=registry.root/bucket
    output,boundary=output_location(stage,train,{'backend':'mujoco','storage_name':'new_run'},registry.root)
    assert boundary==expected/'new_run'
    assert train['collection']=={'source':folder.parent.name,'variant':folder.name,'partition':'train'}
    assert not boundary.exists()


def test_origin_copy_preserves_common_mp4_bytes_and_lineage(tmp_path,episode):
    from soarm101_lab.workflow.dataset import DatasetWriter,EpisodeDataset
    values,attrs,videos,args,contract=episode
    source=tmp_path/'old/source.hdf5'
    with DatasetWriter(source,args,contract) as writer:writer.commit(values,attrs,videos)
    registry=Registry(tmp_path/'workflow');parent=registry.register('source',source)
    before=fingerprint(source)
    copied=registry.ensure_source_origin(source)
    assert copied['parents']==[parent['id']] and fingerprint(source)==before
    assert source_catalog(source)[0]==source_catalog(copied['path'])[0]
    original_values=EpisodeDataset(source).episode('demo_0')[0]
    copied_values=EpisodeDataset(copied['path']).episode('demo_0')[0]
    import numpy as np
    np.testing.assert_array_equal(original_values['joint_targets'],copied_values['joint_targets'])
    with h5py.File(source,'r') as old,h5py.File(copied['path'],'r') as new:
        def compare(name,value):
            if isinstance(value,h5py.Dataset):np.testing.assert_array_equal(value[...],new[name][...])
        old.visititems(compare)
    old_videos=EpisodeDataset(source).episode('demo_0')[2]
    new_videos=EpisodeDataset(copied['path']).episode('demo_0')[2]
    for camera in old_videos:
        old=(source.parent/old_videos[camera]['file']).read_bytes()
        new=(Path(copied['path']).parent/new_videos[camera]['file']).read_bytes()
        assert old==new
    assert registry.ensure_source_origin(source)['id']==copied['id']
    registry.verify(copied)


def test_flat_annotations_use_existing_worker_and_feed_datagen_folder(collection,monkeypatch):
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.workflow.ui import WorkflowPanel
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3);popen=subprocess.Popen
    inputs=[a for a in registry.all() if a.get('collection',{}).get('variant')==folder.name]
    def fixture_worker(argv,**kwargs):
        if argv[0]=='git':return popen(argv,**kwargs)
        script='''import sys,json,shutil,h5py
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());out=Path(request['output']);out.mkdir(parents=True)
src=Path(request['input']);dest=out/'dataset.hdf5'
shutil.copyfile(src,dest);shutil.copyfile(str(src)+'.so101.json',str(dest)+'.so101.json')
with h5py.File(dest,'a') as h:
 for g in h['data'].values():g.create_dataset('obs/datagen_info/subtask_term_signals/grasp',data=[0,0,1,1])
(out/'report.json').write_text(json.dumps({'passed':True,'dataset':str(dest)}))
'''
        return popen([sys.executable,'-c',script,argv[argv.index('--request')+1]],**kwargs)
    monkeypatch.setattr('soarm101_lab.workflow.operations.subprocess.Popen',fixture_worker)
    for artifact in inputs:
        before=fingerprint(artifact['path'])
        result=Operations(registry).run('annotate',artifact,registry.latest({'workspace'})['path'],
            {'backend':'mujoco','python':sys.executable})
        assert 'collection' not in result and result['parents']==[artifact['id']]
        assert Path(result['path']).parent.parent==registry.root/'annotations'
        assert fingerprint(artifact['path'])==before
    annotation_folder=registry.root/'annotations'
    panel=WorkflowPanel(lambda *args:pytest.fail('Input selection must not launch an engine'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('datagen'));panel.select_data(annotation_folder)
        assert panel.inputs.count()==2
        assert {registry.get(panel.inputs.itemData(i))['source_partition']['split'] for i in range(2)}=={'train','val'}
        panel.select_data(Path(result['path']).parent)
        assert panel.inputs.count()==1 and panel.inputs.currentData()==result['id']
        panel.storage_name.setText('내_생성본')
        output,_=output_location('datagen',result,panel.stage_options(),registry.root)
        assert output==registry.root/'datagen/내_생성본'
    finally:panel.close()


def test_failed_split_does_not_publish_partial_version(tmp_path,monkeypatch):
    from soarm101_lab.workflow import curation
    source=tmp_path/'source.hdf5';source_fixture(source,4);before=source.read_bytes()
    export=curation.export_subset;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(args)
        if len(calls)==2:raise OSError('injected val save failure')
        return export(*args,**kwargs)
    monkeypatch.setattr(curation,'export_subset',fail_second)
    destination=tmp_path/'split_1'
    with pytest.raises(OSError,match='val save failure'):export_source_splits(source,destination,nested=True)
    assert not destination.exists() and source.read_bytes()==before
