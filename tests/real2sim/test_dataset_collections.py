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
    dialog.train_ratio.setValue(ratio);dialog.split_seed.setValue(seed)
    dialog.preview_source_split();dialog.save_source_split()
    return Path(dialog.saved_collection)


def test_same_origin_has_independent_numbered_splits_and_origin_reopens(collection,monkeypatch):
    app,registry,original,dialog=collection
    before=fingerprint(original)
    first=save_split(dialog,70,1);first_bytes={str(p):p.read_bytes() for p in first.rglob('*') if p.is_file()}
    second=save_split(dialog,50,2)
    assert first.name=='split_1' and second==first.with_name('split_2')
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
    assert registry.next_split(origin)==first.with_name('split_3')
    opened=[];monkeypatch.setattr(dialog,'load_data',opened.append)
    dialog.open_latest();assert opened==[str(origin)]


def test_annotation_folder_shows_only_train_val_of_selected_split(collection):
    app,registry,original,dialog=collection
    first=save_split(dialog,70,1);second=save_split(dialog,50,2)
    from soarm101_lab.workflow.ui import WorkflowPanel
    panel=WorkflowPanel(lambda *args:pytest.fail('Input selection must not launch an engine'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'));panel.select_collection(second)
        assert panel.inputs.count()==2
        selected=[registry.get(panel.inputs.itemData(i)) for i in range(panel.inputs.count())]
        assert {Path(a['path']) for a in selected}=={second/s/'dataset.hdf5' for s in ('train','val')}
        assert {panel.inputs.itemText(i).split(' · ')[0] for i in range(2)}=={'Train','Val'}
        assert panel.collection_browse.text()=='묶음: sources / split_2'
        for i,artifact in enumerate(selected):
            panel.inputs.setCurrentIndex(i)
            assert panel.task_selector.isHidden()
            assert panel._ready_request[2]['id']==artifact['id']
            assert panel.task_selector.selection()==artifact['tasks']
        panel.stage.setCurrentIndex(panel.stage.findData('datagen'))
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'))
        assert panel.inputs.count()==2 and panel.input_folders['annotate']==str(second)
        panel.select_collection(first)
        assert all(Path(registry.get(panel.inputs.itemData(i))['path']).is_relative_to(first) for i in range(2))
        empty=first.parent/'empty';empty.mkdir();panel.select_collection(empty)
        assert panel.inputs.count()==0 and not panel.run.isEnabled()
        assert panel.inputs.toolTip()==''
        saved=json.loads((registry.root/'ui.json').read_text())
        assert saved['input_folders']['annotate']==str(empty)
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
        panel.browse_collection();panel.browse()
        assert Path(directories[0][2])==folder
        assert Path(files[0][2]).parent==folder
        assert Path(files[0][2]).name in ('train','val')
        assert registry.collection_directory('train')==registry.root/'datasets'
        assert registry.collection_directory('datagen')==registry.root/'annotations'
    finally:panel.close()


def test_unregistered_flat_legacy_split_is_discovered_without_rewriting(collection,tmp_path):
    app,registry,original,dialog=collection
    folder=tmp_path/'external_split_2';result=export_source_splits(original,folder,.5,4)
    before={p:fingerprint(p) for p in result['outputs'].values()}
    from soarm101_lab.workflow.ui import WorkflowPanel
    panel=WorkflowPanel(lambda *args:None)
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('annotate'));panel.select_collection(folder)
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
            panel.stage.setCurrentIndex(panel.stage.findData(stage));panel.select_collection(folder)
            artifact=next(a for a in registry.all() if Path(a['path'])==folder/role/'dataset.hdf5')
            panel.inputs.setCurrentIndex(panel.inputs.findData(artifact['id']));expected[stage]=artifact['id']
        unchanged=dict(panel.input_folders)
        for name in ('annotations_A','annotations_B'):
            other=tmp_path/name;other.mkdir()
            panel.stage.setCurrentIndex(panel.stage.findData('datagen'));panel.select_collection(other)
            dialogs=[]
            monkeypatch.setattr('soarm101_lab.workflow.ui.QFileDialog.getOpenFileName',lambda *args:(dialogs.append(args) or ('','')))
            panel.browse();assert Path(dialogs[-1][2])==other
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
    assert registry.collection_directory(stage,artifact)==registry.root/bucket


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
def test_stage_output_keeps_source_split_and_role(collection,stage,bucket):
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3)
    train=next(a for a in registry.all() if Path(a['path'])==folder/'train/dataset.hdf5')
    expected=registry.root/bucket/folder.parent.name/'split_1'
    if stage!='train':expected=expected/'train'
    output,boundary=output_location(stage,train,{'backend':'mujoco','storage_name':'new_run'},registry.root)
    assert boundary==expected/'new_run'
    assert train['collection']=={'source':folder.parent.name,'variant':'split_1','partition':'train'}
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


def test_grouped_annotation_uses_existing_worker_and_feeds_datagen_folder(collection,monkeypatch):
    from soarm101_lab.workflow.operations import Operations
    from soarm101_lab.workflow.ui import WorkflowPanel
    app,registry,original,dialog=collection
    folder=save_split(dialog,70,3);popen=subprocess.Popen
    inputs=[a for a in registry.all() if a.get('collection',{}).get('variant')=='split_1']
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
            {'backend':'mujoco','python':sys.executable,'storage_name':'annotation_1'})
        assert result['collection']==artifact['collection'] and result['parents']==[artifact['id']]
        assert fingerprint(artifact['path'])==before
    annotation_folder=registry.root/'annotations'/folder.parent.name/'split_1'
    panel=WorkflowPanel(lambda *args:pytest.fail('Input selection must not launch an engine'))
    try:
        panel.stage.setCurrentIndex(panel.stage.findData('datagen'));panel.select_collection(annotation_folder)
        assert panel.inputs.count()==2
        assert {registry.get(panel.inputs.itemData(i))['source_partition']['split'] for i in range(2)}=={'train','val'}
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
