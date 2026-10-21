"""Direct native collections, independent Stage selection and immutable Curation inputs."""
import json
from pathlib import Path
import pytest
from test_artifact_storage import storage_workspace
from test_source_partitions import source_fixture
from soarm101_lab.workflow.curation import export_splits
from soarm101_lab.workflow.paths import discover_inputs,input_directory,output_location,source_collection
from soarm101_lab.utils.artifact_identity import fingerprint


def until(predicate,timeout=5):
    import time
    from PyQt6.QtTest import QTest
    end=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<end
        QTest.qWait(10)


@pytest.fixture
def collection(tmp_path,storage_workspace):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.ui import WorkflowPanel
    from soarm101_lab.real2sim.workspaces.package import load
    app=QApplication.instance() or QApplication([])
    root=tmp_path/'workflow';origin=root/'sources/테스트_1/origin/dataset.hdf5';origin.parent.mkdir(parents=True)
    source_fixture(origin,20,load(storage_workspace))
    p=WorkflowPanel(lambda *a:pytest.fail('No automatic execution'),root=root)
    p.workspace_path=str(storage_workspace)
    yield app,p,root,origin
    p.timer.stop();p.close();app.processEvents()


def test_same_origin_has_independent_named_splits_and_no_copies(collection):
    app,p,root,origin=collection;before=fingerprint(origin)
    first=origin.parent.parent/'분할_1';second=origin.parent.parent/'분할_2'
    a=export_splits(origin,first,.7,3)
    b=export_splits(origin,second,.6,9)
    assert a['counts']=={'train':14,'val':6} and b['counts']=={'train':12,'val':8}
    assert source_collection(origin,root)==origin.parent.parent
    found=discover_inputs(root/'sources',category='dataset')
    assert len(found)==5 and str(origin) in {v['path'] for v in found}
    assert fingerprint(origin)==before and not (root/'artifacts').exists()


@pytest.mark.parametrize('stage',['replay','annotate','datagen','convert','train'])
def test_collection_selection_is_path_and_stage_local(collection,stage):
    app,p,root,origin=collection
    folder=origin.parent.parent/'분할_1';export_splits(origin,folder,.7)
    p.stage.setCurrentIndex(p.stage.findData(stage));p.select_data(folder)
    until(lambda:p._input_request is None and p._input_future is None)
    expected={v['path'] for v in discover_inputs(folder,category='dataset')}
    selected={p.inputs.itemData(i) for i in range(p.inputs.count()) if p.inputs.itemData(i)}
    assert selected==(set() if stage=='train' else expected)
    assert p.input_folders[stage]==str(folder)
    assert not (root/'artifacts').exists()


def test_overlapping_stage_paths_are_independent_and_restored(collection):
    app,p,root,origin=collection
    for stage in ('replay','annotate','convert'):
        p.stage.setCurrentIndex(p.stage.findData(stage));p.select_data(origin)
        until(lambda:p.inputs.currentData()==str(origin))
        assert p.input_selections[stage]=={'path':str(origin)}
    p.stage.setCurrentIndex(p.stage.findData('replay'))
    until(lambda:p.inputs.currentData()==str(origin))
    assert set(p.input_folders)=={'replay','annotate','convert'}
    saved=json.loads((root/'ui.json').read_text())
    assert all('artifact' not in v for v in saved['input_selections'].values())


@pytest.mark.parametrize('stage,bucket',[('datagen','annotations'),('convert','datagen'),('train','datasets'),('sim_eval','models')])
def test_group_directory_uses_stage_store(tmp_path,stage,bucket):
    assert input_directory(stage,tmp_path)==tmp_path/bucket


@pytest.mark.parametrize('stage,bucket',[('annotate','annotations'),('datagen','datagen'),('convert','datasets'),('train','models'),('sim_eval','evaluations')])
def test_stage_output_is_flat_and_has_no_source_group(tmp_path,stage,bucket):
    output,boundary=output_location(stage,'arbitrary_train/origin',{'backend':'mujoco','storage_name':'val_green'},tmp_path)
    assert output==boundary==tmp_path/bucket/'val_green'


def test_curation_preserves_managed_origin_and_never_registers(collection,monkeypatch):
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.curation import analyze
    app,p,root,origin=collection;before=fingerprint(origin)
    d=CurationDialog(root);d.rows=analyze(origin);d.path=str(origin);d.kind='source'
    d.storage_name.setText('선별_1');d.excluded={d.rows[-1]['name']}
    errors=[];monkeypatch.setattr(d,'error',errors.append);monkeypatch.setattr(d,'run_job',lambda fn,done,**kw:done(fn()));d.job=type('Job',(),{'isInterruptionRequested':lambda self:False,'isRunning':lambda self:False})()
    dest=d.curation_destination();d.save()
    assert not errors and dest.is_file() and len(analyze(dest))==19
    assert fingerprint(origin)==before and not (root/'artifacts').exists()
    d.close()


def test_failed_split_does_not_publish_partial_version(tmp_path,monkeypatch):
    from soarm101_lab.workflow import episodes,curation
    original=tmp_path/'original.hdf5';source_fixture(original,6);before=fingerprint(original)
    real=curation.export_subset;calls=[]
    def fail_second(*a,**kw):
        calls.append(a)
        if len(calls)==2:raise RuntimeError('disk failure')
        return real(*a,**kw)
    monkeypatch.setattr(curation,'export_subset',fail_second)
    output=tmp_path/'split'
    with pytest.raises(RuntimeError):curation.export_splits(original,output,.5)
    assert not output.exists() and fingerprint(original)==before
