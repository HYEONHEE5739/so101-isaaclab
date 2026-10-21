"""Source partition evidence, copy/renumber lineage, and Mimic metadata; synthetic motion only."""
import copy,json,shutil
from pathlib import Path
import h5py,numpy as np,pytest
from soarm101_lab.workflow.episodes import source_catalog,validate_partition_episode
from soarm101_lab.workflow.curation import split_dataset,export_splits
from soarm101_lab.workflow.curation import export_subset,analyze
from soarm101_lab.so101_dataset_contract import current_contract,MIMIC_SPACE,write_contract,read_contract


def source_fixture(path,count=30,workspace=None):
    args={'control_dt':1/30,'ee_frame':'robot_base'}
    contract=current_contract(MIMIC_SPACE)
    if workspace:
        from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
        from soarm101_lab.real2sim.workspaces.package import metadata
        from soarm101_lab.real2sim.workspaces.tasks import configured
        workspace=configured(workspace);contract=coordinate_contract(workspace)
        args.update(workspace=metadata(workspace),workspace_path=str(workspace['root']),scene_configuration=workspace['scene_configuration'],task_definitions=[workspace['task_definition']],so101_coordinates=contract)
    with h5py.File(path,'w') as h:
        data=h.create_group('data');data.attrs['env_args']=json.dumps(args)
        for i in range(count):
            g=data.create_group(f'demo_{i}');g.attrs['success']=True
            q=np.full((4,6),i*.001,np.float32);q[:,-1]=.4
            g.create_dataset('actions',data=np.zeros((4,7),np.float32));g.create_dataset('joint_targets',data=q)
            g.create_dataset('obs/policy/joint_pos',data=q)
            ee=np.tile([.1,.1,.1,1,0,0,0],(4,1));ee[:,0]+=np.arange(4)*.001
            g.create_dataset('obs/policy/ee_state',data=ee)
            g.create_dataset('obs/policy/side_cam',data=np.full((4,24,32,3),30+i,np.uint8))
            g.create_dataset('obs/policy/wrist_cam',data=np.full((4,24,32,3),70+i,np.uint8))
    write_contract(path,contract);return args


def historical_plan(source,assignments=None):
    # Fixture for immutable old proofs, not a production splitting path.
    from soarm101_lab.real2sim.precision import fingerprint
    records,_=source_catalog(source)
    assignments=assignments or {n:'train' if i<2 else 'val' for i,n in enumerate(records)}
    inventory={r['source_id']:r['source'] for r in records.values()}
    by_id={records[n]['source_id']:v for n,v in assignments.items()};parts={}
    for role in ('train','val'):
        value=dict(schema='so101.source-partition/1',split=role,seed=0,train_fraction=.5,inventory=inventory,assignments=by_id,source_ids=[k for k,v in by_id.items() if v==role])
        parts[role]={**value,'sha256':fingerprint(value)}
    return dict(records=records,partitions=parts)


def test_free_selection_split_and_repeat_preserve_order(tmp_path):
    src=tmp_path/'source.hdf5';source_fixture(src,50);before=src.read_bytes()
    names=[f'demo_{i}' for i in range(45)]
    subset=export_subset(src,tmp_path/'subset.hdf5',list(reversed(names)))
    with h5py.File(src) as a,h5py.File(subset) as b:
        assert len(b['data'])==45
        for i in range(45):np.testing.assert_array_equal(a[f'data/demo_{i}/joint_targets'],b[f'data/demo_{i}/joint_targets'])
    result=export_splits(subset,tmp_path/'split',.6,3)
    assert result['counts']=={'train':27,'val':18}
    assert split_dataset(subset,.6,3)['assignments']==result['assignments']
    again=export_splits(result['outputs']['train'],tmp_path/'again',.5)
    assert sum(again['counts'].values())==27
    assert not source_catalog(result['outputs']['train'])[1] and src.read_bytes()==before


def test_shared_group_warns_but_never_forces_assignment(tmp_path):
    src=tmp_path/'source.hdf5';source_fixture(src,4)
    with h5py.File(src,'a') as h:
        for i in (0,1):h[f'data/demo_{i}'].attrs['source_group']='same'
    assignments={'demo_0':'train','demo_1':'val','demo_2':'train','demo_3':'val'}
    plan=split_dataset(src,assignments=assignments)
    assert plan['warnings'] and plan['assignments']==assignments
    result=export_splits(src,tmp_path/'split',assignments=assignments)
    assert result['counts']=={'train':2,'val':2}


def test_historical_partition_is_not_resigned_or_used_as_current_role(tmp_path):
    src=tmp_path/'source.hdf5';source_fixture(src,4);plan=historical_plan(src)
    # Construct a historical Train dataset with its original signed proof.
    with h5py.File(src,'a') as h:
        args=json.loads(h['data'].attrs['env_args']);args['source_partition']=plan['partitions']['train']
        h['data'].attrs['env_args']=json.dumps(args)
        for name in list(h['data']):
            if name not in ('demo_0','demo_1'):del h['data'][name]
            else:h['data'][name].attrs['source_lineage']=json.dumps(plan['records'][name])
    before=src.read_bytes();result=export_splits(src,tmp_path/'resplit',assignments={'demo_0':'val','demo_1':'train'})
    with h5py.File(result['outputs']['val']) as h:
        assert json.loads(h['data'].attrs['env_args'])['source_partition']==plan['partitions']['train']
        assert json.loads(h['data'].attrs['curation'])['role']=='val'
    subset=export_subset(result['outputs']['val'],tmp_path/'one.hdf5',['demo_0'])
    assert len(analyze(subset))==1 and analyze(subset)[0]['split'] is None
    assert src.read_bytes()==before


def training_entry(plan,name,split,id):
    return {'id':id,'group_id':'UNKNOWN:original_subtasks','instruction':'pick','split':split,
            'source_partition':plan['partitions'][split],'source_lineage':plan['records'][name]}




def test_mimic_generated_unknown_requires_real_input_partition_evidence(tmp_path):
    from soarm101_lab.real2sim.precision import fingerprint
    src=tmp_path/'source.hdf5';source_fixture(src,4);p=historical_plan(src)['partitions']['train']
    lineage={'kind':'partition','partition_sha256':p['sha256'],'exact_source_ids':None}
    with pytest.raises(ValueError,match='unverifiable'):validate_partition_episode(p,lineage)
    proof={'verification':'source_only_input','sha256':'recorded_input_bytes','source_ids':p['source_ids']}
    proof['evidence_sha256']=fingerprint(proof)
    validate_partition_episode(p,lineage,proof)
    proof['source_ids']=['unrelated']
    with pytest.raises(ValueError,match='unverifiable'):validate_partition_episode(p,lineage,proof)


def test_curation_split_preview_and_manual_ui(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([]);src=tmp_path/'source.hdf5';source_fixture(src)
    dialog=CurationDialog(tmp_path/'workflow')
    monkeypatch.setattr(dialog,'run_job',lambda fn,done,**kw:done(fn()))
    dialog.rows=analyze(src);dialog.path=str(src);dialog.kind='source';dialog.count.setMaximum(30)
    dialog.preview_source_split();assert dialog.split_plan['counts']=={'train':24,'val':6}
    name=next(n for n,s in dialog.split_plan['assignments'].items() if s=='val')
    dialog.scatter.selected=next(i for i,r in enumerate(dialog.rows) if r['name']==name)
    dialog.source_assignment.setCurrentIndex(0);dialog.assign_source_split()
    assert dialog.split_plan['counts']=={'train':25,'val':5}
    assert any('val' in dialog.list.item(i).text() for i in range(dialog.list.count()))
    dialog.close()


def test_curation_selection_split_preview_state_is_clear_and_not_saved(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog,SPLIT_COLORS
    app=QApplication.instance() or QApplication([]);src=tmp_path/'source.hdf5';source_fixture(src,20);before=src.read_bytes()
    d=CurationDialog(tmp_path/'workflow')
    monkeypatch.setattr(d,'run_job',lambda fn,done,**kw:done(fn()));d.job=type('Job',(),{'isInterruptionRequested':lambda self:False,'isRunning':lambda self:False})()
    d.rows=analyze(src);d.path=str(src);d.kind='source';d.count.setMaximum(20);d.count.setValue(20);d.refresh_list()
    d.auto_select();assert not d.excluded # Keeping all 20 does not create Train/Val assignments.
    assert not d.split_plan and not any(r.get('split') for r in d.rows)
    d.count.setValue(10);d.auto_select();assert len(d.excluded)==10 and not d.split_plan
    d.train_ratio.setValue(70);d.split_seed.setValue(1);assert d.val_ratio.text()=='검증 30%'
    d.preview_source_split();assert d.split_plan['counts']=={'train':7,'val':3}
    assert '저장 전' in d.split_status.text() and '7개' in d.split_status.text() and '3개' in d.split_status.text()
    assert d.export_button.isEnabled() and d.export_button.text()=='Train / Val 저장' and d.analysis_toggle.isChecked()
    assert {d.scatter.point_color(r,True) for r in d.rows if r['name'] not in d.excluded}==set(SPLIT_COLORS.values())
    assignments=dict(d.split_plan['assignments']);d.preview_source_split();assert d.split_plan['assignments']==assignments
    d.split_seed.setValue(2)
    assert not d.split_plan and d.export_button.isEnabled() and d.export_button.text()=='데이터 저장'
    assert not any(r.get('split') for r in d.rows)
    assert len(d.excluded)==10 # Changing split settings preserves episode selection.
    d.preview_source_split();d.cancel_source_split();assert not d.split_plan and len(d.excluded)==10
    d.preview_source_split();d.restore_all()
    assert not d.excluded and d.count.value()==20 and not d.split_plan
    assert not any(r.get('split') for r in d.rows) and src.read_bytes()==before
    d.close()


def test_curation_ignores_split_preview_computed_for_old_settings(tmp_path,monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([]);src=tmp_path/'source.hdf5';source_fixture(src,20)
    d=CurationDialog(tmp_path/'workflow');d.rows=analyze(src);d.path=str(src);queued=[]
    monkeypatch.setattr(d,'run_job',lambda fn,done:queued.append((fn,done)))
    d.preview_source_split();fn,done=queued.pop();old_plan=fn();d.train_ratio.setValue(70);done(old_plan)
    assert not d.split_plan and not any(r.get('split') for r in d.rows)
    d.close()


@pytest.mark.parametrize('mode',['subset','planned','locked'])
def test_curation_single_save_exports_selection_or_plan_and_preserves_partition(tmp_path,monkeypatch,mode):
    from PyQt6.QtWidgets import QApplication,QFileDialog,QPushButton
    from soarm101_lab.workflow.curation_ui import CurationDialog
    app=QApplication.instance() or QApplication([])
    original=tmp_path/'source.hdf5';source_fixture(original,6);original_bytes=original.read_bytes()
    src=Path(export_splits(original,tmp_path/'existing_split',.5)['outputs']['train']) if mode=='locked' else original
    before=src.read_bytes();records,partition=source_catalog(src)
    registry=tmp_path/'workflow'
    d=CurationDialog(registry);errors=[]
    monkeypatch.setattr(d,'run_job',lambda fn,done,**kw:done(fn()));d.job=type('Job',(),{'isInterruptionRequested':lambda self:False,'isRunning':lambda self:False})()
    monkeypatch.setattr(d,'error',errors.append)
    assert not d.export_button.isEnabled()
    d.rows=analyze(src);d.path=str(src);d.kind='source'
    d.set_excluded({d.rows[-1]['name']})
    names=[r['name'] for r in d.rows if r['name'] not in d.excluded]
    if mode=='planned':d.preview_source_split()
    d.storage_name.setText('분할_1' if mode=='planned' else '선별_1')
    assert not errors and d.export_button.isEnabled()
    assert [b for b in d.findChildren(QPushButton) if b.text() in ('데이터 저장','Train / Val 저장')]==[d.export_button]
    assert d.export_button.text()==('Train / Val 저장' if mode=='planned' else '데이터 저장')
    dest=tmp_path/('new_split' if mode=='planned' else 'curated.hdf5');requests=[]
    def choose_destination(*args):
        requests.append(args);return str(dest),''
    monkeypatch.setattr(QFileDialog,'getSaveFileName',choose_destination)
    d.export_button.click()
    dest=Path(d.saved_collection) if mode=='planned' else d.curation_destination()
    assert not errors and not requests and dest.exists()
    if mode=='planned':
        result=json.loads((dest/'curation_split.json').read_text())
        assert result['assignments']==d.split_plan['assignments']
        outputs={s:source_catalog(dest/s/'dataset.hdf5') for s in ('train','val')}
        for split,(saved,saved_partition) in outputs.items():
            expected={records[n]['source_id'] for n in names if d.split_plan['assignments'][n]==split}
            assert {r['source_id'] for r in saved.values()}==expected
            assert saved_partition is None
    else:
        saved,saved_partition=source_catalog(dest)
        expected={records[n]['source_id'] for n in names}
        assert {r['source_id'] for r in saved.values()}==expected
        if partition:
            assert saved_partition['split']==partition['split']
            assert saved_partition==partition
        else:assert saved_partition is None
    assert not (registry/'artifacts').exists()
    assert src.read_bytes()==before and original.read_bytes()==original_bytes
    d.close()


def test_cancel_collision_and_failure_do_not_publish(tmp_path,monkeypatch):
    from soarm101_lab.workflow import curation
    from soarm101_lab.so101_dataset_contract import sidecar
    src=tmp_path/'source.hdf5';source_fixture(src,4);before=src.read_bytes()
    out=tmp_path/'out.hdf5';counter=[]
    def cancel():counter.append(1);return len(counter)>3
    with pytest.raises(InterruptedError):export_subset(src,out,['demo_0','demo_1'],cancel=cancel)
    assert not out.exists() and not sidecar(out).exists()
    import soarm101_lab.so101_dataset_contract as contract
    def fail(*args):raise OSError('disk full')
    with monkeypatch.context() as patch:
        patch.setattr(contract,'write_contract',fail)
        with pytest.raises(OSError,match='disk full'):export_subset(src,out,['demo_0'])
    assert not out.exists() and not list(tmp_path.glob('.out*'))
    export_subset(src,out,['demo_0']);saved=out.read_bytes()
    with pytest.raises(ValueError):export_subset(src,out,['demo_1'])
    assert out.read_bytes()==saved and src.read_bytes()==before
    with pytest.raises(ValueError):export_subset(src,src,['demo_1'])
    with pytest.raises(ValueError):split_dataset(src,names=['demo_0','demo_0'])
    with pytest.raises(ValueError):export_subset(src,tmp_path/'empty.hdf5',[])


def test_async_ui_save_reopen_and_cancel(tmp_path,monkeypatch):
    import threading
    from PyQt6.QtWidgets import QApplication
    from test_curation import wait_qt
    from soarm101_lab.workflow import curation_ui
    app=QApplication.instance() or QApplication([])
    src=tmp_path/'source.hdf5';source_fixture(src,6)
    d=curation_ui.CurationDialog(tmp_path/'workflow');errors=[]
    monkeypatch.setattr(d,'error',errors.append)
    try:
        d.load_data(str(src));wait_qt(lambda:d.job is None and len(d.rows)==6)
        d.exclude_all.click();assert len(d.excluded)==6 and not d.export_button.isEnabled()
        d.restore_all();d.set_excluded({'demo_1','demo_3'})
        d.output_parent=str(tmp_path/'chosen');d.storage_name.setText('saved');d.save()
        wait_qt(lambda:d.job is None)
        saved=tmp_path/'chosen/saved.hdf5';assert saved.exists() and not errors
        d.load_data(str(saved));wait_qt(lambda:d.job is None and len(d.rows)==4)
        d.preview_source_split();wait_qt(lambda:d.job is None)
        assert d.split_plan and d.plan_split.isEnabled()
        d.cancel_source_split();d.set_excluded({'demo_0'});d.storage_name.setText('cancelled')
        entered=threading.Event();release=threading.Event();real=curation_ui.export_subset
        def blocked(*args,**kwargs):entered.set();release.wait(3);return real(*args,**kwargs)
        monkeypatch.setattr(curation_ui,'export_subset',blocked)
        d.save();wait_qt(entered.is_set);assert d.cancel_job.isEnabled()
        d.cancel_job.click();release.set();wait_qt(lambda:d.job is None)
        assert not (tmp_path/'chosen/cancelled.hdf5').exists() and not errors
        assert '취소' in d.info.text()
    finally:
        if d.job is not None:d.job.requestInterruption();wait_qt(lambda:d.job is None)
        d.close();app.processEvents()
