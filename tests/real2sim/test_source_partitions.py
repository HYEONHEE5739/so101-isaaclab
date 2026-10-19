"""Source partition evidence, copy/renumber lineage, and Mimic metadata; synthetic motion only."""
import copy,json,shutil
from pathlib import Path
import h5py,numpy as np,pytest
from soarm101_lab.workflow.episodes import source_split,source_catalog,export_source_splits,split_episodes,validate_partition_episode
from soarm101_lab.workflow.curation import export_subset,analyze
from soarm101_lab.so101_dataset_contract import current_contract,MIMIC_SPACE,write_contract,read_contract


def source_fixture(path,count=30,workspace=None):
    args={'control_dt':1/30,'ee_frame':'robot_base'}
    contract=current_contract(MIMIC_SPACE)
    if workspace:
        from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
        from soarm101_lab.real2sim.workspaces.package import metadata
        from soarm101_lab.workflow.tasks import configured
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


def test_source_30_split_seed_export_registry_and_renumber(tmp_path):
    from soarm101_lab.workflow.artifacts import Registry
    src=tmp_path/'source.hdf5';source_fixture(src);before=src.read_bytes()
    a=source_split(src,seed=9);b=source_split(src,seed=9)
    assert a==b and a['counts']=={'train':24,'val':6}
    assert source_split(src,seed=10)['assignments']!=a['assignments']
    result=export_source_splits(src,tmp_path/'split',seed=9)
    sets=[set(result['partitions'][s]['source_ids']) for s in ('train','val')]
    assert not sets[0]&sets[1] and src.read_bytes()==before
    registry=Registry(tmp_path/'registry')
    for split,path in result['outputs'].items():
        artifact=registry.register('source',path);assert artifact['source_partition']['split']==split
        records,p=source_catalog(path);assert len(records)==result['counts'][split]
        chosen=list(records)[-2:];out=export_subset(path,tmp_path/(split+'_curated.hdf5'),chosen)
        after,_=source_catalog(out)
        assert {x['source_id'] for x in after.values()}=={records[n]['source_id'] for n in chosen}
        assert read_contract(src)==read_contract(out)
    copied=tmp_path/'renamed.hdf5';shutil.copyfile(src,copied)
    assert {r['source_id'] for r in source_catalog(copied)[0].values()}=={r['source_id'] for r in source_catalog(src)[0].values()}


def test_manual_assignments_duplicate_group_leakage_and_locked_partition(tmp_path):
    src=tmp_path/'source.hdf5';source_fixture(src,4)
    with h5py.File(src,'a') as h:
        for i in (0,1):h[f'data/demo_{i}'].attrs['source_group']='same_original'
    assignments={'demo_0':'train','demo_1':'val','demo_2':'train','demo_3':'val'}
    with pytest.raises(ValueError,match='leakage'):source_split(src,assignments=assignments)
    assignments['demo_1']='train';plan=source_split(src,assignments=assignments)
    result=export_source_splits(src,tmp_path/'split',assignments=assignments)
    with pytest.raises(ValueError,match='locked'):source_split(result['outputs']['train'])
    assert plan['counts']=={'train':3,'val':1}
    assert plan['group_members']['demo_0']==['demo_0','demo_1']


def test_identical_source_content_copied_into_another_file_has_same_identity(tmp_path):
    first=tmp_path/'a.hdf5';second=tmp_path/'b.hdf5';source_fixture(first,4);source_fixture(second,3)
    with h5py.File(first) as a,h5py.File(second,'a') as b:
        del b['data/demo_0'];a.copy(a['data/demo_2'],b['data'],name='demo_0')
    a,_=source_catalog(first);b,_=source_catalog(second)
    assert a['demo_2']['source_id']==b['demo_0']['source_id']
    plan=source_split(second,seed=2)
    duplicate_names=[n for n,r in plan['records'].items() if r['source_id']==a['demo_2']['source_id']]
    assert len({plan['assignments'][n] for n in duplicate_names})==1


def test_seed_partition_is_stable_after_unsplit_subset_renumber(tmp_path):
    source=tmp_path/'source.hdf5';source_fixture(source,6)
    copied=export_subset(source,tmp_path/'copied.hdf5',[f'demo_{i}' for i in reversed(range(6))])
    plans=[source_split(p,.5,seed=17) for p in (source,copied)]
    by_source=[{plan['records'][n]['source_id']:split for n,split in plan['assignments'].items()} for plan in plans]
    assert by_source[0]==by_source[1]


def training_entry(plan,name,split,id):
    return {'id':id,'group_id':'UNKNOWN:original_subtasks','instruction':'pick','split':split,
            'source_partition':plan['partitions'][split],'source_lineage':plan['records'][name]}


def test_shared_original_across_independently_created_partitions_is_rejected(tmp_path):
    src=tmp_path/'source.hdf5';source_fixture(src,4)
    a=source_split(src,assignments={'demo_0':'train','demo_1':'train','demo_2':'val','demo_3':'val'})
    b=source_split(src,assignments={'demo_0':'val','demo_1':'train','demo_2':'train','demo_3':'val'})
    with pytest.raises(ValueError,match='leakage'):
        split_episodes([training_entry(a,'demo_0','train','a'),training_entry(b,'demo_0','val','b')])
    entries,_=split_episodes([training_entry(a,'demo_0','train','a'),training_entry(a,'demo_2','val','b')])
    assert [e['split'] for e in entries]==['train','val']
    assert all(e['source_group']=='UNKNOWN:original_subtasks' for e in entries)
    with pytest.raises(ValueError,match='overwritten'):split_episodes(entries,{'a':'val','b':'train'})
    with pytest.raises(ValueError,match='evidence'):split_episodes([{**e,'source_lineage':None} for e in entries])


def test_mimic_generated_unknown_requires_real_input_partition_evidence(tmp_path):
    from soarm101_lab.real2sim.precision import fingerprint
    src=tmp_path/'source.hdf5';source_fixture(src,4);p=source_split(src)['partitions']['train']
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
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([]);src=tmp_path/'source.hdf5';source_fixture(src)
    dialog=CurationDialog(Registry(tmp_path/'registry'))
    monkeypatch.setattr(dialog,'run_job',lambda fn,done:done(fn()))
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
    from soarm101_lab.workflow.artifacts import Registry
    d=CurationDialog(Registry(tmp_path/'registry'))
    monkeypatch.setattr(d,'run_job',lambda fn,done:done(fn()))
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
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([]);src=tmp_path/'source.hdf5';source_fixture(src,20)
    d=CurationDialog(Registry(tmp_path/'registry'));d.rows=analyze(src);d.path=str(src);queued=[]
    monkeypatch.setattr(d,'run_job',lambda fn,done:queued.append((fn,done)))
    d.preview_source_split();fn,done=queued.pop();old_plan=fn();d.train_ratio.setValue(70);done(old_plan)
    assert not d.split_plan and not any(r.get('split') for r in d.rows)
    d.close()


@pytest.mark.parametrize('mode',['subset','planned','locked'])
def test_curation_single_save_exports_selection_or_plan_and_preserves_partition(tmp_path,monkeypatch,mode):
    from PyQt6.QtWidgets import QApplication,QFileDialog,QPushButton
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.artifacts import Registry
    app=QApplication.instance() or QApplication([])
    original=tmp_path/'source.hdf5';source_fixture(original,6);original_bytes=original.read_bytes()
    src=Path(export_source_splits(original,tmp_path/'existing_split',.5)['outputs']['train']) if mode=='locked' else original
    before=src.read_bytes();records,partition=source_catalog(src)
    registry=Registry(tmp_path/'registry');parent=registry.register('source',str(src))
    d=CurationDialog(registry);errors=[]
    monkeypatch.setattr(d,'run_job',lambda fn,done:done(fn()))
    monkeypatch.setattr(d,'error',errors.append)
    assert not d.export_button.isEnabled()
    d.rows=analyze(src);d.path=str(src);d.kind='source'
    d.set_excluded({d.rows[-1]['name']})
    names=[r['name'] for r in d.rows if r['name'] not in d.excluded]
    if mode=='planned':d.preview_source_split()
    d.storage_name.setText('분할_1' if mode=='planned' else '선별_1')
    assert not errors and d.export_button.isEnabled()
    assert [b for b in d.findChildren(QPushButton) if '저장' in b.text()]==[d.export_button]
    assert d.export_button.text()==('Train / Val 저장' if mode=='planned' else '데이터 저장')
    dest=tmp_path/('new_split' if mode=='planned' else 'curated.hdf5');requests=[]
    def choose_destination(*args):
        requests.append(args);return str(dest),''
    monkeypatch.setattr(QFileDialog,'getSaveFileName',choose_destination)
    d.export_button.click()
    dest=Path(d.saved_collection) if mode=='planned' else Path(registry.latest({'source'})['path'])
    assert not errors and not requests and dest.exists()
    if mode=='planned':
        result=json.loads((dest/'source_split.json').read_text())
        assert result['assignments']==d.split_plan['assignments']
        outputs={s:source_catalog(dest/s/'dataset.hdf5') for s in ('train','val')}
        for split,(saved,saved_partition) in outputs.items():
            expected={records[n]['source_id'] for n in names if d.split_plan['assignments'][n]==split}
            assert {r['source_id'] for r in saved.values()}==expected
            assert saved_partition==d.split_plan['partitions'][split]
    else:
        saved,saved_partition=source_catalog(dest)
        expected={records[n]['source_id'] for n in names}
        assert {r['source_id'] for r in saved.values()}==expected
        if partition:
            assert saved_partition['split']==partition['split']
            assert saved_partition['parent_partition_sha256']==partition['sha256']
            assert set(saved_partition['source_ids'])==expected
        else:assert saved_partition is None
    registered=[a for a in registry.all() if a['id']!=parent['id'] and not a.get('execution',{}).get('origin_source_id')]
    assert len(registered)==(2 if mode=='planned' else 1)
    origin=next((a for a in registry.all() if a.get('execution',{}).get('origin_source_id')==parent['id']),parent)
    assert all(a['parents']==[origin['id']] for a in registered)
    assert src.read_bytes()==before and original.read_bytes()==original_bytes
    d.close()
