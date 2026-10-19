"""Evaluation quotas, policy-column isolation and reviewable outcome datasets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from soarm101_lab.workflow.evaluation import EvaluationProgress, validate_evaluation_options
from soarm101_lab.workflow.policy import batch_actions, MiniVLAJointPolicy
from soarm101_lab.workflow.dataset import DatasetWriter, EpisodeDataset, validate_dataset
from test_common_mimic import episode


@pytest.mark.parametrize('value',[0,3,True,'4',16,-1])
def test_invalid_parallel_count_rejected(value):
    with pytest.raises(ValueError,match='num_envs'):validate_evaluation_options({'num_envs':value})


def test_evaluation_counts_finished_failures_without_retrying():
    progress=EvaluationProgress(5,4)
    reasons=['','grasp_failure','drop_failure','place_failure','']
    for episode,reason in enumerate(reasons):
        env=episode%4;progress.begin(env,episode,'red')
        progress.finish(env,dict(evaluation_index=episode,steps=30,success=not reason,failure_reason=reason))
    status=progress.progress()
    assert status['completed_episodes']==status['target_episodes']==5
    assert status['attempts']==5 and status['successes']==2 and status['failures']==3
    assert status['failure_breakdown']==dict(grasp_failure=1,drop_failure=1,place_failure=1)
    status['env_status'][0]['successes']=99
    assert progress.workers[0]['successes']==2
    from soarm101_lab.workflow.presentation import datagen_status_text
    text=datagen_status_text(status)
    assert '전체 평가 5/5회 완료' in text and '이동 중 놓침' in text and 'Sim 평가 완료' in text


def test_legacy_camera_diagnostics_do_not_relax_mimic_recording(tmp_path):
    import threading
    from soarm101_lab.workflow.adapters import SimulationAdapter
    workspace={'task':{'pick':'cube','place':'cup','observation_cameras':['front']},
               'profile':{'robot_base':{'T_world':np.eye(4).tolist()},'objects':{'cube':{'dimensions_m':[.024]*3}}}}
    with pytest.raises(ValueError,match='Side and Wrist'):
        SimulationAdapter(workspace,30,tmp_path,0,threading.Event())
    diagnostic=SimulationAdapter(workspace,30,tmp_path,0,threading.Event(),require_standard_cameras=False)
    assert diagnostic.camera_names==['front']


def test_batch_pre_post_normalization_and_slot_order():
    class Policy:
        def select_action(self,batch):
            assert batch['task']==['instruction']*4
            assert batch['observation.images.side_cam'].shape==(4,3,12,16)
            return batch['observation.state']+2
    states=np.arange(24).reshape(4,6)
    rgb=[{'side':np.full((12,16,3),i,np.uint8)} for i in range(4)]
    out=batch_actions(Policy(),lambda b:{**b,'observation.state':b['observation.state']+3},
                      lambda a:a-1,states,rgb,'instruction','cpu')
    np.testing.assert_array_equal(out,states+4)


def test_minivla_batch_shares_weights_and_isolates_chunks_and_guards():
    from soarm101_lab.learning.minivla.policy import ActionGuard
    class Model:
        cfg=SimpleNamespace(cameras=['side'],horizon=3,replan_steps=2)
        def predict(self,images,state,instruction):
            return np.repeat(np.asarray(state)[None],3,axis=0),{'fixture':True}
    policy=MiniVLAJointPolicy.__new__(MiniVLAJointPolicy)
    policy.policy=Model();policy.device='cpu'
    policy.coordinates=SimpleNamespace(to_policy=lambda q,**kw:np.asarray(q),to_model=lambda q:q)
    policy.guard=ActionGuard([-3]*5+[0],[3]*5+[1],[2]*6,30);policy.reset()
    q=torch.tensor([[.2]*5+[.2],[-.3]*5+[.7]])
    batch={'observation.state':q,'observation.images.side_cam':torch.zeros(2,3,12,16),'task':['a','b']}
    torch.testing.assert_close(policy.select_action(batch),q)
    slots=policy._batch_slots
    assert slots[0].policy is slots[1].policy is policy.policy
    assert slots[0].guard is not slots[1].guard and slots[0].remaining is not slots[1].remaining
    policy.reset();assert policy._batch_slots==[]


@pytest.fixture
def evaluated_dataset(tmp_path,episode):
    values,attrs,videos,args,contract=copy.deepcopy(episode)
    values['obs']['policy']['ee_state']=np.tile([0,0,0,1,0,0,0],(len(values['actions']),1))
    args.update(evaluation_schema='so101.evaluation-episode/1',grasp_object='cube',
                object_telemetry={'cube':{'path':'obs/datagen_info/object_pose/cube','frame':'robot_base','pose_format':'matrix4'}})
    root=tmp_path/'evaluation';items=[]
    for outcome,success in (('success',True),('failure',False)):
        path=root/outcome/'dataset.hdf5'
        row={**attrs,'success':success,'success_source':'simulation_place_condition',
             'evaluation':json.dumps({'env_id':int(success),'success':success}),
             'failure_reason':'' if success else 'drop_failure'}
        with DatasetWriter(path,args,contract) as writer:writer.commit(values,row,videos)
        items.append({'path':str(path.relative_to(root)),'outcome':outcome,'episodes':1})
    (root/'evaluation_datasets.json').write_text(json.dumps({'schema':'so101.sim-evaluation/1','datasets':items}))
    return root


def test_outcomes_keep_video_trajectory_and_failed_labels(evaluated_dataset):
    from soarm101_lab.workflow.curation import analyze,EpisodeReader
    rows=analyze(evaluated_dataset)
    assert {r['name'] for r in rows}=={'success/demo_0','failure/demo_0'}
    assert {r['recorded_success'] for r in rows}=={True,False}
    for row in rows:
        reader=EpisodeReader(row)
        payload=reader.frame(2,['side','wrist'])
        assert len(payload['ee'])==3 and set(payload['rgb_frames'])=={'side','wrist'}
        assert row['xy'] is not None or row['object_telemetry']['cube']
        assert validate_dataset(EpisodeDataset(row['data_ref']['path']))['episodes']==1
    failure=next(r for r in rows if not r['recorded_success'])
    assert failure['failure_reason']=='drop_failure'


def test_evaluation_manifest_cannot_escape_run(evaluated_dataset,tmp_path):
    from soarm101_lab.workflow.curation import analyze
    marker=evaluated_dataset/'evaluation_datasets.json';report=json.loads(marker.read_text())
    report['datasets'][0]['path']='../../outside.hdf5';marker.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='outside'):analyze(evaluated_dataset)


def test_legacy_evaluation_folder_is_discovered_without_rewriting(evaluated_dataset):
    from soarm101_lab.workflow.curation import analyze
    from soarm101_lab.workflow.artifacts import fingerprint
    path=evaluated_dataset/'failure';before=fingerprint(path)
    rows=analyze(path)
    assert len(rows)==1 and not rows[0]['recorded_success']
    assert rows[0]['failure_reason']=='drop_failure' and rows[0]['name']=='dataset/demo_0'
    assert fingerprint(path)==before and not (path/'evaluation_datasets.json').exists()


def test_failed_evaluation_playback_does_not_bypass_source_gate(evaluated_dataset,tmp_path):
    from soarm101_lab.workflow.unified import replay
    reader=EpisodeDataset(evaluated_dataset/'failure/dataset.hdf5')
    values,_,_=reader.episode(reader.names[0])
    class Adapter:
        backend=SimpleNamespace(step=lambda command:None)
        workspace={'task':{'dynamic':['cube']}}
        base_world=np.eye(4)
        buffer=SimpleNamespace(abort=lambda:None)
        tracker=SimpleNamespace(completed=False)
        def restore(self,*args):self.step_index=0
        def state(self):return {'joint_pos':np.zeros(6),'object_poses':{'cube':np.eye(4)}}
        def update_grasp(self):self.step_index+=1
        def check_success(self):return False
        def place_success(self):return False
        def failure_category(self):return 'grasp_failure'
    assert not replay(Adapter(),reader,tmp_path/'strict')['passed']
    result=replay(Adapter(),reader,tmp_path/'playback',evaluation_playback=True)
    assert result['passed'] and not result['episodes'][0]['recorded_evaluation_success']
    assert not result['episodes'][0]['replay_task_success']


def test_private_video_relocation_preserves_published_episode(tmp_path,episode):
    from soarm101_lab.utils.async_video_writer import VideoEpisode
    values,attrs,_,args,contract=episode
    video=VideoEpisode(tmp_path,30)
    with pytest.raises(ValueError,match='finalized'):video.relocate(tmp_path/'failure')
    for _ in range(len(values['actions'])):
        video.append({k:np.zeros((480,640,3),np.uint8) for k in ('side_cam','wrist_cam')})
    specs=video.finish(len(values['actions']));video.relocate(tmp_path/'failure')
    specs={k:{**v,'file':str(video.folder/v['file'])} for k,v in specs.items()}
    path=tmp_path/'failure/dataset.hdf5'
    with DatasetWriter(path,args,contract) as writer:writer.commit(values,attrs,specs,video_staging=video)
    video.abort()
    assert validate_dataset(EpisodeDataset(path))['episodes']==1


def test_evaluation_setting_and_curation_review(evaluated_dataset,tmp_path,monkeypatch):
    pytest.importorskip('PyQt6')
    from PyQt6.QtWidgets import QApplication
    from soarm101_lab.workflow.settings_form import SettingsForm
    from soarm101_lab.workflow.curation_ui import CurationDialog
    from soarm101_lab.workflow.artifacts import Registry
    from soarm101_lab.workflow.curation import analyze
    app=QApplication.instance() or QApplication([])
    form=SettingsForm();form.set_stage('sim_eval');form.setPlainText('{"num_envs":4,"episodes":5}')
    assert json.loads(form.toPlainText())['num_envs']==4
    assert form.editors['num_envs'][0].text()=='동시 평가 환경 수'
    dialog=CurationDialog(Registry(tmp_path/'registry'));dialog.select=lambda *a:None
    dialog.loaded(str(evaluated_dataset),analyze(evaluated_dataset))
    assert dialog.kind=='evaluation_dataset' and dialog.filters['failure_reason'].count()==3
    assert not dialog.train_ratio.isEnabled() and not dialog.export_button.isEnabled()
    assert dialog.scatter.point_color(dialog.rows[0],False)!=dialog.scatter.point_color(dialog.rows[1],False)
    dialog.close();form.close();app.processEvents()
