"""Replay/Annotation native Path boundaries, without IDs or registration."""
import copy,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import h5py
import pytest

from test_common_mimic import episode
from test_workspaces import published
from soarm101_lab.workflow.operations import Operations,execute_request
from soarm101_lab.workflow.worker_request import episode_context
from soarm101_lab.workflow.paths import discover_inputs
from soarm101_lab.utils.artifact_identity import fingerprint


@pytest.fixture
def source(episode,published,tmp_path):
    from soarm101_lab.real2sim.workspaces.package import load,metadata
    from soarm101_lab.real2sim.workspaces.tasks import configured
    from soarm101_lab.real2sim.workspaces.demo import coordinate_contract
    from soarm101_lab.data.episodes import DatasetWriter
    w=configured(load(published[3]));values,attrs,videos,args,_=episode
    contract=coordinate_contract(w)
    args={**args,'so101_coordinates':contract,'workspace':metadata(w),'workspace_path':str(w['root']),
          'scene_configuration':w['scene_configuration'],'task_definitions':[w['task_definition']]}
    attrs={**attrs,'task_definition':json.dumps(w['task_definition']),'scene_configuration':json.dumps(w['scene_configuration'])}
    path=tmp_path/'원본 공백.hdf5'
    with DatasetWriter(path,args,contract) as writer:writer.commit(values,attrs,videos)
    return path,w,values,attrs,videos,args,contract


@pytest.mark.parametrize('stage',['annotate'])
def test_plan_and_full_worker_without_registry(source,tmp_path,monkeypatch,stage):
    from soarm101_lab.workflow import operations
    from soarm101_lab.simulation import environment
    path,w,*_=source;root=tmp_path/'독립 root';before=fingerprint(path)
    assert not hasattr(operations,'Registry')
    service=Operations(root=root);opts={'backend':'mujoco','storage_name':'train_자유_이름','num_envs':2}
    plan=service.plan(stage,str(path),str(w['root']),opts,dry=True)
    assert plan['context']['tasks']==[w['task_definition']]
    run=root/'runs/probe';run.mkdir(parents=True)
    plan=service.plan(stage,str(path),str(w['root']),opts,run_dir=run)
    request=json.loads((run/'request.json').read_text())
    assert request['input']==str(path) and not isinstance(request['input'],dict)
    def engine(**kw):raise RuntimeError('native input validated before engine')
    monkeypatch.setattr(environment,'create_mujoco',engine)
    with pytest.raises(RuntimeError,match='before engine'):execute_request(request)
    assert not (root/'artifacts').exists() and fingerprint(path)==before
    assert not list(Path(plan['output']).glob('*.hdf5'))


@pytest.mark.parametrize('stage',['annotate'])
@pytest.mark.parametrize('stopped',[False,True])
def test_operations_lifecycle_publishes_paths_and_preserves_source(source,tmp_path,monkeypatch,stage,stopped):
    from soarm101_lab.workflow import operations
    from soarm101_lab.data.episodes import DatasetWriter
    path,w,values,attrs,videos,args,contract=source;root=tmp_path/'storage'
    service=Operations(root=root);before=fingerprint(path)
    class Process:
        pid=123
        def wait(self,**kw):return 0
    def launch(argv,**kw):
        request=json.loads(Path(argv[-1]).read_text());target=Path(request['output']);target.mkdir()
        report={'passed':True,'stopped':stopped,'episodes':[]}
        if stage=='annotate':
            with DatasetWriter(target/'dataset.hdf5',args,contract) as writer:writer.commit(values,attrs,videos)
            report['dataset']=str(target/'dataset.hdf5')
        (target/'report.json').write_text(json.dumps(report))
        if stopped:service.cancelled=True
        return Process()
    monkeypatch.setattr(operations.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='fixture'))
    monkeypatch.setattr(operations.subprocess,'Popen',launch)
    result=service.run(stage,str(path),str(w['root']),{'backend':'mujoco','storage_name':'val_한글_임의'})
    assert result['status']==('STOPPED' if stopped else 'SUCCEEDED') and 'id' not in result
    assert not (root/'artifacts').exists() and fingerprint(path)==before
    assert discover_inputs(Path(result['report']).parent)
    state=json.loads(next((root/'runs').glob('*/state.json')).read_text())
    assert state['source']==str(path) and state['result_path']==result['path']
    assert not list(root.rglob('*.workflow-lock'))
    with pytest.raises(FileExistsError):service.run(stage,str(path),str(w['root']),{'storage_name':'val_한글_임의'})


@pytest.mark.parametrize('defect',['missing','corrupt','empty','workspace','scene','smoke','generated','locked','partial'])
def test_rejected_native_headers_without_registry(source,tmp_path,defect):
    path,w,*_=source;workspace=str(w['root'])
    if defect=='missing':path.unlink()
    elif defect=='corrupt':path.write_bytes(b'not HDF5')
    elif defect=='empty':
        with h5py.File(path,'a') as h:del h['data/demo_0']
    elif defect=='locked':path.with_name('.'+path.name+'.workflow-lock').write_text('busy')
    else:
        with h5py.File(path,'a') as h:
            args=json.loads(h['data'].attrs['env_args'])
            if defect=='workspace':args['workspace']['sha256']='wrong'
            if defect=='scene':args['scene_configuration']['sha256']='wrong'
            if defect=='smoke':args['validation_only']=True
            if defect=='generated':args['source_dataset']='recorded annotation'
            if defect=='partial':h['data/demo_0'].attrs['committed']=False
            h['data'].attrs['env_args']=json.dumps(args)
    service=Operations(root=tmp_path/'independent')
    state,reason=service.readiness('annotate',str(path),workspace,{'storage_name':'new'})
    assert state=='NOT_READY' and reason and not (service.root/'artifacts').exists()


def test_worker_revalidates_after_plan_and_no_environment_on_corruption(source,tmp_path,monkeypatch):
    from soarm101_lab.simulation import environment
    path,w,*_=source;service=Operations(root=tmp_path/'root');run=tmp_path/'run';run.mkdir()
    service.plan('annotate',str(path),str(w['root']),{'storage_name':'fresh'},run_dir=run)
    request=json.loads((run/'request.json').read_text())
    with h5py.File(path,'a') as h:h['data/demo_0/joint_targets'][0,0]=float('nan')
    monkeypatch.setattr(environment,'create_mujoco',lambda **kw:pytest.fail('Corrupt input reached engine'))
    with pytest.raises(ValueError):execute_request(request)


def test_path_inputs_do_not_accept_ids_and_manual_parallel_rejected(source,tmp_path):
    path,w,*_=source;service=Operations(root=tmp_path/'root')
    assert service.readiness('annotate',{'id':'old','path':str(path)},str(w['root']),{})[0]=='NOT_READY'
    assert service.readiness('annotate',str(path),str(w['root']),{'auto':False,'num_envs':2,'storage_name':'new'})[0]=='NOT_READY'


def test_cli_ready_uses_path_without_registry(source,tmp_path):
    path,w,*_=source;root=tmp_path/'独立 CLI';opts=tmp_path/'opts.json';opts.write_text('{"backend":"mujoco"}')
    script=Path(__file__).resolve().parents[2]/'scripts/real2sim/workflow.py'
    result=subprocess.run([sys.executable,str(script),'ready','--stage','annotate','--input',str(path),'--workspace',str(w['root']),
                           '--root',str(root),'--options',str(opts),'--storage-name','自由'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==['READY',''] and not root.exists()


def test_worker_cannot_publish_or_compress_the_input_as_its_result(source,tmp_path,monkeypatch):
    from soarm101_lab.workflow import operations
    path,w,*_=source;before=fingerprint(path);service=Operations(root=tmp_path/'control')
    class Process:
        def wait(self,**kw):return 0
    def launch(argv,**kw):
        request=json.loads(Path(argv[-1]).read_text());target=Path(request['output']);target.mkdir()
        (target/'report.json').write_text(json.dumps({'passed':True,'dataset':str(path)}))
        return Process()
    monkeypatch.setattr(operations.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='fixture'))
    monkeypatch.setattr(operations.subprocess,'Popen',launch)
    with pytest.raises(ValueError,match='outside reserved output'):
        service.run('annotate',str(path),str(w['root']),{'backend':'mujoco','storage_name':'protected'})
    assert fingerprint(path)==before and not list(service.root.rglob('*.workflow-lock'))


def test_direct_worker_refuses_output_inside_workspace(source,tmp_path):
    path,w,*_=source;target=w['root']/'must_not_create'
    with pytest.raises(ValueError,match='outside input Dataset'):
        execute_request({'backend':'mujoco','stage':'annotate','input':str(path),'workspace':str(w['root']),
                         'output':str(target),'options':{'generation_seed':1047}})
    assert not target.exists()


@pytest.mark.parametrize('change',['scope','files'])
def test_asset_description_is_historical_but_content_identity_is_required(source,tmp_path,change):
    path,w,*_=source
    sidecar=Path(str(path)+'.so101.json');contract=json.loads(sidecar.read_text())
    if change=='scope':contract['robot_asset']['scope']='Historical description of the same files'
    else:contract['robot_asset']['files'][next(iter(contract['robot_asset']['files']))]='different-content'
    sidecar.write_text(json.dumps(contract))
    with h5py.File(path,'a') as h:
        args=json.loads(h['data'].attrs['env_args']);args['so101_coordinates']=contract;h['data'].attrs['env_args']=json.dumps(args)
    state,reason=Operations(root=tmp_path/'root').readiness('annotate',str(path),str(w['root']),{'storage_name':'new'})
    assert state==('READY' if change=='scope' else 'NOT_READY')
    if change=='files':assert 'robot_asset' in reason


def test_default_output_cannot_escape_stage_storage_via_symlink(source,tmp_path):
    path,w,*_=source;root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'annotations').symlink_to(outside,target_is_directory=True)
    state,reason=Operations(root=root).readiness('annotate',str(path),str(w['root']),{'storage_name':'new'})
    assert state=='NOT_READY' and 'symlink' in reason and not list(outside.iterdir())


def test_cli_interrupt_cancels_only_owned_child_and_releases_lock(source,tmp_path,monkeypatch):
    from soarm101_lab.workflow import operations
    path,w,*_=source;service=Operations(root=tmp_path/'control');signals=[]
    class Process:
        pid=12345
        calls=0
        def poll(self):return None
        def wait(self,**kw):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt()
            return 130
    monkeypatch.setattr(operations.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='fixture'))
    monkeypatch.setattr(operations.subprocess,'Popen',lambda *a,**kw:Process())
    monkeypatch.setattr(operations.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(RuntimeError,match='cancelled'):
        service.run('annotate',str(path),str(w['root']),{'backend':'mujoco','storage_name':'interrupt'})
    assert signals==[(12345,operations.signal.SIGINT)]
    assert service.process is None and not list(service.root.rglob('*.workflow-lock'))
