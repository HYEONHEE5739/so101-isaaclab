"""Shared UI/CLI service: explicit stage -> argv -> validate -> register.

No shell interpolation. Each transition explicitly runs one requested stage.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import uuid
import signal
import re
from datetime import datetime
from .artifacts import Registry, inspect_artifact, output_location, resumes_in_place, validate_storage_name
from .worker_request import MUJOCO_STAGES, mujoco_request
from ..learning.minivla.config import MODEL_ROOT
from ..real2sim.storage import atomic
from ..real2sim.launcher import worker_environment
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from ..representation import require_compatible
from ..so101_dataset_contract import JOINT_SPACE, MIMIC_SPACE

STAGES = {
    'source': ({'workspace'}, 'source'),
    'replay': ({'source', 'annotated', 'datagen', 'evaluation_dataset'}, 'replay'),
    'annotate': ({'source'}, 'annotated'),
    'datagen': ({'annotated'}, 'datagen'),
    'convert': ({'datagen', 'source', 'annotated'}, 'dataset'),
    'merge': ({'dataset'}, 'dataset'),
    'train': ({'dataset'}, 'policy'),
    'hf_dataset': ({'dataset'}, 'hf_dataset'),
    'hf_policy': ({'policy'}, 'hf_policy'),
    'sim_eval': ({'policy', 'hf_policy'}, 'sim_eval'),
    'real_eval': ({'policy', 'hf_policy'}, 'real_eval'),
}


def successful_demo_target(options):
    if ('num_successful_demos' in options and 'trials' in options
            and options['num_successful_demos'] != options['trials']):
        raise ValueError('num_successful_demos와 기존 trials 값이 다릅니다. 하나만 지정하세요.')
    target = options.get('num_successful_demos', options.get('trials', 10))
    if type(target) is not int or target < 1:
        raise ValueError('num_successful_demos는 생성할 성공 데이터 수인 양의 정수여야 합니다.')
    return target


def default_python():
    # Explicitly configurable; do not use the GUI-only Python for Isaac.
    candidate = Path.home() / 'miniconda3/envs/lerobot-arena/bin/python'
    return str(candidate) if candidate.exists() else sys.executable


def run_folder_name(stage, artifact, workspace='', options=None):
    """Human-readable local timestamp and semantic task ID, never language text."""
    from .tasks import from_artifact
    from ..real2sim.workspaces.tasks import catalog
    opts = options or {}
    definitions = from_artifact(artifact)
    if stage in ('source', 'sim_eval', 'real_eval'):
        definitions = opts.get('tasks') or definitions
    label = 'no_task'
    if stage in ('sim_eval', 'real_eval') and opts.get('task_scope') == 'all':
        label = 'all_tasks'
    else:
        if not definitions:
            try:
                path = artifact['path'] if stage == 'source' else workspace
                w = load(path) if path else None
                if w:
                    definitions = from_artifact(artifact, w)
                    if not definitions and stage in ('source', 'sim_eval', 'real_eval'):
                        definitions = [t for t in catalog(w) if t['pick_object_id'] == w['task']['pick']
                                       and t['target_object_id'] == w['task']['place']]
            except (ValueError, OSError, KeyError):
                pass  # plan() records the actual validation failure in state.json.
        if len(definitions) == 1:
            label = definitions[0].get('task_id', 'no_task')
        elif definitions:
            label = f'multi_{len(definitions)}_tasks'
    safe = lambda value: re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value)).strip('_')[:80] or 'unknown'
    stage_label = 'annotation' if stage == 'annotate' else stage
    return f'{datetime.now():%Y%m%d_%H%M%S}_{safe(stage_label)}_{safe(label)}_{uuid.uuid4().hex[:8]}'


class Operations:
    def __init__(self, registry=None):
        self.registry = registry or Registry()
        self.process = None
        self.cancelled = False
        self.session = None
        self.cancel_time = None

    def cancel(self):
        self.cancelled = True
        self.cancel_time = self.cancel_time or time.monotonic()
        if self.process is not None and self.process.poll() is None:
            try: os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError: pass

    def readiness(self, stage, artifact, workspace='', options=None):
        try:
            self.plan(stage, artifact, workspace, options or {}, dry=True)
            return 'READY', ''
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            return 'NOT_READY', str(exc)

    def plan(self, stage, artifact, workspace, options, dry=False, run_dir=None):
        if stage not in STAGES or artifact['type'] not in STAGES[stage][0]:
            raise ValueError('Stage input artifact type mismatch')
        if dry:
            inspect_artifact(artifact['type'], artifact['path'])
        else:
            self.registry.verify(artifact)
        opts = dict(options)
        if stage=='sim_eval':
            from .evaluation import validate_evaluation_options
            validate_evaluation_options(opts)
        output,boundary=output_location(stage,artifact,opts,self.registry.root)
        immutable=artifact['path'] if stage=='source' else workspace
        if immutable and (Path(immutable).expanduser()/'manifest.json').is_file() and output.is_relative_to(Path(immutable).expanduser().resolve()):
            raise ValueError('Output must be outside immutable workspace')
        continuing=resumes_in_place(stage,opts,output)
        if boundary.exists() and not continuing:
            raise FileExistsError(f'이미 존재하는 결과물입니다: {boundary} · 새 저장 이름을 입력하세요. 이어 실행은 Resume 입력과 새 출력 이름을 지정하세요.')
        opts['output']=str(output)
        opts['artifact_root']=str(boundary)
        if not continuing and not options.get('output'):opts.setdefault('storage_name',boundary.stem if boundary.suffix else boundary.name)
        if stage=='merge':
            from .lerobot_merge import validate_inputs
            identifiers=opts.get('merge_datasets')
            if not isinstance(identifiers,list) or not identifiers or identifiers[0]!=artifact['id']:
                raise ValueError('merge_datasets에 첫 입력을 포함한 데이터셋 ID 목록을 지정하세요.')
            records=[self.registry.get(identifier) for identifier in identifiers]
            for record in records:
                if dry:inspect_artifact('dataset',record['path'])
                else:self.registry.verify(record)
            validate_inputs(records)
            opts['merge_artifacts']=records
        if stage=='train' and opts.get('architecture')=='minivla':
            from ..learning.minivla.config import training_config
            config_path=opts.get('model_config') or str(MODEL_ROOT/'configs/default.json')
            if opts.get('resume_checkpoint'):
                from .policy import resume_output
                config_path=str(resume_output(opts['resume_checkpoint'],'minivla')/'config.json')
            if 'save_freq' in opts and (type(opts['save_freq']) is not int or opts['save_freq']<1):raise ValueError('체크포인트 저장 간격은 1 이상의 정수여야 합니다.')
            training_config(config_path,opts)
            partition=(artifact.get('provenance',{}).get('source_partition') or {}).get('split')
            if partition and not opts.get('validation_dataset') and not opts.get('resume_checkpoint'):
                raise ValueError(f'선택한 데이터는 {partition.upper()} 전용입니다. 학습에는 Train과 Validation이 모두 필요합니다. Train 데이터를 입력으로 선택하고 Validation Dataset에서 별도로 변환한 Val 데이터를 선택하세요.')
        if stage=='train' and opts.get('validation_dataset'):
            if opts.get('architecture')!='minivla' or artifact['type']!='dataset' or artifact.get('format')=='so101.canonical/1':
                raise ValueError('Separate Validation artifact requires MiniVLA LeRobot inputs; select the converted MP4/Parquet datasets')
            validation=self.registry.get(opts['validation_dataset'])
            if validation['type']!='dataset':raise ValueError('Validation input must be a converted dataset artifact')
            if dry:inspect_artifact(validation['type'],validation['path'])
            else:self.registry.verify(validation)
            from .episodes import split_episodes, partition_evidence
            evidence=[]
            for role,record in (('train',artifact),('val',validation)):
                provenance=record.get('provenance',{});partition=provenance.get('source_partition')
                if not partition or partition.get('split')!=role:raise ValueError('Select verified train and val Source Partition datasets')
                for index in range(record['format']['total_episodes']):
                    part,proof=partition_evidence(provenance,index)
                    evidence.append({'id':f'{role}_{index}','group_id':provenance.get('source_groups',{}).get(str(index),'UNKNOWN'),
                        'source_partition':part,'source_lineage':provenance.get('episode_source_lineage',{}).get(str(index)),
                        'source_partition_input':proof})
            split_episodes(evidence)
            opts['validation_artifact']=validation
        if stage=='train' and opts.get('architecture')=='minivla' and isinstance(artifact.get('format'),dict) and not opts.get('resume_checkpoint'):
            from .episodes import validate_training_inputs
            validate_training_inputs(artifact,opts)
        if artifact.get('status') == 'SUPERSEDED':
            raise ValueError('Select the annotated artifact; this source has been replaced')
        backend = opts.setdefault('backend', 'isaac')
        if backend not in ('isaac', 'mujoco'):
            raise ValueError('Unknown simulation backend')
        opts.setdefault('python', default_python())
        if stage in ('replay','annotate','datagen'):
            return unified_plan(stage,artifact,workspace,opts,dry,run_dir)
        if stage == 'source' and artifact.get('backend', backend) != backend:
            raise ValueError('Artifact simulation backend differs from selected backend')
        if stage in ('source','replay','annotate','datagen','sim_eval','real_eval'):
            from .tasks import bind_scene_options
            opts = bind_scene_options(stage, artifact, load(artifact['path'] if stage=='source' else workspace), opts)
        canonical = artifact.get('format') in ('so101.canonical/1', 'so101.minivla-policy/1')
        if (backend == 'mujoco' and stage in ('source', 'sim_eval')) or (canonical and stage == 'train'):
            return mujoco_plan(stage, artifact, workspace, opts, dry, run_dir)
        if any(k.lower() in ('token', 'api_key', 'password', 'hf_token') for k in opts):
            raise ValueError('Use local Hugging Face login/environment authentication, not workflow options')
        mode = opts.get('mode', 'operator' if opts.get('headless', True) else 'developer')
        if mode not in ('operator', 'developer'):
            raise ValueError('mode must be operator or developer')
        opts.update(mode=mode, headless=mode == 'operator')
        path = artifact['path']
        if stage=='sim_eval' and artifact.get('format')=='so101.minivla-policy/1':
            from .policy import selected_minivla_checkpoint
            opts['checkpoint_file']=str(selected_minivla_checkpoint(path,opts.get('checkpoint_file')))
        if stage == 'source':
            workspace = path
        needs_workspace = stage in {'source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval'}
        w = load(workspace) if needs_workspace or (workspace and stage in ('convert', 'train')) else None
        if w and w['task'].get('mode') == 'geometry_review':
            raise ValueError('Geometry review Workspace: use Real/Sim preview; explicit workflow metadata is required for task execution')
        if w and artifact.get('workspace') != metadata(w):
            raise ValueError('Input workspace identity differs from selected workspace')
        from .tasks import from_artifact
        from ..real2sim.workspaces.tasks import catalog, validate
        definitions = from_artifact(artifact, w)
        if stage in ('source', 'sim_eval', 'real_eval'):
            if opts.get('task_scope') == 'all' and stage != 'source':
                definitions = [t for t in catalog(w) if t['pick_object_id'] in opts.get('active_object_ids',w['task']['dynamic'])]
            else:
                definitions = opts['tasks'] if 'tasks' in opts else definitions or [t for t in catalog(w)
                    if t['pick_object_id'] == w['task']['pick'] and t['target_object_id'] == w['task']['place']]
        elif opts.get('tasks') is not None and opts['tasks'] != definitions:
            raise ValueError('Derived stages must inherit input artifact tasks; do not relabel demonstrations')
        if w:
            definitions = [validate(t, w) for t in definitions]
            if stage == 'source':
                from ..real2sim.workspaces.reset import source_workspace
                w = source_workspace(w,opts.get('resume_data'))
            elif stage in ('replay','annotate','datagen'):
                from .tasks import recorded_scene
                from ..real2sim.workspaces.tasks import configured
                w = configured(w,reset=recorded_scene(artifact,w)['reset'])
            if stage in ('source','sim_eval','real_eval','replay','annotate','datagen'):
                from ..real2sim.workspaces.tasks import configured
                for definition in definitions:
                    configured(w,definition,opts['active_object_ids'],backend)
                if definitions:opts['scene_configuration']=configured(w,definitions[0],opts['active_object_ids'],backend)['scene_configuration']
        if stage in ('source', 'annotate', 'datagen') and len(definitions) != 1:
            raise ValueError('Select one explicit task; annotation/datagen input must be task-homogeneous')
        if stage in ('convert', 'train') and not definitions:
            raise ValueError('Input has no explicit Task Definition; select its workspace and re-import task metadata')
        if stage in ('sim_eval', 'real_eval') and (not definitions or int(opts.get('episodes', 1)) < 1 or int(opts.get('max_steps', 300)) < 1):
            raise ValueError('Evaluation requires selected tasks and positive episodes/max_steps')
        if definitions:
            if stage in ('source', 'sim_eval', 'real_eval') and len({t['task_id'] for t in definitions}) != len(definitions):
                raise ValueError('Duplicate task_id in selection')
            opts['tasks'] = definitions
            opts['task'] = definitions[0]['language_instruction']
        if w and artifact.get('coordinates') and stage!='convert':
            semantic = JOINT_SPACE if stage in ('sim_eval', 'real_eval', 'train') else MIMIC_SPACE
            require_compatible(artifact['coordinates'], coordinate_contract(w), semantic)
        if stage in ('train', 'hf_dataset', 'hf_policy', 'sim_eval', 'real_eval') and not (canonical and stage.startswith('hf_')):
            coords = artifact.get('coordinates')
            if not coords or coords.get('action_space') != JOINT_SPACE:
                raise ValueError('Policy stages require model joint-radian actions')
        if stage in ('source', 'datagen') and opts.get('resume_data'):
            from .data_resume import validate as validate_resume
            count = validate_resume(stage, opts['resume_data'], metadata(w), definitions,
                                    path if stage == 'datagen' else None, opts.get('scene_configuration'), w)
            key = 'episodes' if stage == 'source' else 'num_successful_demos'
            target_count = int(opts.get(key, 1))
            if target_count <= count:
                raise ValueError(f'총 목표 episode 수는 기존 {count}개보다 커야 합니다.')
            opts['resume_completed'] = count
            opts['resume_target'] = target_count
            opts[key] = target_count - count
        if stage == 'train' and opts.get('resume_checkpoint') and opts.get('architecture') != 'minivla':
            from .policy import resume_checkpoint
            resume_checkpoint(opts['resume_checkpoint'], artifact, opts)
        required = {'source': ('port',),
                    'convert': ('repo_id',), 'hf_dataset': ('repo_id',), 'hf_policy': ('repo_id',),
                    'real_eval': ('port', 'follower_id')}.get(stage, ())
        for name in required:
            if not opts.get(name):
                raise ValueError(f'Missing option: {name}')
        if stage == 'source':
            import math
            for key, default, positive in [('episode_time_s', 20, True), ('reset_time_s', 5, False)]:
                value = float(opts.get(key, default))
                if not math.isfinite(value) or value < 0 or (positive and value < .1):
                    raise ValueError('Invalid source timing: ' + key)
        if stage == 'real_eval':
            for name in w['task'].get('observation_cameras',list(w['profile']['cameras'])):
                if not (opts.get('camera_devices',{}).get(name) or opts.get(name+'_device')):
                    raise ValueError('Real camera device required: '+name)
        if stage == 'real_eval' and not opts.get('enable_motion'):
            raise ValueError('Real evaluation requires explicit enable_motion=true')
        if stage in ('hf_dataset', 'hf_policy') and type(opts.get('private')) is not bool:
            raise ValueError('Choose private=true/false explicitly')
        if stage == 'train': opts['workspace'] = str(w['root']) if w else workspace
        if dry:
            return {'stage': stage}
        run_dir = Path(run_dir)
        python = opts.get('python') or default_python()
        device = opts.get('device', 'cuda:0')
        output = Path(opts['output']).expanduser().resolve()
        if stage == 'train' and opts.get('resume_checkpoint'):
            from .policy import resume_output
            original_output = resume_output(opts['resume_checkpoint'], opts.get('architecture', 'smolvla'))
            if opts.get('architecture') != 'minivla':
                opts['resume_output'] = str(output)
            elif output.resolve() == original_output.resolve():
                raise ValueError('MiniVLA Resume은 기존 체크포인트를 보존하도록 새 출력 폴더를 선택하세요.')
        cli = [python, '-u']
        ws = str(w['root']) if w else str(workspace)
        headless = ['--headless'] if opts['headless'] else []
        if stage == 'source':
            cli += ['scripts/real2sim/workspace.py', 'source-demo', ws, '--port', opts['port'],
                    '--dataset_file', str(output), '--num_episodes', str(opts.get('episodes', 1)),
                    '--episode_time_s', str(opts.get('episode_time_s', 20)), '--reset_time_s', str(opts.get('reset_time_s', 5)), '--device', device, *headless]
            if opts.get('identity_sim_for_validation'):cli.append('--identity_sim_for_validation')
            if opts.get('resume_data'):cli+=['--resume_data',str(opts['resume_data'])]
        elif stage == 'convert':
            cli += ['scripts/tools/convert_isaac2lerobot.py', '--input_file', path, '--root', str(output),
                    '--repo_id', opts['repo_id'], '--task', opts.get('task', ''), '--action_source', 'joint_targets']
            if opts.get('include_failed'):cli.append('--include_failed')
            if artifact.get('format')=='so101.common-episode/2':
                fps=1/float(artifact['env_args']['control_dt'])
                if abs(fps-round(fps))>1e-8:raise ValueError('This LeRobot version requires integer FPS; refusing to change recorded timing')
                cli+=['--fps',str(round(fps))]
        else:
            spec = {'stage': stage, 'input': artifact, 'workspace': ws, 'options': opts, 'output': str(output)}
            atomic(run_dir / 'request.json', spec)
            cli += ['scripts/real2sim/workflow.py', 'worker', '--request', str(run_dir / 'request.json')]
        return {'argv': cli, 'output': str(output), 'type': STAGES[stage][1], 'workspace': ws, 'options': opts}

    def run(self, stage, artifact, workspace='', options=None, notify=lambda message: None):
        self.cancelled = False
        self.cancel_time = None
        run_dir = self.registry.root / 'runs' / run_folder_name(stage, artifact, workspace, options)
        run_dir.mkdir(parents=True)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
        state = {'code_commit': commit.stdout.strip(), 'code_dirty': bool(dirty.stdout), 'id': run_dir.name, 'stage': stage, 'source': artifact['id'], 'status': 'READY', 'created_at': time.time()}
        reservation=None
        try:
            plan = self.plan(stage, artifact, workspace, options or {}, run_dir=run_dir)
            if not (run_dir/'request.json').exists():
                atomic(run_dir/'request.json',{'stage':stage,'input':artifact,'workspace':workspace,'options':plan['options'],'output':plan['output']})
            target = Path(plan['output'])
            boundary=Path(plan['options'].get('artifact_root',target))
            boundary.parent.mkdir(parents=True,exist_ok=True)
            lock=boundary.with_name('.'+boundary.name+'.workflow-lock')
            try:
                with lock.open('x') as stream:stream.write(str(run_dir))
                reservation=lock
            except FileExistsError:
                raise FileExistsError(f'다른 실행이 이 저장 위치를 사용 중입니다: {boundary}')
            continuing=resumes_in_place(stage,plan['options'],target)
            if boundary.exists() and not continuing:
                raise FileExistsError(f'이미 존재하는 결과물입니다: {boundary} · 새 저장 이름을 입력하세요.')
            if target.exists() and not continuing:
                raise FileExistsError('Choose a new output path; artifact overwrite is disabled')
            target.parent.mkdir(parents=True, exist_ok=True)
            state.update(status='RUNNING', plan=plan)
            atomic(run_dir / 'state.json', state)
            env = worker_environment(ROOT)
            self.session = run_dir / 'presentation'
            self.session.mkdir()
            atomic(self.session / 'config.json', {'stage': stage, 'mode': plan['options']['mode'], 'tasks': plan['options'].get('tasks', []),'started_at':state['created_at'],
                   'real_cameras': {} if stage == 'real_eval' else plan['options'].get('real_cameras', {})})
            from .stages import SIM_STAGES
            if stage in SIM_STAGES and plan['options'].get('backend', 'isaac') == 'isaac' and sys.platform.startswith('linux'):
                env.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
                env.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
            if plan['options'].get('backend') == 'mujoco':
                for key in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', '__NV_PRIME_RENDER_OFFLOAD', '__GLX_VENDOR_LIBRARY_NAME', 'LIBGL_ALWAYS_SOFTWARE', '__EGL_VENDOR_LIBRARY_FILENAMES'):
                    env.pop(key, None)
            env['PYTHONUNBUFFERED'] = '1'
            env['SO101_WORKFLOW_SESSION'] = str(self.session)
            env.pop('SO101_TASK_SELECTION', None)
            if plan['options'].get('tasks'):
                atomic(run_dir / 'tasks.json', {'tasks': plan['options']['tasks'],
                    'active_object_ids': plan['options'].get('active_object_ids')})
                env['SO101_TASK_SELECTION'] = str(run_dir / 'tasks.json')
            env.pop('HEADLESS', None)
            if plan['options']['headless']:
                env['HEADLESS'] = '1'
            notify('실행: ' + json.dumps(plan['argv'], ensure_ascii=False))
            with (run_dir / 'process.log').open('w') as log:
                process = subprocess.Popen(plan['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.process = process
                while True:
                    try:
                        code = process.wait(timeout=.5)
                        break
                    except subprocess.TimeoutExpired:
                        if self.cancelled and self.cancel_time and stage not in ('source', 'datagen'):
                            elapsed = time.monotonic() - self.cancel_time
                            grace=60 if stage=='train' and plan['options'].get('architecture')=='minivla' else 5
                            if elapsed > grace:
                                try: os.killpg(process.pid, signal.SIGKILL if elapsed > grace+5 else signal.SIGTERM)
                                except ProcessLookupError: pass
            resumable_stop = self.cancelled and stage in ('source', 'datagen')
            if self.cancelled and not resumable_stop:
                raise RuntimeError('Workflow cancelled; partial outputs were not registered')
            if resumable_stop:
                notify('중지 완료 · 저장된 episode 검증 중 (진행 중 episode는 제외)')
            report_path=target/'report.json'
            if stage=='datagen' and plan.get('worker_report') and report_path.is_file():
                stopped_report=json.loads(report_path.read_text())
                if resumable_stop and stopped_report.get('stopped') and stopped_report.get('successes')==0:
                    state.update(status='STOPPED')
                    notify('중지됨 · 완료된 성공 데이터 0개 · 새 데이터는 등록하지 않았습니다.')
                    return {'status':'STOPPED','no_artifact':True}
            if stage in ('source', 'datagen') and not plan.get('worker_report') and (code == 0 or resumable_stop):
                inspect_artifact(stage, target)
                if plan['options'].get('resume_data'):
                    from .data_resume import combine
                    combine(plan['options']['resume_data'], target)
                if resumable_stop:
                    code = 0
            if code:
                raise RuntimeError(f'Process exit {code}; log: {run_dir / "process.log"}')
            if plan.get('worker_report'):
                report_path = target / 'report.json'
                if not report_path.exists() and (target/'failure.json').is_file():
                    failure=json.loads((target/'failure.json').read_text())
                    raise ValueError(f"{failure['error']} · {target/'failure.json'}")
                report = json.loads(report_path.read_text())
                if not report.get('passed'):
                    raise ValueError('Worker validation failed: ' + str(report_path))
                key = plan.get('result_key')
                target = Path(report[key]) if key else report_path
                if target.suffix == '.pt': target = target.parent
            elif stage == 'train':
                if plan['options'].get('architecture') == 'minivla':
                    target = Path(plan['options'].get('resume_output', target / 'policy'))
                else:
                    target = target / 'checkpoints/last/pretrained_model'
            if target.suffix in ('.hdf5', '.h5') and stage in ('source', 'annotate', 'datagen'):
                from .data_resume import compress
                saving = compress(target)
                if saving['saved_bytes']:
                    notify(f"HDF5 무손실 압축: {saving['before_bytes']/2**20:.1f} → {saving['after_bytes']/2**20:.1f} MiB")
            # Validate artifact content even when an Isaac shutdown masks a process failure.
            completed_state = {**state, 'status': 'STOPPED' if resumable_stop else 'SUCCEEDED'}
            partitioned=artifact.get('source_partition') or artifact.get('provenance',{}).get('source_partition')
            replace_annotation = stage == 'annotate' and artifact['type'] == 'source' and not partitioned and not plan['options'].get('artifact_root')
            if replace_annotation and target.suffix in ('.hdf5','.h5'):
                import h5py
                with h5py.File(target,'r') as h:
                    if any(g.attrs.get('external_videos') for n,g in h['data'].items() if n.startswith('demo_')):
                        replace_annotation=False
                        notify('외부 영상 Annotation · Source 원본 보존 · 별도 annotated 데이터로 진행')
            if replace_annotation and inspect_artifact('annotated', target)['episode_count'] != artifact['episode_count']:
                notify('일부 episode만 Annotation 통과 · source 원본 보존 · 통과한 별도 annotated 데이터로 진행')
                replace_annotation = False
            if replace_annotation:
                result = self.registry.promote_annotation(artifact, target, extra=completed_state)
                notify('Annotation 검증 완료 · 기존 source 경로를 annotated 데이터로 전환')
                if plan.get('worker_report'):
                    report[plan['result_key']] = result['path']
                    atomic(report_path, report)
                target = Path(result['path'])
            else:
                parents=plan['options'].get('merge_datasets',[artifact['id']]) if stage=='merge' else [artifact['id']]
                if plan['options'].get('validation_artifact'):parents.append(plan['options']['validation_artifact']['id'])
                display_name=plan['options'].get('storage_name')
                if continuing and not display_name:
                    for previous in reversed(self.registry.all()):
                        if previous['type']!='policy' or not Path(previous['path']).is_relative_to(Path(plan['output'])):continue
                        try:display_name=validate_storage_name(previous.get('display_name'))
                        except ValueError:continue
                        break
                result = self.registry.register(plan['type'], target, parents=parents, extra=completed_state,display_name=display_name)
            if stage=='sim_eval':
                details=json.loads(target.read_text())
                for item in details.get('datasets',[]):
                    data_path=(target.parent/item['path']).resolve()
                    if not data_path.is_relative_to(target.parent.resolve()):raise ValueError('Evaluation dataset outside output')
                    portable=lambda value:''.join(c if c.isalnum() or c in '_-' else '_' for c in value)
                    name=portable(result['display_name']).encode('utf-8')[:135].decode('utf-8','ignore')[:60]
                    label='성공' if item['outcome']=='success' else '실패'
                    self.registry.register('evaluation_dataset',data_path,parents=[result['id']],
                        display_name=label+'_'+name+'_'+portable(data_path.parent.parent.name)[:10])
            state.update(status='STOPPED' if resumable_stop else 'SUCCEEDED', artifact=result['id'])
            notify('완료: ' + str(target))
            return result
        except Exception as exc:
            state.update(status='FAILED', error=str(exc))
            raise
        finally:
            self.process = None
            if reservation:reservation.unlink(missing_ok=True)
            atomic(run_dir / 'state.json', state)



def mujoco_plan(stage, artifact, workspace, options, dry=False, run_dir=None):
    """Translate common stage names to the existing engine worker protocol."""
    from .tasks import from_artifact
    from ..real2sim.workspaces.tasks import catalog, validate
    if stage in ('hf_dataset', 'hf_policy', 'real_eval'):
        raise ValueError('This operation is handled by the common workflow')
    workspace = artifact['path'] if stage == 'source' else workspace
    w = load(workspace)
    if artifact.get('workspace') != metadata(w):
        raise ValueError('Input workspace identity differs from selected workspace')
    from .tasks import bind_scene_options
    opts = bind_scene_options(stage,artifact,w,options)
    if stage == 'train' and opts.get('resume_checkpoint'):
        from .policy import resume_output
        # Non-MiniVLA training of canonical data delegates to its LeRobot child.
        architecture = opts.get('architecture', 'minivla')
        opts['resume_output'] = str(resume_output(opts['resume_checkpoint'], architecture))
        if architecture == 'minivla':
            provenance = json.loads((Path(opts['resume_output']) / 'project_provenance.json').read_text())
            if Path(provenance['source_dataset']).resolve() != Path(artifact['path']).resolve():
                raise ValueError('Resume must use the original MiniVLA training dataset')
            opts['model_config'] = str(Path(opts['resume_output']) / 'config.json')
            opts.pop('resume_output')  # MiniVLA resumes into this run's fresh policy directory.
        elif opts.get('output'):
            opts['resume_output']=str(Path(opts['output']).expanduser().resolve())
    if stage == 'sim_eval' and artifact['type'] == 'hf_policy':
        opts['remote_policy'] = json.loads(Path(artifact['path']).read_text())
    opts.setdefault('mode', 'operator'); opts['headless'] = opts['mode'] == 'operator'
    if opts['mode'] not in ('operator', 'developer'):
        raise ValueError('mode must be operator or developer')
    if any(k.lower() in ('token', 'api_key', 'password', 'hf_token') for k in opts):
        raise ValueError('Use local Hugging Face login/environment authentication, not workflow options')
    opts.setdefault('preview_cameras', True)
    definitions = from_artifact(artifact, w)
    if stage in ('source', 'sim_eval'):
        definitions = opts.get('tasks') or definitions or [t for t in catalog(w)
            if t['pick_object_id'] == w['task']['pick'] and t['target_object_id'] == w['task']['place']]
    if stage == 'sim_eval' and opts.get('task_scope') == 'all': definitions = [t for t in catalog(w) if t['pick_object_id'] in opts.get('active_object_ids',w['task']['dynamic'])]
    definitions = [validate(t, w) for t in definitions]
    if stage == 'source':
        from ..real2sim.workspaces.reset import source_workspace
        w = source_workspace(w,opts.get('resume_data'))
    elif stage in ('replay','annotate','datagen'):
        from .tasks import recorded_scene
        from ..real2sim.workspaces.tasks import configured
        w = configured(w,reset=recorded_scene(artifact,w)['reset'])
    if stage in ('source','sim_eval','replay','annotate','datagen'):
        from ..real2sim.workspaces.tasks import configured
        for definition in definitions:configured(w,definition,opts['active_object_ids'], 'mujoco')
        if definitions:opts['scene_configuration']=configured(w,definitions[0],opts['active_object_ids'],'mujoco')['scene_configuration']
    if not definitions or (stage in ('source', 'annotate', 'datagen') and len(definitions) != 1):
        raise ValueError('Source/Annotation/Datagen require one task')
    opts.update(tasks=definitions, task_definition=definitions[0], instruction=definitions[0]['language_instruction'])
    if stage=='source' and opts.get('resume_data'):
        from .data_resume import validate as validate_resume
        count=validate_resume('source',opts['resume_data'],metadata(w),definitions,
                              scene_configuration=opts.get('scene_configuration'),workspace_definition=w)
        target=int(opts.get('episodes',1))
        if target<=count:
            raise ValueError(f'총 목표 Episodes는 기존 {count}개보다 커야 합니다. {count+1} 이상을 입력하세요.')
        opts.update(resume_completed=count,resume_target=target)
    renderer = opts.get('renderer', 'egl')
    config = opts.get('model_config', str(MODEL_ROOT / 'configs/default.json'))
    if stage == 'sim_eval' and artifact.get('format') == 'so101.minivla-policy/1':
        config = str(Path(artifact['path']) / 'config.json')
    if stage == 'sim_eval':
        from .evaluation import evaluation_seed
        opts['seed'] = evaluation_seed(opts)
    target_stage = {'sim_eval': 'rollout'}.get(stage, stage)
    checkpoint = str(Path(artifact['path']) / 'best.pt') if artifact.get('format') == 'so101.minivla-policy/1' else artifact['path']
    if stage=='sim_eval' and artifact.get('format')=='so101.minivla-policy/1':
        from .policy import selected_minivla_checkpoint
        checkpoint=str(selected_minivla_checkpoint(artifact['path'],opts.get('checkpoint_file')))
    request = mujoco_request(target_stage, opts['python'], workspace, artifact['path'], config, checkpoint, renderer, opts)
    if dry:
        return {'stage': stage}
    run_dir = Path(run_dir)
    output = output_location(stage,artifact,opts,run_dir.parent.parent)[0]
    request['output'] = str(output)
    request['backend'] = 'mujoco'
    atomic(run_dir / 'request.json', request)
    result_kind = {'convert': 'dataset', 'train': 'policy', 'sim_eval': 'sim_eval', 'annotate': 'annotated'}.get(stage, stage)
    result_key = {'source': 'dataset', 'annotate': 'dataset', 'datagen': 'dataset', 'convert': 'dataset', 'train': 'checkpoint'}.get(stage)
    return {'argv': [opts['python'], '-u', 'scripts/real2sim/workflow.py', 'worker', '--request', str(run_dir / 'request.json')],
            'output': str(output), 'type': result_kind, 'workspace': str(w['root']), 'options': opts,
            'worker_report': True, 'result_key': result_key}


def unified_plan(stage,artifact,workspace,options,dry=False,run_dir=None):
    """Plan common episode stages without importing a simulation or Torch in the GUI."""
    from .tasks import bind_scene_options, from_artifact
    from ..real2sim.workspaces.tasks import validate
    if stage=='datagen':
        from ..simulation.ik import validate_solver,validate_failure_policy
        options={**options,'ik_solver':validate_solver(options.get('ik_solver','scipy_ls')),'ik_failure_policy':validate_failure_policy(options.get('ik_failure_policy','strict'))}
    if any(k.lower() in ('token','api_key','password','hf_token') for k in options):
        raise ValueError('Do not store credentials in workflow options')
    w=load(workspace)
    if artifact.get('workspace')!=metadata(w):raise ValueError('Input Workspace identity differs')
    if w['task'].get('mode')=='geometry_review':raise ValueError('Geometry review Workspace requires explicit workflow task metadata')
    opts=bind_scene_options(stage,artifact,w,options)
    definitions=[validate(t,w) for t in from_artifact(artifact,w)]
    if len(definitions)!=1:raise ValueError('Select one homogeneous task')
    if options.get('tasks') is not None and options['tasks']!=definitions:
        raise ValueError('Derived stages must inherit input artifact tasks; do not relabel demonstrations')
    opts.update(tasks=definitions,task_definition=definitions[0],instruction=definitions[0]['language_instruction'])
    opts.setdefault('mode','operator');opts['headless']=opts['mode']=='operator'
    if opts['mode'] not in ('operator','developer'):raise ValueError('Unknown execution mode')
    opts.setdefault('num_envs',1);opts.setdefault('max_attempts',10000)
    if type(opts['num_envs']) is not int or opts['num_envs'] not in (1,2,4,8):raise ValueError('Number of environments must be 1,2,4 or 8')
    if stage=='annotate' and not opts.get('auto',True) and opts['num_envs']!=1:
        raise ValueError('Manual Annotation requires one environment')
    if type(opts['max_attempts']) is not int or opts['max_attempts']<1:raise ValueError('Positive max_attempts required')
    if stage=='datagen':
        opts['num_successful_demos']=successful_demo_target(opts)
        if opts.get('resume_data'):
            from .data_resume import inspect_resume_input
            previous=inspect_resume_input('datagen',opts['resume_data'])
            if previous.get('backend')!=opts['backend'] or previous.get('workspace')!=artifact.get('workspace') or previous.get('tasks')!=definitions:
                raise ValueError('DataGen 이어 생성 데이터의 엔진/Workspace/Task가 현재 입력과 다릅니다.')
            proof=previous['env_args'].get('generation_provenance',{})
            if not artifact.get('sha256') or proof.get('source_sha256')!=artifact['sha256']:
                raise ValueError('DataGen 이어 생성은 이전 결과를 만든 동일 Annotation 입력을 선택해야 합니다.')
            if opts['num_successful_demos']<=previous['episode_count']:
                raise ValueError(f"총 목표 성공 수는 기존 {previous['episode_count']}개보다 커야 합니다.")
    if dry:return {'stage':stage}
    run_dir=Path(run_dir);output=output_location(stage,artifact,opts,run_dir.parent.parent)[0]
    request={'backend':opts['backend'],'stage':stage,'workspace':str(w['root']),'input':artifact['path'],
        'options':opts,'output':str(output),'renderer':opts.get('renderer','egl'),'episode_schema':'so101.common-episode/2'}
    atomic(run_dir/'request.json',request)
    return {'argv':[opts['python'],'-u','scripts/real2sim/workflow.py','worker','--request',str(run_dir/'request.json')],
        'output':str(output),'type':'annotated' if stage=='annotate' else stage,'workspace':str(w['root']),
        'options':opts,'worker_report':True,'result_key':'dataset' if stage!='replay' else None}


def execute_request(request):
    """Worker dispatch; GUI planning does not load engines or policy libraries."""
    if request.get('stage')=='merge':
        from .lerobot_merge import merge
        return merge(request['options']['merge_artifacts'],request['output'],request['options'].get('repo_id','local/so101_merged'))
    if request.get('stage') in ('replay','annotate','datagen'):
        from .unified import execute
        return execute(request)
    backend = request.get('backend', 'common')
    if backend == 'mujoco':
        from .mujoco import execute
        return execute(request)
    if backend != 'common':
        raise ValueError('Unknown workflow worker backend: ' + str(backend))
    stage, artifact, options = request['stage'], request['input'], request['options']
    output = Path(request['output'])
    if stage in ('hf_dataset', 'hf_policy'):
        from .hub import publish
        publish(artifact, options, output)
    elif stage == 'train':
        from .policy import train
        train(artifact, options, output)
    elif stage in ('sim_eval', 'real_eval'):
        from .evaluation import evaluate
        evaluate(request)
    else:
        raise ValueError(stage)
