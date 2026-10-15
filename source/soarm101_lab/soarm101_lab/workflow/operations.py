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
from .artifacts import Registry, inspect_artifact
from ..learning.minivla.config import MODEL_ROOT
from ..real2sim.storage import atomic
from ..real2sim.launcher import worker_environment
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from ..representation import require_compatible
from ..so101_dataset_contract import JOINT_SPACE, MIMIC_SPACE

STAGES = {
    'source': ({'workspace'}, 'source'),
    'replay': ({'source', 'annotated', 'datagen'}, 'replay'),
    'annotate': ({'source'}, 'annotated'),
    'datagen': ({'annotated'}, 'datagen'),
    'convert': ({'datagen', 'source', 'annotated'}, 'dataset'),
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
    from .tasks import from_artifact, catalog
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
        if artifact.get('status') == 'SUPERSEDED':
            raise ValueError('Select the annotated artifact; this source has been replaced')
        backend = opts.setdefault('backend', 'isaac')
        if backend not in ('isaac', 'mujoco'):
            raise ValueError('Unknown simulation backend')
        opts.setdefault('python', default_python())
        if stage in ('source', 'replay', 'annotate', 'datagen') and artifact.get('backend', backend) != backend:
            raise ValueError('Artifact simulation backend differs from selected backend')
        if stage in ('source','replay','annotate','datagen','sim_eval','real_eval'):
            from .tasks import bind_scene_options
            opts = bind_scene_options(stage, artifact, load(artifact['path'] if stage=='source' else workspace), opts)
        canonical = artifact.get('format') in ('so101.canonical/1', 'so101.minivla-policy/1')
        if (backend == 'mujoco' and stage in ('source', 'replay', 'annotate', 'datagen', 'convert', 'sim_eval')) or (canonical and stage in ('train', 'convert')):
            if stage == 'datagen':
                opts['num_successful_demos'] = successful_demo_target(opts)
            return mujoco_plan(stage, artifact, workspace, opts, dry, run_dir)
        if stage == 'datagen':
            opts['num_successful_demos'] = successful_demo_target(opts)
            opts.pop('trials', None)
        if any(k.lower() in ('token', 'api_key', 'password', 'hf_token') for k in opts):
            raise ValueError('Use local Hugging Face login/environment authentication, not workflow options')
        mode = opts.get('mode', 'operator' if opts.get('headless', True) else 'developer')
        if mode not in ('operator', 'developer'):
            raise ValueError('mode must be operator or developer')
        opts.update(mode=mode, headless=mode == 'operator')
        path = artifact['path']
        if stage == 'source':
            workspace = path
        needs_workspace = stage in {'source', 'replay', 'annotate', 'datagen', 'sim_eval', 'real_eval'}
        w = load(workspace) if needs_workspace or (workspace and stage in ('convert', 'train')) else None
        if w and w['task'].get('mode') == 'geometry_review':
            raise ValueError('Geometry review Workspace: use Real/Sim preview; explicit workflow metadata is required for task execution')
        if w and artifact.get('workspace') != metadata(w):
            raise ValueError('Input workspace identity differs from selected workspace')
        from .tasks import catalog, validate, from_artifact
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
            if stage in ('source','sim_eval','real_eval','replay','annotate','datagen'):
                from .tasks import configured
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
        if w and artifact.get('coordinates'):
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
        output = Path(opts['output']).expanduser().resolve() if opts.get('output') else run_dir / STAGES[stage][1]
        if stage == 'train' and opts.get('resume_checkpoint'):
            from .policy import resume_output
            original_output = resume_output(opts['resume_checkpoint'], opts.get('architecture', 'smolvla'))
            if not opts.get('output'):
                output = original_output
            opts['resume_output'] = str(output)
        if stage in ('source', 'annotate', 'datagen') and not opts.get('output'):
            output = output.with_suffix('.hdf5')
        if stage in ('replay', 'sim_eval', 'real_eval', 'hf_dataset', 'hf_policy') and not opts.get('output'):
            output = output.with_suffix('.json')
        cli = [python, '-u']
        ws = str(w['root']) if w else str(workspace)
        headless = ['--headless'] if opts['headless'] else []
        if stage == 'source':
            cli += ['scripts/real2sim/workspace.py', 'source-demo', ws, '--port', opts['port'],
                    '--dataset_file', str(output), '--num_episodes', str(opts.get('episodes', 1)),
                    '--episode_time_s', str(opts.get('episode_time_s', 20)), '--reset_time_s', str(opts.get('reset_time_s', 5)), '--device', device, *headless]
            if opts.get('identity_sim_for_validation'):cli.append('--identity_sim_for_validation')
            if opts.get('resume_data'):cli+=['--resume_data',str(opts['resume_data'])]
        elif stage == 'replay':
            cli += ['scripts/real2sim/workspace.py', 'replay', ws, '--dataset', path, '--report', str(output), '--device', device, *headless]
        elif stage in ('annotate', 'datagen'):
            script = 'annotate_demos.py' if stage == 'annotate' else 'generate_dataset.py'
            cli += ['scripts/mimic/' + script, '--workspace', ws, '--input_file', path, '--output_file', str(output), '--device', device, *headless]
            if stage == 'annotate' and opts.get('auto', True):
                cli += ['--auto']
            if stage == 'datagen':
                cli += ['--generation_num_trials', str(opts['num_successful_demos']), '--num_envs', '1']
                if opts.get('generation_seed') is not None:
                    cli += ['--generation_seed', str(opts['generation_seed'])]
        elif stage == 'convert':
            cli += ['scripts/tools/convert_isaac2lerobot.py', '--input_file', path, '--root', str(output),
                    '--repo_id', opts['repo_id'], '--task', opts.get('task', ''), '--action_source', 'joint_targets']
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
        try:
            plan = self.plan(stage, artifact, workspace, options or {}, run_dir=run_dir)
            target = Path(plan['output'])
            if target.exists() and not (stage == 'train' and plan['options'].get('resume_checkpoint')):
                raise FileExistsError('Choose a new output path; artifact overwrite is disabled')
            target.parent.mkdir(parents=True, exist_ok=True)
            state.update(status='RUNNING', plan=plan)
            atomic(run_dir / 'state.json', state)
            env = worker_environment(ROOT)
            self.session = run_dir / 'presentation'
            self.session.mkdir()
            atomic(self.session / 'config.json', {'stage': stage, 'mode': plan['options']['mode'], 'tasks': plan['options'].get('tasks', []),
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
                            if elapsed > 5:
                                try: os.killpg(process.pid, signal.SIGKILL if elapsed > 10 else signal.SIGTERM)
                                except ProcessLookupError: pass
            resumable_stop = self.cancelled and stage in ('source', 'datagen')
            if self.cancelled and not resumable_stop:
                raise RuntimeError('Workflow cancelled; partial outputs were not registered')
            if resumable_stop:
                notify('중지 완료 · 저장된 episode 검증 중 (진행 중 episode는 제외)')
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
            replace_annotation = stage == 'annotate' and artifact['type'] == 'source'
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
                result = self.registry.register(plan['type'], target, parents=[artifact['id']], extra=completed_state)
            state.update(status='STOPPED' if resumable_stop else 'SUCCEEDED', artifact=result['id'])
            notify('완료: ' + str(target))
            return result
        except Exception as exc:
            state.update(status='FAILED', error=str(exc))
            raise
        finally:
            self.process = None
            atomic(run_dir / 'state.json', state)


MUJOCO_STAGES = dict(environment='Environment', source='Source', replay='Replay',
              prepare='Prepare', convert='Convert', import_='Import', annotate='Annotation',
              datagen='Datagen', train='Train', rollout='Evaluation')
MUJOCO_STAGES['import'] = MUJOCO_STAGES.pop('import_')

def mujoco_request(stage, python, workspace, input_path, config, checkpoint, renderer, options):
    if stage not in MUJOCO_STAGES:
        raise ValueError('Unsupported workflow worker stage')
    if not isinstance(options, dict):
        raise ValueError('Options must be a JSON object')
    if not Path(python).is_file():
        raise ValueError('Worker Python executable missing')
    if renderer not in ('egl', 'software-egl', 'osmesa', 'none'):
        raise ValueError('Unknown renderer')
    if renderer == 'none' and stage in ('source', 'rollout'):
        raise ValueError('This stage requires RGB rendering')
    if stage in ('environment', 'source', 'replay', 'rollout', 'import'):
        load(workspace)
    if stage in ('prepare', 'convert', 'annotate', 'datagen', 'train', 'replay'):
        if not (Path(input_path) / 'manifest.json').is_file():
            raise ValueError('Input must be a canonical MiniVLA dataset directory')
    if stage == 'import' and not Path(input_path).is_file():
        raise ValueError('HDF5 import requires an explicit recipe JSON')
    if stage == 'train':
        data = json.loads(Path(config).read_text())
        available = json.loads((Path(input_path)/'manifest.json').read_text()).get('cameras',['side'])
        preset = Path(config).resolve().parent == (MODEL_ROOT/'configs').resolve()
        if not preset and data.get('cameras') != available:
            raise ValueError('Model config camera contract differs from dataset: '+str(available))
    if stage == 'source' and (int(options.get('episodes', 1)) < 1 or float(options.get('episode_time_s', 20)) <= 0 or float(options.get('reset_time_s', 5)) < 0):
        raise ValueError('Invalid source count/timing')
    if stage == 'source' and not options.get('port'):
        raise ValueError('Leader port required; no automatic hardware discovery')
    if stage == 'convert' and not options.get('repo_id'):
        raise ValueError('Dataset repo_id required')
    if stage == 'rollout' and not Path(checkpoint).exists():
        raise ValueError('A trained policy checkpoint is required')
    return {'stage': stage, 'workspace': workspace, 'input': input_path, 'config': config,
            'checkpoint': checkpoint, 'renderer': renderer, 'options': options}


def mujoco_plan(stage, artifact, workspace, options, dry=False, run_dir=None):
    """Translate common stage names to the existing engine worker protocol."""
    from .tasks import catalog, validate, from_artifact
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
    if stage in ('source','sim_eval','replay','annotate','datagen'):
        from .tasks import configured
        for definition in definitions:configured(w,definition,opts['active_object_ids'], 'mujoco')
        if definitions:opts['scene_configuration']=configured(w,definitions[0],opts['active_object_ids'],'mujoco')['scene_configuration']
    if not definitions or (stage in ('source', 'annotate', 'datagen') and len(definitions) != 1):
        raise ValueError('Source/Annotation/Datagen require one task')
    opts.update(tasks=definitions, task_definition=definitions[0], instruction=definitions[0]['language_instruction'])
    renderer = opts.get('renderer', 'egl')
    config = opts.get('model_config', str(MODEL_ROOT / 'configs/default.json'))
    if stage == 'sim_eval' and artifact.get('format') == 'so101.minivla-policy/1':
        config = str(Path(artifact['path']) / 'config.json')
    if stage == 'sim_eval':
        from .evaluation import evaluation_seed
        opts['seed'] = evaluation_seed(opts)
    target_stage = {'sim_eval': 'rollout'}.get(stage, stage)
    checkpoint = str(Path(artifact['path']) / 'best.pt') if artifact.get('format') == 'so101.minivla-policy/1' else artifact['path']
    request = mujoco_request(target_stage, opts['python'], workspace, artifact['path'], config, checkpoint, renderer, opts)
    if dry:
        return {'stage': stage}
    run_dir = Path(run_dir)
    output = Path(opts['output']).expanduser().resolve() if opts.get('output') else run_dir / stage
    request['output'] = str(output)
    request['backend'] = 'mujoco'
    atomic(run_dir / 'request.json', request)
    result_kind = {'convert': 'dataset', 'train': 'policy', 'sim_eval': 'sim_eval', 'annotate': 'annotated'}.get(stage, stage)
    result_key = {'source': 'dataset', 'annotate': 'dataset', 'datagen': 'dataset', 'convert': 'dataset', 'train': 'checkpoint'}.get(stage)
    return {'argv': [opts['python'], '-u', 'scripts/real2sim/workflow.py', 'worker', '--request', str(run_dir / 'request.json')],
            'output': str(output), 'type': result_kind, 'workspace': str(w['root']), 'options': opts,
            'worker_report': True, 'result_key': result_key}


def execute_request(request):
    """Worker dispatch; GUI planning does not load engines or policy libraries."""
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
