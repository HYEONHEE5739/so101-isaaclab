"""Shared UI/CLI service: explicit stage -> argv -> validate -> register.

No shell interpolation, no implicit next-stage run, no implicit upload or hardware motion.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import uuid
import signal
from .artifacts import Registry, inspect_artifact
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import ROOT, load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from ..representation import require_compatible
from ..so101_dataset_contract import JOINT_SPACE, MIMIC_SPACE

STAGES = {
    'accept': ({'revision'}, 'revision'),
    'publish': ({'revision'}, 'workspace'),
    'source': ({'workspace'}, 'source'),
    'replay': ({'source', 'datagen'}, 'replay'),
    'annotate': ({'source'}, 'annotated'),
    'datagen': ({'annotated'}, 'datagen'),
    'convert': ({'datagen', 'source'}, 'dataset'),
    'train': ({'dataset'}, 'policy'),
    'hf_dataset': ({'dataset'}, 'hf_dataset'),
    'hf_policy': ({'policy'}, 'hf_policy'),
    'sim_eval': ({'policy', 'hf_policy'}, 'sim_eval'),
    'real_eval': ({'policy', 'hf_policy'}, 'real_eval'),
}


def default_python():
    # Explicitly configurable; do not use the GUI-only Python for Isaac.
    candidate = Path.home() / 'miniconda3/envs/lerobot-arena/bin/python'
    return str(candidate) if candidate.exists() else sys.executable


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
        if w and artifact.get('workspace') != metadata(w):
            raise ValueError('Input workspace identity differs from selected workspace')
        from .tasks import catalog, validate, from_artifact
        definitions = from_artifact(artifact, w)
        if stage in ('source', 'sim_eval', 'real_eval'):
            if opts.get('task_scope') == 'all' and stage != 'source':
                definitions = catalog(w)
            else:
                definitions = opts['tasks'] if 'tasks' in opts else definitions or [t for t in catalog(w)
                    if t['pick_object_id'] == w['task']['pick'] and t['target_object_id'] == w['task']['place']]
        elif opts.get('tasks') is not None and opts['tasks'] != definitions:
            raise ValueError('Derived stages must inherit input artifact tasks; do not relabel demonstrations')
        if w:
            definitions = [validate(t, w) for t in definitions]
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
        if stage in ('train', 'hf_dataset', 'hf_policy', 'sim_eval', 'real_eval'):
            coords = artifact.get('coordinates')
            if not coords or coords.get('action_space') != JOINT_SPACE:
                raise ValueError('Policy stages require model joint-radian actions')
        if stage == 'publish' and artifact.get('revision', {}).get('status', '').upper() != 'ACCEPTED':
            raise ValueError('Accept revision first')
        required = {'publish': ('workspace_id', 'task_file'), 'source': ('port',),
                    'convert': ('repo_id',), 'hf_dataset': ('repo_id',), 'hf_policy': ('repo_id',),
                    'real_eval': ('port', 'follower_id', 'side_device', 'wrist_device')}.get(stage, ())
        for name in required:
            if not opts.get(name):
                raise ValueError(f'Missing option: {name}')
        if stage == 'source':
            import math
            for key, default, positive in [('episode_time_s', 20, True), ('reset_time_s', 5, False)]:
                value = float(opts.get(key, default))
                if not math.isfinite(value) or value < 0 or (positive and value < .1):
                    raise ValueError('Invalid source timing: ' + key)
        if stage == 'publish' and not Path(opts['task_file']).is_file():
            raise FileNotFoundError(opts['task_file'])
        if stage == 'real_eval' and not opts.get('enable_motion'):
            raise ValueError('Real evaluation requires explicit enable_motion=true')
        if stage in ('hf_dataset', 'hf_policy') and type(opts.get('private')) is not bool:
            raise ValueError('Choose private=true/false explicitly')
        if dry:
            return {'stage': stage}
        run_dir = Path(run_dir)
        python = opts.get('python') or default_python()
        device = opts.get('device', 'cuda:0')
        output = Path(opts['output']).expanduser().resolve() if opts.get('output') else run_dir / STAGES[stage][1]
        if stage == 'accept' and not opts.get('output'):
            output = run_dir / 'accepted_revision/profile.json'
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
        elif stage == 'replay':
            cli += ['scripts/real2sim/workspace.py', 'replay', ws, '--dataset', path, '--report', str(output), '--device', device, *headless]
        elif stage in ('annotate', 'datagen'):
            script = 'annotate_demos.py' if stage == 'annotate' else 'generate_dataset.py'
            cli += ['scripts/mimic/' + script, '--workspace', ws, '--input_file', path, '--output_file', str(output), '--device', device, *headless]
            if stage == 'annotate' and opts.get('auto', True):
                cli += ['--auto']
            if stage == 'datagen':
                cli += ['--generation_num_trials', str(opts.get('trials', 10)), '--num_envs', '1']
        elif stage == 'convert':
            cli += ['scripts/tools/convert_isaac2lerobot.py', '--input_file', path, '--root', str(output),
                    '--repo_id', opts['repo_id'], '--task', opts.get('task', ''), '--action_source', 'joint_targets']
        elif stage == 'publish':
            output = ROOT / 'outputs/real2sim/environments' / opts['workspace_id'] / ('v' + str(opts.get('version', 1)))
            cli += ['scripts/real2sim/publish_workspace.py', '--revision', path, '--workspace-id', opts['workspace_id'],
                    '--version', str(opts.get('version', 1)), '--task', opts['task_file']]
        else:
            spec = {'stage': stage, 'input': artifact, 'workspace': ws, 'options': opts, 'output': str(output)}
            atomic(run_dir / 'request.json', spec)
            cli += ['scripts/real2sim/workflow_stage.py', str(run_dir / 'request.json')]
        return {'argv': cli, 'output': str(output), 'type': STAGES[stage][1], 'workspace': ws, 'options': opts}

    def run(self, stage, artifact, workspace='', options=None, notify=lambda message: None):
        self.cancelled = False
        self.cancel_time = None
        run_dir = self.registry.root / 'runs' / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
        state = {'code_commit': commit.stdout.strip(), 'code_dirty': bool(dirty.stdout), 'id': run_dir.name, 'stage': stage, 'source': artifact['id'], 'status': 'READY', 'created_at': time.time()}
        try:
            plan = self.plan(stage, artifact, workspace, options or {}, run_dir=run_dir)
            target = Path(plan['output'])
            if target.exists() and stage != 'publish':
                raise FileExistsError('Choose a new output path; artifact overwrite is disabled')
            target.parent.mkdir(parents=True, exist_ok=True)
            state.update(status='RUNNING', plan=plan)
            atomic(run_dir / 'state.json', state)
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'source/soarm101_lab') + os.pathsep + os.environ.get('PYTHONPATH', ''))
            self.session = run_dir / 'presentation'
            self.session.mkdir()
            atomic(self.session / 'config.json', {'stage': stage, 'mode': plan['options']['mode'], 'tasks': plan['options'].get('tasks', []),
                   'real_cameras': {} if stage == 'real_eval' else plan['options'].get('real_cameras', {})})
            env['PYTHONUNBUFFERED'] = '1'
            env['SO101_WORKFLOW_SESSION'] = str(self.session)
            env.pop('SO101_TASK_SELECTION', None)
            if plan['options'].get('tasks'):
                atomic(run_dir / 'tasks.json', {'tasks': plan['options']['tasks']})
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
                        if self.cancelled and self.cancel_time:
                            elapsed = time.monotonic() - self.cancel_time
                            if elapsed > 5:
                                try: os.killpg(process.pid, signal.SIGKILL if elapsed > 10 else signal.SIGTERM)
                                except ProcessLookupError: pass
            if self.cancelled:
                raise RuntimeError('Workflow cancelled; partial outputs were not registered')
            if code:
                raise RuntimeError(f'Process exit {code}; log: {run_dir / "process.log"}')
            if stage == 'train':
                target = target / 'checkpoints/last/pretrained_model'
            # Validate artifact content even when an Isaac shutdown masks a process failure.
            result = self.registry.register(plan['type'], target, parents=[artifact['id']], extra=state)
            state.update(status='SUCCEEDED', artifact=result['id'])
            notify('완료: ' + str(target))
            return result
        except Exception as exc:
            state.update(status='FAILED', error=str(exc))
            raise
        finally:
            self.process = None
            atomic(run_dir / 'state.json', state)
