"""Run-local camera presentation and controls; never owns simulation or joint mapping.

Both UI modes attach this adapter to the same environment. Images are diagnostic
copies of observations, never fed back into the recorder or policy.
"""
import json
import os
import time
import uuid
from pathlib import Path

from ..real2sim.storage import atomic

SESSION_ENV = 'SO101_WORKFLOW_SESSION'


def training_status_text(row):
    """Compact LeRobot-style display; structured measurements remain in JSONL."""
    step=row['step'];target=row.get('target_steps')
    fields=[f"step:{step:,}"+(f"/{target:,}" if target else '')]
    if 'samples' in row:fields.append(f"smpl:{row['samples']:,}")
    if 'batch_size' in row:fields.append(f"bs:{row['batch_size']}")
    fields.append(f"loss:{row.get('training_loss',row.get('loss')):.3f}")
    if 'grad_norm' in row:fields.append(f"grdn:{row['grad_norm']:.3f}")
    rates=row.get('learning_rate',row.get('lr'))
    if rates is not None:
        rates=rates if isinstance(rates,list) else [rates]
        fields.append('lr:'+','.join(f'{value:.1e}' for value in rates))
    for key,label in (('update_s','updt_s'),('dataloading_s','data_s'),('preprocessing_s','prep_s'),('seconds','step_s')):
        if key in row:fields.append(f'{label}:{row[key]:.3f}')
    if 'peak_allocated_gib' in row:fields.append(f"mem_gb:{row['peak_allocated_gib']:.2f}")
    validation=row.get('validation_loss',row.get('validation_action_loss'))
    if validation is not None:fields.append(f'val:{validation:.3f}')
    if 'best_validation_loss' in row:fields.append(f"best_val:{row['best_validation_loss']:.3f}")
    return '[TRAIN] '+' '.join(fields)


def send(root, command):
    identifier = f'{time.time_ns():020d}_{uuid.uuid4().hex}'
    atomic(Path(root) / 'commands' / (identifier + '.json'), {'command': command})
    return identifier


class Presentation:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / 'config.json').read_text())
        self.callbacks = {}
        self.pending = set()
        self.paused = False
        self.single_step = False
        self.steps = 0
        self.last_publish = 0.
        self.frames = {}
        tasks = self.config.get('tasks', [])
        self.details = {'task_id': tasks[0]['task_id']} if len(tasks) == 1 else {}
        self.readers = {}
        self.closed = False
        self.idle_render = None
        self.env = None
        self.progress_provider = None
        self.status('STARTING')

    def status(self, state='RUNNING'):
        if self.progress_provider: self.details.update(self.progress_provider())
        atomic(self.root / 'status.json', dict(state=state, time=time.time(), steps=self.steps,
               stage=self.config['stage'], mode=self.config['mode'], frames=self.frames, **self.details))

    def start_datagen(self,target,num_envs):
        self.progress_provider=None
        self.details.update(workflow_phase='preparing',target_successes=target,successes=0,attempts=0,failures=0,
            success_rate=None,failure_breakdown={},episode_outcome='환경 준비 중',outcome_reason='',
            env_status={i:{'state':'waiting','attempts':0,'successes':0,'failures':{},'steps':0} for i in range(num_envs)})
        self.status()

    def source_progress(self,completed,total,episode=None,frame=0,frames=0):
        self.details.update(workflow_phase='source_validation',source_completed=completed,source_total=total,
            source_episode=episode,source_frame=frame,source_frames=frames,
            episode_outcome='Source 사전 검증 중',outcome_reason='검증 완료 후 병렬 생성을 시작합니다.')
        for index,worker in self.details.get('env_status',{}).items():
            worker.update(state='source_validation' if int(index)==0 and episode else 'waiting',
                steps=frame if int(index)==0 else 0)

    def bind_generation(self,scheduler):
        for key in ('episode','episode_index','episode_total'):self.details.pop(key,None)
        self.details['workflow_phase']='generation'
        self.progress_provider=scheduler.progress
        self.status()

    def poll(self):
        for path in sorted((self.root / 'commands').glob('*.json')):
            command = json.loads(path.read_text())['command']
            accepted = True
            if command in self.callbacks:
                self.callbacks[command]()
            elif command == 'pause':
                self.paused = True
            elif command == 'resume':
                self.paused = False
            elif command == 'step':
                self.paused = True
                self.single_step = True
            elif command in ('save', 'discard', 'quit') and self.config['stage'] == 'source':
                self.pending.add(command)
            else:
                accepted = False
            atomic(self.root / 'replies' / path.name, {'command': command, 'accepted': accepted})
            path.unlink()

    def before_step(self):
        boundary=getattr(self,'interrupt_boundary',None)
        if boundary:boundary()
        self.poll()
        while self.paused and not self.single_step:
            if boundary:boundary()
            self.status('PAUSED')
            if self.idle_render:
                self.idle_render()
            time.sleep(.03)
            self.poll()
        self.single_step = False

    def publish(self, images=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < 1 / 30:
            return
        self.last_publish = now
        import numpy as np
        from PIL import Image
        images = dict(images or {})
        if self.env is not None:
            sensors = getattr(getattr(self.env, 'scene', None), 'sensors', {})
            if 'camera_overview' in sensors:
                images['sim_overview'] = sensors['camera_overview'].data.output['rgb']
        # Optional real cameras reuse the existing reader, opened only on explicit configuration.
        for view, reader in self.readers.items():
            try:
                packet = reader.nearest()
                if packet and time.monotonic_ns() - packet['read_end']['monotonic_ns'] < 500_000_000:
                    images['real_' + view] = packet['rgb']
            except Exception as exc:
                self.details[view + '_camera_error'] = str(exc)
        for name, rgb in images.items():
            if hasattr(rgb, 'detach'):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
            target = self.root / (name + '.jpg')
            tmp = target.with_suffix('.pending')
            Image.fromarray(rgb[..., :3]).save(tmp, format='JPEG', quality=90)
            tmp.replace(target)
            self.frames[name] = time.time()
        self.status('PAUSED' if self.paused else 'RUNNING')

    def observe(self, result, force=False):
        obs = result[0] if isinstance(result, tuple) else result
        policy = obs.get('policy', {}) if isinstance(obs, dict) else {}
        if isinstance(obs, dict):
            def flag(group, key):
                value = obs.get(group, {}).get(key) if isinstance(obs.get(group), dict) else None
                if value is None: return None
                return bool(value.reshape(-1)[0].item())
            signals = {'grasp': flag('subtask_terms', 'grasp'), 'place': flag('task_state', 'place_success')}
            if signals['place'] is None: signals['place'] = flag('task', 'success')
            if self.env is not None:
                workspace = getattr(self.env, '_workflow_task_workspace', None) or getattr(getattr(self.env, 'cfg', None), 'workflow_workspace', None)
                if workspace:
                    try:
                        tcp = self.env.scene['ee_frame'].data.target_pos_w[0, 0]
                        obj = self.env.scene[workspace['task']['pick']].data.root_pos_w[0]
                        signals['tcp_distance_mm'] = float((tcp - obj).norm().item() * 1000)
                    except (KeyError, AttributeError): pass
            self.details['task_signals'] = signals
        self.publish({'sim_' + key[:-4]: value for key,value in policy.items() if key.endswith('_cam')}, force)

    def start_real_cameras(self):
        from ..real2sim.cameras import CameraReader
        for view, device in self.config.get('real_cameras', {}).items():
            if not isinstance(view, str) or not view.isidentifier() or not device:
                continue
            reader = None
            try:
                reader = CameraReader(device)
                reader.start()
                self.readers[view] = reader
            except Exception as exc:
                self.details[view + '_camera_error'] = str(exc)
                if reader is not None:
                    reader.close()

    def close(self):
        if not self.closed:
            self.closed = True
            for reader in self.readers.values():
                reader.close()
            self.status('FINISHED')


def current():
    root = os.environ.get(SESSION_ENV)
    if not root:
        return None
    global _current
    if _current is None or _current.root != Path(root):
        _current = Presentation(root)
    return _current


_current = None


def datagen_status_text(status):
    """One backend-independent summary for preparation, source validation and generation."""
    phases={'preparing':'환경 준비 중','source_validation':'Source 사전 검증 중','generation':'데이터 생성 중'}
    states={'waiting':'대기','idle':'대기','resetting':'초기화','running':'실행 중','saving':'저장 중',
        'source_validation':'Source 검증','success':'성공','interrupted':'중단','target_already_reached':'목표 도달',
        'ik_failure':'IK 계산 탈락','planning_failure':'계획 실패','simulation_failure':'시뮬레이션 실패',
        'grasp_failure':'집기 실패','drop_failure':'이동 중 놓침','place_failure':'컵 배치 실패','timeout':'시간 초과','storage_failure':'저장 실패'}
    phase=status.get('workflow_phase','generation');title=phases.get(phase,phase)
    if status.get('target_successes') and status.get('successes',0)>=status['target_successes']:title='목표 달성'
    elif status.get('state')=='FINISHED':title='실행 종료'
    lines=[title]
    if status.get('ik_solver'):lines.append('IK: '+status['ik_solver']+' · Policy: '+status.get('ik_failure_policy','strict'))
    if phase=='source_validation':
        lines.append(f"Source 검증 완료 {status.get('source_completed',0)}/{status.get('source_total',0)} · "
            f"현재 {status.get('source_episode') or '준비 중'} · frame {status.get('source_frame',0)}/{status.get('source_frames',0)}")
    rate=status.get('success_rate');rate=f'{rate*100:.1f}%' if rate is not None else '—'
    lines.append(f"전체 성공 {status.get('successes',0)}/{status.get('target_successes','—')} · 시도 {status.get('attempts',0)} · 실패 {status.get('failures',0)} · 성공률 {rate}")
    elapsed=status.get('target_completed_seconds')
    completed=elapsed is not None
    if elapsed is None:elapsed=status.get('wall_seconds')
    if elapsed is not None:
        minutes,seconds=divmod(max(0,int(elapsed)),60)
        label=f"목표 {status['target_successes']}개 저장 완료" if completed else '이번 생성 경과'
        lines.append(f'{label}: {minutes}분 {seconds:02d}초 (환경 준비 제외·일시정지 포함)')
        if status.get('resumed_successes'):lines.append(f"이어 수집: 기존 {status['resumed_successes']}개 제외, 이번 실행 시간")
    if status.get('latest_env_id') is not None:
        category=states.get(status.get('latest_category'),status.get('episode_outcome',''))
        lines.append(f"최근 결과: Env {status['latest_env_id']} · 시도 {status['latest_attempt']} · {category} {status.get('outcome_reason','')}")
    for index,worker in sorted(status.get('env_status',{}).items(),key=lambda item:int(item[0])):
        lines.append(f"Env {index}: {states.get(worker['state'],worker['state'])} · 시도 {worker['attempts']} · 성공 {worker['successes']} · 실패 {sum(worker.get('failures',{}).values())} · step {worker.get('steps',0)}")
        if worker.get('failures'):
            lines.append('  실패 원인: '+' · '.join(f'{states.get(key,key)} {value}' for key,value in worker['failures'].items()))
    if status.get('failure_breakdown'):
        lines.append('실패 원인: '+' · '.join(f'{states.get(key,key)}: {value}' for key,value in status['failure_breakdown'].items()))
    return '\n'.join(lines)


def attach(env):
    """Preserve the environment's methods and returned observations in both modes."""
    bridge = current()
    if bridge is None or getattr(env, '_workflow_presentation', None):
        return bridge
    env._workflow_presentation = bridge
    bridge.env = env
    bridge.idle_render = getattr(getattr(env, 'sim', None), 'render', None)
    step, reset, close = env.step, env.reset, env.close

    def presented_step(*args, **kwargs):
        bridge.before_step()
        result = step(*args, **kwargs)
        bridge.steps += 1
        bridge.observe(result)
        return result

    def presented_reset(*args, **kwargs):
        result = reset(*args, **kwargs)
        bridge.observe(result, force=True)
        return result

    def presented_close(*args, **kwargs):
        try:
            bridge.close()
        finally:
            close(*args, **kwargs)

    env.step, env.reset, env.close = presented_step, presented_reset, presented_close
    bridge.start_real_cameras()
    return bridge


class RecorderControls:
    """Same recorder input API as KeyboardControl; optional Developer keyboard."""
    def __init__(self, bridge, keyboard=None):
        self.bridge, self.keyboard = bridge, keyboard

    def consume(self, name):
        self.bridge.poll()
        present = name in self.bridge.pending
        self.bridge.pending.discard(name)
        key = getattr(self.keyboard, 'consume_' + name)() if self.keyboard else False
        return present or key

    def consume_save(self):
        return self.consume('save')

    def consume_discard(self):
        return self.consume('discard')

    def should_quit(self):
        self.bridge.poll()
        return 'quit' in self.bridge.pending or bool(self.keyboard and self.keyboard.should_quit())

    def destroy(self):
        if self.keyboard:
            self.keyboard.destroy()
        self.bridge.close()


def annotation_keyboard(adapter):
    """Standalone Isaac CLI controls for the same common replay/annotation loop."""
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
    keyboard=Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=.1,rot_sensitivity=.1))
    adapter.annotation_paused=True
    keyboard.add_callback('N',lambda:setattr(adapter,'annotation_paused',False))
    keyboard.add_callback('B',lambda:setattr(adapter,'annotation_paused',True))
    keyboard.add_callback('S',lambda:adapter.annotation_marks.append(adapter.annotation_position[0]))
    keyboard.add_callback('Q',lambda:adapter.annotation_skipped.append(True))
    keyboard.reset()
    print('[WORKFLOW] Annotation: N play, B pause, S grasp boundary, Q skip',flush=True)
    return keyboard


def framing(workspace):
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    p = workspace['profile']
    base = np.asarray(p['robot_base']['T_world'])[:3, 3]
    transform = np.asarray(p['workspace']['T_world'])
    points = [base, transform[:3, 3]]
    for name, obj in p['objects'].items():
        if 'T_workspace' in obj and not name.startswith('background_'):
            points.append((transform @ np.asarray(obj['T_workspace']))[:3, 3])
    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
    urdf = ET.parse(workspace['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf')
    reach = sum(np.linalg.norm(np.fromstring(o.get('xyz', '0 0 0'), sep=' '))
                for o in urdf.findall('./joint/origin'))
    radius = max(np.linalg.norm(np.asarray(points) - center, axis=1).max(), reach, .1)
    center[2] = max(center[2], base[2] + reach * .3)
    # View from the same open side as the existing camera, not behind a backdrop.
    camera = p['cameras'].get('side') or next((c for c in p['cameras'].values() if c['parent']=='world'), next(iter(p['cameras'].values())))
    direction = -np.asarray(camera['T_parent_camera'])[:3, 2].copy()
    if np.linalg.norm(direction[:2]) < 1e-6:
        direction[:2] = (1., 1.)
    direction[2] = max(direction[2], .5)
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius * 1.5
    forward = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    q = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return tuple(eye), (float(q[3]), *map(float, q[:3]))


class GenerationProgress:
    def __init__(self, generation, target=None):
        self.target = target
        self.generation = generation
        self.previous = (0, 0)
        self.outcome = '진행 중'

    def __call__(self):
        g = self.generation
        attempts, successes = g.num_attempts, g.num_success
        if attempts > self.previous[0]:
            self.outcome = '성공 (최근 완료 시도)' if successes > self.previous[1] else '실패 (최근 완료 시도)'
        self.previous = attempts, successes
        return dict(success_rate=successes / attempts if attempts else None, target_successes=self.target, attempts=attempts, successes=successes, failures=g.num_failures,
                    episode_outcome=self.outcome, outcome_reason=(f'성공 데이터 {successes}/{self.target}개' if self.target is not None else 'Mimic generator 판정'))
