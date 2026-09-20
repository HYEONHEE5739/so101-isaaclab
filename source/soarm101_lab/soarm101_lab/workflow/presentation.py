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
        self.poll()
        while self.paused and not self.single_step:
            self.status('PAUSED')
            if self.idle_render:
                self.idle_render()
            time.sleep(.03)
            self.poll()
        self.single_step = False

    def publish(self, images=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < .1:
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
        self.publish({'sim_' + view: policy[view + '_cam'] for view in ('side', 'wrist')
                      if view + '_cam' in policy}, force)

    def start_real_cameras(self):
        from ..real2sim.cameras import CameraReader
        for view, device in self.config.get('real_cameras', {}).items():
            if view not in ('side', 'wrist') or not device:
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
