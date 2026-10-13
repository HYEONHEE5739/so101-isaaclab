"""Non-blocking launch of the selected Real/Sim runtime; no physics here."""
from pathlib import Path
import os
import ctypes.util
import sys
import subprocess
import time
import signal


def worker_environment(root, extra_python_paths=()):
    """Use the selected worker Python's packages, never the GUI's site-packages."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in
        (Path(root) / 'source/soarm101_lab', *extra_python_paths))
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def resolve_live_renderer(renderer):
    """Choose a usable live-preview CPU path before the worker imports MuJoCo."""
    if renderer not in ('egl', 'software-egl', 'osmesa'):
        raise ValueError('Unknown live renderer')
    if renderer == 'osmesa' and ctypes.util.find_library('OSMesa') is None:
        return 'software-egl'
    return renderer


class RuntimeLauncher:
    def __init__(self, bus, workspace, root):
        self.bus = bus
        self.workspace = Path(workspace)
        self.root = Path(root)
        self.process = None
        self.pending = None
        self.started = None
        self.old_session = None
        self.backend = 'isaac'
        self.log_path = self.workspace / 'runtime_startup.log'
        self.replacement = None
        self.stopping = False
        self.stop_started = None
        self.stop_pids = []

    def worker_pids(self):
        """Match only this entry point/workspace, including workers from an older UI."""
        expected = (self.root / 'scripts/real2sim/real2sim_sim.py').resolve()
        result = []
        for folder in Path('/proc').glob('[0-9]*'):
            try:
                if folder.stat().st_uid != os.getuid():
                    continue
                argv = (folder / 'cmdline').read_bytes().decode().split('\0')
                if str(expected) not in argv or '--workspace' not in argv:
                    continue
                workspace = argv[argv.index('--workspace') + 1]
                if Path(workspace).resolve() == self.workspace.resolve():
                    result.append(int(folder.name))
            except (OSError, ValueError, IndexError):
                continue
        return result

    def interrupt_build(self, status):
        """Cancel engine construction only when no real devices have been connected."""
        if status.get('state') != 'BUILDING_ENVIRONMENT' or status.get('real_connected'):
            raise RuntimeError('장치 연결 중에는 강제 종료하지 않습니다. 연결 완료 후 Disconnect하세요.')
        pids = self.worker_pids()
        if not pids:
            raise RuntimeError('이 workspace의 runtime 프로세스를 확인하지 못했습니다. 실행 터미널에서 종료하세요.')
        for pid in pids:
            try:
                for entry in (Path('/proc') / str(pid) / 'fd').iterdir():
                    device = os.readlink(entry)
                    if device.startswith(('/dev/video', '/dev/tty', '/dev/serial', '/dev/bus/usb')):
                        raise RuntimeError('실제 장치가 열린 runtime은 강제 종료하지 않습니다.')
            except FileNotFoundError:
                continue
        # Recheck after inspecting process identity/FDs; never use a bare saved PID.
        latest = self.bus.status()
        if latest.get('session') != status.get('session') or latest.get('state') != 'BUILDING_ENVIRONMENT':
            raise RuntimeError('Runtime 상태가 바뀌었습니다. 다시 Disconnect하세요.')
        for pid in pids:
            if pid in self.worker_pids():
                os.kill(pid, signal.SIGTERM)
        self.stop_pids = pids

    def connect(self, payload, python, headless=True, backend='isaac', renderer='egl'):
        if backend not in ('isaac', 'mujoco', 'real', 'calibration'):
            raise ValueError('Unknown simulation backend')
        if self.pending is not None or self.replacement is not None or self.stopping:
            raise RuntimeError('Sim 시작 중입니다. 준비되면 Connect를 자동 실행합니다.')
        if not Path(python).is_file():
            raise ValueError('Workspace Pipeline의 실행 Python 경로를 확인하세요: ' + python)
        if backend == 'mujoco':
            renderer = resolve_live_renderer(renderer)
        if self.bus.owner_alive():
            status = self.bus.status()
            # Recreating an Isaac context/RTX cameras in the same Kit process can
            # stall after a failed hardware connection. Tear down the worker first.
            restart_isaac = backend == 'isaac' and status.get('sim_connected')
            # Older workers do not publish their renderer. Reusing them can keep
            # an unavailable OSMesa choice even after Stop changes ERROR to IDLE.
            restart_mujoco = backend == 'mujoco' and (status.get('state') == 'ERROR' or
                status.get('renderer') != renderer)
            # A failed Capture worker may retain stale imported device classes even
            # after the UI is reopened. Retry only after its normal disconnect exits.
            restart_real = backend == 'real' and status.get('state') == 'ERROR'
            if status.get('backend', 'isaac') != backend or status.get('state') == 'BUILDING_ENVIRONMENT' or restart_isaac or restart_mujoco or restart_real:
                self.disconnect()
                self.replacement = dict(payload=payload, python=python, headless=headless,
                                        backend=backend, renderer=renderer)
                return None
            self.backend = backend
            return self.bus.send('connect', payload)
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError('자동 실행한 Sim 프로세스가 아직 준비되지 않았습니다. 로그: ' + str(self.log_path))
        self.old_session = self.bus.status().get('session')
        self.backend = backend
        env = worker_environment(self.root)
        env.pop('HEADLESS', None)
        if backend == 'isaac' and sys.platform.startswith('linux'):
            env.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
            env.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
        if backend == 'mujoco':
            for key in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', '__NV_PRIME_RENDER_OFFLOAD',
                        '__GLX_VENDOR_LIBRARY_NAME', 'LIBGL_ALWAYS_SOFTWARE', '__EGL_VENDOR_LIBRARY_FILENAMES'):
                env.pop(key, None)
        argv = [python, '-u', str(self.root / 'scripts/real2sim/real2sim_sim.py'),
                '--workspace', str(self.workspace), '--backend', backend]
        if backend == 'mujoco':
            argv += ['--renderer', renderer]
        self.workspace.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('w') as log:
            self.process = subprocess.Popen(
                argv + (['--headless'] if headless else []),
                cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.pending = payload
        self.started = time.monotonic()
        return None

    def poll(self):
        if self.stopping:
            if self.process is not None:
                self.process.poll()  # Reap UI-owned children before checking the workspace lease.
            alive = (self.process is not None and self.process.poll() is None) or bool(self.worker_pids()) or self.bus.owner_alive()
            if alive:
                if time.monotonic() - self.stop_started > 20:
                    self.replacement = None
                    raise RuntimeError('Runtime 종료 대기 중입니다. 장치 재연결 없이 종료 상태를 확인하세요.')
                return None
            self.stopping = False
            self.process = None
            replacement, self.replacement = self.replacement, None
            if replacement:
                return self.connect(**replacement)
        if self.pending is None:
            return None
        code = self.process.poll()
        if code is not None:
            self.pending = None
            raise RuntimeError(f'{self.backend} 시작 실패 (exit {code}). 로그: {self.log_path}')
        status = self.bus.status()
        if (status.get('session') != self.old_session and self.bus.owner_alive()
                and self.bus.responsive(status) and status.get('state') not in self.bus.CONNECTING_STATES):
            if status.get('backend', 'isaac') != self.backend:
                self.pending = None
                raise RuntimeError('Runtime backend differs from requested backend')
            payload = self.pending
            self.pending = None
            return self.bus.send('connect', payload)
        if time.monotonic() - self.started > 300:
            self.pending = None
            raise RuntimeError(f'{self.backend} 준비 대기 5분 초과. 자동 연결을 취소했습니다. 프로세스는 유지됩니다. 로그: {self.log_path}')
        return None

    def cancel_connect(self):
        # UI closure must never connect hardware later as a delayed side effect.
        self.pending = None
        self.replacement = None

    def disconnect(self):
        self.replacement = None
        if self.stopping:
            return None
        if self.pending is not None:
            self.cancel_connect()
            # No Connect was sent: only the UI-owned startup process may be terminated.
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.stopping = True
            self.stop_started = time.monotonic()
            return None
        if not self.bus.owner_alive():
            return None
        status = self.bus.status()
        if status.get('state') == 'BUILDING_ENVIRONMENT':
            self.interrupt_build(status)
            identifier = None
        else:
            identifier = self.bus.send('disconnect')
        self.stopping = True
        self.stop_started = time.monotonic()
        return identifier
