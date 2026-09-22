"""Non-blocking launch of the existing Isaac runtime; no simulation logic here."""
from pathlib import Path
import os
import subprocess
import time


class RuntimeLauncher:
    def __init__(self, bus, workspace, root):
        self.bus = bus
        self.workspace = Path(workspace)
        self.root = Path(root)
        self.process = None
        self.pending = None
        self.started = None
        self.old_session = None
        self.log_path = self.workspace / 'runtime_startup.log'

    def connect(self, payload, python, headless=True):
        if self.pending is not None:
            raise RuntimeError('Isaac 시작 중입니다. 준비되면 Connect를 자동 실행합니다.')
        if self.bus.owner_alive():
            return self.bus.send('connect', payload)
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError('자동 실행한 Isaac 프로세스가 아직 준비되지 않았습니다. 로그: ' + str(self.log_path))
        if not Path(python).is_file():
            raise ValueError('Workspace Pipeline의 실행 Python 경로를 확인하세요: ' + python)
        self.old_session = self.bus.status().get('session')
        env = dict(os.environ)
        env.pop('HEADLESS', None)
        self.workspace.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('w') as log:
            self.process = subprocess.Popen(
                [python, '-u', str(self.root / 'scripts/real2sim/real2sim_sim.py'), '--workspace', str(self.workspace)] + (['--headless'] if headless else []),
                cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.pending = payload
        self.started = time.monotonic()
        return None

    def poll(self):
        if self.pending is None:
            return None
        code = self.process.poll()
        if code is not None:
            self.pending = None
            raise RuntimeError(f'Isaac 시작 실패 (exit {code}). 로그: {self.log_path}')
        status = self.bus.status()
        if (status.get('session') != self.old_session and self.bus.owner_alive()
                and self.bus.responsive(status) and status.get('state') not in self.bus.CONNECTING_STATES):
            payload = self.pending
            self.pending = None
            return self.bus.send('connect', payload)
        if time.monotonic() - self.started > 300:
            self.pending = None
            raise RuntimeError(f'Isaac 준비 대기 5분 초과. 자동 연결을 취소했습니다. 프로세스는 유지됩니다. 로그: {self.log_path}')
        return None

    def cancel_connect(self):
        # UI closure must never connect hardware later as a delayed side effect.
        self.pending = None

    def disconnect(self):
        if self.pending is not None:
            self.cancel_connect()
            # No Connect was sent: only the UI-owned startup process may be terminated.
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return None
        return self.bus.send('disconnect')
