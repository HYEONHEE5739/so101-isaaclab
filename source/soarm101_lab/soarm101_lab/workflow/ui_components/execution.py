"""One execution owner; only immutable plain snapshots enter validation/worker threads."""
import copy
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class WorkflowExecutionController(QObject):
    readinessChanged = pyqtSignal(object, str, str)
    validationChoices = pyqtSignal(object, object)
    busyChanged = pyqtSignal(bool)
    started = pyqtSignal(object)
    finished = pyqtSignal(object, object)

    def __init__(self, service, context, run_worker, parent=None):
        super().__init__(parent)
        self.service, self.context, self.run_worker = service, context, run_worker
        self.busy = False
        self.active = None
        self.generation = 0
        self.pending = self.future = None
        self.executor = None
        self.closed = False
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(200)
        self.debounce.timeout.connect(self._start_check)

    def invalidate(self):
        self.generation += 1
        self.pending = None
        self.debounce.stop()

    def request_ready(self, owner, snapshot=None, error=None, validation_candidates=()):
        self.invalidate()
        if self.busy:
            return
        if error is not None:
            self.readinessChanged.emit(owner, 'NOT_READY', str(error))
            return
        self.pending = (self.generation, owner, copy.deepcopy(snapshot), copy.deepcopy(validation_candidates))
        self.readinessChanged.emit(owner, 'CHECKING', '실행 설정 확인 중')
        self.debounce.start()

    @staticmethod
    def _check(service, snapshot, candidates):
        compatible = []
        if candidates:
            from ..episodes import validate_training_inputs
            for record in candidates:
                try:
                    validate_training_inputs(snapshot['artifact'], {'validation_artifact': record})
                except (ValueError, KeyError, OSError, TypeError):
                    continue
                compatible.append(record)
        state, reason = service.readiness(snapshot['stage'], snapshot['artifact'],
                                          snapshot['workspace'], snapshot['options'])
        return state, reason, compatible

    def _start_check(self):
        if self.closed or self.busy or self.future is not None or self.pending is None:
            return
        generation, owner, snapshot, candidates = self.pending
        self.pending = None
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workflow-readiness')
            executor = self.executor
            self.destroyed.connect(lambda: executor.shutdown(wait=False, cancel_futures=True))
        self.future = (generation, owner, self.executor.submit(self._check, self.service, snapshot, candidates))

    def poll(self):
        if self.future is None:
            return
        generation, owner, future = self.future
        if not future.done():
            return
        self.future = None
        try:
            state, reason, choices = future.result()
        except Exception as exc:
            state, reason, choices = 'NOT_READY', str(exc), []
        if generation == self.generation and not self.busy and not self.closed:
            self.validationChoices.emit(owner, choices)
            self.readinessChanged.emit(owner, state, reason)
        if self.pending is not None and not self.debounce.isActive():
            self.debounce.start(0)

    def execute(self, owner, snapshot):
        if self.busy:
            raise ValueError('이미 Workflow를 실행 중입니다.')
        if self.closed:
            raise ValueError('닫힌 Workflow입니다.')
        self.invalidate()
        self.active = copy.deepcopy(dict(snapshot, owner=owner))
        active = self.active
        self.service.session = None
        self.service.cancelled = False
        self.cancel_requested = False
        self.busy = True
        self.context.set_locked(True)
        self.context.save()
        self.busyChanged.emit(True)
        self.started.emit(active)

        def operation(notify):
            if self.cancel_requested:
                return {'ok': False, 'error': 'Workflow cancelled before worker startup'}
            try:
                artifact = self.service.run(active['stage'], active['artifact'], active['workspace'],
                                            active['options'], notify)
                return {'ok': True, 'artifact': artifact}
            except Exception as exc:
                return {'ok': False, 'error': str(exc)}

        def done(result):
            if self.closed:
                return
            self.busy = False
            self.context.set_locked(False)
            self.busyChanged.emit(False)
            self.finished.emit(active, result)
        try:
            self.run_worker(operation, done)
        except Exception as exc:
            done({'ok': False, 'error': str(exc)})
            raise

    def cancel(self):
        if self.busy:
            self.cancel_requested = True
            self.service.cancel()

    def close(self):
        self.closed = True
        self.invalidate()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self.future = None
