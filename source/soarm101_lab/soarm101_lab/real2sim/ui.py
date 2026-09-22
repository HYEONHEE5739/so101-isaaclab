"""Desktop control, field provenance, raw-image landmarks and numerical fitting."""

import argparse
import json
from pathlib import Path
import sys
import time
from threading import Event
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QComboBox,
    QInputDialog,
    QDialog,
    QScrollArea,
)
from .profile import load, validate, upgrade_legacy, digest
from .storage import Bus, atomic, revision
from . import references
from . import calibration

ROOT = Path(__file__).resolve().parents[4]


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageLabel(QLabel):
    clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 160)

    def show_image(self, path):
        pix = QPixmap(str(path))
        if not pix.isNull():
            self.source = pix
            self.refresh()

    def refresh(self):
        if not self.source.isNull():
            self.setPixmap(
                self.source.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

    def mousePressEvent(self, e):
        pix = self.pixmap()
        if pix and not self.source.isNull():
            x = e.position().x() - (self.width() - pix.width()) / 2
            y = e.position().y() - (self.height() - pix.height()) / 2
            if 0 <= x < pix.width() and 0 <= y < pix.height():
                self.clicked.emit(x * self.source.width() / pix.width(), y * self.source.height() / pix.height())


class Window(QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.bus = Bus(self.workspace / "ipc")
        from .launcher import RuntimeLauncher
        self.runtime_launcher = RuntimeLauncher(self.bus, self.workspace, ROOT)
        self.worker = None
        self.cancel = Event()
        self.pending = {}
        self.reference_files = []
        self.setWindowTitle("SO-101 Real2Sim Control / Measurement")
        self.resize(1280, 980)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.status = QLabel("Isaac runtime 연결 대기")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        reference = QWidget()
        rl = QVBoxLayout(reference)
        self.tabs.addTab(reference, "1 · Reference")
        rl.addWidget(QLabel("실제 환경 사진 등록 → 외부 Codex로 provisional profile 준비 → Real / Sim에서 확인·Capture → Inspect"))
        rl.addWidget(QLabel("실제 사진만 등록하세요. Sim screenshot과 paired capture는 별도 데이터입니다."))
        self.reference_view = QComboBox()
        self.reference_view.addItems(references.VIEWS)
        self.reference_view.setCurrentText("unknown")
        rl.addWidget(QLabel("이번에 선택할 사진 전체의 view (파일명에서 추측하지 않습니다)"))
        rl.addWidget(self.reference_view)
        self.button(rl, "실제 사진 가져오기 (여러 장)", self.import_media)
        self.button(rl, "Reference 목록 새로고침", self.reload_references)
        location = QLabel(str(self.workspace / "references"))
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(location)
        self.reference_summary = QLabel()
        rl.addWidget(self.reference_summary)
        self.reference_list = QPlainTextEdit()
        self.reference_list.setReadOnly(True)
        rl.addWidget(self.reference_list)
        live = QWidget()
        live_l = QVBoxLayout(live)
        self.tabs.addTab(live, "2 · Real / Sim")
        grid = QGridLayout()
        live_l.addLayout(grid, 1)
        self.views = {}
        for i, (name, title) in enumerate(
            [
                ("real_side", "REAL SIDE"),
                ("sim_side", "SIM SIDE"),
                ("real_wrist", "REAL WRIST"),
                ("sim_wrist", "SIM WRIST"),
            ]
        ):
            tile = QWidget()
            vl = QVBoxLayout(tile)
            vl.addWidget(QLabel(title))
            label = ImageLabel()
            label.setText("No frame")
            vl.addWidget(label, 1)
            grid.addWidget(tile, i // 2, i % 2)
            self.views[name] = label
        row = QHBoxLayout()
        live_l.addLayout(row)
        for title, fn in [
            ("Connect", self.connect_hw),
            ("Start", lambda: self.send("start")),
            ("Stop", self.stop),
            ("Capture", lambda: self.send("capture")),
            ("Disconnect / Sim 종료", self.disconnect_sim),
        ]:
            self.button(row, title, fn)
        settings = QWidget()
        form = QFormLayout(settings)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        settings_scroll.setMaximumHeight(260)
        live_l.addWidget(settings_scroll)
        self.live_details = QLabel()
        self.live_details.setWordWrap(True)
        live_l.addWidget(self.live_details)
        self.capture_details = QLabel()
        self.capture_details.setWordWrap(True)
        self.capture_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.capture_details.setToolTip("영상을 보며 자세가 안정된 뒤 Capture를 누르세요. 최종 품질 판정은 runtime이 수행합니다.")
        capture_scroll = QScrollArea()
        capture_scroll.setWidgetResizable(True)
        capture_scroll.setWidget(self.capture_details)
        capture_scroll.setMaximumHeight(110)
        # Keep capture feedback immediately below the camera/control area.
        live_l.insertWidget(2, capture_scroll)
        inspect = QWidget()
        il = QVBoxLayout(inspect)
        self.tabs.addTab(inspect, "3 · Inspect")
        self.profile_details = QLabel("Profile 미선택")
        self.profile_details.setWordWrap(True)
        il.addWidget(self.profile_details)
        self.button(il, "Profile 열기", self.open_profile)
        self.button(il, "Profile 적용", lambda: self.send("apply_profile", {"profile": self.profile_path()}))
        self.button(il, "USD + Profile 내보내기", lambda: self.send("export", {"profile": self.profile_path()}))
        self.advanced = QTabWidget()
        il.addWidget(QLabel("Advanced / optional · 수동 JSON 및 측정·수치 보정"))
        il.addWidget(self.advanced)
        self.fields = {}
        for key, title, default in [
            ("profile", "Profile JSON", ""),
            ("leader_port", "Leader port", ""),
            ("follower_port", "Follower port", ""),
            ("follower_id", "Follower calibration ID", ""),
            ("leader_calibration", "Leader calibration JSON (absolute path or existing filename)", "so101_leader.json"),
            ("follower_calibration_dir", "Follower calibration directory", ""),
            ("side_camera", "Side camera index/path", ""),
            ("wrist_camera", "Wrist camera index/path", ""),
        ]:
            edit = QLineEdit(default)
            self.fields[key] = edit
            form.addRow(title, edit)
        for title, fn in [
            ("Profile 열기", self.open_profile),
            ("Sim 미리보기 (장비 연결 없음)", lambda: self.connect_hw(True)),
            ("USD + Profile 내보내기", lambda: self.send("export", {"profile": self.profile_path()})),
        ]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, f=fn: self.guard(f))
            form.addRow(b)
        editor = QWidget()
        el = QVBoxLayout(editor)
        self.advanced.addTab(editor, "Profile JSON")
        self.button(el, "알려진 치수 템플릿 열기", self.template)
        self.editor = QPlainTextEdit()
        el.addWidget(self.editor)
        el.addWidget(QLabel("필드별 provenance를 유지합니다. 초기 pose/K/조명은 provisional이며, null은 unmeasured입니다."))
        r = QHBoxLayout()
        el.addLayout(r)
        self.button(r, "새 revision 저장", self.save_profile)
        calibrate = QWidget()
        cl = QVBoxLayout(calibrate)
        self.advanced.addTab(calibrate, "측정 / 수치 보정")
        cl.addWidget(QLabel("기존 optimizer의 intrinsic 선행 조건은 유지됩니다. 체커보드 기능은 선택적 도구이며 bootstrap 등록 조건이 아닙니다."))
        self.dataset = QLineEdit(str(self.workspace / "landmarks.json"))
        cl.addWidget(QLabel("측정 landmark dataset"))
        cl.addWidget(self.dataset)
        self.group = QComboBox()
        self.group.addItems(["side", "robot_base", "wrist", "workspace", "table"])
        cl.addWidget(self.group)
        for title, fn in [
            ("선택 도구: Side 체커보드 보정", lambda: self.intrinsics("side")),
            ("선택 도구: Wrist 체커보드 보정", lambda: self.intrinsics("wrist")),
            ("사진에서 측정점 지정", self.annotate),
            ("정량 오차 계산", self.metrics),
            ("선택 그룹 최적화 + 재렌더", self.optimize),
            ("현재 Profile로 데이터 재렌더", self.rerender),
            ("Profile 적용", lambda: self.send("apply_profile", {"profile": self.profile_path()})),
        ]:
            b = QPushButton(title)
            b.clicked.connect(lambda checked=False, f=fn: self.guard(f))
            cl.addWidget(b)
        cl.addWidget(
            QLabel(
                "정지한 여러 자세에서 캡처 → raw 사진에 알려진 3D 점 지정.\n고정 world 기준점으로 side부터 보정. capture 전체를 train/validation으로 분리.\nStop은 마지막 목표를 유지합니다. Disconnect는 토크를 해제합니다."
            )
        )
        cl.addStretch()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        layout.addWidget(self.log)
        settings_path = self.workspace / "ui_settings.json"
        if settings_path.exists():
            for k, v in json.loads(settings_path.read_text()).items():
                if k in self.fields:
                    self.fields[k].setText(v)
        from ..workflow.ui import WorkflowPanel
        self.workflow_panel = WorkflowPanel(self.work, self)
        self.tabs.addTab(self.workflow_panel, "4 · Workspace Pipeline")
        self.tabs.currentChanged.connect(lambda _: self.log.setVisible(self.tabs.currentWidget() is not self.workflow_panel))
        self.reload_references()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)
        self.refresh()

    def guard(self, fn):
        try:
            return fn()
        except Exception as exc:
            self.log.appendPlainText(str(exc))
            QMessageBox.warning(self, "확인 필요", str(exc))

    def button(self, row, title, fn):
        b = QPushButton(title)
        b.clicked.connect(lambda checked=False: self.guard(fn))
        row.addWidget(b)
        return b

    def profile_path(self):
        path = self.fields["profile"].text().strip()
        if not path:
            raise ValueError("측정 Profile을 먼저 선택하거나 저장하세요")
        return str(Path(path).resolve())

    def profile(self):
        return load(self.profile_path())

    def selected(self, path):
        self.fields["profile"].setText(str(path))
        p = load(path)
        self.editor.setPlainText(json.dumps(p, indent=2, ensure_ascii=False))
        self.group.clear()
        self.group.addItems(
            ["side", "robot_base", "wrist", "workspace", "table", *["object:" + n for n in p["objects"]]]
        )

    def send(self, name, payload=None):
        if name in ("start", "connect", "apply_profile") and self.worker and self.worker.isRunning():
            raise ValueError("보정 작업 중에는 Start/장면 변경을 할 수 없습니다")
        identifier = self.bus.send(name, payload)
        self.pending[identifier] = name
        return identifier

    def stop(self):
        self.cancel.set()
        if self.runtime_launcher.pending is not None:
            self.runtime_launcher.cancel_connect()
            self.log.appendPlainText("자동 Connect 취소 · Isaac 프로세스는 유지됩니다.")
            return
        self.send("stop")

    def connect_hw(self, preview=False):
        if self.worker and self.worker.isRunning():
            raise ValueError("진행 중인 보정 작업이 끝난 뒤 연결하세요")
        hardware = {k: v.text().strip() for k, v in self.fields.items() if k != "profile"}
        payload = {"profile": self.profile_path(), "hardware": hardware, "preview_only": preview}
        identifier = self.runtime_launcher.connect(payload, self.workflow_panel.python.text().strip(),
                                                   headless=self.workflow_panel.mode.currentData() == 'operator')
        if identifier:
            self.pending[identifier] = "connect"
        else:
            self.log.appendPlainText("Isaac 자동 시작 · 준비 후 Connect 실행 · 로그: " + str(self.runtime_launcher.log_path))
        atomic(self.workspace / "ui_settings.json", {k: v.text() for k, v in self.fields.items()})

    def disconnect_sim(self):
        if self.worker and self.worker.isRunning():
            raise ValueError("보정 작업이 끝난 뒤 Disconnect를 실행하세요.")
        identifier = self.runtime_launcher.disconnect()
        if identifier:
            self.pending[identifier] = "disconnect"
        else:
            self.log.appendPlainText("자동 연결 취소 · 시작 중인 Isaac 종료 요청")

    def refresh(self):
        atomic(self.bus.root / "ui_heartbeat.json", {"time": time.time()})
        try:
            identifier = self.runtime_launcher.poll()
            if identifier:
                self.pending[identifier] = "connect"
                self.log.appendPlainText("Isaac 준비 완료 · Connect 요청 전송")
        except Exception as exc:
            self.log.appendPlainText("ERROR: " + str(exc))
        s = self.bus.status()
        fresh = self.bus.responsive(s)
        self.status.setText(
            (
                f"{s.get('state')} · Capture {s.get('capture_number', 0):06d} · Real {s.get('real_connected')} / Sim {s.get('sim_connected')}\n{s.get('message', '')}\nJoint: {s.get('joint_error')}\n{s.get('error') or ''}"
                if fresh
                else "Isaac 연결 끊김 / 시작 대기 · 마지막 영상은 실시간이 아닙니다"
            )
        )
        if self.runtime_launcher.pending is not None:
            self.status.setText("Isaac 자동 시작 중 · 준비 후 연결됩니다.\n로그: " + str(self.runtime_launcher.log_path))
        availability = s.get("availability", {}) if fresh else {}
        quality = s.get("capture_quality") if fresh else None
        self.live_details.setText(
            f"Hardware: {availability.get('robot_state', '확인 불가')} · "
            f"Side: {availability.get('real_side', '확인 불가')} · Wrist: {availability.get('real_wrist', '확인 불가')}\n"
            f"적용 profile hash: {s.get('profile_hash') if fresh else '확인 불가'}")
        self.capture_details.setText(
            f"Session: {s.get('capture_session', '확인 불가')} · 저장 수: {s.get('capture_number', 0)}\n"
            f"상태 (최근 관측 기준): {availability or 'runtime 확인 불가'}\n"
            f"Capture 품질: {json.dumps(quality, ensure_ascii=False) if quality else '아직 평가되지 않음'}\n"
            f"최근 저장: {s.get('last_capture') or '없음'}")
        self.profile_details.setText(
            f"선택 profile/revision: {self.fields['profile'].text() or '미선택'}\n"
            f"Runtime 적용 hash: {s.get('profile_hash') if fresh else '확인 불가'}\n"
            "Provenance와 보정 상태는 아래 Profile JSON에서 확인하고, metrics 결과는 하단 로그에서 확인하세요.")
        if not fresh:
            for label in self.views.values():
                label.source = QPixmap()
                label.setText("No live frame")
        if fresh:
            for name, label in self.views.items():
                f = self.bus.root / (name + ".jpg")
                connected = s.get("real_connected") if name.startswith("real_") else s.get("sim_connected")
                if connected and f.exists() and time.time() - f.stat().st_mtime < 2:
                    label.show_image(f)
                else:
                    label.source = QPixmap()
                    label.setText("No live frame")
        for identifier, name in list(self.pending.items()):
            path = self.bus.root / ("reply_" + identifier + ".json")
            if path.exists():
                reply = json.loads(path.read_text())
                self.log.appendPlainText(name + ": " + json.dumps(reply, ensure_ascii=False))
                path.unlink()
                del self.pending[identifier]

    def work(self, fn, done=None):
        if self.worker and self.worker.isRunning():
            raise ValueError("다른 작업이 진행 중입니다")
        self.cancel.clear()
        self.worker = Worker(fn)
        self.worker.progress.connect(self.log.appendPlainText)
        self.worker.failed.connect(lambda s: self.log.appendPlainText("ERROR: " + s))
        self.worker.done.connect(
            done or (lambda result: self.log.appendPlainText(json.dumps(result, ensure_ascii=False, indent=2)))
        )
        self.worker.start()

    def open_profile(self):
        from ..workflow.artifacts import browse_directory
        path, _ = QFileDialog.getOpenFileName(self, "Profile 선택", browse_directory("revision"), "JSON (*.json)")
        if path:
            self.selected(path)

    def template(self):
        path = ROOT / "docs/real2sim/measurements.example.json"
        self.editor.setPlainText(path.read_text())
        self.tabs.setCurrentIndex(2)
        self.advanced.setCurrentIndex(0)

    def save_profile(self):
        p = json.loads(self.editor.toPlainText())
        p = upgrade_legacy(p)
        parent = self.fields["profile"].text().strip()
        p["revision"] = {"kind": "manual", "parent": digest(load(parent)) if parent else None}
        path = revision(self.workspace, validate(p))
        self.selected(path)
        self.log.appendPlainText("Saved " + str(path))

    def reload_references(self):
        records, warnings = references.discover(self.workspace / "references")
        self.reference_files = [r["path"] for r in records if r["role"] == "real_reference"]
        counts = {v: sum(r["role"] == "real_reference" and r["view"] == v for r in records)
                  for v in references.VIEWS}
        legacy = sum(r["role"] == "unclassified" for r in records)
        self.reference_summary.setText(" · ".join(f"{v}: {n}" for v, n in counts.items()) + f" · legacy 미분류: {legacy}")
        self.reference_list.setPlainText("\n".join(
            [f"{r['role']} / {r['view']} · {r['original_filename']} · {r['path']}" for r in records]
            + ["확인 필요: " + w for w in warnings]))

    def import_media(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "실제 환경 사진 선택 (선택한 전체 batch에 view 적용)", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not files:
            return
        view = self.reference_view.currentText()
        self.work(
            lambda notify: references.import_images(self.workspace / "references", files, view),
            lambda result: (self.reload_references(), self.log.appendPlainText(f"실제 reference {len(result['items'])}장 등록")),
        )

    def intrinsics(self, view):
        files, _ = QFileDialog.getOpenFileNames(
            self, "같은 카메라의 체커보드 사진 (10장 이상)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not files:
            return
        text, ok = QInputDialog.getText(self, "보드 규격", "내부 코너 열, 행, 칸 한 변(m) 예: 9,6,0.025")
        if not ok:
            return
        cols, rows, square = map(float, text.split(","))
        p = self.profile()

        def task(notify):
            p["cameras"][view].update(calibration.calibrate_intrinsics(files, int(cols), int(rows), square))
            p["cameras"][view]["extrinsics_calibrated"] = False
            for key in ("K", "distortion"):
                p["cameras"][view]["provenance"][key] = "numerically_fitted"
            p["cameras"][view]["provenance"]["resolution"] = "derived"
            return revision(self.workspace, p, p["cameras"][view]["intrinsics_report"])

        self.work(task, self.selected)

    def annotate(self):
        folder = QFileDialog.getExistingDirectory(
            self, "COMPLETE가 있는 capture 폴더 선택", str(self.workspace / "captures")
        )
        if not folder:
            return
        if not (Path(folder) / "COMPLETE").exists():
            raise ValueError("완료된 캡처가 아닙니다")
        view, ok = QInputDialog.getItem(self, "Camera", "Raw real camera", ["side", "wrist"], 0, False)
        if not ok:
            return
        dataset_path = Path(self.dataset.text())
        dataset = {"observations": []}
        if dataset_path.exists():
            dataset = json.loads(dataset_path.read_text())
        existing = [
            r["split"]
            for r in dataset["observations"]
            if str((dataset_path.parent / r["capture"]).resolve()) == str(Path(folder).resolve())
        ]
        split = (
            existing[0]
            if existing
            else QInputDialog.getItem(self, "데이터 분할", "이 capture 전체의 용도", ["train", "validation"], 0, False)[
                0
            ]
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Raw pixel 선택 → 실측 3D 좌표 입력")
        dialog.resize(1000, 800)
        layout = QVBoxLayout(dialog)
        label = ImageLabel()
        label.show_image(Path(folder) / ("real_" + view + ".png"))
        layout.addWidget(label)
        info = QLabel("점 좌표 frame: world / workspace / table / object:이름 / robot:링크이름")
        layout.addWidget(info)

        def clicked(u, v):
            text, ok = QInputDialog.getText(dialog, "실측점", "frame, x(m), y(m), z(m) 예: world,0.1,0.2,0.6")
            if not ok:
                return
            try:
                parts = text.split(",")
                frame = parts[0].strip()
                xyz = list(map(float, parts[1:]))
                if len(xyz) != 3:
                    raise ValueError("XYZ 3개를 입력하세요")
                dataset["observations"].append(
                    {
                        "capture": str(Path(folder).resolve()),
                        "view": view,
                        "split": split,
                        "frame": frame,
                        "point_m": xyz,
                        "uv": [u, v],
                    }
                )
                atomic(dataset_path, dataset)
                info.setText(f"저장 {len(dataset['observations'])}점 · pixel ({u:.1f}, {v:.1f})")
            except Exception as exc:
                QMessageBox.warning(dialog, "입력 오류", str(exc))

        label.clicked.connect(clicked)
        dialog.exec()

    def metrics(self):
        p = self.profile()
        path = self.dataset.text()
        self.work(lambda notify: calibration.evaluate(p, calibration.load_dataset(path)))

    def ensure_stopped(self):
        s = self.bus.status()
        if s.get("state") not in ("STOPPED", "CONNECTED"):
            raise ValueError("장비 연결 후 Stop 상태에서 실행하세요")

    def optimize(self):
        self.ensure_stopped()
        p = self.profile()
        path = self.dataset.text()
        group = self.group.currentText()

        def task(notify):
            rows = calibration.load_dataset(path)
            candidate, report = calibration.optimize(p, rows, group)
            dest = revision(self.workspace, candidate, report)
            if not report["accepted"]:
                notify(json.dumps(report))
                return None
            identifier = self.bus.send(
                "render_dataset", {"profile": str(dest), "captures": sorted({r["capture"] for r in rows})}
            )
            result = self.bus.wait(identifier, 300)
            notify(json.dumps(result))
            return dest

        self.work(
            task,
            lambda result: (
                self.selected(result) if result else self.log.appendPlainText("후보 거부: 기존 profile 유지")
            ),
        )

    def rerender(self):
        self.ensure_stopped()
        path = self.dataset.text()
        profile = self.profile_path()
        self.work(
            lambda notify: self.bus.wait(
                self.bus.send(
                    "render_dataset",
                    {"profile": profile, "captures": sorted({r["capture"] for r in calibration.load_dataset(path)})},
                ),
                300,
            )
        )

    def closeEvent(self, event):
        self.runtime_launcher.cancel_connect()
        self.cancel.set()
        self.workflow_panel.service.cancel()
        try:
            self.bus.send("stop")
        except RuntimeError:
            pass
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "작업 종료 대기", "중지 요청을 보냈습니다. 현재 계산 작업이 끝난 뒤 닫아주세요."
            )
            event.ignore()
            return
        atomic(self.workspace / "ui_settings.json", {k: v.text() for k, v in self.fields.items()})
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(ROOT / "outputs/real2sim"))
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    from ..workflow.theme import STYLE
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = Window(args.workspace)
    w.show()
    sys.exit(app.exec())
