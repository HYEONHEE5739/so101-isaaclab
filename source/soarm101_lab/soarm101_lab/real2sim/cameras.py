"""Timestamped LeRobot OpenCVCamera readers. Host receipt times, NOT exposure times."""

from collections import deque
import math
from pathlib import Path
from threading import Event, Lock, Thread
from .core import stamp


def device_key(device):
    if str(device).isdigit():
        return str(Path("/dev/video" + str(device)).resolve())
    return str(Path(device).resolve())


class CameraReader:
    def __init__(self, device, width=640, height=480, fps=30, fourcc=None):
        from .hardware import audit_camera_sources

        audit = audit_camera_sources()
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, ColorMode

        self.camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=int(device) if str(device).isdigit() else str(device),
                width=width,
                height=height,
                fps=fps,
                color_mode=ColorMode.RGB,
                fourcc=fourcc,
            )
        )
        self.frames = deque(maxlen=90)
        self.lock = Lock()
        self.stop = Event()
        self.thread = None
        self.error = None
        self.metadata = {
            "api_audit": audit,
            "device": str(device),
            "resolved_device": device_key(device),
            "requested_size": [width, height],
            "requested_fps": fps,
            "fourcc": fourcc,
            "color": "RGB",
            "rotation": "none",
            "timestamp_kind": "host_read_interval",
            "hardware_exposure_timestamp_available": False,
        }

    def start(self):
        import cv2

        self.camera.connect()
        cap = self.camera.videocapture
        # Best effort; backend may ignore this. Always report the result, never promise zero buffering.
        self.metadata["buffer_size_request_accepted"] = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
        self.metadata["reported"] = {
            name: float(cap.get(key))
            for name, key in [
                ("width", cv2.CAP_PROP_FRAME_WIDTH),
                ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                ("fps", cv2.CAP_PROP_FPS),
                ("exposure", cv2.CAP_PROP_EXPOSURE),
                ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
            ]
        }
        self.metadata["reported"] = {k: v if math.isfinite(v) else None for k, v in self.metadata["reported"].items()}
        self.thread = Thread(target=self._run, daemon=True, name="real-camera-" + str(self.metadata["device"]))
        self.thread.start()

    def _run(self):
        seq = 0
        try:
            while not self.stop.is_set():
                begin = stamp()
                rgb = self.camera.read()
                end = stamp()
                seq += 1
                packet = {"rgb": rgb.copy(), "sequence": seq, "read_start": begin, "read_end": end}
                with self.lock:
                    self.frames.append(packet)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = repr(exc)

    def nearest(self, target_ns=None):
        if self.error:
            raise RuntimeError(f"Camera failed: {self.metadata['device']}: {self.error}")
        with self.lock:
            if not self.frames:
                return None
            frame = (
                self.frames[-1]
                if target_ns is None
                else min(self.frames, key=lambda f: abs(f["read_end"]["monotonic_ns"] - target_ns))
            )
            return {**frame, "rgb": frame["rgb"].copy()}

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.camera.is_connected:
            self.camera.disconnect()
        if self.thread:
            self.thread.join(timeout=1.0)
