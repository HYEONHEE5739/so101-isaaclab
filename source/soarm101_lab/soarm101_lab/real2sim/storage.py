"""Atomic publication, snapshot durability helpers and file IPC."""

import json
import os
from pathlib import Path
import time
import uuid
import fcntl


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temp.open("w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new_dir(root, prefix):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        path = root / (prefix + "_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Unable to reserve unique directory")


class Bus:
    CONNECTING_STATES = {"CONNECTING", "BUILDING_ENVIRONMENT"}

    def __init__(self, root, owner=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = None
        self.session = None
        if owner:
            self.lock = (self.root / "owner.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lock.close()
                raise RuntimeError("Another Real2Sim runtime owns this workspace")
            self.session = uuid.uuid4().hex

    def publish(self, **status):
        atomic(self.root / "status.json", {"session": self.session, "timestamp": time.time(), **status})

    def status(self):
        try:
            return json.loads((self.root / "status.json").read_text())
        except (OSError, ValueError):
            return {}

    def owner_alive(self):
        """Probe the existing local runtime lock; a dead process releases it."""
        try:
            with (self.root / "owner.lock").open("r") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            pass
        return False

    def responsive(self, status, timeout=3):
        if status.get("state") == "CLOSED":
            return False
        if status.get("state") in self.CONNECTING_STATES:
            return self.owner_alive()
        return time.time() - status.get("timestamp", 0) < timeout

    def send(self, command, payload=None):
        s = self.status()
        if not self.responsive(s, timeout=5):
            raise RuntimeError("Simulation runtime is not responding")
        if s.get("state") in self.CONNECTING_STATES:
            raise RuntimeError("Sim 환경 연결/생성 중입니다. 완료 후 명령을 보내세요.")
        if len(list(self.root.glob("cmd_*.json"))) >= 64:
            raise RuntimeError("Command queue is full")
        identifier = uuid.uuid4().hex
        atomic(
            self.root / f"cmd_{time.time_ns()}_{identifier}.json",
            {
                "id": identifier,
                "session": s["session"],
                "time": time.time(),
                "command": command,
                "payload": payload or {},
            },
        )
        return identifier

    def commands(self):
        for path in sorted(self.root.glob("cmd_*.json"))[:64]:
            try:
                d = json.loads(path.read_text())
                if d.get("session") == self.session and 0 <= time.time() - d.get("time", 0) < 30:
                    yield d
            finally:
                path.unlink(missing_ok=True)

    def reply(self, identifier, ok, result):
        atomic(self.root / ("reply_" + identifier + ".json"), {"ok": ok, "result": result})

    def wait(self, identifier, timeout=120):
        end = time.monotonic() + timeout
        path = self.root / ("reply_" + identifier + ".json")
        while time.monotonic() < end:
            if path.exists():
                result = json.loads(path.read_text())
                path.unlink()
                if not result["ok"]:
                    raise RuntimeError(result["result"])
                return result["result"]
            time.sleep(0.1)
        raise TimeoutError("Isaac command timed out: " + identifier)

    def close(self):
        if self.lock:
            self.publish(state="CLOSED")
            self.lock.close()
            self.lock = None
