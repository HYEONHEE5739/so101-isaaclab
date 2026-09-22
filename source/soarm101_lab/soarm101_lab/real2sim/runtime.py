"""Isaac process state machine. All hardware writes stay in this process."""

import copy
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER, require_capture_mapping
from .profile import load, render_k, digest, validate
from .storage import Bus, CaptureStore, atomic, new_dir
from .core import JOINTS, CONVENTION, stamp, to_sim, from_sim, joint_error, Stability, assess_pair
from .hardware import Leader, Follower
from .cameras import CameraReader, device_key
from .scene import build_usd, configure_environment, apply_profile
from .calibration import image_metrics


def tf(pos, q):
    t = np.eye(4)
    t[:3, 3] = pos
    t[:3, :3] = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
    return t


class Runtime:
    def __init__(self, args, app):
        self.args = args
        self.app = app
        self.bus = Bus(Path(args.workspace) / "ipc", owner=True)
        self.env = None
        self.leader = None
        self.follower = None
        self.cameras = {}
        self.running = False
        self.shutdown_requested = False
        self.activated = False
        self.state = "IDLE"
        self.message = "Choose profile with field provenance and Connect in PyQt"
        self.profile = None
        self.store = CaptureStore(Path(args.workspace) / "captures")
        self.last_capture = None
        self.pending = None
        self.capture_requests = []
        self.writer = ThreadPoolExecutor(max_workers=1)
        self.images = {}
        self.last_preview = 0.0
        self.error = None
        self.step = 0
        self.last_cycle = None
        self.real_history = Stability()
        self.sim_history = Stability()
        self.snapshot_data = None
        self.publish()

    def publish(self):
        self.bus.publish(
            state=self.state,
            message=self.message,
            error=self.error,
            capture_number=self.store.index,
            capture_session=self.store.session_id,
            capture_quality=(self.snapshot_data or {}).get("quality"),
            availability={
                "robot_state": bool(self.follower and self.snapshot_data),
                "sim": self.env is not None,
                **{name: ("error: " + camera.error if camera.error else
                           "frame received" if name in (self.snapshot_data or {}).get("timestamps", {}).get("real_cameras", {})
                           else "waiting for frame") for name, camera in self.cameras.items()},
            },
            last_capture=self.last_capture,
            real_connected=bool(self.follower),
            sim_connected=self.env is not None,
            joint_error=(self.snapshot_data or {}).get("real_sim_joint_error"),
            profile_hash=digest(self.profile) if self.profile else None,
        )

    def disconnect_hardware(self):
        self.running = False
        self.activated = False
        for item in [self.follower, self.leader, *self.cameras.values()]:
            if item:
                try:
                    item.close()
                except Exception:
                    traceback.print_exc()
        self.follower = self.leader = None
        self.cameras = {}
        self.snapshot_data = None

    def connect(self, payload):
        if self.pending:
            raise ValueError("Wait for capture save to finish")
        self.state = "CONNECTING"
        self.message = "Isaac 연결 준비 중 · heartbeat 일시 정지, runtime 잠금으로 생존 확인 (진행 여부는 확인 불가)"
        self.error = None
        self.publish()
        self.disconnect_hardware()
        p = load(payload["profile"])
        cfg = payload["hardware"]
        self.config = cfg
        if not payload.get("preview_only") and not cfg.get("follower_id"):
            raise ValueError("Enter the actual follower robot id")
        if (
            not payload.get("preview_only")
            and Path(cfg["leader_port"]).resolve() == Path(cfg["follower_port"]).resolve()
        ):
            raise ValueError("Leader/follower ports must differ")
        if not payload.get("preview_only") and device_key(cfg["side_camera"]) == device_key(cfg["wrist_camera"]):
            raise ValueError("Side/wrist cameras must differ")
        self.state = "BUILDING_ENVIRONMENT"
        self.message = "Isaac 환경 생성 중 · heartbeat 일시 정지, runtime 잠금으로 생존 확인 (진행 여부는 확인 불가)"
        self.publish()
        if self.env is not None:
            self.env.close()
            self.env = None
        from isaaclab.envs import ManagerBasedEnv
        from soarm101_lab.tasks.manager_based.soarm101_lab import SO101TeleopEnvCfg

        ec = SO101TeleopEnvCfg()
        ec.scene.num_envs = 1
        ec.sim.device = self.args.device
        ec.actions.joint_positions.joint_names = list(JOINTS)
        ec.actions.joint_positions.preserve_order = True
        ec.sim.render_interval = ec.decimation
        asset_dir = new_dir(Path(self.args.workspace) / "generated", "scene")
        usd = build_usd(asset_dir / "workspace.usda", p)
        configure_environment(ec, p, usd, calibration=True)
        if not Path(ec.scene.robot.spawn.usd_path).is_file():
            raise FileNotFoundError("Install the repository SO101 robot USD at " + str(ec.scene.robot.spawn.usd_path))
        self.env = ManagerBasedEnv(cfg=ec)
        self.env.reset()
        from .scene import apply_robot_appearance
        from isaaclab.sim import get_current_stage
        apply_robot_appearance(get_current_stage(), p)
        self.profile = p
        for _ in range(5):
            self.env.sim.render()
        robot = self.env.scene["robot"]
        self.ids = [robot.joint_names.index(n) for n in JOINTS]
        if list(self.env.action_manager.get_term("joint_positions")._joint_names) != list(JOINTS):
            raise RuntimeError("Unexpected joint action mapping")
        self.limits = robot.data.soft_joint_pos_limits[0, self.ids].detach().cpu().numpy().copy()
        self.robot_hash = hashlib.sha256(Path(ec.scene.robot.spawn.usd_path).read_bytes()).hexdigest()
        if payload.get("preview_only"):
            self.state = "PREVIEW"
            self.message = "Simulation preview only; no hardware ports opened"
            self.error = None
            return
        self.state = "CONNECTING"
        self.message = "Isaac 환경 생성 완료 · 장비 연결 중 (새 follower 목표 전송 없음)"
        self.publish()
        # Constructors may read files; overridden automatic calibration can never run.
        self.leader = Leader(self.env, cfg["leader_port"], cfg["leader_calibration"], False)
        self.follower = Follower(
            cfg["follower_port"], cfg["follower_id"], cfg.get("follower_calibration_dir") or None, False
        )
        self.leader.connect()
        self.follower.connect()
        self.leader.activate()  # Release leader only after both calibrations match.
        for view in ("side", "wrist"):
            w, h = p["cameras"][view]["resolution"]
            c = CameraReader(cfg[view + "_camera"], w, h, int(cfg.get("fps", 30)))
            self.cameras["real_" + view] = c
            c.start()
        self.hardware_identity = digest(
            {
                "robot": self.robot_hash,
                "leader": self.leader.metadata["calibration"],
                "follower": self.follower.metadata["calibration"],
                "convention": CONVENTION,
                "cameras": {
                    n: {"device": c.metadata["resolved_device"], "size": c.metadata["requested_size"]}
                    for n, c in self.cameras.items()
                },
            }
        )
        self.real_history = Stability()
        self.sim_history = Stability()
        self.error = None
        self.state = "CONNECTED"
        self.message = "Connected: leader torque released; no follower goal sent. Press Start to follow leader targets."
        self.last_cycle = None

    def start(self):
        if not self.follower:
            raise ValueError("Connect hardware first")
        if not self.activated:
            self.leader.activate()
            self.follower.activate()
            self.activated = True
        self.running = True
        self.state = "RUNNING"
        self.message = "Leader controls Real + Sim"
        self.last_cycle = None

    def stop(self):
        self.running = False
        self.state = "STOPPED" if self.env else "IDLE"
        self.message = "Stopped: no new physical targets. Follower holds last target until Disconnect."
        for c in self.capture_requests:
            self.bus.reply(c["id"], False, "Capture cancelled by Stop")
        self.capture_requests = []

    def check_target(self, native):
        q = np.array([to_sim(native)[n] for n in JOINTS])
        return q

    def set_measured_pose(self, native):
        import torch

        q = self.check_target(native)
        robot = self.env.scene["robot"]
        full = robot.data.joint_pos.clone()
        full[0, self.ids] = torch.tensor(q, device=self.env.device, dtype=full.dtype)
        robot.write_joint_state_to_sim(full, torch.zeros_like(full))
        robot.set_joint_position_target(full)
        self.env.sim.forward()
        self.env.scene.update(self.env.step_dt)
        for _ in range(2):
            self.env.sim.render()
        for v in ("side", "wrist"):
            self.env.scene["camera_" + v + "view"].update(self.env.step_dt, force_recompute=True)

    def snapshot(self):
        begin = stamp()
        env = self.env
        robot = env.scene["robot"]
        images = {}
        cameras = {}
        base = tf(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        links = {
            n: (
                np.linalg.inv(base)
                @ tf(robot.data.body_pos_w[0, i].cpu().numpy(), robot.data.body_quat_w[0, i].cpu().numpy())
            ).tolist()
            for i, n in enumerate(robot.body_names)
        }
        for v in ("side", "wrist"):
            sensor = env.scene["camera_" + v + "view"]
            d = sensor.data
            rgb = d.output["rgb"][0, :, :, :3].cpu().numpy().copy()
            if rgb.dtype != np.uint8:
                raise TypeError("Unexpected render dtype")
            images["sim_" + v] = rgb
            k = d.intrinsic_matrices[0].cpu().numpy()
            if not np.allclose(k, render_k(self.profile["cameras"][v]), atol=0.5):
                raise RuntimeError("Isaac rendered intrinsics do not match rectified target K")
            cameras[v] = {
                "K": k.tolist(),
                "T_world_optical": tf(d.pos_w[0].cpu().numpy(), d.quat_w_ros[0].cpu().numpy()).tolist(),
                "resolution": [rgb.shape[1], rgb.shape[0]],
                "frame": int(sensor.frame[0].item()),
                "distortion": [0] * 5,
            }
        q = dict(zip(JOINTS, robot.data.joint_pos[0, self.ids].cpu().tolist()))
        end = stamp()
        return (
            images,
            {"start": begin, "end": end, "sim_time": float(env.sim.current_time)},
            q,
            cameras,
            links,
            base.tolist(),
        )

    def tick(self):
        import torch

        if not self.env:
            return
        if not self.follower:
            self.env.sim.render()
            for v in ("side", "wrist"):
                self.env.scene["camera_" + v + "view"].update(self.env.step_dt, force_recompute=True)
            images, *_ = self.snapshot()
            for n, rgb in images.items():
                path = self.bus.root / (n + ".pending")
                Image.fromarray(rgb).save(path, format="JPEG")
                path.replace(self.bus.root / (n + ".jpg"))
            return
        native, leader_interval = self.leader.read()
        real, real_interval = self.follower.read()
        sent = None
        send_interval = None
        pose_error = None
        if self.running:
            try:
                q = self.check_target(native)
                sent, send_interval = self.follower.send(native)
                q = self.check_target(sent)
                self.env.step(torch.tensor(q, device=self.env.device, dtype=torch.float32)[None])
            except ValueError as exc:
                self.stop()
                self.message = str(exc) + " — paused; correct pose then Start"
        else:
            try:
                self.set_measured_pose(real)
            except ValueError as exc:
                pose_error = str(exc)
                self.message = pose_error + " — Sim shows last valid pose; capture unavailable"
                self.env.sim.render()
                for v in ("side", "wrist"):
                    self.env.scene["camera_" + v + "view"].update(self.env.step_dt, force_recompute=True)
        self.step += 1
        sim_images, sim_interval, sim_q, sim_cameras, links, base = self.snapshot()
        real, real_interval = self.follower.read()
        packets = {n: c.nearest(sim_interval["end"]["monotonic_ns"]) for n, c in self.cameras.items()}
        self.real_history.add(real_interval["end"]["monotonic_ns"], real)
        self.sim_history.add(sim_interval["end"]["monotonic_ns"], from_sim(sim_q))
        error = joint_error(real, sim_q)
        quality = assess_pair(
            packets,
            sim_interval["end"],
            real_interval,
            self.real_history.assess(),
            self.sim_history.assess(),
            error,
            time.monotonic_ns(),
        )
        if pose_error:
            quality["eligible_for_calibration"] = False
            quality["reasons"].append("Measured pose cannot be represented in Sim: " + pose_error)
        self.images = {**sim_images, **{n: p["rgb"] for n, p in packets.items() if p is not None}}
        self.snapshot_data = {
            "leader_target": native,
            "physical_follower_measured_joint_state": real,
            "sim_measured_joint_state_rad": sim_q,
            "real_sim_joint_error": error,
            "joint_names": list(JOINTS),
            "joint_convention": CONVENTION,
            "sim_joint_mapping": DEFAULT_MAPPER.metadata(),
            "active_profile": copy.deepcopy(self.profile),
            "profile_hash": digest(self.profile),
            "sim_cameras": sim_cameras,
            "real_camera_metadata": {n: c.metadata for n, c in self.cameras.items()},
            "links_base": links,
            "robot_base_transform": base,
            "workspace_transform": self.profile["workspace"]["T_world"],
            "objects": self.profile["objects"],
            "quality": quality,
            "hardware_identity": self.hardware_identity,
            "hardware": {"leader": self.leader.metadata, "follower": self.follower.metadata},
            "timestamps": {
                "leader": leader_interval,
                "follower": real_interval,
                "send": send_interval,
                "sim": sim_interval,
                "real_cameras": {n: {k: v for k, v in p.items() if k != "rgb"} for n, p in packets.items() if p},
            },
            "sent_target": sent,
        }
        if self.capture_requests and self.pending is None:
            command = self.capture_requests.pop(0)
            if not quality["eligible_for_calibration"]:
                self.bus.reply(command["id"], False, "Capture rejected: " + ", ".join(quality["reasons"]))
            else:
                # Re-render the measured physical pose for calibration FK, not just the target.
                self.set_measured_pose(real)
                si, interval, sq, sc, links, base = self.snapshot()
                after, after_interval = self.follower.read()
                err = joint_error(after, sq)
                pairs = {n: c.nearest(interval["end"]["monotonic_ns"]) for n, c in self.cameras.items()}
                self.real_history.add(after_interval["end"]["monotonic_ns"], after)
                quality = assess_pair(
                    pairs,
                    interval["end"],
                    after_interval,
                    self.real_history.assess(),
                    self.sim_history.assess(),
                    err,
                    time.monotonic_ns(),
                )
                if not quality["eligible_for_calibration"]:
                    self.bus.reply(
                        command["id"], False, "Exact-pose capture timing/motion rejected: " + str(quality["reasons"])
                    )
                else:
                    images = {**si, **{n: p["rgb"] for n, p in pairs.items()}}
                    state = copy.deepcopy(self.snapshot_data)
                    state.update(
                        physical_follower_measured_joint_state=after,
                        sim_measured_joint_state_rad=sq,
                        real_sim_joint_error=err,
                        sim_cameras=sc,
                        links_base=links,
                        robot_base_transform=base,
                        quality=quality,
                        capture_mode="measured_pose_render",
                    )
                    state["timestamps"].update(
                        sim=interval,
                        follower=after_interval,
                        real_cameras={n: {k: v for k, v in p.items() if k != "rgb"} for n, p in pairs.items()},
                        request=command["time"],
                    )
                    self.pending = (self.writer.submit(self.store.save, images, state), command["id"])
        if time.monotonic() - self.last_preview > 0.1:
            self.last_preview = time.monotonic()
            for n, rgb in self.images.items():
                path = self.bus.root / (n + ".pending")
                Image.fromarray(rgb).save(path, format="JPEG", quality=85)
                path.replace(self.bus.root / (n + ".jpg"))

    def apply(self, p):
        validate(p)
        if self.running:
            raise ValueError("Stop before updating scene")
        if not self.env:
            raise ValueError("Connect scene first")
        # Changes in topology/dimensions need rebuild; optimizer only adjusts transforms.
        old = self.profile
        if p.get("wrist_mount") != old.get("wrist_mount"):
            raise ValueError("Wrist mount changed; reconnect")
        if set(p["objects"]) != set(old["objects"]):
            raise ValueError("Object list changed; reconnect")
        for a, b in [(p["table"], old["table"]), *[(p["objects"][n], old["objects"][n]) for n in p["objects"]]]:
            if any(a.get(k) != b.get(k) for k in ("shape", "dimensions_m", "wall_m", "render_enabled", "bottom_diameter_m", "print_texture")):
                raise ValueError("Geometry dimensions changed; reconnect")
        for view in ("side", "wrist"):
            if p["cameras"][view]["resolution"] != old["cameras"][view]["resolution"]:
                raise ValueError("Resolution changed; reconnect before applying")
        try:
            apply_profile(self.env, p)
        except Exception:
            apply_profile(self.env, old)
            raise
        self.profile = p
        self.snapshot_data = None
        self.real_history = Stability()
        self.sim_history = Stability()

    def render_dataset(self, payload):
        if self.running:
            raise ValueError("Stop first")
        previous = copy.deepcopy(self.profile)
        p = load(payload["profile"])
        folder = new_dir(Path(self.args.workspace) / "renders", "render")
        results = []
        try:
            self.apply(p)
            for name in payload["captures"]:
                self.publish()
                source = Path(name)
                if not (source / "COMPLETE").exists():
                    raise ValueError("Incomplete capture")
                state = json.loads((source / "state.json").read_text())
                require_capture_mapping(state)
                if state["hardware_identity"] != self.hardware_identity:
                    raise ValueError("Robot calibration/asset changed")
                if not state["quality"]["eligible_for_calibration"]:
                    raise ValueError("Invalid capture")
                self.set_measured_pose(state["physical_follower_measured_joint_state"])
                ims, _, _, cams, _, _ = self.snapshot()
                dest = new_dir(folder, "pose")
                record = {"capture": str(source), "profile_hash": digest(p), "views": {}}
                for v in ("side", "wrist"):
                    real = np.asarray(Image.open(source / ("real_" + v + ".png")).convert("RGB"))
                    metric, rect = image_metrics(real, ims["sim_" + v], p["cameras"][v])
                    Image.fromarray(ims["sim_" + v]).save(dest / ("sim_" + v + ".png"))
                    Image.fromarray(rect).save(dest / ("rectified_real_" + v + ".png"))
                    Image.blend(Image.fromarray(rect), Image.fromarray(ims["sim_" + v]), 0.5).save(
                        dest / ("overlay_" + v + ".png")
                    )
                    record["views"][v] = metric
                atomic(dest / "metrics.json", record)
                results.append(record)
            atomic(folder / "summary.json", results)
            return {"directory": str(folder), "poses": len(results)}
        except BaseException:
            self.apply(previous)
            raise

    def command(self, c):
        name = c["command"]
        p = c["payload"]
        if name == "connect":
            self.connect(p)
            return "Connected; follower not commanded"
        if name == "start":
            self.start()
            return "Started"
        if name == "stop":
            self.stop()
            return "Stopped"
        if name == "disconnect":
            self.stop()
            self.disconnect_hardware()
            self.state = "CLOSING"
            self.shutdown_requested = True
            return "Disconnected; Isaac runtime shutting down"
        if name == "capture":
            if not self.follower:
                raise ValueError("Connect first")
            if len(self.capture_requests) >= 32:
                raise ValueError("Capture queue full")
            self.capture_requests.append(c)
            return None
        if name == "apply_profile":
            self.apply(load(p["profile"]))
            return "Profile applied"
        if name == "render_dataset":
            return self.render_dataset(p)
        if name == "export":
            if self.running:
                raise ValueError("Stop first")
            profile = load(p["profile"])
            folder = new_dir(Path(self.args.workspace) / "exports", "workspace")
            usd = build_usd(folder / "workspace.usd", profile)
            atomic(folder / "profile.json", profile)
            return {"usd": usd, "profile": str(folder / "profile.json")}
        raise ValueError("Unsupported command: " + name)

    def run(self):
        try:
            while self.app.is_running():
                begin = time.monotonic()
                if self.running:
                    heartbeat = self.bus.root / "ui_heartbeat.json"
                    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 3:
                        self.stop()
                        self.message = "UI heartbeat lost; physical commands stopped"
                    if self.last_cycle and begin - self.last_cycle > 0.75:
                        self.stop()
                        self.message = "Control loop stalled; press Start after inspection"
                self.last_cycle = begin
                for c in self.bus.commands():
                    try:
                        result = self.command(c)
                        if result is not None:
                            self.error = None
                            self.bus.reply(c["id"], True, result)
                    except Exception as exc:
                        if c["command"] == "connect":
                            self.disconnect_hardware()
                            self.state = "ERROR"
                        self.error = str(exc)
                        self.bus.reply(c["id"], False, str(exc))
                        traceback.print_exc()
                    finally:
                        # Publish completion/error before the first potentially slow render.
                        self.publish()
                    if self.shutdown_requested:
                        break
                if self.shutdown_requested:
                    break
                try:
                    self.tick()
                except Exception as exc:
                    self.disconnect_hardware()
                    self.state = "ERROR"
                    self.error = str(exc)
                    traceback.print_exc()
                if self.pending and self.pending[0].done():
                    future, identifier = self.pending
                    try:
                        self.last_capture = str(future.result())
                        self.bus.reply(identifier, True, self.last_capture)
                    except Exception as exc:
                        self.bus.reply(identifier, False, str(exc))
                    self.pending = None
                self.publish()
                if not self.env:
                    self.app.update()
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - begin)))
        finally:
            self.disconnect_hardware()
            self.writer.shutdown(wait=True)
            if self.env:
                self.env.close()
            self.bus.close()
