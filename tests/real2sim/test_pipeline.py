import copy
import json
from pathlib import Path
import multiprocessing
import numpy as np
import pytest
from soarm101_lab.real2sim import profile, calibration, storage, scene

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def p():
    p = profile.load(ROOT / "docs/real2sim/measurements.example.json")
    # Explicit synthetic full geometry for the existing transform-fitting regressions.
    p["table"].update(dimensions_m=[0.6, 0.6, 0.05], T_workspace=np.eye(4).tolist(), render_enabled=True)
    p["table"]["provenance"] = {"dimensions_m": "user_measured", "T_workspace": "provisional"}
    p["objects"] = {"container": copy.deepcopy(p["table"])}
    p["objects"]["container"]["dimensions_m"] = [0.06, 0.06, 0.08]
    p["relations"] = {}
    for c in p["cameras"].values():
        c["intrinsics_calibrated"] = True
        c["extrinsics_calibrated"] = True
        for key in ("K", "distortion", "T_parent_camera"):
            c["provenance"][key] = "numerically_fitted"
    p["robot_base"]["calibrated"] = True
    p["workspace"]["calibrated"] = True
    for key in ("robot_base", "workspace"):
        p[key]["provenance"]["T_world"] = "numerically_fitted"
    return p


def make_rows(p, frame="world", view="side"):
    rng = np.random.default_rng(9)
    out = []
    for capture in range(5):
        link = profile.matrix([0.02 * capture, 0, 0.05 * capture, 0.05 * capture, 0.08 * capture, -0.05 * capture])
        state = {
            "links_base": {"gripper": link.tolist(), "tool0": link.tolist()},
            "physical_follower_measured_joint_state": dict(
                zip(
                    ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                    [capture * 12, 0, 0, 0, 0, 40],
                )
            ),
        }
        for i in range(12):
            point = rng.uniform([-0.15, -0.15, 0.5], [0.15, 0.15, 0.8])
            row = {
                "frame": frame,
                "point_m": point.tolist(),
                "view": view,
                "split": "validation" if capture == 4 else "train",
                "capture": str(capture),
                "_state": state,
            }
            row["uv"] = calibration.project(p, row)[0].tolist()
            out.append(row)
    return out


@pytest.mark.parametrize(
    "group,frame,view",
    [
        ("side", "world", "side"),
        ("robot_base", "robot:tool0", "side"),
        ("workspace", "workspace", "side"),
        ("table", "table", "side"),
        ("object:container", "object:container", "side"),
        ("wrist", "world", "wrist"),
    ],
)
def test_fit_recovers_transform(p, group, frame, view):
    rows = make_rows(p, frame, view)
    wrong = copy.deepcopy(p)
    x = profile.pose6(profile.get_group(p, group))
    x += np.array([0.015, -0.01, 0.008, 0.012, -0.018, 0.009])
    profile.set_group(wrong, group, profile.matrix(x))
    fitted, report = calibration.optimize(wrong, rows, group)
    assert report["accepted"], report
    profile.validate(fitted)
    assert report["after"]["validation"]["rmse_px"] < 1e-4
    assert np.allclose(profile.get_group(fitted, group), profile.get_group(p, group), atol=1e-5)


def test_unobservable_and_holdout_rejected(p):
    rows = make_rows(p)
    for r in rows:
        r["point_m"] = [0, 0, 0.2]
        r["uv"] = calibration.project(p, r)[0].tolist()
    wrong = copy.deepcopy(p)
    wrong["cameras"]["side"]["T_parent_camera"][0][3] += 0.01
    _, report = calibration.optimize(wrong, rows, "side")
    assert not report["accepted"]
    rows = make_rows(p)
    for r in rows:
        if r["split"] == "validation":
            r["uv"] = calibration.project(wrong, r)[0].tolist()
    _, report = calibration.optimize(wrong, rows, "side")
    assert not report["accepted"]


def test_gauge_and_pose_coverage(p):
    p["cameras"]["side"]["extrinsics_calibrated"] = False
    with pytest.raises(ValueError, match="side camera"):
        calibration.optimize(p, make_rows(p, "robot:tool0"), "robot_base")
    with pytest.raises(ValueError, match="captures"):
        calibration.optimize(p, make_rows(p)[:12], "side")


def test_invalid_transform_and_agent_geometry(p):
    wrong = copy.deepcopy(p)
    wrong["robot_base"]["T_world"][0][0] = -1
    with pytest.raises(ValueError):
        profile.validate(wrong)
    style = {
        "styles": [{"id": "table", "color": [0.1, 0.2, 0.3], "roughness": 0.7}],
        "light_intensity": 1000,
        "notes": "visual approximation",
        "robot_base": [999, 999, 999],
    }
    result = profile.initial(p, style)
    assert result["robot_base"] == p["robot_base"]
    assert result["table"]["dimensions_m"] == p["table"]["dimensions_m"]
    p["geometry_source"] = "example"
    with pytest.raises(ValueError):
        profile.initial(p)


def _save(root):
    store = storage.CaptureStore(root)
    store.save(
        {n: np.zeros((8, 8, 3), dtype=np.uint8) for n in ("real_side", "sim_side", "real_wrist", "sim_wrist")},
        {"quality": {"eligible_for_calibration": True}},
    )


def test_capture_concurrent_restart_failure(tmp_path, monkeypatch):
    processes = [multiprocessing.Process(target=_save, args=(tmp_path,)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        assert p.exitcode == 0
    assert len(storage.captures(tmp_path)) == 4
    store = storage.CaptureStore(tmp_path)
    images = {n: np.zeros((8, 8, 3), dtype=np.uint8) for n in ("real_side", "sim_side", "real_wrist", "sim_wrist")}

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(storage.Image.Image, "save", fail)
    with pytest.raises(OSError):
        store.save(images, {})
    assert len(storage.captures(tmp_path)) == 4
    assert (store.root / "capture_000001").exists()


def test_ipc_stale_session_and_exclusive_owner(tmp_path):
    b = storage.Bus(tmp_path, True)
    b.publish(state="STOPPED")
    with pytest.raises(RuntimeError):
        storage.Bus(tmp_path, True)
    client = storage.Bus(tmp_path)
    id = client.send("capture")
    cmd = list(b.commands())
    assert cmd[0]["id"] == id
    b.reply(id, True, "saved")
    assert client.wait(id) == "saved"
    id = client.send("start")
    b.close()
    b = storage.Bus(tmp_path, True)
    assert not list(b.commands())
    b.close()


def test_usd_reopens_with_hollow_cup_and_transforms(tmp_path, p):
    from pxr import Usd, UsdGeom, UsdPhysics

    cup = p["objects"]["container"]
    cup.update(shape="cup_proxy", wall_m=0.003, approximation="Synthetic legacy preview only")
    cup["provenance"].update(dimensions_m="provisional", wall_m="provisional")
    path = scene.build_usd(tmp_path / "workspace.usd", p)
    s = Usd.Stage.Open(path)
    assert s.GetDefaultPrim().GetPath() == "/Real2Sim"
    assert UsdGeom.GetStageMetersPerUnit(s) == 1
    cup = s.GetPrimAtPath("/Real2Sim/Workspace/container")
    assert len([x for x in cup.GetChildren() if x.GetName().startswith("Wall_")]) == 48
    assert not s.GetPrimAtPath("/Real2Sim/Robot")
    assert s.GetPrimAtPath("/Real2Sim/Workspace/table/Shape").HasAPI(UsdPhysics.CollisionAPI)
    assert json.loads(s.GetDefaultPrim().GetCustomDataByKey("profile_json")) == p
    with pytest.raises(FileExistsError):
        scene.build_usd(tmp_path / "workspace.usd", p)


def test_metric_rectification(p):
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:200, 100:200] = 255
    stats, _ = calibration.image_metrics(image, image, p["cameras"]["side"])
    assert stats["appearance_rgb_mae"] == 0
    assert stats["edge_chamfer_px"] == 0


def test_camera_calibration_synthetic(monkeypatch):
    import cv2

    cols, rows = 9, 6
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * 0.025
    k = np.array([[610.0, 0, 317.0], [0, 620.0, 242.0], [0, 0, 1.0]])
    corners = []
    for i in range(12):
        r = np.array([-0.25 + i * 0.04, 0.15 - i * 0.02, 0.02 * i])
        t = np.array([-0.1 + i * 0.008, -0.06, 0.5 + i * 0.025])
        uv, _ = cv2.projectPoints(obj, r, t, k, np.zeros(5))
        corners.append(uv.astype(np.float32))
    monkeypatch.setattr(cv2, "imread", lambda *a: np.zeros((480, 640), np.uint8))
    it = iter(corners)
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a: (True, next(it)))
    result = calibration.calibrate_intrinsics([str(i) for i in range(12)], cols, rows, 0.025)
    assert np.allclose(result["K"], k, atol=0.1)
    assert max(result["intrinsics_report"]["heldout_rmse_px"]) < 0.01


def test_duplicate_intrinsic_views_rejected(monkeypatch):
    import cv2

    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    k = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    uv, _ = cv2.projectPoints(obj, np.zeros(3), np.array([0.0, 0.0, 0.7]), k, np.zeros(5))
    monkeypatch.setattr(cv2, "imread", lambda *a: np.zeros((480, 640), np.uint8))
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a: (True, uv))
    with pytest.raises(ValueError, match="tilted"):
        calibration.calibrate_intrinsics([str(i) for i in range(12)], 9, 6, 0.025)
