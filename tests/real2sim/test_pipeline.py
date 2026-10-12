import copy
import json
from pathlib import Path
import numpy as np
import pytest
from soarm101_lab.real2sim import profile, storage, scene

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def p():
    p = profile.load(ROOT / "tests/real2sim/fixtures/partial_profile.json")
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


def test_ipc_stale_session_and_exclusive_owner(tmp_path):
    b = storage.Bus(tmp_path, True)
    b.publish(state="STOPPED")
    with pytest.raises(RuntimeError):
        storage.Bus(tmp_path, True)
    client = storage.Bus(tmp_path)
    id = client.send("capture_snapshot")
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
