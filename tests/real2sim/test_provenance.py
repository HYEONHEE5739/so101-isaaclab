"""Bootstrap/schema regressions. No hardware, renderer, or network connection."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from soarm101_lab.real2sim import calibration, profile, scene, storage

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs/real2sim/measurements.example.json"


def bootstrap():
    return profile.load(TEMPLATE)


def legacy():
    p = bootstrap()
    p["schema"] = "so101.real2sim/1"
    p["geometry_source"] = "user_measured"
    p.pop("relations")
    p["table"].update(dimensions_m=[0.6, 0.6, 0.05], T_workspace=np.eye(4).tolist())
    p["objects"] = {"container": {
        "shape": "cup", "dimensions_m": [0.06, 0.06, 0.08], "wall_m": 0.003,
        "T_workspace": np.eye(4).tolist(), "appearance": {"color": [0.7] * 3, "roughness": 0.6},
    }}
    return p


def test_supplied_measurements_and_computed_values():
    p = bootstrap()
    assert p["robot_asset"]["source"] == "model_asset"
    assert p["objects"]["cube"]["dimensions_m"] == [0.024] * 3
    assert p["workspace"]["size_m"] == [0.150] * 2
    for name in ("cup_a", "cup_b"):
        cup = p["objects"][name]
        for key, value in (("top_outer_diameter_m", 0.070), ("wall_thickness_m", 0.002), ("height_m", 0.065)):
            assert cup[key] == value
            assert cup["provenance"][key] == "user_measured"
        for key in ("bottom_outer_diameter_m", "bottom_inner_diameter_m", "bottom_thickness_m", "taper_profile", "T_workspace"):
            assert cup[key] is None
            assert cup["provenance"][key] == "unmeasured"
        assert "top_inner_diameter_m" not in cup
    assert p["relations"]["cup_spacing"]["center_to_center_distance_m"] == 0.090
    before = copy.deepcopy(p)
    derived = profile.derived_measurements(p)
    assert derived["cup_a.top_inner_diameter_m"]["value"] == pytest.approx(0.066)
    assert derived["cup_spacing.top_edge_gap_m"]["value"] == pytest.approx(0.020)
    assert all(v["source"] == "derived" for v in derived.values())
    assert p == before


def test_unknown_state_is_not_promoted_by_saving(tmp_path):
    p = bootstrap()
    path = storage.revision(tmp_path, p)
    saved = profile.load(path)
    assert saved == p
    assert "geometry_source" not in saved
    for name in ("robot_base", "workspace"):
        assert saved[name]["provenance"]["T_world"] == "provisional"
        assert not saved[name]["calibrated"]
    for camera in saved["cameras"].values():
        assert all(v == "provisional" for v in camera["provenance"].values())
        assert not camera["intrinsics_calibrated"]
        assert not camera["extrinsics_calibrated"]
    assert saved["table"]["dimensions_m"] is None
    assert saved["table"]["T_workspace"] is None
    assert all(o["T_workspace"] is None for o in saved["objects"].values())
    assert saved["appearance"]["provenance"]["light_intensity"] == "provisional"


def test_blanket_label_and_false_calibration_are_rejected(tmp_path):
    p = bootstrap()
    p["geometry_source"] = "user_measured"
    with pytest.raises(ValueError, match="Whole-profile"):
        storage.revision(tmp_path, p)
    assert not (tmp_path / "revisions").exists()
    p = bootstrap()
    p["cameras"]["side"]["intrinsics_calibrated"] = True
    with pytest.raises(ValueError, match="provisional"):
        profile.validate(p)
    p = bootstrap()
    p["objects"]["cup_a"]["provenance"]["bottom_outer_diameter_m"] = "user_measured"
    with pytest.raises(ValueError, match="null/unmeasured"):
        profile.validate(p)


def test_legacy_migration_preserves_file_and_distrusts_blanket_label(tmp_path):
    old = legacy()
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    new = profile.load(path)
    assert path.read_bytes() == original
    assert new["schema"] == profile.SCHEMA
    assert "geometry_source" not in new
    assert new["table"]["provenance"]["dimensions_m"] == "provisional"
    assert new["objects"]["container"]["shape"] == "cup_proxy"
    assert new["objects"]["container"]["dimensions_m"] == old["objects"]["container"]["dimensions_m"]
    assert not new["cameras"]["side"]["intrinsics_calibrated"]
    with pytest.raises(ValueError, match="excluded"):
        calibration.world_point(new, {"frame": "object:container", "_state": {}, "point_m": [0, 0, 0]})


def test_partial_cup_and_unknown_table_cannot_be_rendered_or_fit():
    p = bootstrap()
    with pytest.raises(ValueError, match="excluded"):
        calibration.world_point(p, {"frame": "object:cup_a", "_state": {}, "point_m": [0, 0, 0]})
    p["objects"]["cup_a"]["T_workspace"] = np.eye(4).tolist()
    p["objects"]["cup_a"]["provenance"]["T_workspace"] = "provisional"
    p["objects"]["cup_a"]["render_enabled"] = True
    with pytest.raises(ValueError, match="Partial cup"):
        profile.validate(p)
    p = bootstrap()
    p["table"].update(render_enabled=True, T_workspace=np.eye(4).tolist())
    p["table"]["provenance"]["T_workspace"] = "provisional"
    with pytest.raises(ValueError, match="Unknown dimensions"):
        profile.validate(p)


def test_derived_values_cannot_be_added_as_independent_measurements():
    p = bootstrap()
    p["objects"]["cup_a"]["top_inner_diameter_m"] = 0.066
    with pytest.raises(ValueError, match="derived"):
        profile.validate(p)
    p = bootstrap()
    p["relations"]["cup_spacing"]["top_edge_gap_m"] = 0.020
    with pytest.raises(ValueError, match="derived"):
        profile.validate(p)


def test_runtime_render_toggle_requires_reconnect():
    from soarm101_lab.real2sim.runtime import Runtime

    r = Runtime.__new__(Runtime)
    r.running = False
    r.env = object()
    r.profile = bootstrap()
    changed = copy.deepcopy(r.profile)
    cube = changed["objects"]["cube"]
    cube.update(T_workspace=np.eye(4).tolist(), render_enabled=True)
    cube["provenance"]["T_workspace"] = "provisional"
    with pytest.raises(ValueError, match="reconnect"):
        r.apply(changed)
    assert r.profile["objects"]["cube"]["T_workspace"] is None


def test_existing_rerender_command_with_bootstrap_and_mock_sensor(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from soarm101_lab.real2sim.runtime import Runtime

    p = bootstrap()
    saved_profile = storage.revision(tmp_path, p)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    images = {name: image for name in ("real_side", "sim_side", "real_wrist", "sim_wrist")}
    joints = {name: 0.0 for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    capture = storage.CaptureStore(tmp_path / "captures").save(images, {
        "hardware_identity": "synthetic", "quality": {"eligible_for_calibration": True},
        "physical_follower_measured_joint_state": joints,
    })
    r = Runtime.__new__(Runtime)
    r.running = False
    r.args = SimpleNamespace(workspace=str(tmp_path))
    r.profile = p
    r.hardware_identity = "synthetic"
    r.apply = Mock()
    r.publish = Mock()
    r.set_measured_pose = Mock()
    r.snapshot = Mock(return_value=({"sim_side": image, "sim_wrist": image}, None, None, {}, None, None))
    result = r.command({"command": "render_dataset", "payload": {
        "profile": str(saved_profile), "captures": [str(capture)],
    }})
    r.set_measured_pose.assert_called_once_with(joints)
    assert result["poses"] == 1
    summary = json.loads((Path(result["directory"]) / "summary.json").read_text())
    assert summary[0]["profile_hash"] == profile.digest(p)
    assert summary[0]["views"]["side"]["appearance_rgb_mae"] == 0


def test_bootstrap_usd_and_explicit_cube_preview(tmp_path):
    from pxr import Usd

    p = bootstrap()
    path = scene.build_usd(tmp_path / "bootstrap.usda", p)
    stage = Usd.Stage.Open(path)
    assert stage.GetPrimAtPath("/Real2Sim/Workspace")
    assert not stage.GetPrimAtPath("/Real2Sim/Workspace/table")
    assert not stage.GetPrimAtPath("/Real2Sim/Workspace/cup_a")
    assert not stage.GetPrimAtPath("/Real2Sim/Workspace/cube")
    assert json.loads(stage.GetDefaultPrim().GetCustomDataByKey("profile_json")) == p
    cube = p["objects"]["cube"]
    cube.update(T_workspace=np.eye(4).tolist(), render_enabled=True)
    cube["provenance"]["T_workspace"] = "provisional"
    path = scene.build_usd(tmp_path / "cube.usda", p)
    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath("/Real2Sim/Workspace/cube/Shape")
    assert np.allclose(prim.GetAttribute("xformOp:scale").Get(), [0.024] * 3)
    assert p["objects"]["cube"]["provenance"]["T_workspace"] == "provisional"


def test_ui_is_api_free_and_saves_honest_revision(tmp_path):
    # Fresh interpreter proves UI does not transitively import the retained agent module.
    code = '''
import os, sys, json, urllib.request
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
os.environ.pop("OPENAI_API_KEY", None)
def no_network(*a, **k):
    raise AssertionError("UI attempted an HTTP request")
urllib.request.urlopen = no_network
from PyQt6.QtWidgets import QApplication, QPushButton
from soarm101_lab.real2sim.ui import Window
app = QApplication([])
w = Window(sys.argv[2])
w.template()
w.save_profile()
p = w.profile()
assert "geometry_source" not in p
assert p["objects"]["cup_a"]["bottom_outer_diameter_m"] is None
assert "model" not in w.fields and not hasattr(w, "measured")
assert "soarm101_lab.real2sim.agent" not in sys.modules
assert not any("GPT" in b.text() or "OpenAI" in b.text() for b in w.findChildren(QPushButton))
assert all(hasattr(w, n) for n in ("connect_hw", "stop", "annotate", "metrics", "optimize", "rerender"))
assert len(w.views) == 4
w.close()
'''
    subprocess.run([sys.executable, "-B", "-c", code, json.dumps(sys.path), str(tmp_path)], check=True)
