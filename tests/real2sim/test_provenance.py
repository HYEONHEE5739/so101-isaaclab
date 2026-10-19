"""Profile provenance, scene authoring and UI isolation; no real hardware."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from soarm101_lab.real2sim import profile, scene, storage

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tests/real2sim/fixtures/partial_profile.json"


def partial_profile():
    return profile.load(TEMPLATE)


def legacy():
    p = partial_profile()
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
    p = partial_profile()
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
    p = partial_profile()
    path = tmp_path / "profile.json"
    storage.atomic(path, profile.validate(p))
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
    p = partial_profile()
    p["geometry_source"] = "user_measured"
    with pytest.raises(ValueError, match="Whole-profile"):
        profile.validate(p)
    assert not (tmp_path / "revisions").exists()
    p = partial_profile()
    p["cameras"]["side"]["intrinsics_calibrated"] = True
    with pytest.raises(ValueError, match="provisional"):
        profile.validate(p)
    p = partial_profile()
    p["objects"]["cup_a"]["provenance"]["bottom_outer_diameter_m"] = "user_measured"
    with pytest.raises(ValueError, match="null/unmeasured"):
        profile.validate(p)


def test_retired_profile_schema_rejected_without_rewriting(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy()))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="so101.real2sim/2"):
        profile.load(path)
    assert path.read_bytes() == original


def test_partial_cup_and_unknown_table_cannot_be_rendered():
    p = partial_profile()
    p["objects"]["cup_a"]["T_workspace"] = np.eye(4).tolist()
    p["objects"]["cup_a"]["provenance"]["T_workspace"] = "provisional"
    p["objects"]["cup_a"]["render_enabled"] = True
    with pytest.raises(ValueError, match="Partial cup"):
        profile.validate(p)
    p = partial_profile()
    p["table"].update(render_enabled=True, T_workspace=np.eye(4).tolist())
    p["table"]["provenance"]["T_workspace"] = "provisional"
    with pytest.raises(ValueError, match="Unknown dimensions"):
        profile.validate(p)


def test_derived_values_cannot_be_added_as_independent_measurements():
    p = partial_profile()
    p["objects"]["cup_a"]["top_inner_diameter_m"] = 0.066
    with pytest.raises(ValueError, match="derived"):
        profile.validate(p)
    p = partial_profile()
    p["relations"]["cup_spacing"]["top_edge_gap_m"] = 0.020
    with pytest.raises(ValueError, match="derived"):
        profile.validate(p)


def test_runtime_render_toggle_requires_reconnect():
    from soarm101_lab.real2sim.runtime import Runtime

    r = Runtime.__new__(Runtime)
    r.running = False
    r.env = object()
    r.profile = partial_profile()
    changed = copy.deepcopy(r.profile)
    cube = changed["objects"]["cube"]
    cube.update(T_workspace=np.eye(4).tolist(), render_enabled=True)
    cube["provenance"]["T_workspace"] = "provisional"
    with pytest.raises(ValueError, match="reconnect"):
        r.apply(changed)
    assert r.profile["objects"]["cube"]["T_workspace"] is None


def test_partial_profile_usd_and_explicit_cube_preview(tmp_path):
    from pxr import Usd

    p = partial_profile()
    path = scene.build_usd(tmp_path / "partial.usda", p)
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


def test_ui_is_api_free_and_loads_profile(tmp_path):
    # Fresh interpreter proves UI does not transitively import any agent module.
    code = '''
import os, sys, json, urllib.request
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
os.environ.pop("OPENAI_API_KEY", None)
def no_network(*a, **k):
    raise AssertionError("UI attempted an HTTP request")
urllib.request.urlopen = no_network
from PyQt6.QtWidgets import QApplication, QPushButton
from soarm101_lab.workflow import ui
from soarm101_lab.workflow.artifacts import Registry
from soarm101_lab.real2sim.ui import Window
ui.Registry = lambda: Registry(Path(sys.argv[2]) / 'workflow_registry')
app = QApplication([])
w = Window(sys.argv[2])
from soarm101_lab.real2sim.profile import load
assert not w.fields["profile"].text()
w.selected(Path(sys.argv[3]))
p = load(w.profile_path())
assert "geometry_source" not in p
assert p["objects"]["cup_a"]["bottom_outer_diameter_m"] is None
assert "model" not in w.fields and not hasattr(w, "measured")
assert "soarm101_lab.real2sim.agent" not in sys.modules
assert not any("GPT" in b.text() or "OpenAI" in b.text() for b in w.findChildren(QPushButton))
assert all(hasattr(w, n) for n in ("connect_hw", "stop", "capture_snapshot", "profile_path"))
assert len(w.views) == 4
w.close()
'''
    subprocess.run([sys.executable, "-B", "-c", code, json.dumps(sys.path), str(tmp_path), str(TEMPLATE)], check=True)


HYPOTHESIS = ROOT / "tests/real2sim/fixtures/scene_profile.json"


def test_reviewed_geometry_and_provenance():
    p = profile.load(HYPOTHESIS)
    assert p['robot_asset']['source'] == 'model_asset'
    assert p['workspace']['size_m'] == [.15, .15]
    for name in ['cube_red', 'cube_blue', 'cube_green']:
        o = p['objects'][name]
        assert o['dimensions_m'] == [.024]*3
        assert o['provenance']['dimensions_m'] == 'user_measured'
        assert np.all(np.abs(np.asarray(o['T_workspace'])[:2, 3]) + .012 < .075)
        assert o['provenance']['T_workspace'] == 'provisional'
    for name in ['cup_a', 'cup_b']:
        o = p['objects'][name]
        assert o['top_outer_diameter_m'] == .07 and o['height_m'] == .065
        assert o['bottom_outer_diameter_m'] is None
        assert o['provenance']['bottom_outer_diameter_m'] == 'unmeasured'
    a,b = [np.asarray(p['objects'][n]['T_workspace'])[:3,3] for n in ['cup_a','cup_b']]
    assert np.linalg.norm(a-b) == pytest.approx(.09)
    for c in p['cameras'].values():
        assert not c['intrinsics_calibrated'] and not c['extrinsics_calibrated']
    assert profile.derived_measurements(p)['cup_a.top_inner_diameter_m']['value'] == pytest.approx(.066)


def test_deterministic_scene(tmp_path):
    from pxr import Usd
    p=profile.load(HYPOTHESIS)
    a=scene.build_usd(tmp_path/'a.usda',p);b=scene.build_usd(tmp_path/'b.usda',p)
    assert Path(a).read_text() == Path(b).read_text()
    stage=Usd.Stage.Open(a)
    for name in ['table','marker','cube_red','cube_blue','cube_green','cup_a_visual','cup_b_visual','background_roller']:
        assert stage.GetPrimAtPath('/Real2Sim/Workspace/'+name)
    assert not stage.GetPrimAtPath('/Real2Sim/Workspace/cup_a')


def test_taper_mesh_and_validation(tmp_path):
    from pxr import Usd, UsdGeom
    p=profile.load(HYPOTHESIS)
    cup=p['objects']['cup_a_visual']
    cup['bottom_diameter_m']=.052
    cup['provenance']['bottom_diameter_m']='provisional'
    path=scene.build_usd(tmp_path/'cup.usda',p)
    stage=Usd.Stage.Open(path)
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Real2Sim/Workspace/cup_a_visual/Surface_0'))
    pts=np.asarray(mesh.GetPointsAttr().Get())
    assert np.linalg.norm(pts[0,:2]) == pytest.approx(.026)
    assert np.linalg.norm(pts[96,:2]) == pytest.approx(.035)
    assert pts[:,2].max()-pts[:,2].min() == pytest.approx(.065)
    assert not mesh.GetPrim().HasAPI(__import__('pxr.UsdPhysics',fromlist=['CollisionAPI']).CollisionAPI)
    cup['provenance']['bottom_diameter_m']='user_measured'
    with pytest.raises(ValueError,match='provisional'):
        profile.validate(p)


def test_mount_validation_and_visual_only(tmp_path):
    from pxr import Usd, UsdPhysics
    p=profile.load(HYPOTHESIS)
    part=copy.deepcopy(p['objects']['cube_red'])
    part['provenance']['dimensions_m']='provisional'
    p['wrist_mount']={'parent':'gripper','source':'provisional','parts':{'body':part}}
    profile.validate(p)
    child=copy.deepcopy(p);child.pop('wrist_mount');child['objects']={'body':part};child['relations']={}
    path=scene.build_usd(tmp_path/'mount.usda',child,visual_only=True)
    stage=Usd.Stage.Open(path)
    assert not stage.GetPrimAtPath('/Real2Sim/Workspace/body/Shape').HasAPI(UsdPhysics.CollisionAPI)
    p['wrist_mount']['parent']='world'
    with pytest.raises(ValueError,match='gripper'):
        profile.validate(p)
