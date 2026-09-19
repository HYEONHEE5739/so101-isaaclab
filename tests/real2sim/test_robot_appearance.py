from pathlib import Path
import pytest
from pxr import Usd, UsdShade, Sdf, Gf
from soarm101_lab.real2sim.scene import apply_robot_appearance
from soarm101_lab.real2sim.profile import load, validate


def test_local_color_override_restores_asset_and_preserves_motor(tmp_path):
    source = Usd.Stage.CreateNew(str(tmp_path / 'robot.usda'))
    for material, color in [('material_a_d_printed', (.1, 0, .2)), ('motor', (.1, .1, .1))]:
        shader = UsdShade.Shader.Define(source, '/Robot/Looks/' + material + '/Shader')
        shader.CreateInput('diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    source.GetRootLayer().Save()
    original = (tmp_path / 'robot.usda').read_bytes()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/envs/env_0/Robot').GetReferences().AddReference(str(tmp_path / 'robot.usda'), '/Robot')
    attr = stage.GetPrimAtPath('/World/envs/env_0/Robot/Looks/material_a_d_printed/Shader').GetAttribute('inputs:diffuse_color_constant')
    before = attr.Get()
    apply_robot_appearance(stage, {'appearance': {'robot_print_color': [.2, .05, .4]}})
    assert list(attr.Get()) == pytest.approx([.2, .05, .4])
    motor = stage.GetPrimAtPath('/World/envs/env_0/Robot/Looks/motor/Shader').GetAttribute('inputs:diffuse_color_constant')
    assert list(motor.Get()) == pytest.approx([.1, .1, .1])
    apply_robot_appearance(stage, {'appearance': {}})
    assert attr.Get() == before
    assert (tmp_path / 'robot.usda').read_bytes() == original


def test_color_requires_valid_value_and_provenance():
    p = load(Path(__file__).resolve().parents[2] / 'docs/real2sim/measurements.example.json')
    p['appearance']['robot_print_color'] = [.1, .2, .3]
    with pytest.raises(ValueError):
        validate(p)
    p['appearance']['provenance']['robot_print_color'] = 'numerically_fitted'
    validate(p)
    for color in [[float('nan'), 0, 0], [1.1, 0, 0], [.1, .2]]:
        p['appearance']['robot_print_color'] = color
        with pytest.raises(ValueError):
            validate(p)
