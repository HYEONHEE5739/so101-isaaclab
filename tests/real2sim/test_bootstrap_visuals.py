import copy
import numpy as np
import pytest
from soarm101_lab.real2sim import profile, scene
from test_bootstrap import HYPOTHESIS


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
