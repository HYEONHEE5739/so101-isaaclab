"""Session-only recovery of missing imported finger collision geometry."""
from pathlib import Path
import hashlib
import numpy as np
import pytest


def test_missing_finger_meshes_restored_without_changing_asset():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    from soarm101_lab.real2sim.workspaces.environment import restore_missing_finger_collisions
    from soarm101_lab.real2sim.workspaces.package import ROOT
    import xml.etree.ElementTree as ET
    import trimesh
    urdf = ROOT/'assets/SO101/urdf/so101_isaaclab.urdf'
    original = urdf.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
    for name in ('gripper', 'jaw'):
        UsdGeom.Xform.Define(stage, '/Robot/'+name+'/collisions')
    restore_missing_finger_collisions(robot, urdf)
    first = stage.GetRootLayer().ExportToString()
    restore_missing_finger_collisions(robot, urdf)
    assert stage.GetRootLayer().ExportToString() == first
    for name in ('gripper', 'jaw'):
        nodes = ET.fromstring(original).find(f"link[@name='{name}']").findall('collision')
        for index, node in enumerate(nodes):
            mesh = UsdGeom.Mesh.Get(stage, f'/Robot/{name}/URDFCollision_{index}')
            assert mesh and UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'convexHull'
            source = trimesh.load_mesh(urdf.parent/node.find('geometry/mesh').get('filename'), process=False)
            scale = np.fromstring(node.find('geometry/mesh').get('scale', '1 1 1'), sep=' ')
            np.testing.assert_allclose(mesh.GetPointsAttr().Get(), source.vertices*scale, atol=1e-8)
            assert mesh.GetVisibilityAttr().Get() == 'invisible'
    assert hashlib.sha256(urdf.read_bytes()).digest() == hashlib.sha256(original).digest()


def test_existing_finger_collision_mesh_preserved():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from soarm101_lab.real2sim.workspaces.environment import restore_missing_finger_collisions
    from soarm101_lab.real2sim.workspaces.package import ROOT
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
    for name in ('gripper', 'jaw'):
        UsdGeom.Mesh.Define(stage, '/Robot/'+name+'/collisions/authored')
    before = stage.GetRootLayer().ExportToString()
    restore_missing_finger_collisions(robot, ROOT/'assets/SO101/urdf/so101_isaaclab.urdf')
    assert stage.GetRootLayer().ExportToString() == before


def test_instanced_finger_mesh_is_not_mistaken_for_missing(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from soarm101_lab.real2sim.workspaces.environment import restore_missing_finger_collisions
    from soarm101_lab.real2sim.workspaces.package import ROOT
    asset=Usd.Stage.CreateNew(str(tmp_path/'shared.usda'))
    root=UsdGeom.Xform.Define(asset,'/Shared').GetPrim();asset.SetDefaultPrim(root)
    UsdGeom.Mesh.Define(asset,'/Shared/mesh');asset.GetRootLayer().Save()
    stage=Usd.Stage.CreateInMemory();robot=UsdGeom.Xform.Define(stage,'/Robot').GetPrim()
    for name in ('gripper','jaw'):
        p=UsdGeom.Xform.Define(stage,f'/Robot/{name}/collisions').GetPrim()
        p.GetReferences().AddReference(str(tmp_path/'shared.usda'));p.SetInstanceable(True)
    before=stage.GetRootLayer().ExportToString()
    restore_missing_finger_collisions(robot,ROOT/'assets/SO101/urdf/so101_isaaclab.urdf')
    assert stage.GetRootLayer().ExportToString()==before
