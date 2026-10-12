import copy
import json
from pathlib import Path
import numpy as np
import pytest
from pxr import Usd,UsdPhysics
from soarm101_lab.real2sim.profile import load as profile_load
from soarm101_lab.real2sim.workspaces import package
from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer


@pytest.fixture(scope='module')
def published(tmp_path_factory):
    from test_builder_workspace import make_export
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    root = tmp_path_factory.mktemp('workspace')
    source = make_export(root)
    path = import_scene(source, 'workspace_test', registry=root/'registry')
    w = package.load(path)
    return root, path/'profile.json', w['task'], path


def test_import_idempotent_and_deterministic(published):
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    root,source,task,path=published
    before=source.read_bytes()
    assert import_scene(root/'export','workspace_test',task=task,registry=root/'registry')==path
    second=import_scene(root/'export','workspace_test',task=task,registry=root/'other')
    assert package.load(path)['manifest']==package.load(second)['manifest']
    assert source.read_bytes()==before
    changed=copy.deepcopy(task);changed['place']='cup_b'
    with pytest.raises(FileExistsError):import_scene(root/'export','workspace_test',task=changed,registry=root/'registry')


def test_missing_and_modified_asset_fail(published):
    _,_,_,path=published;asset=path/'scene/static.usda';data=asset.read_bytes()
    try:
        asset.write_bytes(data+b'\n# modified')
        with pytest.raises(ValueError,match='asset'):package.load(path)
        asset.unlink()
        with pytest.raises(ValueError,match='asset'):package.load(path)
    finally:asset.write_bytes(data)


def test_schema_identity_and_task_validation(published):
    root,source,task,path=published;profile=json.loads(source.read_text())
    from soarm101_lab.real2sim.workspaces.builder import import_scene
    with pytest.raises(ValueError):import_scene(root/'export','../bad',registry=root)
    with pytest.raises(ValueError):import_scene(root/'export','good',version=0,registry=root)
    for patch in [{'pick':'unknown'},{'place':'unknown'},{'dynamic':['cube_red','cube_red']}]:
        bad=copy.deepcopy(task);bad.update(patch)
        with pytest.raises(ValueError):package.validate_task(bad,profile)
    bad=copy.deepcopy(task);bad['reset']['margin_m']=1
    with pytest.raises(ValueError):package.validate_task(bad,profile)


def test_dynamic_not_duplicated_and_target_collision_composed(published):
    _,_,task,path=published;s=Usd.Stage.Open(str(path/'scene/static.usda'))
    for name in task['dynamic']:assert not s.GetPrimAtPath('/Real2Sim/Workspace/'+name)
    cup=Usd.Stage.Open(str(path/'scene/cup_a.usda'))
    assert cup.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    collisions=[p for p in cup.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                and p.GetAttribute('physics:collisionEnabled').Get() is not False]
    assert len(collisions)==1  # Synthetic export has one collider; real cup topology belongs to Builder.
    assert not cup.GetPrimAtPath('/Real2Sim/CollisionTop')


def test_randomization_local_bounds_separation_and_repeatability(published):
    w=package.load(published[3]);a=WorkspaceRandomizer(w);b=WorkspaceRandomizer(w)
    for _ in range(100):
        state=a.sample();assert state==b.sample()
        positions=list(state['cube_positions'].values())
        for pos in positions:
            assert max(abs(pos[0]),abs(pos[1]))<=.075-.012-.01
            assert pos[2]==pytest.approx(.0005+.012+.003)
        for i in range(3):
            for j in range(i):assert max(abs(np.array(positions[i][:2])-positions[j][:2]))>.024+.005


def test_demo_metadata_rejects_wrong_workspace(published,tmp_path):
    import h5py
    from soarm101_lab.real2sim.workspaces.demo import verify_dataset,coordinate_contract
    path=published[3];w=package.load(path);meta=package.metadata(w);demo=tmp_path/'demo.hdf5'
    with h5py.File(demo,'w') as h:h.create_group('data').attrs['env_args']=json.dumps({'workspace':meta})
    assert verify_dataset(demo,path)[0]['manifest']['workspace_id']=='workspace_test'
    with h5py.File(demo,'a') as h:
        meta['version']=999;h['data'].attrs['env_args']=json.dumps({'workspace':meta})
    with pytest.raises(ValueError,match='differs'):verify_dataset(demo,path)
    assert coordinate_contract(w)['mapping']==w['manifest']['mapper']


def test_legacy_camera_disabled_only_in_workspace_session():
    from pxr import UsdGeom
    from soarm101_lab.real2sim.workspaces.environment import deactivate_legacy_wrist_camera
    asset = Usd.Stage.Open(str(package.ROOT / 'assets/SO101/usd/so101_isaaclab.usd'))
    before = asset.GetRootLayer().ExportToString()
    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim('/World/Robot', 'Xform')
    robot.GetReferences().AddReference(asset.GetRootLayer().identifier)
    legacy = stage.GetPrimAtPath('/World/Robot/gripper/Camera_WristView')
    assert legacy and legacy.IsActive()
    wrist = UsdGeom.Camera.Define(stage, '/World/Robot/gripper/Camera_WorkspaceWrist').GetPrim()
    side = UsdGeom.Camera.Define(stage, '/World/Camera_Side').GetPrim()
    overview = UsdGeom.Camera.Define(stage, '/World/Camera_Overview').GetPrim()
    root_before = stage.GetRootLayer().ExportToString()
    deactivate_legacy_wrist_camera(robot)
    deactivate_legacy_wrist_camera(robot)  # Idempotent.
    assert not legacy.IsActive()
    assert all(p.IsActive() for p in (wrist, side, overview))
    assert stage.GetRootLayer().ExportToString() == root_before
    assert asset.GetRootLayer().ExportToString() == before
    assert asset.GetPrimAtPath('/so101_calibrated/gripper/Camera_WristView').IsActive()
    deactivate_legacy_wrist_camera(stage.DefinePrim('/World/NoCamera', 'Xform'))
