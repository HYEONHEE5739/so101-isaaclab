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
    root=tmp_path_factory.mktemp('workspace');p=profile_load(package.ROOT/'docs/real2sim/measurements.example.json')
    def box(size,xyz):
        t=np.eye(4);t[:3,3]=xyz
        return {'shape':'box','dimensions_m':size,'T_workspace':t.tolist(),'render_enabled':True,
                'provenance':{'dimensions_m':'provisional','T_workspace':'provisional'},
                'appearance':{'color':[.3,.3,.3],'roughness':.6,'provenance':{'color':'provisional','roughness':'provisional'}}}
    p['objects']={n:box([.024]*3,[0,0,.013]) for n in ['cube_red','cube_green','cube_blue']}
    p['objects']['marker']=box([.15,.15,.0005],[0,0,.00025])
    for n,x in [('cup_a',-.05),('cup_b',.05)]:
        p['objects'][n]=box([.07,.07,.065],[x,.14,.0325]);p['objects'][n]['render_enabled']=False
        c=box([.07,.07,.065],[x,.14,.0325]);c.update(shape='cup_proxy',wall_m=.002,approximation='Synthetic test cup')
        c['provenance']['wall_m']='provisional'
        p['objects'][n+'_visual']=c
    p['table']=box([.6,.6,.04],[0,0,-.02]);p['relations']={};p['revision']={'status':'ACCEPTED'}
    source=root/'revision_test/profile.json';source.parent.mkdir();source.write_text(json.dumps(p))
    task=package.default_task(p);path=package.publish(source,'workspace_test',task,registry=root/'registry')
    return root,source,task,path


def test_publish_idempotent_and_deterministic(published):
    root,source,task,path=published
    before=source.read_bytes()
    assert package.publish(source,'workspace_test',task,registry=root/'registry')==path
    second=package.publish(source,'workspace_test',task,registry=root/'other')
    assert package.load(path)['manifest']==package.load(second)['manifest']
    assert (path/'profile.json').read_bytes()==before==source.read_bytes()
    changed=copy.deepcopy(task);changed['place']='cup_b'
    with pytest.raises(FileExistsError):package.publish(source,'workspace_test',changed,registry=root/'registry')


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
    with pytest.raises(ValueError):package.publish(source,'../bad',task,registry=root)
    with pytest.raises(ValueError):package.publish(source,'good',task,version=0,registry=root)
    for patch in [{'pick':'unknown'},{'place':'unknown'},{'dynamic':['cube_red','cube_red']}]:
        bad=copy.deepcopy(task);bad.update(patch)
        with pytest.raises(ValueError):package.validate_task(bad,profile)
    bad=copy.deepcopy(task);bad['reset']['margin_m']=1
    with pytest.raises(ValueError):package.validate_task(bad,profile)


def test_dynamic_not_duplicated_and_target_collision_open(published):
    _,_,task,path=published;s=Usd.Stage.Open(str(path/'scene/static.usda'))
    for name in task['dynamic']:assert not s.GetPrimAtPath('/Real2Sim/Workspace/'+name)
    cup=Usd.Stage.Open(str(path/'scene/cup_a.usda'))
    assert cup.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    collisions=[p for p in cup.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(collisions)==129
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
