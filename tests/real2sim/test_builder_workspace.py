"""Builder import contracts independent of the Builder application/Isaac runtime."""
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from soarm101_lab.real2sim.workspaces import builder, package
from soarm101_lab.real2sim.workspaces.reset import WorkspaceRandomizer
from soarm101_lab.so101_joint_mapping import DEFAULT_MAPPER
from soarm101_lab.workflow.tasks import catalog


def val(x): return {'value': x, 'provenance': 'estimated', 'evidence': []}


def make_export(tmp_path):
    root = tmp_path/'export'; root.mkdir()
    def obj(name, role, size, xyz):
        t = np.eye(4); t[:3, 3] = xyz
        return dict(id=name, role=role, frame='world', pose=val(t.tolist()), dimensions=val(size),
                    visual={'type': 'box'}, collision={'type': 'box'}, color=val([.4,.3,.2]))
    objects = [obj('table','support_surface',[.6,.6,.03],[0,0,-.015]),
               obj('marker','workspace',[.15,.15,.0005],[0,0,.00025]),
               obj('cube_red','manipulable_object',[.024]*3,[0,0,.0125]),
               obj('cup_a','receptacle',[.07,.07,.065],[.12,0,.0325])]
    objects[-1]['collision'] = {'type':'container','round':True,'bottom_diameter_m':.052,'wall':val(.0015)}
    for name, x in [('cube_green', -.04), ('cube_blue', .04)]:
        objects.append(obj(name,'manipulable_object',[.024]*3,[x,0,.0125]))
    cup=copy.deepcopy(objects[3]);cup['id']='cup_b';cup['pose']['value'][0][3]=-.12;objects.append(cup)
    mount=obj('camera_case','fixture',[.04,.03,.02],[0,.04,.02]);mount['frame']='robot_link_gripper'
    ink=obj('cup_print','fixture',[.02,.001,.01],[.12,-.035,.03]);ink['parent_object']='cup_a'
    objects.extend([mount,ink])
    cameras=[]
    for name in ('side','wrist'):
        t=np.eye(4); t[:3,3]=[.2,-.3,.4] if name=='side' else [0,.06,-.01]
        cameras.append(dict(id=name,frame='world' if name=='side' else 'robot_link_gripper',pose=val(t.tolist()),
                             K=val([[500,0,320],[0,500,240],[0,0,1]]),resolution=[640,480],distortion=val([0.]*5)))
    (root/'assets').mkdir();shutil.copy(package.ROOT/'assets/SO101/urdf/so101_isaaclab.urdf',root/'assets/robot.urdf')
    scene={'schema':'real2sim.scene/1','objects':objects,'cameras':cameras,
           'frames':{'world':{},'robot_base':{'parent':'world','transform':val(np.eye(4).tolist())},
                     'robot_link_gripper':{'parent':'robot_base','transform':val(np.eye(4).tolist())}},
           'robot_model':{'format':'urdf','path':'assets/robot.urdf','joint_names':DEFAULT_MAPPER.metadata()['joint_order'],
                          'mapping':DEFAULT_MAPPER.metadata()}}
    (root/'scene.yaml').write_text(yaml.safe_dump(scene));(root/'SCENE.md').write_text('Synthetic test scene')
    xml=ET.Element('mujoco');ET.SubElement(xml,'asset');world=ET.SubElement(xml,'worldbody')
    usd=['#usda 1.0','(defaultPrim = "Scene")','def Xform "Scene" {']
    for o in objects:
        b=ET.SubElement(world,'body',name=o['id'])
        for kind in ('visual','collision'):ET.SubElement(b,'geom',name=o['id']+'_'+kind+'_0',type='box',size='.01 .01 .01')
        usd += [f'def Material "material_{o["id"]}" {{}}',f'def Xform "{o["id"]}" {{',
                'def Cube "visual_0" {}','def Cube "collision_0" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}','}']
    usd.append('}');(root/'scene.usd').write_text('\n'.join(usd));ET.ElementTree(xml).write(root/'scene.xml')
    manifest={'schema':'real2sim.package/1','units':'m','canonical':'scene.yaml','status':'UNVALIDATED','validation_scope':'VISUAL_ONLY',
              'files':{p.relative_to(root).as_posix():package.sha(p) for p in root.rglob('*') if p.is_file()}}
    (root/'scene_manifest.yaml').write_text(yaml.safe_dump(manifest))
    return root


@pytest.fixture
def exported(tmp_path):
    return make_export(tmp_path)


def test_import_portable_provenance_and_task(exported,tmp_path):
    dest=builder.import_scene(exported,registry=tmp_path/'registry');w=package.load(dest)
    assert builder.import_scene(exported,registry=tmp_path/'registry')==dest
    assert w['manifest']['builder']['status']=='UNVALIDATED'
    assert package.metadata(w)['builder_scene']['physics_status']=='PROVISIONAL_TASK_PHYSICS'
    assert catalog(w)[0]['task_id']=='cube_red_to_cup_a'
    assert w['task']['physics']['provenance']=='provisional'
    states=[WorkspaceRandomizer(w,seed).sample() for seed in range(10)]
    assert len({tuple(x['cube_positions']['cube_red']) for x in states})==10
    shutil.rmtree(exported)
    assert package.load(dest)['task']==w['task']  # No dependency on original export or Builder checkout.
    assert '../builder/scene.usd' in (dest/'scene/cup_a.usda').read_text()
    assert 'PhysicsRigidBodyAPI' in (dest/'scene/cube_red.usda').read_text()


def test_tamper_rejected(exported,tmp_path):
    (exported/'scene.xml').write_text('<mujoco/>')
    with pytest.raises(ValueError,match='Modified Builder export'):builder.import_scene(exported,registry=tmp_path)


def test_paths_and_frame_cycles_rejected(exported):
    with pytest.raises(ValueError,match='unsafe'):builder._file(exported,'../outside')
    s=builder.inspect_export(exported)[2];s['frames']['robot_base']['parent']='robot_link_gripper'
    with pytest.raises(ValueError,match='frame graph'):builder.frame(s,'robot_base')


def test_mapping_mismatch_rejected(exported):
    s=yaml.safe_load((exported/'scene.yaml').read_text());s['robot_model']['mapping']['arm_sign'][0]=-1
    (exported/'scene.yaml').write_text(yaml.safe_dump(s));m=yaml.safe_load((exported/'scene_manifest.yaml').read_text())
    m['files']['scene.yaml']=package.sha(exported/'scene.yaml');(exported/'scene_manifest.yaml').write_text(yaml.safe_dump(m))
    with pytest.raises(ValueError,match='mapping differs'):builder.inspect_export(exported)


def test_generic_task_roles(exported):
    s=builder.inspect_export(exported)[2]
    for o in s['objects']:
        if o['id']=='cube_red':o['id']='block'
        if o['id']=='cup_a':o['id']='bin'
    p,t=builder.profile_and_task(s)
    assert catalog({'profile':p,'task':t})[0]['task_id']=='block_to_bin'


def test_mujoco_live_owners_and_physics_without_isaac(exported,tmp_path):
    import os,subprocess,sys
    path=builder.import_scene(exported,registry=tmp_path/'registry')
    code = """
import sys,numpy as np
from soarm101_lab.simulation.mujoco_backend import MujocoBackend
e=MujocoBackend(sys.argv[2],sys.argv[1],render_backend='none',full_workspace=True)
for geom,body in [('builder_camera_case_visual_0','robot_gripper'),('builder_cup_print_visual_0','cup_a')]:
 g=e.mj.mj_name2id(e.model,e.mj.mjtObj.mjOBJ_GEOM,geom)
 b=e.mj.mj_name2id(e.model,e.mj.mjtObj.mjOBJ_BODY,body)
 assert e.model.geom_bodyid[g]==b
 assert e.model.geom_contype[g]==0
assert e.model.opt.enableflags & e.mj.mjtEnableBit.mjENBL_MULTICCD
assert e.model.nu==6
assert len(e.object_indices)==3
for _ in range(20):e.step(e.home)
assert np.isfinite(e.data.qpos).all()
assert not set(sys.modules)&{'isaaclab','omni','pxr'}
e.close()
"""
    env={**os.environ,'PYTHONPATH':str(package.ROOT/'source/soarm101_lab')}
    result=subprocess.run([sys.executable,'-B','-c',code,str(path),str(tmp_path/'runtime')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture
def fruit_export(tmp_path):
    root=make_export(tmp_path)
    scene=yaml.safe_load((root/'scene.yaml').read_text())
    rename={'table':'bench','cube_red':'apple','cube_green':'banana','cup_a':'basket'}
    removed={'cube_blue','cup_b'}
    scene['objects']=[o for o in scene['objects'] if o['id'] not in removed]
    for o in scene['objects']:
        o['id']=rename.get(o['id'],o['id'])
        if o.get('parent_object'):o['parent_object']=rename.get(o['parent_object'],o['parent_object'])
        if o['id']=='apple':
            o['visual']={'type':'sphere'};o['collision']={'type':'sphere'};o['dimensions']=val([.03]*3)
            o['physics']={'mass_kg':{'value':.1,'provenance':'measured'},'friction':{'value':.8,'provenance':'measured'}}
        if o['id']=='banana':
            o['visual']={'type':'compound'};o['collision']={'type':'compound'};o['dimensions']=val([.05,.02,.02])
        if o['id']=='basket':
            o['collision']={'type':'container','round':False,'wall':val(.004)};o['dimensions']=val([.16,.14,.1])
            o['pose']['value'][2][3]=.05
    scene['cameras']=[scene['cameras'][0]];scene['cameras'][0]['id']='front'
    scene['workflow']={'schema':'so101.scene-workflow/1','support_surface':'bench','observation_cameras':['front'],
        'physics':{'provenance':'provisional'},'object_physics':{'banana':{'mass_kg':.06,'provenance':'provisional'}},
        'tasks':[{'id':'fruit_sort','pick':'apple','place':'basket','instruction':'Place the apple in the basket.',
                  'success':{'relation':'inside_region_and_settled','bounds_m':[[-.08,-.07,-.047],[.08,.07,.1]],'max_speed_m_s':.03}}]}
    (root/'scene.yaml').write_text(yaml.safe_dump(scene))
    xml=ET.parse(root/'scene.xml');world=xml.getroot().find('worldbody')
    for body in list(world):
        old=body.get('name')
        if old in removed:world.remove(body);continue
        name=rename.get(old,old);body.set('name',name)
        for g in body.findall('geom'):g.set('name',g.get('name').replace(old+'_',name+'_'))
        if name=='apple':
            for g in body.findall('geom'):g.set('type','sphere');g.set('size','.015')
        if name=='banana':
            original=body.find('geom');body.remove(original)
            for g in list(body.findall('geom')):body.remove(g)
            for kind in ('visual','collision'):
                for i in range(3): ET.SubElement(body,'geom',name=f'banana_{kind}_{i}',type='box',size='.009 .01 .01',pos=f'{(i-1)*.016} 0 0')
        if name=='basket':
            for g in body.findall('geom'):g.set('size','.08 .07 .002');g.set('pos','0 0 -.048')
    xml.write(root/'scene.xml')
    usd=(root/'scene.usd').read_text()
    for old,new in rename.items():usd=usd.replace(old,new)
    (root/'scene.usd').write_text(usd)
    manifest=yaml.safe_load((root/'scene_manifest.yaml').read_text())
    manifest['files']={p.relative_to(root).as_posix():package.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='scene_manifest.yaml'}
    (root/'scene_manifest.yaml').write_text(yaml.safe_dump(manifest))
    return root


def test_metadata_driven_one_camera_fruit_scene(fruit_export,tmp_path):
    root=builder.import_scene(fruit_export,registry=tmp_path/'registry');w=package.load(root)
    assert list(w['profile']['cameras'])==['front']
    assert w['profile']['table']['source_id']=='bench'
    assert w['task']['dynamic']==['apple','banana']
    assert catalog(w)[0]['task_id']=='fruit_sort'
    assert w['task']['object_physics']['apple']['mass_kg']==.1
    import subprocess,sys,os
    code='''
import sys,numpy as np
from soarm101_lab.simulation.mujoco_backend import MujocoBackend
with MujocoBackend(sys.argv[2],sys.argv[1]) as e:
 assert e.metadata['primary_camera']=='front'
 assert e.model.ncam==2 # front + inspection overview
 assert e.model.geom('builder_apple_collision_0').type==e.mj.mjtGeom.mjGEOM_SPHERE
 assert sum('builder_banana_collision_' in e.mj.mj_id2name(e.model,e.mj.mjtObj.mjOBJ_GEOM,i) for i in range(e.model.ngeom))==3
 assert abs(e.model.body('apple').mass-.1)<1e-9
 assert abs(e.model.body('banana').mass-.06)<1e-9
 for _ in range(20):e.step(e.home)
 pose=e.get_object_pose('basket');pose[2,3]=.019
 e.set_object_pose('apple',pose)
 for _ in range(150):e.step(e.home)
 assert e.is_success(),e.success_details()
 assert not set(sys.modules)&{'isaaclab','pxr','omni'}
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(root),str(tmp_path/'sim')],env={**os.environ,'PYTHONPATH':str(package.ROOT/'source/soarm101_lab')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_missing_workflow_success_rejected(fruit_export):
    scene=builder.inspect_export(fruit_export)[2]
    scene['workflow']['tasks'][0]['success'].pop('bounds_m')
    with pytest.raises(ValueError,match='bounds_m'):builder.profile_and_task(scene)


def test_exported_dynamic_body_has_one_runtime_rigid_owner(exported,tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdPhysics
    text=(exported/'scene.usd').read_text().replace('def Xform "cube_red" {', 'def Xform "cube_red" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {')
    (exported/'scene.usd').write_text(text)
    m=yaml.safe_load((exported/'scene_manifest.yaml').read_text());m['files']['scene.usd']=package.sha(exported/'scene.usd');(exported/'scene_manifest.yaml').write_text(yaml.safe_dump(m))
    root=builder.import_scene(exported,registry=tmp_path/'registry')
    stage=Usd.Stage.Open(str(root/'scene/cube_red.usda'))
    assert stage.GetPrimAtPath('/Real2Sim').HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath('/Real2Sim/cube_red').HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath('/Real2Sim/cube_red/collision_0').HasAPI(UsdPhysics.CollisionAPI)


def test_workflow_evidence_survives_runtime_conversion(fruit_export,tmp_path):
    scene=builder.inspect_export(fruit_export)[2]
    evidence={k:{'source':'estimated','status':'UNVALIDATED','evidence':['fixture']} for k in ('tasks','physics','reset','success')}
    scene['workflow']['provenance']=evidence
    p,t=builder.profile_and_task(scene)
    assert t['workflow_provenance']==evidence
    from soarm101_lab.real2sim.workspaces.physics import compile_physics
    physics=compile_physics(p,t)['objects']
    assert physics['apple']['field_provenance']['mass_kg']['provenance']=='measured'
    assert physics['apple']['workflow_provenance']['status']=='UNVALIDATED'


def test_usd_simulation_zero_frames_and_original_model_preserved(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    root=tmp_path/'workspace';shutil.copytree(package.ROOT/'assets/SO101',root/'assets/SO101')
    original=package.sha(root/'assets/SO101/urdf/so101_isaaclab.urdf')
    original_usd=package.sha(root/'assets/SO101/usd/so101_isaaclab.usd')
    before=Usd.Stage.Open(str(root/'assets/SO101/usd/so101_isaaclab.usd'))
    joint=UsdPhysics.RevoluteJoint(before.GetPrimAtPath('/so101_calibrated/joints/elbow_flex'))
    q=joint.GetLocalRot0Attr().Get();rotation=Rotation.from_quat([*q.GetImaginary(),q.GetReal()])
    node={'value':[0.,0.,.07,0.,0.,0.],'provenance':'optimized','evidence':['Synthetic frame test']}
    relative=builder.calibrate_simulation_usd(root,node)
    derived=Usd.Stage.Open(str(root/relative))
    joint=UsdPhysics.RevoluteJoint(derived.GetPrimAtPath('/so101_calibrated/joints/elbow_flex'))
    q=joint.GetLocalRot0Attr().Get();after=Rotation.from_quat([*q.GetImaginary(),q.GetReal()])
    np.testing.assert_allclose(after.as_matrix(),(rotation*Rotation.from_rotvec([0,0,.07])).as_matrix(),atol=1e-7)
    assert package.sha(root/'assets/SO101/urdf/so101_isaaclab.urdf')==original
    assert package.sha(root/'assets/SO101/usd/so101_isaaclab.usd')==original_usd
    assert relative=='assets/SO101/usd/so101_sim.usda'
    assert (root/'assets/SO101/urdf/so101_sim.urdf').is_file()
    old={j.get('name'):j for j in ET.parse(root/'assets/SO101/urdf/so101_isaaclab.urdf').getroot().findall('joint')}
    new={j.get('name'):j for j in ET.parse(root/'assets/SO101/urdf/so101_sim.urdf').getroot().findall('joint')}
    assert old['elbow_flex'].find('limit').attrib==new['elbow_flex'].find('limit').attrib
    assert old['elbow_flex'].find('origin').attrib!=new['elbow_flex'].find('origin').attrib


def test_new_execution_usd_limits_and_unchanged_wrist_and_default_prim(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdPhysics
    from soarm101_lab.so101_robot_spec import limit_usd,ROBOT_URDF,joint_limits
    source=package.ROOT/'assets/SO101/usd/so101_isaaclab.usd'
    original_hash=package.sha(source)
    result=limit_usd(source,ROBOT_URDF,tmp_path/'limits.usda')
    stage=Usd.Stage.Open(str(result));original=Usd.Stage.Open(str(source))
    assert stage.GetDefaultPrim() and stage.GetDefaultPrim().GetPath()==original.GetDefaultPrim().GetPath()
    for prim in stage.Traverse():
        n=prim.GetName()
        if n not in joint_limits() or not prim.IsA(UsdPhysics.RevoluteJoint):continue
        joint=UsdPhysics.RevoluteJoint(prim)
        values=[joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]
        np.testing.assert_allclose(np.radians(values),joint_limits()[n],atol=1e-5,rtol=0)
        if n in ('wrist_roll','gripper'):
            old=UsdPhysics.RevoluteJoint(original.GetPrimAtPath(prim.GetPath()))
            assert values==[old.GetLowerLimitAttr().Get(),old.GetUpperLimitAttr().Get()]
    assert package.sha(source)==original_hash
    with pytest.raises(ValueError,match='new file'):limit_usd(source,ROBOT_URDF,result)
