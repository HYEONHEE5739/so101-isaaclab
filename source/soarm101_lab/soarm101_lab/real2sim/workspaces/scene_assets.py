"""Deterministic package representations, generated from profile + task."""
import copy
import math
import numpy as np
from ..scene import build_usd


def build_assets(root, profile, task, asset_map):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    p = copy.deepcopy(profile)
    p.pop('wrist_mount',None);p['relations']={}
    removed = set(task['dynamic']) | set(task['targets']) | set(task['targets'].values())
    static = copy.deepcopy(p);static['objects']={k:v for k,v in p['objects'].items() if k not in removed}
    build_usd(root/'scene/static.usda',static)
    for name,visual in task['targets'].items():
        q=copy.deepcopy(p);q['table']['render_enabled']=False;q['workspace']['T_world']=np.eye(4).tolist()
        obj=copy.deepcopy(p['objects'][visual]);obj['T_workspace']=np.eye(4).tolist();q['objects']={name:obj}
        path=root/f'scene/{name}.usda';build_usd(path,q,visual_only=True)
        stage=Usd.Stage.Open(str(path));prim=stage.GetDefaultPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        # Open cup collision: hidden segmented tapered walls plus a bottom, never a top cap.
        h=obj['dimensions_m'][2];rt=obj['dimensions_m'][0]/2;rb=obj.get('bottom_diameter_m',2*rt)/2;wall=obj['wall_m']
        base=UsdGeom.Cylinder.Define(stage,'/Real2Sim/CollisionBottom');base.CreateRadiusAttr(rb);base.CreateHeightAttr(wall)
        base.AddTranslateOp().Set(Gf.Vec3d(0,0,-h/2+wall/2))
        shapes=[base]
        for j in range(4):
            z=-h/2+(j+.5)*h/4;radius=rb+(rt-rb)*(j+.5)/4-wall/2
            for i in range(32):
                angle=2*math.pi*i/32;c=UsdGeom.Cube.Define(stage,f'/Real2Sim/Wall_{j}_{i}');c.CreateSizeAttr(1.)
                c.AddTranslateOp().Set(Gf.Vec3d(radius*math.cos(angle),radius*math.sin(angle),z))
                c.AddRotateZOp().Set(math.degrees(angle));c.AddScaleOp().Set(Gf.Vec3f(wall,2*radius*math.tan(math.pi/32)*1.02,h/4+.0002));shapes.append(c)
        for shape in shapes:
            shape.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        stage.GetRootLayer().Save()
    if profile.get('wrist_mount'):
        q=copy.deepcopy(p);q['table']['render_enabled']=False;q['workspace']['T_world']=np.eye(4).tolist()
        q['objects']=copy.deepcopy(profile['wrist_mount']['parts'])
        build_usd(root/'scene/wrist_mount.usda',q,visual_only=True)

    # Author relative texture references only after profile validation/build.
    from pxr import Sdf
    for path in (root/'scene').glob('*.usda'):
        stage=Usd.Stage.Open(str(path))
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value=attr.Get()
                if isinstance(value,Sdf.AssetPath) and value.path in asset_map:
                    attr.Set(Sdf.AssetPath('../'+asset_map[value.path]))
        stage.GetRootLayer().Save()
