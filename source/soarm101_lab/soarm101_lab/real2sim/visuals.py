"""Profile-driven tapered visual cup. Measured constraints remain separate."""
import math
from pxr import UsdGeom, UsdShade, Sdf, Gf


def tapered_cup(stage, path, obj, material):
    radius = obj['dimensions_m'][0] / 2
    bottom = obj['bottom_diameter_m'] / 2
    height = obj['dimensions_m'][2]
    wall = obj['wall_m']
    count = 96
    # Four rings: outer bottom/top and inner top/bottom. Bottom thickness is provisional wall.
    rings = [(bottom, -height/2), (radius, height/2),
             (radius-wall, height/2), (bottom-wall, -height/2+wall)]
    pts = [(r*math.cos(2*math.pi*i/count), r*math.sin(2*math.pi*i/count), z)
           for r,z in rings for i in range(count)]
    for band, (a,b) in enumerate([(0,1),(1,2),(2,3)]):
        indices, uvs = [], []
        tex = obj.get('print_texture') if band == 0 else None
        rect = tex['uv_rect'] if tex else [0,0,1,1]
        for i in range(count):
            indices.extend([a*count+i,a*count+(i+1)%count,b*count+(i+1)%count,b*count+i])
            # Repeat observed front print twice around circumference; explicitly provisional.
            u0=(i % (count//2))/(count//2);u1=u0+2/count
            uvs.extend([(rect[0]+u*(rect[2]-rect[0]),rect[1]+v*(rect[3]-rect[1]))
                        for u,v in [(u0,0),(u1,0),(u1,1),(u0,1)]])
        mesh=UsdGeom.Mesh.Define(stage,path+f'/Surface_{band}')
        mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr([4]*count)
        mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if tex:
            uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying')
            uv.Set(uvs)
            mat=UsdShade.Material.Define(stage,path+'/PrintMaterial')
            shader=UsdShade.Shader.Define(stage,path+'/PrintMaterial/Surface');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(obj['appearance']['roughness'])
            image=UsdShade.Shader.Define(stage,path+'/PrintMaterial/Image');image.CreateIdAttr('UsdUVTexture')
            image.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(tex['path'])
            image.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            reader=UsdShade.Shader.Define(stage,path+'/PrintMaterial/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            image.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(),'rgb')
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    base=UsdGeom.Cylinder.Define(stage,path+'/Bottom')
    base.CreateRadiusAttr(bottom);base.CreateHeightAttr(wall);base.CreateAxisAttr('Z')
    base.AddTranslateOp().Set(Gf.Vec3d(0,0,-height/2+wall/2))
    UsdShade.MaterialBindingAPI.Apply(base.GetPrim()).Bind(material)
