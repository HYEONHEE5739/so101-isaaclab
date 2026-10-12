"""Expand the minimal MJCF using the same published profile/task as Isaac."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .mjcf import numbers, pose_attrs, transform
from ..real2sim.workspaces.physics import compile_physics
from ..real2sim.workspaces.package import ROOT
from .mjcf import RGB_PROFILE, MATERIAL_RESPONSE


def extend_scene(scene, robot, workspace, urdf, width, height, physics_profile='isaac_matched'):
    p, task = workspace['profile'], workspace['task']
    world = scene.find('worldbody')
    visual = scene.find('visual')
    visual.find('quality').set('offsamples', '4')
    visual.find('quality').set('shadowsize', '2048')
    ET.SubElement(visual, 'headlight', ambient='.45 .45 .45', diffuse='.2 .2 .2', specular='0 0 0')
    key = world.find('light')
    key.set('dir', '-.3 -.4 -1'); key.set('ambient', '.25 .25 .25')
    key.set('diffuse', '.5 .5 .5'); key.set('specular', '.05 .05 .05'); key.set('castshadow', 'false')
    world_pose = transform(p['workspace']['T_world'])
    physics = compile_physics(p, task)['objects']
    dynamic = list(task['dynamic'])
    targets = list(task['targets'])
    included = ['table', task['pick'], task['place']]

    def primitive(parent, name, obj, collision=False):
        body = ET.SubElement(parent, 'body', name=name, **pose_attrs(obj['T_workspace']))
        dims = np.asarray(obj['dimensions_m'], float)
        shape = obj['shape']
        if shape == 'box': size = dims / 2
        elif shape == 'cylinder': size = [dims[0] / 2, dims[2] / 2]
        else: raise ValueError(f'Unsupported workspace primitive {name}: {shape}')
        ET.SubElement(body, 'geom', name=name + '_geom', type=shape, size=numbers(size),
                      rgba=numbers([*obj['appearance']['color'], 1 if obj.get('render_enabled', True) else 0]),
                      contype='1' if collision else '0', conaffinity='3' if collision else '0')
        return body

    # Existing minimal bodies must use the published physics roles too.
    for body in world.findall('body'):
        name = body.get('name')
        if name in physics:
            for geom in body.findall('geom'):
                geom.set('contype', '1' if physics[name]['collision'] else '0')
                geom.set('conaffinity', '3' if physics[name]['collision'] else '0')
        if name not in included: included.append(name)
    existing = {b.get('name') for b in world.findall('body')}
    # Rebuild each cup with its own dimensions, not a scaled copy of the selected cup.
    for name in targets:
        if name in existing: continue
        obj = p['objects'][task['targets'][name]]
        body = ET.SubElement(world, 'body', name=name, **pose_attrs(world_pose @ transform(obj['T_workspace'])))
        if not workspace.get('manifest', {}).get('builder'): _cup(body, name, obj)
        included.append(name)
    textures = {}
    for name in targets:
        body = world.find(f"body[@name='{name}']")
        if not workspace.get('manifest', {}).get('builder'):
            textures.update(_cup_visual(scene, body, name, p['objects'][task['targets'][name]], workspace))
    target_visuals = set(task['targets'].values())
    for name, obj in p['objects'].items():
        if name in existing or name in targets or name in target_visuals: continue
        spec = physics[name]
        if not obj.get('render_enabled', True) and not spec['collision']: continue
        mapped = {**obj, 'T_workspace': world_pose @ transform(obj['T_workspace'])}
        body = primitive(world, name, mapped, spec['collision'])
        if name in dynamic:
            ET.SubElement(body, 'freejoint', name='object_free_' + name)
            size = np.asarray(obj['dimensions_m'], float)
            mass = spec['mass_kg']
            ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass),
                          diaginertia=numbers(mass / 12 * (size @ size - size**2)))
        included.append(name)

    # The same local material override used by Isaac's printed robot parts.
    color = p.get('appearance', {}).get('robot_print_color')
    if color is not None:
        for link in ET.parse(urdf).getroot().findall('link'):
            for i, visual in enumerate(link.findall('visual')):
                mat = visual.find('material')
                if mat is not None and mat.get('name') == '3d_printed':
                    geom = robot.find(f".//geom[@name='{link.get('name')}_visual_{i}']")
                    if geom is not None: geom.set('rgba', numbers([*color, 1]))

    gripper = robot.find(".//body[@name='robot_gripper']")
    contacts_path = ROOT / 'assets/SO101/mujoco/usd_contact_pads.json'
    contacts = json.loads(contacts_path.read_text())
    asset_root = urdf.parent.parent
    for relative, digest in contacts['source_sha256'].items():
        if hashlib.sha256((asset_root/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('USD contact pad source changed; regenerate offline contact pad export: ' + relative)
    for pad in contacts['pads']:
        body = robot.find(f".//body[@name='robot_{pad['link']}']")
        contact = {'priority': '1', 'condim': '4', 'solref': '.004 1', 'solimp': '.95 .99 .001'} if physics_profile == 'isaac_matched' else {}
        ET.SubElement(body, 'geom', name=pad['link']+'_contact_pad', type='box',
                      size=numbers(np.asarray(pad['dimensions_m'])/2), **pose_attrs(pad['T_link']),
                      rgba=numbers(pad['rgba']), contype='2' if pad['collision'] else '0',
                      conaffinity='1' if pad['collision'] else '0',
                      friction=numbers([pad['dynamic_friction'], .005, .0001]),
                      **contact)
    robot.find(".//site[@name='tool0']").set('rgba', '0 0 0 0')
    mount = p.get('wrist_mount', {})
    for name, obj in mount.get('parts', {}).items():
        if obj.get('render_enabled', True):
            primitive(gripper, 'wrist_mount_' + name, obj)
            included.append('wrist_mount_' + name)

    cameras = {}
    for cam in list(world.findall('camera')): world.remove(cam)
    for name, camera in p['cameras'].items():
        parent = gripper if camera['parent'] == 'gripper' else world
        cameras[name] = add_camera(parent, name, camera, width, height)
    if workspace.get('root'):
        from ..workflow.presentation import framing
        pos, quat = framing(workspace)
        optical = np.eye(4)
        optical[:3, 3] = pos
        optical[:3, :3] = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
        optical[:3, :3] = optical[:3, :3] @ np.diag([1., -1., -1.])
        ET.SubElement(world, 'camera', name='overview', fovy='50', **pose_attrs(optical))
        cameras['overview'] = {'purpose': 'inspection only; same workspace framing as Isaac'}
    for name, response in MATERIAL_RESPONSE.items():
        if name in ('robot_print', 'cup'): continue
        body = world.find(f"body[@name='{name}']")
        if body is not None:
            for geom in body.findall('geom'):
                color = np.fromstring(geom.get('rgba', '1 1 1 1'), sep=' ')
                color[:3] *= response
                geom.set('rgba', numbers(color))
    for geom in robot.findall('.//geom'):
        color = np.fromstring(geom.get('rgba', '1 1 1 1'), sep=' ')
        if color[2] > color[0]*1.5 and color[3] > 0:
            color[:3] *= MATERIAL_RESPONSE['robot_print']; geom.set('rgba', numbers(color))
    return {'dynamic_objects': dynamic, 'target_objects': targets, 'scene_subset': included + ['SO101'],
            'cameras': cameras, 'physics_roles': physics, 'texture_sha256': textures,
            'contact_pads': contacts,
            'render_profile': RGB_PROFILE,
            'render_settings': {'msaa_samples': 4, 'output_encoding': 'sRGB transfer of classic renderer RGB',
                                'texture_vflip': True, 'lighting': 'key + ambient fill; image-matched approximation',
                                'material_response': MATERIAL_RESPONSE,
                                'fit_capture': 'session_20260919_121858/capture_000001 side/wrist',
                                'validation_captures': ['capture_000007', 'capture_000009']}}


def _cup_visual(scene, body, name, obj, workspace):
    """Same 96-segment tapered visual / repeated front-print UV recipe as Isaac."""
    assets = scene.find('asset')
    if assets is None: assets = ET.SubElement(scene, 'asset')
    radius, bottom = obj['dimensions_m'][0] / 2, obj.get('bottom_diameter_m', obj['dimensions_m'][0]) / 2
    height, wall = obj['dimensions_m'][2], obj['wall_m']
    rings = [(bottom, -height/2), (radius, height/2), (radius-wall, height/2), (bottom-wall, -height/2+wall)]
    for geom in body.findall('geom'):
        if geom.get('type') == 'box':
            geom.set('rgba', '0 0 0 0'); geom.set('group', '3')
        else:
            color = np.fromstring(geom.get('rgba'), sep=' ')
            color[:3] *= MATERIAL_RESPONSE['cup']; geom.set('rgba', numbers(color))
    textures = {}
    for band, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        tex = obj.get('print_texture') if band == 0 else None
        rect = tex['uv_rect'] if tex else [0, 0, 1, 1]
        points, faces, uv = [], [], []
        count = 96
        for i in range(count):
            start = len(points)
            for ring, index in [(a, i), (a, i+1), (b, i+1), (b, i)]:
                r, z = rings[ring]; angle = 2*np.pi*index/count
                points.append([r*np.cos(angle), r*np.sin(angle), z])
            faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
            u0 = (i % (count//2))/(count//2); u1 = u0 + 2/count
            uv.extend([[rect[0]+u*(rect[2]-rect[0]), rect[1]+v*(rect[3]-rect[1])]
                      for u, v in [(u0, 0), (u1, 0), (u1, 1), (u0, 1)]])
        mesh_name = f'{name}_surface_{band}'
        ET.SubElement(assets, 'mesh', name=mesh_name, vertex=numbers(points), inertia='shell',
                      face=' '.join(str(n) for f in faces for n in f), texcoord=numbers(uv))
        attrs = {'rgba': numbers([*(np.array(obj['appearance']['color'])*MATERIAL_RESPONSE['cup']), 1])}
        if tex:
            texture_path = Path(tex['path'])
            if workspace.get('root'):
                provenance = workspace['root'] / 'provenance.json'
                if provenance.is_file():
                    texture_path = workspace['root'] / json.loads(provenance.read_text())['asset_map'][tex['path']]
            digest = hashlib.sha256(texture_path.read_bytes()).hexdigest()
            if digest != tex['image_sha256']: raise ValueError('Workspace print texture hash mismatch')
            textures[str(texture_path)] = digest
            ET.SubElement(assets, 'texture', name=name+'_print', type='2d', file=str(texture_path), vflip='true')
            ET.SubElement(assets, 'material', name=name+'_print', texture=name+'_print', texuniform='false',
                          specular='0', shininess='0', rgba='.70 .70 .70 1')
            attrs = {'material': name+'_print'}
        ET.SubElement(body, 'geom', name=mesh_name, type='mesh', mesh=mesh_name,
                      contype='0', conaffinity='0', group='2', **attrs)
    return textures


def add_camera(parent, name, camera, width, height):
    if np.any(np.asarray(camera['distortion']) != 0):
        raise ValueError('MuJoCo camera requires undistorted workspace input')
    k = np.asarray(camera['K'], float).copy()
    k[0] *= width / camera['resolution'][0]; k[1] *= height / camera['resolution'][1]
    if not np.allclose(k[2], [0,0,1]) or abs(k[0,1]) > 1e-9:
        raise ValueError('Camera requires zero-skew pinhole intrinsics')
    pose = transform(camera['T_parent_camera']).copy()
    pose[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])
    ET.SubElement(parent, 'camera', name=name, **pose_attrs(pose), resolution=f'{width} {height}',
                  sensorsize=f'{width} {height}', focalpixel=numbers([k[0, 0], k[1, 1]]),
                  principalpixel=numbers([width/2-k[0,2], k[1,2]-height/2]))
    return {**copy.deepcopy(camera), 'resolution': [width, height], 'K': k.tolist()}


def _cup(body, name, obj):
    h, top, bottom, wall = (float(obj['dimensions_m'][2]), float(obj['dimensions_m'][0]) / 2,
                            float(obj.get('bottom_diameter_m', obj['dimensions_m'][0])) / 2, float(obj['wall_m']))
    if min(h, top, bottom, wall) <= 0 or wall >= min(bottom, top, h / 2):
        raise ValueError('Invalid target dimensions: ' + name)
    color = numbers([*obj['appearance']['color'], 1])
    ET.SubElement(body, 'geom', name=name + '_bottom', type='cylinder',
                  pos=numbers([0, 0, -h / 2 + wall / 2]), size=numbers([bottom, wall / 2]), rgba=color)
    for ring in range(4):
        radius = bottom + (top - bottom) * (ring + .5) / 4 - wall / 2
        for segment in range(32):
            angle = 2 * np.pi * segment / 32
            ET.SubElement(body, 'geom', name=f'{name}_wall_{ring}_{segment}', type='box',
                          pos=numbers([radius * np.cos(angle), radius * np.sin(angle), -h / 2 + (ring + .5) * h / 4]),
                          quat=numbers([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
                          size=numbers([wall / 2, radius * np.tan(np.pi / 32) * 1.02, h / 8 + .0001]), rgba=color)


def apply_builder_visuals(scene, robot, workspace):
    """Overlay immutable exported visuals onto the existing articulated physics scene.

    Reference robot snapshots are never spawned as a second robot. Visual parts
    follow live links, dynamic objects and receptacles through local transforms.
    """
    from ..real2sim.workspaces.builder import frame, mounted, world_pose
    import yaml
    root = workspace['root']
    canonical = yaml.safe_load((root / workspace['manifest']['builder']['scene']).read_text())
    source_path = root / workspace['manifest']['builder']['mjcf']
    exported = ET.parse(source_path).getroot()
    assets = scene.find('asset')
    if assets is None: assets = ET.SubElement(scene, 'asset')
    rename = {}
    for asset in exported.find('asset'):
        rename[asset.get('name')] = 'builder_' + asset.get('name')
    for original in exported.find('asset'):
        asset = copy.deepcopy(original)
        asset.set('name', rename[asset.get('name')])
        for field in ('texture', 'material', 'mesh'):
            if asset.get(field) in rename: asset.set(field, rename[asset.get(field)])
        if asset.get('file'):
            path = (source_path.parent / asset.get('file')).resolve()
            if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Unsafe Builder mesh/texture')
            asset.set('file', str(path))
        assets.append(asset)
    world = scene.find('worldbody'); task = workspace['task']
    objects = {o['id']: o for o in canonical['objects']}
    source_bodies = {b.get('name'): b for b in exported.find('worldbody').findall('body')}
    # Builder owns object collision geometry; runtime owns joints, mass and contact material.
    physics = compile_physics(workspace['profile'], task)['objects']
    for body in world.findall('body'):
        for geom in list(body.findall('geom')): body.remove(geom)
        name = body.get('name')
        if name in task['dynamic']:
            # Explicit conservative inertia from canonical local bounds and supplied mass.
            inertial = body.find('inertial')
            size = np.asarray(workspace['profile']['objects'][name]['dimensions_m'])
            mass = physics[name]['mass_kg']
            inertial.set('mass', str(mass))
            inertial.set('diaginertia', numbers(mass / 12 * (size @ size - size**2)))
    replaced_links = {o['link'] for o in canonical['objects'] if o['role'] == 'reference_robot'}
    for body in robot.findall('.//body'):
        if body.get('name', '').removeprefix('robot_') not in replaced_links: continue
        for geom in body.findall('geom'):
            if '_visual_' in geom.get('name', ''): body.remove(geom)
    added = []
    for identity, obj in objects.items():
        if obj['role'] == 'reference_robot':
            body = robot.find(f".//body[@name='robot_{obj['link']}']")
            local = transform(obj['link_local_pose'])
        elif mounted(canonical, obj):
            body = robot.find(".//body[@name='robot_gripper']")
            local = np.linalg.inv(frame(canonical, 'robot_link_gripper')) @ world_pose(canonical, obj)
        elif obj.get('parent_object'):
            parent = obj['parent_object']; body = world.find(f"body[@name='{parent}']")
            local = np.linalg.inv(world_pose(canonical, objects[parent])) @ world_pose(canonical, obj)
        else:
            owner = 'table' if identity == workspace['profile']['table'].get('source_id','table') else identity
            body = world.find(f"body[@name='{owner}']")
            local = np.eye(4)
        if body is None: raise ValueError('Builder visual has no runtime owner: '+identity)
        for original in source_bodies[identity].findall('geom'):
            visual_geom = '_visual_' in original.get('name', '')
            physical_id = 'table' if identity == workspace['profile']['table'].get('source_id','table') else identity
            collision = (not visual_geom and physical_id in physics and physics[physical_id].get('collision', False))
            if not visual_geom and not collision: continue
            geom = copy.deepcopy(original)
            t = np.eye(4); t[:3, 3] = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            q = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
            t[:3, :3] = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
            geom.attrib.update(pose_attrs(local @ t)); geom.set('name', 'builder_'+geom.get('name'))
            for field in ('mesh', 'material'):
                if geom.get(field): geom.set(field, rename[geom.get(field)])
            geom.set('contype', '1' if collision else '0'); geom.set('conaffinity', '3' if collision else '0')
            geom.set('mass', '0'); geom.set('group', '3' if collision else '2')
            if collision:
                geom.set('rgba', '0 0 0 0')
                geom.set('friction', numbers([physics[physical_id]['material']['dynamic_friction'], .005, .0001]))
            body.append(geom)
        added.append(identity)
    return {'builder_visuals': added, 'builder_source': copy.deepcopy(workspace['manifest']['builder']),
            'render_profile': RGB_PROFILE,
            'render_settings': {'output_encoding': 'sRGB transfer of classic renderer RGB',
                                'lighting': 'existing runtime key + ambient; not pixel-equivalent to Isaac',
                                'material_response': 'canonical Builder colors; historical response coefficients not applied'},
            'physics_roles': compile_physics(workspace['profile'], task)['objects']}
