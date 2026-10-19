"""URDF/STL -> in-memory MJCF. No generated scene files or workspace edits."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..so101_joint_mapping import JOINTS
from ..so101_robot_spec import SO101_POSITION_GAINS, SO101_POSITION_FORCE_LIMIT
from ..real2sim.workspaces.package import ROOT, default_task, load, metadata
from ..real2sim.profile import validate as validate_profile

# Configured initial gains, based on the existing Isaac gains, NOT measured motor parameters.
KP = tuple(SO101_POSITION_GAINS[n][0] for n in JOINTS)
KD = tuple(SO101_POSITION_GAINS[n][1] for n in JOINTS)


RGB_PROFILE = 'so101.mujoco-rgb/2'
RGB_WIDTH = 640
RGB_HEIGHT = 480

# Classic renderer response fit at capture_000001 (side/wrist), keeping profile
# geometry and collision untouched. These are renderer adapters, not measured albedo.
MATERIAL_RESPONSE = {
    'table': (.73, .73, .73), 'marker': (.84, .75, .72),
    'cube_red': (.96, .80, .75), 'cube_green': (.70, 1., .90),
    'cube_blue': (.65, .80, 1.), 'background_roller': (.1, .1, .1),
    'robot_print': (1., .80, 1.), 'cup': (.72, .72, .72),
}


def numbers(values):
    return ' '.join(f'{float(v):.12g}' for v in np.asarray(values).reshape(-1))


def transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Expected finite rigid 4x4 transform')
    return t


def pose_attrs(value):
    t = transform(value)
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return {'pos': numbers(t[:3, 3]), 'quat': numbers(q[[3, 0, 1, 2]])}


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def load_inputs(workspace=None):
    if isinstance(workspace, dict):
        # Unpublished Real/Sim calibration profiles use the same scene compiler.
        from ..real2sim.profile import digest
        w = copy.deepcopy(workspace)
        validate_profile(w['profile'])
        w.setdefault('task', default_task(w['profile']))
        urdf = Path(w.pop('urdf', Path(w['root'])/'assets/SO101/urdf/so101_isaaclab.urdf' if w.get('manifest') else ROOT/'assets/SO101/urdf/so101_isaaclab.urdf'))
        return w, urdf, metadata(w) if w.get('manifest',{}).get('content_sha256') else {'source': 'real2sim_profile', 'profile_sha256': digest(w['profile'])}
    if workspace:
        w = load(workspace)
        return w, w['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf', metadata(w)
    raise ValueError('Select an explicit Workspace or scene profile; no bootstrap scene is created')


def xml_bytes(node):
    ET.indent(node)
    return ET.tostring(node, encoding='utf-8', xml_declaration=True)


def robot_xml(urdf, base_pose, physics_profile='isaac_matched', joint_offsets=None):
    from ..so101_robot_spec import simulation_joint_offsets
    offsets = dict(zip(JOINTS, simulation_joint_offsets(joint_offsets, JOINTS)))
    root = ET.parse(urdf).getroot()
    links = {link.get('name'): link for link in root.findall('link')}
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    bases = set(links) - children
    if len(bases) != 1:
        raise ValueError('URDF must have a single fixed robot base')
    movable = {j.get('name'): j for j in joints if j.get('type') != 'fixed'}
    if set(movable) != set(JOINTS):
        raise ValueError('URDF must have the six named SO101 revolute joints')
    result = ET.Element('mujocoinclude')
    assets = ET.SubElement(result, 'asset')
    meshes = {}
    source_hashes = {str(urdf): hashlib.sha256(urdf.read_bytes()).hexdigest()}
    convex_path=ROOT/'assets/SO101/mujoco/finger_convex_parts.json'
    convex_parts={}
    if physics_profile=='isaac_matched':
        exported=json.loads(convex_path.read_text())
        if exported.get('schema')!='so101.finger-convex-parts/1':
            raise ValueError('Unknown SO101 finger collision export')
        convex_parts={(entry['link'],entry['collision_index']):entry for entry in exported['meshes']}
        source_hashes[str(convex_path)]=hashlib.sha256(convex_path.read_bytes()).hexdigest()
    materials = {m.get('name'): m.find('color').get('rgba') for m in root.findall('material')}
    world = ET.SubElement(result, 'worldbody')

    def add_link(parent, name, frame, joint=None):
        if joint is not None and joint.get('type') != 'fixed':
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * offsets[joint.get('name')]).as_matrix()
            frame = frame @ motion
        if name == 'tool0':
            ET.SubElement(parent, 'site', name='tool0', size='.002', rgba='0 1 0 1', **pose_attrs(frame))
            return  # The URDF's massless fixed tool frame is a site, not a dynamic body.
        body = ET.SubElement(parent, 'body', name='robot_' + name, **pose_attrs(frame))
        if joint is not None and joint.get('type') != 'fixed':
            if joint.get('type') != 'revolute':
                raise ValueError('Only SO101 revolute joints supported')
            limit = joint.find('limit')
            ET.SubElement(body, 'joint', name=joint.get('name'), type='hinge',
                          axis=joint.find('axis').get('xyz'),
                          range=f"{limit.get('lower')} {limit.get('upper')}", damping='0', armature='.001')
        inertial = links[name].find('inertial')
        tensor = inertial.find('inertia').attrib
        inertia = np.array([[float(tensor['ixx']), float(tensor['ixy']), float(tensor['ixz'])],
                            [float(tensor['ixy']), float(tensor['iyy']), float(tensor['iyz'])],
                            [float(tensor['ixz']), float(tensor['iyz']), float(tensor['izz'])]])
        inertial_pose = origin(inertial.find('origin'))
        inertia = inertial_pose[:3, :3] @ inertia @ inertial_pose[:3, :3].T
        ET.SubElement(body, 'inertial', pos=numbers(inertial_pose[:3, 3]),
                      mass=inertial.find('mass').get('value'),
                      fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                           inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        for kind in ('visual', 'collision'):
            for index, geom in enumerate(links[name].findall(kind)):
                mesh = geom.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Expected SO101 mesh geometry')
                path = (urdf.parent / mesh.get('filename')).resolve()
                scale = mesh.get('scale', '1 1 1')
                key = (str(path), scale)
                if key not in meshes:
                    meshes[key] = f'mesh_{len(meshes)}'
                    ET.SubElement(assets, 'mesh', name=meshes[key], file=str(path), scale=scale)
                    source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                attributes = {'name': f'{name}_{kind}_{index}', 'type': 'mesh', 'mesh': meshes[key],
                              **pose_attrs(origin(geom.find('origin')))}
                if kind=='collision' and name in ('gripper','jaw') and physics_profile=='isaac_matched':
                    entry=convex_parts.get((name,index))
                    if not entry or entry['source_sha256']!=source_hashes[str(path)]:
                        raise ValueError('Finger collision STL changed; regenerate offline convex export: '+str(path))
                    if not entry['parts']:raise ValueError('Empty finger collision export')
                    for part_index,part in enumerate(entry['parts']):
                        part_name=f'{name}_collision_{index}_part_{part_index}'
                        ET.SubElement(assets,'mesh',name=part_name,vertex=numbers(part['vertices']),
                                      face=' '.join(str(int(v)) for row in part['faces'] for v in row),scale=scale)
                        ET.SubElement(body,'geom',**{**attributes,'name':part_name,'mesh':part_name,
                                                   'contype':'2','conaffinity':'1','group':'3','rgba':'0 0 0 0'})
                    continue
                if kind == 'visual':
                    material = geom.find('material')
                    attributes.update(contype='0', conaffinity='0', group='2',
                                      rgba=materials.get(material.get('name') if material is not None else '', '.7 .7 .7 1'))
                else:
                    # Robot/robot contacts disabled, matching the existing self-collision setting.
                    # Robot/environment contacts enabled. URDF collision meshes become convex hulls.
                    attributes.update(contype='2', conaffinity='1', group='3', rgba='0 0 0 0')
                ET.SubElement(body, 'geom', **attributes)
        for child_joint in joints:
            if child_joint.find('parent').get('link') == name:
                add_link(body, child_joint.find('child').get('link'), origin(child_joint.find('origin')), child_joint)

    add_link(world, next(iter(bases)), base_pose)
    actuators = ET.SubElement(result, 'actuator')
    limits = []
    for index, name in enumerate(JOINTS):
        limit = movable[name].find('limit')
        limits.append([float(limit.get('lower')), float(limit.get('upper'))])
        # Match the existing Isaac position actuator; URDF effort is retained in hashes.
        effort = SO101_POSITION_FORCE_LIMIT if physics_profile == 'isaac_matched' else float(limit.get('effort'))
        ET.SubElement(actuators, 'position', name='target_' + name, joint=name,
                      kp=str(KP[index]), kv=str(KD[index]),
                      ctrlrange=numbers(limits[-1]), forcerange=numbers([-effort, effort]))
    return result, limits, source_hashes


def build_scene(workspace=None, width=320, height=240, timestep=.002, decimation=10,
                task_definition=None, full_workspace=False, physics_profile='isaac_matched', identity_sim=False, active_object_ids=None, reset=None):
    if (type(width) is not int or type(height) is not int or width < 1 or height < 1
            or type(decimation) is not int or decimation < 1
            or not np.isfinite(timestep) or not 0 < timestep <= .01):
        raise ValueError('Invalid resolution, timestep or decimation')
    if physics_profile not in ('isaac_matched', 'legacy_urdf'):
        raise ValueError('Unknown MuJoCo physics profile')
    w, urdf, source = load_inputs(workspace)
    if identity_sim:
        w = copy.deepcopy(w)
        w['profile'].pop('simulation_joint_offsets_rad', None)
    if task_definition is not None or active_object_ids is not None or w.get('scene_configuration') or reset is not None:
        if workspace is None:
            raise ValueError('Task Definition requires a published workspace')
        from ..workflow.tasks import configured
        w = configured(w, task_definition or w.get('task_definition'), active_object_ids, backend='mujoco',reset=reset)
    p, task = w['profile'], w['task']
    builder_scene = bool(w.get('manifest', {}).get('builder'))
    review = task.get('mode') == 'geometry_review'
    if not review and task['success']['relation'] not in ('contained_and_settled','inside_region_and_settled'):
        raise ValueError('Only contained_and_settled supported')
    pick, target = task['pick'], task['place']
    cube = p['objects'][pick] if not review else None
    cup = p['objects'][task['targets'][target]] if not review else None
    if not builder_scene and (cube['shape'] != 'box' or cup['shape'] not in ('cup', 'cup_proxy')):
        raise ValueError('Phase 2 requires a box and a cup target')
    workspace_pose = transform(p['workspace']['T_world'])
    robot, limits, hashes = robot_xml(urdf, p['robot_base']['T_world'], physics_profile,
                                    p.get('simulation_joint_offsets_rad'))
    scene = ET.Element('mujoco', model='so101_phase2')
    ET.SubElement(scene, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false')
    solver = {'cone': 'elliptic', 'iterations': '100', 'tolerance': '1e-10', 'impratio': '10'} if physics_profile == 'isaac_matched' else {}
    option = ET.SubElement(scene, 'option', timestep=str(timestep), integrator='implicitfast', gravity='0 0 -9.81', **solver)
    if w.get('manifest', {}).get('builder'):
        if physics_profile == 'isaac_matched':
            # Resolve residual tangential slip without inflating material friction.
            option.set('noslip_iterations', '10')
        # Flat box against cylinder bottom needs a contact manifold to avoid rocking.
        ET.SubElement(option, 'flag', multiccd='enable')
    default = ET.SubElement(scene, 'default')
    # PhysX static/dynamic friction cannot be transferred exactly; explicit MuJoCo approximation.
    friction = float(task['physics']['dynamic_friction'])
    ET.SubElement(default, 'geom', friction=numbers([friction, .005, .0001]), condim='3',
                  contype='1', conaffinity='3', solref='.01 1')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'quality', shadowsize='512', offsamples='0')
    ET.SubElement(visual, 'map', znear='.001')
    ET.SubElement(scene, 'include', file='so101.xml')
    world = ET.SubElement(scene, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 1', dir='0 0 -1', directional='true', castshadow='false')

    def rgba(obj):
        return numbers([*obj['appearance']['color'], 1.])

    def box(name, obj):
        body = ET.SubElement(world, 'body', name=name,
                             **pose_attrs(workspace_pose @ transform(obj['T_workspace'])))
        ET.SubElement(body, 'geom', name=name + '_geom', type='box',
                      size=numbers(np.asarray(obj['dimensions_m']) / 2), rgba=rgba(obj))
        return body

    box('table', p['table'])
    support = task['reset']['support_object']
    if support != 'table':
        box(support, p['objects'][support])
    if not review:
        cube_body = box(pick, cube)
        ET.SubElement(cube_body, 'freejoint', name='object_free')
        cube_body.find('geom').set('mass', str(task['physics']['mass_kg']))
        # Explicit inertia for the free box (compiler never derives robot inertia from visuals).
        size = np.asarray(cube['dimensions_m'], dtype=float)
        mass = float(task['physics']['mass_kg'])
        ET.SubElement(cube_body, 'inertial', pos='0 0 0', mass=str(mass),
                      diaginertia=numbers(mass / 12 * (size @ size - size**2)))
        bin_body = ET.SubElement(world, 'body', name=target,
                                **pose_attrs(workspace_pose @ transform(cup['T_workspace'])))
        if not builder_scene:
            h = float(cup['dimensions_m'][2])
            top = float(cup['dimensions_m'][0]) / 2
            bottom = float(cup.get('bottom_diameter_m', top * 2)) / 2
            wall = float(cup['wall_m'])
            if min(h, bottom, top, wall) <= 0 or wall >= min(bottom, top, h / 2):
                raise ValueError('Invalid target dimensions')
            ET.SubElement(bin_body, 'geom', name='bin_bottom', type='cylinder',
                          pos=numbers([0, 0, -h / 2 + wall / 2]), size=numbers([bottom, wall / 2]), rgba=rgba(cup))
            # Primitive tapered rings; open top. Same segmented idea as the existing USD cup compiler.
            for ring in range(4):
                radius = bottom + (top - bottom) * (ring + .5) / 4 - wall / 2
                for segment in range(32):
                    angle = 2 * np.pi * segment / 32
                    ET.SubElement(bin_body, 'geom', name=f'bin_wall_{ring}_{segment}', type='box',
                                  pos=numbers([radius * np.cos(angle), radius * np.sin(angle), -h / 2 + (ring + .5) * h / 4]),
                                  quat=numbers([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
                                  size=numbers([wall / 2, radius * np.tan(np.pi / 32) * 1.02, h / 8 + .0001]), rgba=rgba(cup))
    primary_camera = task.get('observation_cameras', list(p['cameras']))[0]
    camera = p['cameras'][primary_camera]
    if not builder_scene and (camera['parent'] != 'world' or np.any(np.asarray(camera['distortion']) != 0)):
        raise ValueError('Phase 2 supports world-mounted undistorted side camera only')
    k = np.asarray(camera['K'], dtype=float).copy()
    k[0] *= width / camera['resolution'][0]
    k[1] *= height / camera['resolution'][1]
    if not builder_scene and not np.allclose(k, [[k[0, 0], 0, width / 2], [0, k[1, 1], height / 2], [0, 0, 1]]):
        raise ValueError('Phase 2 supports centered, zero-skew pinhole intrinsics only')
    cam_pose = transform(camera['T_parent_camera']).copy()
    cam_pose[:3, :3] = cam_pose[:3, :3] @ np.diag([1., -1., -1.])  # ROS optical -> OpenGL camera.
    ET.SubElement(world, 'camera', name=primary_camera, **pose_attrs(cam_pose),
                  resolution=f'{width} {height}', sensorsize=f'{width} {height}', focalpixel=numbers([k[0, 0], k[1, 1]]))
    extended = {}
    if w.get('manifest', {}).get('builder'):
        full_workspace = True
    if full_workspace:
        from .workspace_scene import extend_scene
        extended = extend_scene(scene, robot, w, urdf, width, height, physics_profile)
        hashes.update(extended.get('texture_sha256', {}))
        if w.get('manifest', {}).get('builder'):
            from .workspace_scene import apply_builder_visuals
            extended.update(apply_builder_visuals(scene, robot, w))
            hashes.update({str(w['root']/k): v for k, v in w['manifest']['files'].items() if k.startswith('builder/')})
        extended['cameras'][primary_camera] = {**copy.deepcopy(camera), 'resolution': [width, height], 'K': k.tolist()}
    scene_data, robot_data = xml_bytes(scene), xml_bytes(robot)
    reduced = copy.deepcopy(w)
    reduced['task']['dynamic'] = list(task['dynamic']) if full_workspace else [pick]
    info = {
        'schema': 'so101.mujoco-scene/1', 'backend': 'mujoco', 'env_count': 1, 'physics_device': 'cpu',
        'source': source, 'asset_sha256': hashes, 'joint_order': list(JOINTS), 'joint_limits_rad': limits,
        'workspace_T_world': workspace_pose.tolist(), 'pick': pick, 'target': target,
        'primary_camera': primary_camera, 'physics_timestep_s': timestep, 'control_timestep_s': timestep * decimation,
        'decimation': decimation, 'camera': {**copy.deepcopy(camera), 'resolution': [width, height], 'K': k.tolist()},
        'task': copy.deepcopy(task), 'scene_subset': [pick, target, 'table', support, 'SO101'],
        'task_definition': copy.deepcopy(w.get('task_definition')),
        'scene_configuration': copy.deepcopy(w.get('scene_configuration')),
        'full_workspace': full_workspace,
        'physics_profile': physics_profile,
        'dynamic_objects': list(reduced['task']['dynamic']),
        'target_objects': list(task['targets']) if full_workspace else [target],
        **extended,
        'approximations': {
            'actuator': ('configured kp/kv and force limit 30 from existing Isaac actuator; not measured'
                         if physics_profile == 'isaac_matched' else 'legacy kp/kv from Isaac; force limit from URDF effort (10), not measured'),
            'robot_collision': ('offline convex decomposition of finger collision STL; other links per-mesh convex hulls; self collision disabled'
                                if physics_profile=='isaac_matched' else 'legacy URDF per-mesh convex hulls; self collision disabled'),
            'bin_collision': ('Builder-exported collision parts; AABB used for success/inertia approximation' if builder_scene else 'open primitive bottom and 4x32 box wall segments; wall provenance from profile'),
            'friction': 'task dynamic_friction as MuJoCo sliding friction; torsion/rolling .005/.0001 provisional',
            'restitution': 'MuJoCo solref .01 1 configured; no direct PhysX restitution equivalence',
            'scene': ('published objects, backgrounds, wrist mount, robot printed color and camera transforms; '
                      'published cup print textures and tapered visual mesh; lighting/material response is approximate' if full_workspace else
                      'one selected cube/target; other objects, textures and background omitted'),
        },
    }
    info['model_sha256'] = hashlib.sha256(
        scene_data + robot_data
        + json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return scene_data.decode('utf-8'), {'so101.xml': robot_data}, reduced, info
