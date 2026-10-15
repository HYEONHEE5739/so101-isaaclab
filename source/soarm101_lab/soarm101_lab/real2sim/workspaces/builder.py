"""Import a portable Builder scene into the existing actionable SO-101 workspace.

No Builder checkout, USD binding, simulator or hardware is imported here. Exported
visuals are referenced; existing robot control, task/reset and collision contracts
remain responsible for dynamics. Reconstruction acceptance is never inferred.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from . import package
from ..profile import SCHEMA, ROBOT_ASSET, validate, transform
from ...so101_joint_mapping import DEFAULT_MAPPER


def _file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing/unsafe Builder asset: ' + str(relative))
    return path


def frame(scene, name, seen=()):
    if name in seen or name not in scene['frames']:
        raise ValueError('Invalid Builder frame graph: ' + name)
    if name == 'world':
        return np.eye(4)
    row = scene['frames'][name]
    return frame(scene, row['parent'], (*seen, name)) @ transform(row['transform']['value'])


def world_pose(scene, obj):
    return frame(scene, obj['frame']) @ transform(obj['pose']['value'])


def mounted(scene, obj):
    """Only gripper-mounted fixtures are supported; never silently freeze a mount."""
    name = obj['frame']
    seen = set()
    while name != 'world':
        if name in seen: raise ValueError('Cyclic Builder frame')
        seen.add(name)
        if name == 'robot_link_gripper': return True
        if name.startswith('robot_link_'):
            raise ValueError('Only world/gripper camera and fixture mounts are supported: '+name)
        name = scene['frames'][name]['parent']
    return False


def inspect_export(source):
    root = Path(source).resolve()
    if root.is_file(): root = root.parent
    manifest = yaml.safe_load(_file(root, 'scene_manifest.yaml').read_text())
    if manifest.get('schema') != 'real2sim.package/1' or manifest.get('units') != 'm':
        raise ValueError('Expected metric real2sim.package/1 export')
    required = {'scene.yaml', 'scene.xml', 'scene.usd', 'SCENE.md'}
    if not required <= manifest.get('files', {}).keys(): raise ValueError('Incomplete Builder export')
    for relative, digest in manifest['files'].items():
        if package.sha(_file(root, relative)) != digest: raise ValueError('Modified Builder export: ' + relative)
    scene = yaml.safe_load(_file(root, manifest['canonical']).read_text())
    if 'workflow' in manifest:
        if 'workflow' in scene and scene['workflow'] != manifest['workflow']: raise ValueError('Conflicting scene/manifest workflow metadata')
        scene['workflow'] = copy.deepcopy(manifest['workflow'])
    ids = [o['id'] for o in scene['objects']]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', n) for n in ids):
        raise ValueError('Invalid/duplicate Builder object identity')
    robot = scene.get('robot_model', {})
    from ...so101_robot_spec import simulation_joint_offsets
    simulation_joint_offsets(robot.get('simulation_joint_offsets_rad'), DEFAULT_MAPPER.metadata()['joint_order'])
    if robot.get('joint_names') != DEFAULT_MAPPER.metadata()['joint_order']:
        raise ValueError('Builder robot joint order/mapping differs from the SO-101 controller')
    package.validate_mapping(robot.get('mapping'), package.ROOT / ROBOT_ASSET['urdf'])
    # Portable exports rename mesh files, but must preserve the controller's kinematic tree.
    def joint_contract(path):
        result = {}
        for j in ET.parse(path).getroot().findall('joint'):
            result[j.get('name')] = (j.get('type'), {c.tag: {k:v for k,v in c.attrib.items() if not (c.tag=='limit' and k in ('lower','upper'))} for c in j})
        return result
    if robot.get('format') != 'urdf' or joint_contract(_file(root, robot['path'])) != joint_contract(package.ROOT / ROBOT_ASSET['urdf']):
        raise ValueError('Builder kinematics differ from the SO-101 controlled robot')
    for obj in scene['objects']: world_pose(scene, obj)
    return root, manifest, scene


def profile_and_task(scene, explicit_task=False):
    objects = {o['id']: o for o in scene['objects']}
    reserved = {'robot','ee_frame','dome_light','workspace_static','workspace_camera_mount','wrist_mount','camera_overview'} | {'camera_'+c['id']+'view' for c in scene['cameras']}
    if set(objects) & reserved: raise ValueError('Builder object ID conflicts with runtime entity')
    integration = copy.deepcopy(scene.get('workflow', {}))
    review = not integration and not explicit_task
    physical_review = review and any(o['role'] == 'manipulable_object' and
        (o.get('physics', {}).get('mass_kg', {}).get('value') or 0) > 0 for o in scene['objects'])
    if integration and integration.get('schema') != 'so101.scene-workflow/1':
        raise ValueError('Unsupported scene workflow metadata')
    allowed = {'schema','support_surface','tasks','reset','physics','object_physics','success','observation_cameras','real_camera_bindings','roles','provenance'}
    if set(integration) - allowed: raise ValueError('Unknown scene workflow fields')
    if 'provenance' in integration:
        evidence = integration['provenance']
        if not {'tasks','success','reset','physics'} <= evidence.keys(): raise ValueError('Incomplete workflow provenance')
        for row in evidence.values():
            if row.get('source') not in ('measured','calibrated','derived','optimized','estimated','unknown') or row.get('status') not in ('UNVALIDATED','VALIDATED','NOT_APPLICABLE') or not isinstance(row.get('evidence'),list):
                raise ValueError('Invalid workflow provenance/status')
    supports = [o for o in scene['objects'] if o['role'] == 'workspace']
    tables = [o for o in scene['objects'] if o['role']=='support_surface']
    table_id = integration.get('support_surface', 'table' if 'table' in objects else tables[0]['id'] if len(tables)==1 else None)
    if len(supports) != 1 or table_id not in objects: raise ValueError('One workspace region and identified support surface required')
    if table_id != 'table' and 'table' in objects: raise ValueError('table is reserved for the runtime support surface')
    if any(o['id']+'_visual' in objects for o in scene['objects'] if o['role']=='receptacle'):
        raise ValueError('Target visual alias conflicts with canonical object ID')
    support = supports[0]
    world = np.eye(4); world[:3, :3] = world_pose(scene, support)[:3, :3]
    # Reset samples in this support frame; both engines transform positions and
    # orientations into world coordinates. Keep measured plane inclination.
    if not review and world[2, 2] <= 0: raise ValueError('Task support must face upward')
    inv = np.linalg.inv(world)
    provenance = lambda *keys: {k: 'provisional' for k in keys}
    def primitive(obj):
        shape = obj['visual']['type']
        if shape not in ('box', 'cylinder'): shape = 'box'  # metadata proxy only; exported visual is retained
        return {'shape': shape, 'dimensions_m': obj['dimensions']['value'],
                'T_workspace': (inv @ world_pose(scene, obj)).tolist(), 'render_enabled': True,
                'provenance': provenance('dimensions_m', 'T_workspace'),
                'appearance': {'color': obj['color']['value'], 'roughness': obj.get('material', {}).get('roughness', {}).get('value', .6),
                               'provenance': provenance('color', 'roughness')}}
    p = {'schema': SCHEMA, 'units': 'metres', 'optical_axes': 'x_right_y_down_z_forward', 'robot_asset': copy.deepcopy(ROBOT_ASSET),
         'robot_base': {'T_world': frame(scene, 'robot_base').tolist(), 'provenance': provenance('T_world')},
         'workspace': {'T_world': world.tolist(), 'size_m': support['dimensions']['value'][:2], 'provenance': provenance('T_world', 'size_m')},
         'table': {**primitive(objects[table_id]), 'source_id':table_id}, 'objects': {}, 'cameras': {}, 'relations': {},
         'appearance': {'light_intensity': 1000., 'provenance': provenance('light_intensity')}}
    dynamics, targets = [], {}
    for obj in scene['objects']:
        if obj['role'] == 'reference_robot' or obj['id'] == table_id or mounted(scene, obj) or obj.get('parent_object'): continue
        name = obj['id']; spec = primitive(obj)
        if obj['role'] == 'manipulable_object' and (not review or physical_review and
                (obj.get('physics', {}).get('mass_kg', {}).get('value') or 0) > 0):
            if obj['collision']['type'] == 'none':
                raise ValueError('Dynamic object requires explicit collision geometry: ' + name)
            spec['geometry_source'] = 'builder_export'
            spec['bounds_source'] = 'canonical dimensions; conservative local bounding box'
            dynamics.append(name)
        if obj['role'] == 'receptacle' and (not review or physical_review):
            targets[name] = name + '_visual'
            if obj['collision']['type'] == 'container' and obj['collision'].get('round'):
                spec.update(shape='cup_proxy', bottom_diameter_m=obj['collision'].get('bottom_diameter_m', obj['dimensions']['value'][0]),
                            wall_m=obj['collision']['wall']['value'], approximation='Builder exported collision')
                spec['provenance'].update(provenance('bottom_diameter_m', 'wall_m'))
            elif not integration:
                raise ValueError('Non-round targets require explicit workflow success-volume metadata')
            spec['geometry_source'] = 'builder_export'
            name += '_visual'
        p['objects'][name] = spec
    for c in scene['cameras']:
        view = c['id']
        if view in p['cameras']: raise ValueError('Duplicate camera ID: '+view)
        attached = mounted(scene, c)
        t = world_pose(scene, c)
        if attached: t = np.linalg.inv(frame(scene, 'robot_link_gripper')) @ t
        p['cameras'][view] = {'parent': 'gripper' if attached else 'world', 'T_parent_camera': t.tolist(),
            'K': c['K']['value'], 'distortion': c['distortion']['value'], 'resolution': c['resolution'],
            'provenance': provenance('T_parent_camera', 'K', 'distortion', 'resolution')}
    if scene['robot_model'].get('simulation_joint_offsets_rad') is not None:
        p['simulation_joint_offsets_rad'] = copy.deepcopy(scene['robot_model']['simulation_joint_offsets_rad'])
    p['wrist_mount'] = {'parent': 'gripper', 'source': 'provisional', 'parts': {}}
    validate(p)
    if review:
        task = package.review_contract(p, support['id'], dynamic=dynamics, targets=targets)
        if physical_review:
            task['physics'].update(mass_kg=next(o['physics']['mass_kg']['value'] for o in scene['objects'] if o['id'] in dynamics),
                note='Explicit canonical object physics; review has no task success or random reset')
            task['object_physics'] = {}
            for obj in scene['objects']:
                if obj['id'] not in [*dynamics, *targets]: continue
                row = {'provenance': 'provisional', 'field_provenance': {}}
                for field, keys in [('mass_kg', ('mass_kg',)), ('friction', ('static_friction', 'dynamic_friction')),
                                    ('restitution', ('restitution',))]:
                    evidence = obj.get('physics', {}).get(field, {})
                    if evidence.get('value') is not None:
                        for key in keys:
                            row[key] = evidence['value']
                            row['field_provenance'][key] = copy.deepcopy(evidence)
                task['object_physics'][obj['id']] = row

        return p, package.validate_task(task, p)
    task = package.default_task(p)
    if not dynamics or not targets: raise ValueError('Builder scene needs manipulands and receptacles for the current workflow')
    task.update(dynamic=dynamics, targets=targets, pick='cube_red' if 'cube_red' in dynamics else dynamics[0], place=next(iter(targets)),
                semantics_source='Builder semantic roles mapped to existing pick/place recipe; not inferred from image language')
    task['reset']['support_object'] = support['id']
    task['physics']['provenance'] = 'provisional'
    task['physics']['note'] = 'Existing 20g / friction 1 template; NOT measured or identified by Builder'
    if len(dynamics) > 4: task['reset']['mode'] = 'uniform'
    if integration:
        for key in ('reset', 'physics', 'object_physics', 'success', 'tasks', 'observation_cameras', 'real_camera_bindings', 'roles'):
            if key in integration:
                if key in ('reset', 'physics'): task[key].update(integration[key])
                else: task[key] = integration[key]
        if not integration.get('tasks'): raise ValueError('workflow requires explicit pick/place tasks')
        first = integration['tasks'][0]
        task.update(pick=first['pick'], place=first['place'], success=first.get('success', task['success']),
                    semantics_source='explicit canonical scene.workflow metadata')
    if task['reset']['support_object'] == table_id: task['reset']['support_object'] = 'table'
    if 'roles' in task and table_id in task['roles']:
        task['roles']['table'] = task['roles'].pop(table_id)
    task['workflow_provenance'] = copy.deepcopy(integration.get('provenance', {
        k:{'source':'unknown','status':'UNVALIDATED','evidence':[]} for k in ('tasks','reset','success','physics')}))
    physics = task.setdefault('object_physics', {})
    for obj in scene['objects']:
        if obj['id'] not in [*dynamics,*targets]: continue
        fields = obj.get('physics', {})
        row = physics.setdefault(obj['id'], {})
        for field,keys in [('mass_kg',('mass_kg',)), ('friction',('static_friction','dynamic_friction')), ('restitution',('restitution',))]:
            value = fields.get(field, {}).get('value')
            if value is not None:
                for key in keys:
                    if key not in row:
                        row[key] = value
                        row.setdefault('field_provenance', {})[key] = copy.deepcopy(fields[field])
        if not row: physics.pop(obj['id'])
    task.setdefault('observation_cameras', list(p['cameras']))
    task.setdefault('real_camera_bindings', {n:n+'_camera' for n in p['cameras'] if n in ('side','wrist')})
    return p, package.validate_task(task, p)


def _matrix(t):
    return '(' + ', '.join('(' + ', '.join(format(float(v), '.12g') for v in row) + ')' for row in np.asarray(t).T) + ')'


def build_usd_assets(root, scene, profile, task):
    """Small composition layers reuse the original exported geometry and materials."""
    source = ET.parse(root/'builder/scene.xml').getroot()
    geometries = {b.get('name'): [g.get('name').split(b.get('name')+'_', 1)[1] for g in b.findall('geom')]
                  for b in source.find('worldbody').findall('body')}
    objs = {o['id']: o for o in scene['objects']}
    material = task['physics']
    from .physics import compile_physics
    physics = compile_physics(profile, task)['objects']
    def document(path, entries, rigid=None):
        spec = physics.get(path.stem, {})
        material = {**task['physics'], **(spec.get('material') or {})}
        if spec.get('mass_kg') is not None: material['mass_kg'] = spec['mass_kg']
        # Entries: source ID, destination child name, parent-relative matrix, collision enabled.
        lines = ['#usda 1.0', '(defaultPrim = "Real2Sim"\n metersPerUnit = 1\n upAxis = "Z")']
        api = ' (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])' if rigid else ''
        lines.append('def Xform "Real2Sim"'+api+' {')
        if rigid:
            lines += ['bool physics:rigidBodyEnabled = true', 'bool physics:kinematicEnabled = '+('true' if rigid == 'target' else 'false')]
            if rigid == 'dynamic': lines.append('float physics:mass = '+str(material['mass_kg']))
        lines += ['def Material "PhysicsMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {',
                  'float physics:staticFriction = '+str(material['static_friction']),
                  'float physics:dynamicFriction = '+str(material['dynamic_friction']),
                  'float physics:restitution = '+str(material['restitution']), '}']
        static = path.name == 'static.usda'
        if static:
            lines += ['def Xform "Workspace" {', 'matrix4d xformOp:transform = '+_matrix(profile['workspace']['T_world']), 'uniform token[] xformOpOrder = ["xformOp:transform"]']
        prefix = '/Real2Sim/Workspace' if static else '/Real2Sim'
        for identity, name, pose, collision in entries:
            matname = 'Material_'+name
            lines += [f'def Material "{matname}" (prepend references = @../builder/scene.usd@</Scene/material_{identity}>) {{}}',
                      f'def Xform "{name}" (prepend references = @../builder/scene.usd@</Scene/{identity}>\n delete apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{',
                      'matrix4d xformOp:transform = '+_matrix(pose), 'uniform token[] xformOpOrder = ["xformOp:transform"]',
                      'bool physics:rigidBodyEnabled = false']
            for geom in geometries[identity]:
                lines.append(f'over "{geom}" {{')
                if geom.startswith('visual_'): lines.append(f'rel material:binding = <{prefix}/{matname}>')
                else:
                    lines += ['bool physics:collisionEnabled = '+('true' if collision else 'false'),
                              'rel material:binding:physics = </Real2Sim/PhysicsMaterial>']
                lines.append('}')
            lines.append('}')
        if static: lines.append('}')
        lines.append('}')
        path.write_text('\n'.join(lines)+'\n')
    root.joinpath('scene').mkdir()
    inv = np.linalg.inv(np.asarray(profile['workspace']['T_world']))
    static, mount = [], []
    for obj in scene['objects']:
        name = obj['id']
        if name == profile['table'].get('source_id','table'): name = 'table'
        if (obj['role'] == 'reference_robot' or name in task['dynamic'] or name in task['targets']
                or obj.get('parent_object') in [*task['dynamic'], *task['targets']]): continue
        if mounted(scene, obj):
            mount.append((name, name, np.linalg.inv(frame(scene, 'robot_link_gripper')) @ world_pose(scene, obj), False))
        else:
            # Table supplies collision; marker/printed edges/background are visual-only like existing workspaces.
            static.append((obj['id'], name, inv @ world_pose(scene, obj), physics.get(name, {}).get('collision', False)))
    document(root/'scene/static.usda', static)
    document(root/'scene/wrist_mount.usda', mount)
    for link in sorted({o['link'] for o in scene['objects'] if o['role'] == 'reference_robot'}):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', link): raise ValueError('Invalid robot link name')
        entries = [(o['id'], o['id'], transform(o['link_local_pose']), False)
                   for o in scene['objects'] if o['role'] == 'reference_robot' and o['link'] == link]
        document(root/f'scene/robot_visual_{link}.usda', entries)
    for name in [*task['dynamic'], *task['targets']]:
        entries = [(name, name, np.eye(4), True)]
        for obj in scene['objects']:
            if obj.get('parent_object') == name:
                entries.append((obj['id'], obj['id'], np.linalg.inv(world_pose(scene, objs[name])) @ world_pose(scene, obj), False))
        document(root/f'scene/{name}.usda', entries, 'dynamic' if name in task['dynamic'] else 'target')


def calibrate_simulation_usd(root, node):
    """Derive zero-frame overrides; original USD/contact-pad evidence stays intact."""
    if node is None:
        return 'assets/SO101/usd/so101_isaaclab.usd'
    from ...so101_robot_spec import simulation_joint_offsets
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    offsets = dict(zip(DEFAULT_MAPPER.metadata()['joint_order'],
                       simulation_joint_offsets(node, DEFAULT_MAPPER.metadata()['joint_order'])))
    urdf = root/'assets/SO101/urdf/so101_isaaclab.urdf'
    tree = ET.parse(urdf)
    for joint in tree.getroot().findall('joint'):
        if joint.get('name') not in offsets:
            continue
        origin = joint.find('origin')
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
        rotation = rotation * Rotation.from_rotvec(axis / np.linalg.norm(axis) * offsets[joint.get('name')])
        origin.set('rpy', ' '.join(format(float(x), '.16g') for x in rotation.as_euler('xyz')))
    tree.write(urdf.with_name('so101_sim.urdf'), encoding='utf-8', xml_declaration=True)
    relative = 'assets/SO101/usd/so101_sim.usda'
    stage = Usd.Stage.CreateNew(str(root/relative))
    stage.GetRootLayer().subLayerPaths = ['so101_isaaclab.usd']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/so101_calibrated'))
    joints = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    # The fixed world-to-base anchor has an empty body0 relationship.
    # It stays untouched; only internal articulation edges are calibrated.
    joints = [j for j in joints if j.GetBody0Rel().GetTargets() and j.GetBody1Rel().GetTargets()]
    for joint in joints:
        name = joint.GetPrim().GetName()
        if name not in offsets:
            continue
        axis = UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get()
        direction = np.eye(3)['XYZ'.index(axis)]
        old = joint.GetLocalRot0Attr().Get()
        quat = Rotation.from_quat([*old.GetImaginary(), old.GetReal()]) * Rotation.from_rotvec(direction * offsets[name])
        q = quat.as_quat()
        joint.GetLocalRot0Attr().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    def local(joint, index):
        t = np.eye(4)
        q = (joint.GetLocalRot0Attr() if index == 0 else joint.GetLocalRot1Attr()).Get()
        t[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        t[:3, 3] = (joint.GetLocalPos0Attr() if index == 0 else joint.GetLocalPos1Attr()).Get()
        return t
    # Rigid links are siblings in the original SO101 USD. Set a consistent zero
    # configuration so PhysX does not begin with violated calibrated joint frames.
    poses = {}
    children = {str(j.GetBody1Rel().GetTargets()[0]) for j in joints}
    for joint in joints:
        parent = str(joint.GetBody0Rel().GetTargets()[0])
        if parent not in children:
            poses[parent] = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(parent)).GetLocalTransformation()).T
    pending = list(joints)
    while pending:
        before = len(pending)
        for joint in pending[:]:
            parent = str(joint.GetBody0Rel().GetTargets()[0]); child = str(joint.GetBody1Rel().GetTargets()[0])
            if parent not in poses:
                continue
            poses[child] = poses[parent] @ local(joint, 0) @ np.linalg.inv(local(joint, 1))
            UsdGeom.Xformable(stage.GetPrimAtPath(child)).MakeMatrixXform().Set(Gf.Matrix4d(poses[child].T.tolist()))
            pending.remove(joint)
        if before == len(pending):
            raise ValueError('Disconnected simulation USD joint graph')
    stage.GetRootLayer().Save()
    return relative


def import_scene(source, workspace_id=None, version=1, task=None, registry=package.REGISTRY):
    root, manifest, scene = inspect_export(source)
    workspace_id = workspace_id or ('builder_' + package.sha(root/'scene_manifest.yaml')[:12])
    if not re.fullmatch(r'[A-Za-z0-9_]+', workspace_id) or type(version) is not int or version < 1: raise ValueError('Invalid workspace identity')
    explicit_task = task is not None
    profile, default = profile_and_task(scene, explicit_task=explicit_task)
    task = package.validate_task(copy.deepcopy(task or default), profile)
    dest = Path(registry)/workspace_id/f'v{version}'
    digest = package.sha(root/'scene_manifest.yaml')
    if dest.exists():
        old = package.load(dest)
        if old['manifest'].get('builder', {}).get('manifest_sha256') == digest and (not explicit_task or old['task'] == task): return dest
        raise FileExistsError('Workspace version already exists; use a new version')
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.builder_', dir=dest.parent))
    try:
        (staging/'builder').mkdir()
        for relative in manifest['files']:
            if relative.startswith('assets/') or relative in ('scene.yaml', 'scene.xml', 'scene.usd', 'SCENE.md'):
                target = staging/'builder'/relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_file(root, relative), target)
        shutil.copy2(root/'scene_manifest.yaml', staging/'builder/scene_manifest.yaml')
        shutil.copytree(package.ROOT/'assets/SO101', staging/'assets/SO101')
        robot_asset = calibrate_simulation_usd(staging, profile.get('simulation_joint_offsets_rad'))
        identity_asset = 'assets/SO101/usd/so101_joint_limits.usda'
        if profile.get('simulation_joint_offsets_rad') is not None:
            from ...so101_robot_spec import limit_usd
            execution_asset = 'assets/SO101/usd/so101_execution_limits.usda'
            limit_usd(staging/robot_asset, staging/'assets/SO101/urdf/so101_isaaclab.urdf', staging/execution_asset)
            robot_asset = execution_asset
        else:
            robot_asset = identity_asset
        package.dump(staging/'profile.json', profile); package.dump(staging/'task.json', task)
        from .physics import compile_physics
        package.dump(staging/'physics.json', compile_physics(profile, task))
        build_usd_assets(staging, scene, profile, task)
        package.dump(staging/'provenance.json', {'asset_map': {}, 'source_builder_path': str(root),
                     'task_semantics_source': task['semantics_source'], 'physics': task['physics'],
                     'workflow_provenance': task.get('workflow_provenance', {}),
                     'reconstruction_status': manifest['status'], 'validation_scope': manifest.get('validation_scope')})
        files = {f.relative_to(staging).as_posix(): package.sha(f) for f in sorted(staging.rglob('*')) if f.is_file()}
        robots = {k:v for k,v in files.items() if k.startswith('assets/SO101/')}
        data = {'schema': package.SCHEMA, 'status': 'READY', 'workspace_id': workspace_id, 'version': version,
                'source_revision': 'builder:'+digest, 'source_profile_sha256': files['profile.json'],
                'robot_asset': robot_asset, 'identity_robot_asset': identity_asset, 'robot_asset_sha256': hashlib.sha256(package.canonical(robots)).hexdigest(),
                'mapper': copy.deepcopy(scene['robot_model']['mapping']), 'task_sha256': files['task.json'], 'files': files,
                'content_sha256': hashlib.sha256(package.canonical(files)).hexdigest(),
                'builder': {'manifest_sha256': digest, 'status': manifest['status'], 'validation_scope': manifest.get('validation_scope'),
                            'physics_status': ('REVIEW_PHYSICS_UNVALIDATED' if task.get('mode') == 'geometry_review' and task['dynamic'] else 'PROVISIONAL_TASK_PHYSICS'),
                            'robot_visual_links': sorted({o['link'] for o in scene['objects'] if o['role'] == 'reference_robot'}),
                            'scene': 'builder/scene.yaml',
                            'mjcf': 'builder/scene.xml', 'usd': 'builder/scene.usd'}}
        package.dump(staging/'manifest.json', data); package.load(staging); staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest
