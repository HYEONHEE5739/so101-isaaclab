"""Independent workspace scene; reuses SO101 action/observation implementations only."""
import copy
import json
import numpy as np
from .package import load, metadata
from .reset import reset_workspace


def deactivate_legacy_wrist_camera(robot_prim):
    """Session-only override; never edit the referenced robot asset on disk."""
    from pxr import Usd, UsdGeom
    stage = robot_prim.GetStage()
    camera = stage.GetPrimAtPath(robot_prim.GetPath().AppendPath('gripper/Camera_WristView'))
    if camera and camera.IsA(UsdGeom.Camera):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            camera.SetActive(False)


def restore_missing_finger_collisions(robot_prim, urdf):
    """Session geometry from the selected URDF; preserve authored nonempty colliders."""
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
    from scipy.spatial.transform import Rotation
    stage = robot_prim.GetStage()
    model = ET.parse(urdf).getroot()
    for name in ('gripper', 'jaw'):
        group = stage.GetPrimAtPath(robot_prim.GetPath().AppendPath(name + '/collisions'))
        if group and any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(group, Usd.TraverseInstanceProxies())):
            continue
        for index, node in enumerate(model.find(f"link[@name='{name}']").findall('collision')):
            geometry = node.find('geometry/mesh')
            if geometry is None:
                raise ValueError('Expected SO101 finger collision mesh')
            mesh = trimesh.load_mesh(Path(urdf).parent / geometry.get('filename'), process=False)
            vertices = np.asarray(mesh.vertices) * np.fromstring(geometry.get('scale', '1 1 1'), sep=' ')
            path = robot_prim.GetPath().AppendPath(f'{name}/URDFCollision_{index}')
            shape = UsdGeom.Mesh.Define(stage, path)
            shape.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
            shape.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            shape.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).ravel().tolist())
            shape.CreateSubdivisionSchemeAttr('none')
            origin = node.find('origin'); T = np.eye(4)
            if origin is not None:
                T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            UsdGeom.Xformable(shape).MakeMatrixXform().Set(Gf.Matrix4d(T.T.tolist()))
            shape.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
            material = UsdShade.Material.Get(stage, robot_prim.GetPath().AppendPath('gripper_collider_material'))
            if material:
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')


def spawn_workspace_robot(prim_path, cfg, translation=None, orientation=None, visual_links=(), collision_urdf=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    robot = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    deactivate_legacy_wrist_camera(robot)
    if collision_urdf:
        restore_missing_finger_collisions(robot, collision_urdf)
    if visual_links:
        from pxr import UsdGeom
        for link in visual_links:
            visual = robot.GetStage().GetPrimAtPath(robot.GetPath().AppendPath(link + "/visuals"))
            if visual: UsdGeom.Imageable(visual).CreateVisibilityAttr("invisible")
    return robot


# Contact/geometry tolerance, in meters; shared by annotation, datagen and evaluation.
WALL_TOLERANCE_M = 0.001


def inside_wall(radial_excess):
    return (radial_excess <= WALL_TOLERANCE_M).all(1)


def task_success(env, workspace):
    return task_success_details(env, workspace)[0]


def task_success_details(env, workspace):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms, quat_apply
    workspace = getattr(env, '_workflow_task_workspace', workspace)
    task=workspace['task'];profile=workspace['profile'];cube=env.scene[task['pick']];target=env.scene[task['place']]
    pos,quat=subtract_frame_transforms(target.data.root_pos_w,target.data.root_quat_w,cube.data.root_pos_w,cube.data.root_quat_w)
    size=profile['objects'][task['pick']]['dimensions_m'];cup=profile['objects'][task['targets'][task['place']]]
    corners=torch.tensor([[a*size[0]/2,b*size[1]/2,c*size[2]/2] for a in [-1,1] for b in [-1,1] for c in [-1,1]],device=env.device)
    pts=quat_apply(quat[:,None,:].expand(-1,8,-1).reshape(-1,4),corners[None].expand(env.num_envs,-1,-1).reshape(-1,3)).reshape(env.num_envs,8,3)+pos[:,None,:]
    if task['success']['relation'] == 'inside_region_and_settled':
        bounds = torch.tensor(task['success']['bounds_m'], device=env.device)
        inside = ((pts >= bounds[0]) & (pts <= bounds[1])).all(2).all(1)
        speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w,dim=-1)
        settled = speed < task['success']['max_speed_m_s']
        return inside & settled, dict(pick=task['pick'], target=task['place'], height_ok=inside,
                                     wall_ok=inside, speed_ok=settled, speed=speed,
                                     speed_limit=task['success']['max_speed_m_s'])
    h=cup['dimensions_m'][2];bottom=cup.get('bottom_diameter_m',cup['dimensions_m'][0])/2;top=cup['dimensions_m'][0]/2;wall=cup['wall_m']
    radius=bottom+(top-bottom)*(pts[:,:,2]+h/2)/h-wall
    radial_excess = torch.linalg.vector_norm(pts[:,:,:2],dim=-1)-radius
    zmin, zmax = pts[:,:,2].min(1).values, pts[:,:,2].max(1).values
    lower, upper = -h/2+wall-.002, h/2
    height_ok = (zmin > lower) & (zmax < upper)
    wall_ok = inside_wall(radial_excess)
    speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w,dim=-1)
    speed_ok = speed < task['success']['max_speed_m_s']
    return height_ok & wall_ok & speed_ok, dict(
        pick=task['pick'], target=task['place'], height_ok=height_ok, wall_ok=wall_ok,
        speed_ok=speed_ok, zmin=zmin, zmax=zmax, lower=lower, upper=upper,
        radial_excess=radial_excess.max(1).values, speed=speed,
        speed_limit=task['success']['max_speed_m_s'], wall_tolerance=WALL_TOLERANCE_M)


def make_config(workspace, device='cuda:0', num_envs=1, task_definition=None, identity_sim=False, active_object_ids=None):
    if num_envs!=1:raise ValueError('Workspace Phase 1 supports num_envs=1')
    from isaaclab.utils import configclass
    from isaaclab.envs import ManagerBasedEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg, FrameTransformerCfg
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
    import isaaclab.sim as sim
    from ...assets.robots.so101 import SO101_FOLLOWER_CFG
    from ...tasks.manager_based.soarm101_lab.base_pick_place_teleop_env_cfg import ActionsCfg, ObservationsCfg
    from ..profile import pose, render_k
    from ...workflow.tasks import configured
    w=configured(workspace if isinstance(workspace,dict) else load(workspace), task_definition or (workspace.get('task_definition') if isinstance(workspace,dict) else None), active_object_ids, backend='isaac');p=w['profile'];task=w['task'];root=w['root']
    class RuntimeCamera(Camera):
        def _update_poses(self, env_ids):
            # USD Xform reads remain at the authored pose with Fabric enabled.
            # For the moving camera read its existing PhysX parent, then compose
            # the unchanged optical mount. The renderer owns the same hierarchy.
            if not self.cfg.prim_path.rsplit('/', 1)[0].endswith('/gripper'):
                return super()._update_poses(env_ids)
            import torch
            from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention
            context = sim.SimulationContext.instance()
            if not hasattr(self, '_mount_view'):
                parent = self.cfg.prim_path.rsplit('/', 1)[0].replace('.*', '*')
                self._mount_view = context.physics_sim_view.create_rigid_body_view(parent)
            state = self._mount_view.get_transforms()[env_ids]
            local_position = torch.tensor(self.cfg.offset.pos, device=self.device, dtype=state.dtype).expand(len(state), -1)
            local_rotation = torch.tensor(self.cfg.offset.rot, device=self.device, dtype=state.dtype).expand(len(state), -1)
            position, rotation = combine_frame_transforms(
                state[:, :3], state[:, [6, 3, 4, 5]], local_position, local_rotation)
            self._data.pos_w[env_ids] = position
            self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
                rotation, origin=self.cfg.offset.convention, target='world')

    @configclass
    class Events:
        reset_episode=EventTermCfg(func=reset_workspace,mode='reset',params={'workspace':w,'random_state':None})
    @configclass
    class TaskObservations(ObservationGroupCfg):
        success=ObservationTermCfg(func=task_success,params={'workspace':w})
    cfg=ManagerBasedEnvCfg();cfg.scene=InteractiveSceneCfg(num_envs=1,env_spacing=2.)
    scene=cfg.scene
    scene.robot=SO101_FOLLOWER_CFG.copy();scene.robot.prim_path='{ENV_REGEX_NS}/Robot'
    scene.robot.spawn.usd_path=str(root/w['manifest']['robot_asset'])
    if identity_sim:
        scene.robot.spawn.usd_path=str(root/w['manifest'].get('identity_robot_asset','assets/SO101/usd/so101_isaaclab.usd'))
    scene.robot.spawn.func=spawn_workspace_robot
    if w['manifest'].get('builder'):
        from functools import partial
        links = w['manifest']['builder'].get('robot_visual_links', [])
        scene.robot.spawn.func = partial(spawn_workspace_robot, visual_links=links, collision_urdf=root/'assets/SO101/urdf/so101_isaaclab.urdf')
        for link in links:
            setattr(scene, 'builder_robot_' + link, AssetBaseCfg(
                prim_path='{ENV_REGEX_NS}/Robot/' + link + '/BuilderVisuals',
                spawn=sim.UsdFileCfg(usd_path=str(root/f'scene/robot_visual_{link}.usda'))))
    scene.robot.init_state.pos,scene.robot.init_state.rot=pose(p['robot_base']['T_world'])
    scene.robot.init_state.joint_pos=copy.deepcopy(task['robot_initial_joint_rad'])
    scene.ee_frame=FrameTransformerCfg(prim_path='{ENV_REGEX_NS}/Robot/base',debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path='{ENV_REGEX_NS}/Robot/tool0',name='tool0')])
    scene.dome_light=AssetBaseCfg(prim_path='/World/Light',spawn=sim.DomeLightCfg(intensity=p['appearance']['light_intensity'],color=(.75,.75,.75)))
    scene.workspace_static=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/WorkspaceStatic',spawn=sim.UsdFileCfg(usd_path=str(root/'scene/static.usda')))
    if p.get('wrist_mount'):
        scene.workspace_camera_mount=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Robot/gripper/WorkspaceMount',spawn=sim.UsdFileCfg(usd_path=str(root/'scene/wrist_mount.usda')))
    compiled = json.loads((root/'physics.json').read_text()) if (root/'physics.json').exists() else None
    for name in task['dynamic']:
        obj=p['objects'][name];pos,rot=pose(np.array(p['workspace']['T_world'])@np.array(obj['T_workspace']));physics=task['physics']
        if compiled:
            spec=compiled['objects'][name]
            physics={**spec['material'], 'mass_kg': spec['mass_kg']}
        setattr(scene,name,RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name,
            spawn=sim.CuboidCfg(size=tuple(obj['dimensions_m']),mass_props=sim.MassPropertiesCfg(mass=physics['mass_kg']),
                rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=16,solver_velocity_iteration_count=1),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=.002,rest_offset=0.),
                physics_material=sim.RigidBodyMaterialCfg(static_friction=physics['static_friction'],dynamic_friction=physics['dynamic_friction'],restitution=physics['restitution']),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(obj['appearance']['color']),roughness=obj['appearance']['roughness'])),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot)))
    # Builder keeps its exported visual geometry while existing task roles own dynamics.
    if w['manifest'].get('builder'):
        for name in task['dynamic']:
            getattr(scene, name).spawn = sim.UsdFileCfg(usd_path=str(root/f'scene/{name}.usda'),
                rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1),
                mass_props=sim.MassPropertiesCfg(mass=compiled['objects'][name]['mass_kg']),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.))
    for name,visual in task['targets'].items():
        pos,rot=pose(np.array(p['workspace']['T_world'])@np.array(p['objects'][visual]['T_workspace']))
        setattr(scene,name,RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name,
            spawn=sim.UsdFileCfg(usd_path=str(root/f'scene/{name}.usda')),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot)))
    for view,c in p['cameras'].items():
        width,height=c['resolution'];k=render_k(c);pos,rot=pose(c['T_parent_camera'])
        setattr(scene,'camera_'+view+'view',CameraCfg(class_type=RuntimeCamera,
            prim_path='{ENV_REGEX_NS}/'+(('Robot/gripper/' if c['parent']=='gripper' else '') + ('Camera_Side' if view=='side' else 'Camera_WorkspaceWrist' if view=='wrist' else 'Camera_'+view)),
            width=width,height=height,data_types=['rgb'],update_latest_camera_pose=True,update_period=0.,
            spawn=sim.PinholeCameraCfg(focal_length=24.,horizontal_aperture=24*width/k[0,0],vertical_aperture=24*height/k[1,1],horizontal_aperture_offset=24*(width/2-k[0,2])/k[0,0],vertical_aperture_offset=24*(k[1,2]-height/2)/k[1,1],clipping_range=(.001,100.),f_stop=0.),
            offset=CameraCfg.OffsetCfg(pos=pos,rot=rot,convention='ros')))
    # Inspection-only camera: same scene/runtime, outside policy observation space.
    from ...workflow.presentation import current
    presentation = current()
    if presentation and presentation.config['stage'] in ('annotate', 'datagen', 'sim_eval'):
        from ...workflow.presentation import framing
        overview_pos, overview_rot = framing(w)
        scene.camera_overview = CameraCfg(prim_path='{ENV_REGEX_NS}/Camera_Overview', width=800, height=600,
            data_types=['rgb'], update_period=.1,
            spawn=sim.PinholeCameraCfg(focal_length=18., horizontal_aperture=36., vertical_aperture=27., clipping_range=(.01, 100.)),
            offset=CameraCfg.OffsetCfg(pos=overview_pos, rot=overview_rot, convention='ros'))
    cfg.actions=ActionsCfg();cfg.observations=ObservationsCfg();cfg.observations.task=TaskObservations(concatenate_terms=False)
    if task.get('mode') == 'geometry_review':
        cfg.observations.task = None
    from isaaclab.managers import SceneEntityCfg
    image_template = copy.deepcopy(cfg.observations.policy.side_cam)
    cfg.observations.policy.side_cam = None; cfg.observations.policy.wrist_cam = None
    for view in task.get('observation_cameras', list(p['cameras'])):
        term = copy.deepcopy(image_template); term.params['sensor_cfg'] = SceneEntityCfg('camera_'+view+'view')
        setattr(cfg.observations.policy, view+'_cam', term)

    if presentation and presentation.config['stage'] == 'sim_eval':
        from ...tasks.manager_based.soarm101_lab.so101_mimic_env_cfg import SO101PickPlaceMimicEnvCfg
        from isaaclab.managers import SceneEntityCfg
        subtask = copy.deepcopy(SO101PickPlaceMimicEnvCfg().observations.subtask_terms)
        original_grasp = subtask.grasp.func
        from functools import wraps
        @wraps(original_grasp)
        def selected_grasp(env, **params):
            active = getattr(env, '_workflow_task_workspace', w)
            params['object_cfg'] = SceneEntityCfg(active['task']['pick'])
            return original_grasp(env, **params)
        subtask.grasp.func = selected_grasp
        subtask.grasp.params['object_cfg'] = SceneEntityCfg(task['pick'])
        cfg.observations.subtask_terms = subtask
    cfg.events=Events();cfg.decimation=2;cfg.sim.dt=1/60;cfg.sim.render_interval=2;cfg.sim.device=device
    if w['manifest'].get('builder'):
        # Keep commands/cameras at 30 Hz; resolve fingertip contact at 240 Hz.
        cfg.sim.dt = 1 / 240
        cfg.decimation = cfg.sim.render_interval = 8
    cfg.workflow_workspace=w
    cfg.workspace_metadata=metadata(w);cfg.workspace_package=str(root)
    return cfg
