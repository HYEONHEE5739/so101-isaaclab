"""Measured static USD asset plus opt-in Isaac Lab config wrapper; no physics ID."""

import copy
import json
from pathlib import Path
import math
import numpy as np
from .profile import validate, transform, pose, render_k, digest, scene_objects


def usd_xform(stage, path, t):
    from pxr import UsdGeom, Gf

    x = UsdGeom.Xform.Define(stage, path)
    x.MakeMatrixXform().Set(Gf.Matrix4d(transform(t).T.tolist()))
    return x


def build_usd(path, p, visual_only=False):
    from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

    validate(p)
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new export filename: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Real2Sim")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root.GetPrim().SetCustomDataByKey("profile_sha256", digest(p))
    usd_xform(stage, "/Real2Sim/Workspace", p["workspace"]["T_world"])

    def primitive(path, o):
        usd_xform(stage, path, o["T_workspace"])
        stage.GetPrimAtPath(path).SetCustomDataByKey("provenance_json", json.dumps(o["provenance"], sort_keys=True))
        if o["shape"] == "cup_proxy":
            stage.GetPrimAtPath(path).SetCustomDataByKey("render_approximation", o["approximation"])
        dims = o["dimensions_m"]
        app = o["appearance"]
        mat = UsdShade.Material.Define(stage, path + "/Material")
        shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*app["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(app["roughness"])
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        def bind(shape):
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
            if not visual_only:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)

        if o["shape"] == "box":
            s = UsdGeom.Cube.Define(stage, path + "/Shape")
            s.CreateSizeAttr(1.0)
            s.AddScaleOp().Set(Gf.Vec3f(*dims))
            bind(s)
        elif o["shape"] == "cylinder":
            s = UsdGeom.Cylinder.Define(stage, path + "/Shape")
            s.CreateRadiusAttr(dims[0] / 2)
            s.CreateHeightAttr(dims[2])
            s.CreateAxisAttr("Z")
            bind(s)
        elif "bottom_diameter_m" in o:
            from .visuals import tapered_cup
            tapered_cup(stage, path, o, mat)
        else:
            # Open container: solid base plus convex wall segments, never a solid capped cylinder.
            radius = dims[0] / 2
            wall = o["wall_m"]
            height = dims[2]
            s = UsdGeom.Cylinder.Define(stage, path + "/Bottom")
            s.CreateRadiusAttr(radius)
            s.CreateHeightAttr(wall)
            s.AddTranslateOp().Set(Gf.Vec3d(0, 0, -height / 2 + wall / 2))
            bind(s)
            count = 48
            for i in range(count):
                angle = i * 2 * math.pi / count
                mid = radius - wall / 2
                s = UsdGeom.Cube.Define(stage, path + f"/Wall_{i:02d}")
                s.CreateSizeAttr(1.0)
                s.AddTranslateOp().Set(Gf.Vec3d(mid * math.cos(angle), mid * math.sin(angle), 0))
                s.AddRotateZOp().Set(math.degrees(angle))
                s.AddScaleOp().Set(Gf.Vec3f(wall, 2 * radius * math.tan(math.pi / count), height))
                bind(s)

    for name, o in scene_objects(p).items():
        primitive("/Real2Sim/Workspace/" + name, o)
    # Camera/robot transforms are metadata carried by the companion profile, not duplicated robots.
    root.GetPrim().SetCustomDataByKey("profile_json", json.dumps(p, sort_keys=True))
    stage.GetRootLayer().Save()
    return str(path.resolve())


def configure_environment(cfg, profile, usd_path, calibration=False):
    """Call before ManagerBasedEnv construction. Training defaults remain opt-in."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim
    from .profile import load

    p = load(profile) if isinstance(profile, (str, Path)) else validate(profile)
    scene = cfg.scene
    if calibration:
        if scene.num_envs != 1:
            raise ValueError("Real2Sim runtime requires one environment")
        for n in [
            "table",
            "ground",
            "pick_zone_top",
            "pick_zone_bottom",
            "pick_zone_left",
            "pick_zone_right",
            "bin_a",
            "bin_b",
            "cube_red",
            "cube_green",
            "cube_blue",
        ]:
            setattr(scene, n, None)
        cfg.events.reset_episode = None
    else:
        # Keep task assets and reset semantics; replace only the old static table.
        scene.table = None
    scene.real2sim_workspace = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Real2Sim", spawn=sim.UsdFileCfg(usd_path=str(Path(usd_path).resolve()))
    )
    if p.get("wrist_mount"):
        mount_profile = copy.deepcopy(p)
        mount = mount_profile.pop("wrist_mount")
        mount_profile["objects"] = mount["parts"]
        mount_profile["relations"] = {}
        mount_profile["table"]["render_enabled"] = False
        mount_profile["workspace"]["T_world"] = np.eye(4).tolist()
        mount_usd = build_usd(Path(usd_path).with_name("wrist_mount.usda"), mount_profile, visual_only=True)
        scene.real2sim_wrist_mount = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/Real2SimMount",
            spawn=sim.UsdFileCfg(usd_path=mount_usd))
    scene.robot.init_state.pos, scene.robot.init_state.rot = pose(p["robot_base"]["T_world"])
    scene.dome_light.spawn.intensity = p["appearance"]["light_intensity"]
    for view in ("side", "wrist"):
        c = p["cameras"][view]
        w, h = c["resolution"]
        k = render_k(c)
        focal = 24.0
        pos, rot = pose(c["T_parent_camera"])
        setattr(
            scene,
            "camera_" + view + "view",
            CameraCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Camera_R2S_Side"
                    if view == "side"
                    else "{ENV_REGEX_NS}/Robot/gripper/Camera_R2S_Wrist"
                ),
                width=w,
                height=h,
                data_types=["rgb"],
                update_period=0.0,
                # Wrist follows an articulation; side pose may change when applying a profile.
                # Otherwise CameraData keeps the initialization pose while RGB continues moving.
                update_latest_camera_pose=True,
                spawn=sim.PinholeCameraCfg(
                    focal_length=focal,
                    horizontal_aperture=focal * w / k[0, 0],
                    vertical_aperture=focal * h / k[1, 1],
                    clipping_range=(0.001, 100.0),
                    f_stop=0.0,
                ),
                offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
            ),
        )
    return p


def camera_prim(stage, sensor):
    """Use initialized camera paths, not the config's environment regex."""
    from pxr import UsdGeom

    view = getattr(sensor, "_view", None)
    paths = list(view.prim_paths) if view is not None else []
    if len(paths) != 1:
        raise RuntimeError(f"Real2Sim requires one initialized camera prim; resolved paths: {paths}")
    prim = stage.GetPrimAtPath(paths[0])
    if not prim or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera prim missing from current stage: {paths[0]}")
    return prim


def apply_robot_appearance(stage, p):
    """Author an opt-in local override; never edit the referenced robot USD."""
    from pxr import Gf

    path = "/World/envs/env_0/Robot/Looks/material_a_d_printed/Shader"
    prim = stage.GetPrimAtPath(path)
    color = p["appearance"].get("robot_print_color")
    if not prim:
        if color is not None:
            raise RuntimeError("Robot printed material missing: " + path)
        return
    attr = prim.GetAttribute("inputs:diffuse_color_constant")
    if color is not None:
        if not attr:
            raise RuntimeError("Robot printed material has no diffuse color input")
        attr.Set(Gf.Vec3f(*color))
    elif attr:
        # Clear only this stage's local opinion; reveal the source asset color.
        attr.Clear()


def apply_profile(env, p):
    """Update poses/K/appearance in a stopped calibration scene. Topology is immutable."""
    import torch
    from pxr import UsdGeom, UsdShade, Gf
    from isaaclab.sim import get_current_stage

    validate(p)
    stage = get_current_stage()
    apply_robot_appearance(stage, p)
    # Check both cameras before changing any scene or robot state.
    camera_prims = {}
    for view in ("side", "wrist"):
        sensor = env.scene["camera_" + view + "view"]
        if (sensor.cfg.width, sensor.cfg.height) != tuple(p["cameras"][view]["resolution"]):
            raise ValueError("Resolution changed: disconnect/reconnect scene")
        camera_prims[view] = camera_prim(stage, sensor)
    base = "/World/envs/env_0/Real2Sim/Workspace"
    for path, t in [(base, p["workspace"]["T_world"])] + [
        (base + "/" + n, o["T_workspace"]) for n, o in scene_objects(p).items()
    ]:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError("Expected generated scene prim: " + path)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(transform(t).T.tolist()))
    for n, o in scene_objects(p).items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(base + "/" + n + "/Material/Shader"))
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*o["appearance"]["color"]))
        shader.GetInput("roughness").Set(o["appearance"]["roughness"])
    light = stage.GetPrimAtPath("/World/Light")
    a = light.GetAttribute("inputs:intensity")
    if a:
        a.Set(p["appearance"]["light_intensity"])
    robot = env.scene["robot"]
    pos, q = pose(p["robot_base"]["T_world"])
    rp = torch.tensor([[*pos, *q]], device=env.device, dtype=torch.float32)
    robot.write_root_pose_to_sim(rp)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    # Local mounting link transform persists as robot moves.
    for view in ("side", "wrist"):
        sensor = env.scene["camera_" + view + "view"]
        c = p["cameras"][view]
        w, h = c["resolution"]
        if (sensor.cfg.width, sensor.cfg.height) != (w, h):
            raise ValueError("Resolution changed: disconnect/reconnect scene")
        sensor.set_intrinsic_matrices(torch.tensor(render_k(c)[None], device=env.device, dtype=torch.float32))
        t = transform(c["T_parent_camera"]) @ np.diag([1, -1, -1, 1])
        UsdGeom.Xformable(camera_prims[view]).MakeMatrixXform().Set(Gf.Matrix4d(t.T.tolist()))
    env.sim.forward()
    env.scene.update(0.0)
