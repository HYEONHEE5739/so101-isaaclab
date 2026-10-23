"""Worker-owned engine construction. Engine imports remain lazy."""


def create_isaac(workspace: dict, scene: dict, coordinates: dict, *,
                 control_hz: float, device: str = "cuda:0", num_envs: int = 1,
                 randomization_seed: int = 1, overview_camera: bool = False):
    """Create and reset a recorded-scene environment after AppLauncher startup.

    Workspace is already loaded/validated by the worker. Inputs are borrowed;
    configured() owns the derived scene copy. The returned environment belongs
    to the worker, which must close it before closing AppLauncher. Reset errors
    close the newly created environment and propagate. No Dataset/UI imports.
    overview_camera requests the existing inspection feed, never a policy input.
    """
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensorCfg
    from ..real2sim.workspaces.environment import make_config
    from ..real2sim.workspaces.tasks import configured
    w=configured(workspace,scene['task_definition'],scene['active_object_ids'],backend='isaac',reset=scene['reset'],
                 **({'scene_objects':scene['object_ids']} if 'object_ids' in scene else {}))
    cfg=make_config(w,device,num_envs,coordinates=coordinates,overview_camera=overview_camera)
    cfg.sim.dt = 1 / (control_hz * cfg.decimation)
    cfg.events.reset_episode.params['randomization_seed']=randomization_seed
    for name,link in [('grasp_fixed_contact','gripper'),('grasp_moving_contact','jaw')]:
        setattr(cfg.scene,name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+link,update_period=0.,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/'+w['task']['pick']]))
    native=ManagerBasedEnv(cfg=cfg)
    try:
        # Construction starts PhysX, but task reset events have not initialized
        # the vector slots yet. Replay subsequently restores only its own slot.
        native.reset()
    except BaseException:
        native.close();raise
    return native


def create_mujoco(workspace: dict, *, renderer='egl', control_hz=30,
                  width=640, height=480, closed_rad=None, open_rad=None,
                  measured_gripper_limits=None, task_definition=None,
                  full_workspace=True, viewer=False, physics_profile='isaac_matched',
                  simulation_configuration=None, active_object_ids=None):
    """Construct a worker-owned MuJoCo policy environment from explicit inputs.

    No Request, Dataset, Resume, calibration selection or Workspace loading here.
    PolicyEnvironment owns its Backend and closes it on post-construction failure;
    the worker owns normal close. Engine/GL imports remain construction-only.
    """
    from .mujoco_backend import PolicyEnvironment
    return PolicyEnvironment(workspace=workspace,renderer=renderer,control_hz=control_hz,
        width=width,height=height,closed_rad=closed_rad,open_rad=open_rad,
        measured_gripper_limits=measured_gripper_limits,task_definition=task_definition,
        full_workspace=full_workspace,viewer=viewer,physics_profile=physics_profile,
        simulation_configuration=simulation_configuration,active_object_ids=active_object_ids)
