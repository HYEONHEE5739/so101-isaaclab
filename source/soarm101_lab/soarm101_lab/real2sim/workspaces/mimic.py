"""Developer physics comparison against the fixed-scene native SO101 Mimic config.

Production Workspace generation uses workflow.mimic and workflow.datagen.
"""
import copy
from .package import load
from .environment import make_config, task_success


def make_mimic_config(workspace, device='cuda:0', num_envs=1, annotation=False, randomization_seed=None, coordinates=None):
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from ...tasks.manager_based.soarm101_lab.so101_mimic_env_cfg import SO101PickPlaceMimicEnvCfg
    from ...workflow.tasks import configured
    w = configured(workspace if isinstance(workspace,dict) else load(workspace))
    base = make_config(w, device, num_envs, coordinates=coordinates)
    from isaaclab.sensors import ContactSensorCfg
    # One sensor per rigid body: Isaac filtered contacts are one-to-many.
    for sensor, link in [('grasp_fixed_contact', 'gripper'), ('grasp_moving_contact', 'jaw')]:
        setattr(base.scene, sensor, ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/' + link, update_period=0.0,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/' + w['task']['pick']]))
    cfg = SO101PickPlaceMimicEnvCfg()
    # Replace the complete scene/reset; never translate legacy world positions.
    for name in ('scene', 'events', 'sim', 'decimation', 'workspace_metadata', 'workspace_package', 'workflow_workspace','simulation_configuration'):
        setattr(cfg, name, copy.deepcopy(getattr(base, name)))
    if randomization_seed is not None:
        cfg.events.reset_episode.params['randomization_seed'] = randomization_seed
    cfg.seed = w['task']['reset']['seed'] if randomization_seed is None else randomization_seed
    if annotation:
        from ...tasks.manager_based.soarm101_lab.mdp.so101_ik_actions import SO101PinocchioIKAction
        class RecordedJointAction(SO101PinocchioIKAction):
            # Annotation replays recorded model-space targets, not another IK reconstruction.
            def apply_actions(self):
                target = getattr(self._env, '_workspace_recorded_arm_target', None)
                if target is None:
                    raise RuntimeError('Workspace annotation requires recorded joint_targets')
                self._joint_position_targets[:] = target
                self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        cfg.actions.arm.class_type = RecordedJointAction
    urdf = w['root'] / base.simulation_configuration['robot_model']['urdf']
    cfg.actions.arm.urdf_path = str(urdf)
    cfg.observations.policy = copy.deepcopy(base.observations.policy)
    from .grasp import object_stably_grasped
    cfg.observations.subtask_terms.grasp.func = object_stably_grasped
    cfg.observations.subtask_terms.grasp.params['object_cfg'] = SceneEntityCfg(w['task']['pick'])
    cfg.observations.task_state.place_success.func = task_success
    cfg.observations.task_state.place_success.params = {'workspace': w}
    cfg.terminations.success = TerminationTermCfg(func=task_success, params={'workspace': w})
    cfg.subtask_configs['tool0'][0].object_ref = w['task']['pick']
    cfg.subtask_configs['tool0'][1].object_ref = w['task']['place']
    cfg.datagen_config.name = w['manifest']['workspace_id']
    cfg.datagen_config.seed = cfg.seed
    cfg.datagen_config.generation_guarantee = True  # Stop at the requested number of successful demonstrations.
    cfg.env_name = 'SO101-Workspace-Mimic-v1'
    return cfg
