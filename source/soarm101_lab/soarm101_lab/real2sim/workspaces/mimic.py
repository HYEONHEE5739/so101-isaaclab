"""Workspace config adapter for the existing SO101 Mimic implementation."""
import copy
import json
from .package import load, metadata
from .environment import make_config, task_success
from .demo import verify_dataset, coordinate_contract


def make_mimic_config(workspace, device='cuda:0', num_envs=1, annotation=False):
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from ...tasks.manager_based.soarm101_lab.so101_mimic_env_cfg import SO101PickPlaceMimicEnvCfg
    from ...workflow.tasks import configured
    w = configured(load(workspace))
    base = make_config(workspace, device, num_envs)
    cfg = SO101PickPlaceMimicEnvCfg()
    # Replace the complete scene/reset; never translate legacy world positions.
    for name in ('scene', 'events', 'sim', 'decimation', 'workspace_metadata', 'workspace_package', 'workflow_workspace'):
        setattr(cfg, name, copy.deepcopy(getattr(base, name)))
    cfg.seed = w['task']['reset']['seed']
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
    cfg.actions.arm.urdf_path = str(w['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf')
    cfg.observations.policy = copy.deepcopy(base.observations.policy)
    cfg.observations.subtask_terms.grasp.params['object_cfg'] = SceneEntityCfg(w['task']['pick'])
    cfg.observations.task_state.place_success.func = task_success
    cfg.observations.task_state.place_success.params = {'workspace': w}
    cfg.terminations.success = TerminationTermCfg(func=task_success, params={'workspace': w})
    cfg.subtask_configs['tool0'][0].object_ref = w['task']['pick']
    cfg.subtask_configs['tool0'][1].object_ref = w['task']['place']
    cfg.datagen_config.name = w['manifest']['workspace_id']
    cfg.datagen_config.seed = w['task']['reset']['seed']
    cfg.datagen_config.generation_guarantee = False  # explicit finite number of attempts
    cfg.env_name = 'SO101-Workspace-Mimic-v1'
    return cfg


def validate_input(path, workspace, annotated=False):
    import h5py
    w, args = verify_dataset(path, workspace)
    from ...workflow.tasks import configured
    w = configured(w)
    if args.get('task_definitions') and args['task_definitions'] != [w['task_definition']]:
        raise ValueError('Input task differs from selected task')
    if args.get('validation_only'):
        raise ValueError('Synthetic smoke demo is not a manipulation demonstration')
    from ...representation import require_compatible
    from ...so101_dataset_contract import read_contract, MIMIC_SPACE
    require_compatible(read_contract(path), coordinate_contract(w), MIMIC_SPACE)
    with h5py.File(path, 'r') as h:
        episodes = [g for n, g in h['data'].items() if n.startswith('demo_')]
        if not episodes:
            raise ValueError('No demonstrations')
        if annotated and any('obs/datagen_info/subtask_term_signals/grasp' not in g for g in episodes):
            raise ValueError('Mimic annotations missing')
    return w, args


def preserve_metadata(source, output, workspace):
    """After the writer closes, retain original provenance and record the new stage."""
    import h5py
    w, args = verify_dataset(source, workspace)
    with h5py.File(output, 'a') as h:
        group = h['data']
        updated = json.loads(group.attrs.get('env_args', '{}'))
        from ...representation import describe
        updated.update(representation=describe(coordinate_contract(w)), workspace=metadata(w), workspace_path=str(w['root']),
                       so101_coordinates=coordinate_contract(w),
                       grasp_object=w['task']['pick'], place_bin=w['task']['place'],
                       source_dataset=str(source), source_env_args=args)
        try:
            from isaaclab.utils.datasets.hdf5_dataset_file_handler import DATASET_FORMAT_VERSION
        except ImportError:
            DATASET_FORMAT_VERSION = 0
        updated['root_quaternion_order'] = 'xyzw' if DATASET_FORMAT_VERSION >= 1 else 'wxyz'
        group.attrs['env_args'] = json.dumps(updated)
    from ...workflow.tasks import configured, stamp_file
    stamp_file(output, configured(w)['task_definition'])
