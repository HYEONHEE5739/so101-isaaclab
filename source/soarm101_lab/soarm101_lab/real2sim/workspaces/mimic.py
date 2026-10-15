"""Workspace config adapter for the existing SO101 Mimic implementation."""
import copy
import json
from .package import load, metadata
from .environment import make_config, task_success
from .demo import verify_dataset, coordinate_contract


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


def validate_input(path, workspace, annotated=False):
    import h5py
    w, args = verify_dataset(path, workspace)
    if args.get('source_partition'):
        from ...workflow.episodes import source_catalog
        source_catalog(path) # The actual file read by Mimic must contain only its declared Source pool.
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


def preserve_metadata(source, output, workspace, generated=False, expected_source_sha256=None, annotation_sources=None):
    """After the writer closes, retain original provenance and record the new stage."""
    import h5py
    from ...workflow.artifacts import fingerprint as artifact_hash
    if expected_source_sha256 and artifact_hash(source)!=expected_source_sha256:raise ValueError('Mimic Source input changed during generation')
    w, args = verify_dataset(source, workspace)
    from ...workflow.episodes import source_catalog,validate_partition
    partition=validate_partition(args['source_partition']) if args.get('source_partition') else None
    records,_=source_catalog(source,allow_unknown=not bool(partition))
    annotation_map={}
    if not generated and (partition or annotation_sources is not None):
        # The recorder renumbers successful exports after failed/skipped input demos.
        with h5py.File(source,'r') as original,h5py.File(output,'r') as completed:
            exported=sorted((n for n in completed['data'] if n.startswith('demo_')),key=lambda n:int(n[5:]))
            if (annotation_sources is None or len(annotation_sources)!=len(exported)
                    or len(set(annotation_sources))!=len(annotation_sources)
                    or not set(annotation_sources).issubset(original['data'])):
                raise ValueError('Annotation Source export order/count is unverifiable')
            annotation_map=dict(zip(exported,annotation_sources))
    with h5py.File(output, 'a') as h:
        group = h['data']
        updated = json.loads(group.attrs.get('env_args', '{}'))
        from ...representation import describe
        coordinates = args.get('so101_coordinates', coordinate_contract(w))
        from ..precision import simulation_configuration
        updated['simulation_configuration']=simulation_configuration(w,coordinates=coordinates,backend='isaac')
        updated.update(representation=describe(coordinates), workspace=metadata(w), workspace_path=str(w['root']),
                       so101_coordinates=coordinates, scene_configuration=w['scene_configuration'],
                       grasp_object=w['task']['pick'], place_bin=w['task']['place'],
                       source_dataset=str(source), source_env_args=args)
        if partition:
            from ...workflow.artifacts import fingerprint
            updated['source_partition']=partition
            updated['source_partition_input']={'path':str(source),'sha256':fingerprint(source),
                'source_ids':sorted({r['source_id'] for r in records.values()}),
                'verification':'source_only_input','lineage_mode':'verified_partition','exact_subtask_sources':'UNKNOWN'}
            from ..precision import fingerprint as contract_hash
            updated['source_partition_input']['evidence_sha256']=contract_hash(updated['source_partition_input'])
        from ...tasks.manager_based.soarm101_lab.mdp.so101_mimic_recorders import telemetry_metadata
        telemetry=telemetry_metadata(w['task']['dynamic'])
        episodes=[g for n,g in group.items() if n.startswith('demo_')]
        telemetry['object_telemetry']={n:spec for n,spec in telemetry['object_telemetry'].items() if episodes and all(spec['path'] in g for g in episodes)}
        if not episodes or not all('obs/timestamp' in g for g in episodes):
            telemetry.pop('observation_timestamp_path')
            updated.pop('observation_timestamp_path',None)
        updated.update(telemetry)
        updated['success_source']='simulation_success_term' if generated else args.get('success_source','unknown')
        from ...workflow.curation import _group_id
        with h5py.File(source,'r') as original:
            for name,episode in group.items():
                if not name.startswith('demo_'):continue
                source_name=annotation_map.get(name,name)
                if name in annotation_map:episode.attrs['annotation_source_episode']=source_name
                if partition:
                    episode.attrs['split']=partition['split']
                    lineage={'kind':'partition','partition_sha256':partition['sha256'],'exact_source_ids':None,'detail':'UNKNOWN: Mimic can recombine subtasks'} if generated else records.get(source_name)
                    if lineage is None:raise ValueError('Annotation episode identity cannot be traced to Source')
                    episode.attrs['source_lineage']=json.dumps(lineage)
                    if generated:episode.attrs['source_group']='UNKNOWN:'+str(source)
                elif not generated and source_name in records:episode.attrs['source_lineage']=json.dumps(records[source_name])
                if 'source_group' in episode.attrs:continue
                if not generated and source_name in original['data']:
                    episode.attrs['source_group']=_group_id(source,source_name,original['data'][source_name].attrs,args)
                else:episode.attrs['source_group']='UNKNOWN:'+str(source)
        try:
            from isaaclab.utils.datasets.hdf5_dataset_file_handler import DATASET_FORMAT_VERSION
        except ImportError:
            DATASET_FORMAT_VERSION = 0
        updated['root_quaternion_order'] = 'xyzw' if DATASET_FORMAT_VERSION >= 1 else 'wxyz'
        group.attrs['env_args'] = json.dumps(updated)
    from ...workflow.tasks import configured, stamp_file
    stamp_file(output, configured(w)['task_definition'])
