"""Shared MuJoCo worker request contract, independent of GUI and engine execution.

Planner and worker validate the same payload. Keep validation order, messages and
option ownership stable: existing callers also import this interface from operations.
"""
import json
from pathlib import Path

from ..learning.minivla.config import MODEL_ROOT
from ..real2sim.workspaces.package import load


MUJOCO_STAGES = dict(environment='Environment', source='Source',
              convert='Convert', import_='Import', annotate='Annotation',
              datagen='Datagen', rollout='Evaluation')
MUJOCO_STAGES['import'] = MUJOCO_STAGES.pop('import_')

EPISODE_STAGES = frozenset({'annotate'})


def episode_context(path, workspace, stage, options=None):
    """Read native input evidence in a worker, without registration or mutation.

    This checks headers/Scene/Coordinate compatibility. Full arrays/video
    validation remains in unified.execute before engine creation. The returned
    reader has context-managed per-read handles; no engine or writer is owned.
    """
    if stage not in EPISODE_STAGES and stage!='datagen':raise ValueError('Not a Path episode stage')
    from .paths import input_format
    from .dataset import EpisodeDataset
    from .tasks import recorded_scene
    from ..real2sim.workspaces.package import metadata
    from ..real2sim.workspaces.tasks import validate
    from ..real2sim.workspaces.demo import coordinate_contract
    from ..representation import require_compatible
    from ..so101_dataset_contract import MIMIC_SPACE
    path=Path(path).expanduser().resolve()
    if input_format(path) not in ('hdf5','canonical'):raise ValueError('Select a committed Common/Legacy demonstration Dataset')
    if any(p.with_name('.'+p.name+'.workflow-lock').exists() for p in (path,*path.parents) if p.name):
        raise ValueError('Input Dataset is still being written')
    reader=EpisodeDataset(path)
    args=reader.args
    provenance=reader.provenance
    if not reader.names:raise ValueError('No committed demonstration episodes')
    if args.get('validation_only'):raise ValueError('Smoke data cannot enter the workflow')
    if args.get('dataset_schema') not in (None,'so101.common-episode/2'):raise ValueError('Unknown episode dataset schema')
    if stage=='annotate' and (args.get('evaluation_schema') or args.get('generation_provenance') or
            args.get('generation_method') or args.get('source_dataset') or provenance.get('generation_method') or provenance.get('source_dataset')):
        raise ValueError('Annotation requires original demonstrations, not generated/evaluation data')
    w=load(workspace or args.get('workspace_path',''))
    if args.get('workspace')!=metadata(w):raise ValueError('Input Workspace identity differs')
    expected=coordinate_contract(w)
    # Historical writers used a different human-readable asset scope sentence.
    # File identities/digest, mapping and every numeric contract remain exact.
    actual_asset={k:v for k,v in (reader.contract.get('robot_asset') or {}).items() if k!='scope'}
    expected_asset={k:v for k,v in expected['robot_asset'].items() if k!='scope'}
    require_compatible({**reader.contract,'robot_asset':actual_asset},{**expected,'robot_asset':expected_asset},MIMIC_SPACE)
    if args.get('so101_coordinates') is not None and args['so101_coordinates']!=reader.contract:
        raise ValueError('Embedded/sidecar coordinate contracts differ')
    scene=recorded_scene(args,w)
    if w['task'].get('mode')=='geometry_review' and not scene.get('scene_only'):raise ValueError('Explicit workflow task metadata is required')
    definition=validate(scene['task_definition'],w) if scene['task_definition'] is not None else None
    if args.get('task_definitions') and args['task_definitions']!=[definition]:raise ValueError('Select one homogeneous recorded Task')
    # Inspect only episode headers here. Legacy entries are already in manifest;
    # calling legacy episode_attributes would perform FK/array loading.
    if reader.legacy:
        headers=reader.manifest['episodes']
    else:
        import h5py
        with h5py.File(path,'r') as h:
            if stage=='annotate' and h['data'].attrs.get('generation_provenance'):
                raise ValueError('Annotation requires original demonstrations')
            for name in reader.names:
                group=h['data'][name]
                state='obs/policy/joint_pos' if 'obs/policy/joint_pos' in group else 'obs/joint_pos'
                if ('actions' not in group or not len(group['actions']) or 'joint_targets' not in group or state not in group):
                    raise ValueError('Nonempty actions and measured joint observations/targets are required')
                if group['joint_targets'].shape[-1]!=6 or group[state].shape[-1]!=6:raise ValueError('Expected six model joints')
                if 'committed' in group.attrs and not group.attrs['committed']:raise ValueError('Episode is not committed')
            headers=[dict(h['data'][name].attrs) for name in reader.names]
    for entry in headers:
        if entry.get('validation_only'):raise ValueError('Smoke episodes cannot enter the workflow')
        for key,expected in (('task_definition',definition),('scene_configuration',scene)):
            value=entry.get(key)
            if isinstance(value,(str,bytes)):value=json.loads(value)
            if value is not None and value!=expected:raise ValueError('Mixed or inconsistent recorded '+key)
    opts=dict(options or {})
    if opts.get('tasks') is not None and opts['tasks']!=([definition] if definition else []):raise ValueError('Derived stages must inherit recorded tasks')
    if 'active_object_ids' in opts and set(opts['active_object_ids'])!=set(scene['active_object_ids']):
        raise ValueError('Derived stages must inherit the recorded active object set')
    from .task_scripts import followup_context
    contract=followup_context(reader,w,scene,opts,stage=stage)
    return {'reader':reader,'workspace':w,'scene':scene,'tasks':[definition] if definition else [],'followup':contract}

def mujoco_request(stage, python, workspace, input_path, config, checkpoint, renderer, options):
    if stage not in MUJOCO_STAGES:
        raise ValueError('Unsupported workflow worker stage')
    if not isinstance(options, dict):
        raise ValueError('Options must be a JSON object')
    if not Path(python).is_file():
        raise ValueError('Worker Python executable missing')
    if renderer not in ('egl', 'software-egl', 'osmesa', 'none'):
        raise ValueError('Unknown renderer')
    if renderer == 'none' and stage in ('source', 'rollout'):
        raise ValueError('This stage requires RGB rendering')
    if stage in ('environment', 'source', 'rollout', 'import'):
        load(workspace)
    if stage in ('convert', 'annotate', 'datagen'):
        if not (Path(input_path) / 'manifest.json').is_file():
            raise ValueError('Input must be a canonical MiniVLA dataset directory')
    if stage == 'import' and not Path(input_path).is_file():
        raise ValueError('HDF5 import requires an explicit recipe JSON')
    if stage == 'source' and (int(options.get('episodes', 1)) < 1 or float(options.get('episode_time_s', 20)) <= 0 or float(options.get('reset_time_s', 5)) < 0):
        raise ValueError('Invalid source count/timing')
    if stage == 'source' and not options.get('port'):
        raise ValueError('Leader port required; no automatic hardware discovery')
    if stage == 'convert' and not options.get('repo_id'):
        raise ValueError('Dataset repo_id required')
    if stage == 'rollout' and not Path(checkpoint).exists():
        raise ValueError('A trained policy checkpoint is required')
    if stage=='rollout':
        from .evaluation import validate_evaluation_options
        validate_evaluation_options(options)
    return {'stage': stage, 'workspace': workspace, 'input': input_path, 'config': config,
            'checkpoint': checkpoint, 'renderer': renderer, 'options': options}


def prepare_mujoco(request: dict, *, coordinates=None, hz=None, workspace=None) -> dict:
    """Resolve worker policy into explicit engine inputs; no engine construction.

    workspace may be the already validated original Workspace. Scene/reset
    selection uses a configured copy, retaining the same compiler inputs as the
    old path loader. Dataset/Resume/Calibration checks remain above Simulation.
    Returned numeric configuration is borrowed synchronously by the constructor.
    """
    from .dataset import recording_provenance
    from ..simulation.mjcf import RGB_WIDTH, RGB_HEIGHT
    options = request['options']
    from ..real2sim.precision import simulation_configuration,calibration_status
    recorded_contract = None
    source = options.get('resume_data') or request.get('input') or request.get('checkpoint')
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file() and path.suffix in ('.hdf5','.h5'):
            from ..so101_dataset_contract import read_contract
            recorded_contract=read_contract(path)
        elif (path/'project_provenance.json').is_file():
            recorded_contract = recording_provenance(path).get('coordinate_contract')
        elif (path/'so101_coordinates.json').is_file():
            from ..so101_dataset_contract import read_contract
            recorded_contract = read_contract(path)
    base_workspace=load(request['workspace']) if workspace is None else workspace
    w=base_workspace
    from ..real2sim.workspaces.tasks import configured
    w=configured(w,options.get('task_definition'),options.get('active_object_ids'),backend='mujoco',scene_objects=options.get('scene_objects',options.get('script_scenes',[None])[0]),
                 scene_only=options.get('scene_only',False),object_configuration=(options.get('script_scene') or {}).get('objects'))
    if request.get('stage')=='source' and not options.get('scene_only'):
        from ..real2sim.workspaces.reset import source_workspace
        w=source_workspace(w,options.get('resume_data'))
    elif recorded_contract and recorded_contract.get('simulation_model'):
        recorded_scene=recorded_contract['simulation_model'].get('scene_configuration')
        if recorded_scene is not None:
            if not isinstance(recorded_scene,dict):raise ValueError('Recorded simulation scene_configuration must be an object or null')
            w=configured(w,reset=recorded_scene.get('reset'),scene_objects=options.get('scene_objects',recorded_scene.get('object_ids')))
    if options.get('scene_reset') is not None:
        w=configured(w,reset=options['scene_reset'])
    simulation=simulation_configuration(w,coordinates=recorded_contract,
        mappings=options.get('_collection_mappings'),identity=options.get('identity_sim_for_validation',False),
        backend='mujoco',scene_configuration=w['scene_configuration'],hardware_role='leader' if '_collection_mappings' in options else None)
    print('[WORKFLOW] '+calibration_status(simulation),flush=True)
    coords = coordinates or {}
    physics_profile = options.get('physics_profile', 'isaac_matched')
    if options.get('resume_data'):
        previous_scene = recording_provenance(options['resume_data'])['scene']
        physics_profile = previous_scene.get('physics_profile', 'legacy_urdf')
    return dict(workspace=w if options.get('scene_only') else base_workspace, renderer=request['renderer'],
                              control_hz=hz or options.get('control_hz', 30),
                              width=options.get('width', RGB_WIDTH), height=options.get('height', RGB_HEIGHT),
                              closed_rad=coords.get('gripper_closed_rad', options.get('gripper_closed_rad')),
                              open_rad=coords.get('gripper_open_rad', options.get('gripper_open_rad')),
                              measured_gripper_limits=coords.get('measured_gripper_limits_rad'),
                              task_definition=options.get('task_definition'),
                              full_workspace=options.get('full_workspace', True),
                              viewer=options.get('mode') == 'developer', physics_profile=physics_profile,
                              simulation_configuration=simulation,
                              active_object_ids=options.get('active_object_ids'))


def stage_input(stage, path, *, full=True):
    """Worker-side native input validation; returns ephemeral format metadata.

    Only Paths cross the process boundary. No IDs, history lookup or persisted
    catalog: Dataset/Policy owners read and validate their own native files.
    """
    if not isinstance(path,(str,Path)) or not str(path):raise ValueError('Select an explicit input Path')
    from .paths import input_format
    from ..learning.datasets import hub_reference
    if hub_reference(str(path)):
        if stage!='train':raise ValueError('Remote Dataset reference supports Training only')
        from ..learning.datasets import inspect_dataset
        return inspect_dataset(path,full=full)
    path=Path(path).expanduser().resolve()
    if any(p.name and p.with_name('.'+p.name+'.workflow-lock').exists() for p in (path,*path.parents)):
        raise ValueError('Input is still being written')
    if stage=='source':
        from ..real2sim.workspaces.package import metadata
        from ..real2sim.workspaces.demo import coordinate_contract
        w=load(path)
        return dict(path=str(w['root']),type='workspace',workspace=metadata(w),coordinates=coordinate_contract(w))
    if stage=='hf_policy' and input_format(path) in ('minivla-checkpoint','minivla-policy'):
        from .policy import inspect_checkpoint
        return inspect_checkpoint(path)
    if stage in ('validate','hf_policy'):
        from ..learning.datasets import inspect_policy
        return inspect_policy(path)
    if stage=='train' or (stage=='hf_dataset' and input_format(path)!='canonical'):
        from ..learning.datasets import inspect_dataset
        return inspect_dataset(path,full=full)
    if stage in ('sim_eval','real_eval'):
        from .policy import inspect_checkpoint
        return inspect_checkpoint(path)
    from .dataset import inspect_dataset,episode_input_kind
    fmt=input_format(path)
    if stage in ('train','merge','hf_dataset'):
        kind='dataset'
        if fmt not in ('lerobot','canonical'):raise ValueError('Select a native LeRobot/Legacy Dataset Path')
    elif stage in ('datagen','convert'):
        if fmt not in ('hdf5','canonical'):raise ValueError('Select an Episode Dataset Path')
        kind=episode_input_kind(path)
        if stage=='datagen' and kind!='annotated':raise ValueError('DataGen requires an annotated demonstration Dataset')
    else:raise ValueError('Unknown Path stage: '+stage)
    result=dict(inspect_dataset(kind,path,full=full),path=str(path),type=kind)
    if stage in ('train','merge','hf_dataset') and fmt=='lerobot':
        from ..so101_dataset_contract import JOINT_SPACE
        if result.get('coordinates',{}).get('action_space')!=JOINT_SPACE:raise ValueError('Dataset requires explicit joint-radian coordinates')
    return result


def validate_task_request(stage, backend, options):
    """Path-only planning/dispatch gate; never execute user code here."""
    if options.get('followup_python'):
        if stage not in ('annotate','datagen') or backend!='mujoco':
            raise ValueError('Followup Python requires MuJoCo Annotation/Mimic')
        from .task_scripts import read_followup
        read_followup(options['followup_python'])
    if not options.get('task_python'):return
    if stage not in ('sim_eval','rollout') or backend!='mujoco':
        raise ValueError('Task Python currently supports MuJoCo Sim Evaluation only')
    if options.get('num_envs',1)!=1:
        raise ValueError('Task Python currently requires num_envs=1')
    if not (options.get('tasks') or options.get('task_definition')):
        raise ValueError('Task Python requires an explicit scene task_definition or tasks selection')
    path=Path(options['task_python']).expanduser()
    if path.suffix!='.py' or not path.is_file():raise ValueError('Select an existing Task Python file')


def load_task_python(options):
    """Worker-only trusted source execution; hash the exact bytes being executed."""
    if not options.get('task_python'):return None,None
    import hashlib
    from ..real2sim.workspaces.tasks import validate_task_module
    path=Path(options['task_python']).expanduser().resolve()
    source=path.read_bytes();digest=hashlib.sha256(source).hexdigest()
    if options.get('task_sha256') and options['task_sha256']!=digest:
        raise ValueError('Task source identity changed since request')
    namespace={'__name__':'__workspace_task__','__file__':str(path)}
    exec(compile(source,str(path),'exec'),namespace)
    validate_task_module(namespace)
    return namespace,{'path':str(path),'sha256':digest,'task_id':namespace['TASK_ID'],
                     'instruction':namespace['INSTRUCTION'],
                     'contracts':namespace.get('TASK_CONTRACTS'),
                     'identity_scope':'entry Python file bytes; imported modules are not pinned'}
