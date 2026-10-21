"""Source metadata, resume policy and save-time simulation handoff.

The existing MuJoCo Source loop owns collection/discard/cancel. Format I/O lives
in data; this module never makes a Writer retain an environment or callback.
"""
import copy
import json
from pathlib import Path
import uuid
import numpy as np
from ..so101_joint_mapping import JOINTS,POLICY_UNITS


def metadata(context:dict,instruction:str)->tuple[dict,dict]:
    manifest=dict(schema_version=1,joint_names=list(JOINTS),units=list(POLICY_UNITS),control_hz=context['control_hz'],
        action_semantics='absolute_target_after_observation',episodes=[],cameras=context['cameras'])
    provenance=dict(schema='so101.minivla-dataset/1',backend='mujoco',workspace=context['scene']['source'],
        scene=context['scene'],coordinates=context['coordinates'],instruction=instruction,source_group_prefix=uuid.uuid4().hex,
        scene_configuration=copy.deepcopy(context['scene'].get('scene_configuration')),workspace_path=context['workspace_path'])
    return manifest,provenance


def resume_legacy(writer,source,workspace)->None:
    """Validate Source/scene/calibration policy before copying immutable entries."""
    from .tasks import recorded_scene
    source=Path(source).resolve();manifest=json.loads((source/'manifest.json').read_text())
    provenance=json.loads((source/'project_provenance.json').read_text());current=writer.provenance
    old=recorded_scene({'provenance':provenance},workspace);scene=current.get('scene_configuration')
    if scene and old['active_object_ids']!=scene['active_object_ids']:raise ValueError('Resume active object set differs')
    if (provenance['workspace']!=current['workspace'] or provenance['scene']['model_sha256']!=current['scene']['model_sha256']
            or manifest['units']!=writer.manifest['units'] or manifest['control_hz']!=writer.manifest['control_hz']
            or any(provenance['coordinates'].get(k)!=current['coordinates'].get(k) for k in ('joint_order','gripper_closed_rad','gripper_open_rad'))
            or provenance['scene'].get('task_definition')!=current['scene'].get('task_definition')):
        raise ValueError('Resume workspace/model/task/coordinates differ')
    if any((provenance.get('coordinate_contract') or {}).get(k)!=(current.get('coordinate_contract') or {}).get(k)
           for k in ('hardware_mapping','simulation_joint_frame','simulation_model')):
        raise ValueError('Resume hardware calibration/collection coordinates differ; use a new dataset')
    writer.copy_episodes(source,manifest)


def resume_common(writer,provenance:dict,source)->None:
    from .dataset import EpisodeDataset
    from .episodes import source_catalog
    source=Path(source).resolve();previous=EpisodeDataset(source)
    if previous.args['workspace']!=provenance['workspace'] or previous.contract!=provenance['coordinate_contract']:
        raise ValueError('Resume Workspace/calibration/model differs')
    if previous.args.get('source_partition'):raise ValueError('Cannot append new Source to an immutable partition')
    records=source_catalog(source,allow_unknown=True)[0]
    for old in previous.names:
        values,attrs,videos=previous.episode(old)
        if not attrs.get('source_lineage') and old in records:attrs['source_lineage']=json.dumps(records[old])
        if not previous.legacy:videos={k:{**v,'file':str(previous.path.parent/v['file'])} for k,v in videos.items()}
        writer.commit(values,attrs,videos)
    provenance['resumed_from']=str(source)


def save_common(writer,recording,env,provenance:dict,outcome:dict)->str:
    """FK remains at save time; preserve historical Common initial-state format."""
    from ..data.episodes import recording_args
    from ..data.recording import source_episode_values
    if not recording.rows:raise ValueError('Cannot save an empty episode')
    writer.args.update(recording_args(provenance,{'control_hz':recording.hz}))
    specs=recording.finish()
    videos={name+'_cam':{**spec,'file':str((recording.video.folder/spec['file']).resolve())} for name,spec in specs.items()}
    targets=np.array([r['raw_action'] for r in recording.rows])
    values=source_episode_values(recording.rows,targets,env.recording_geometry(targets))
    logical=f'episode_{writer.count:04d}'
    attrs=dict(episode_id=logical,source_group=provenance['source_group_prefix']+'/'+logical,
        success=bool(outcome['success']),validation_only=outcome.get('validation_only',False),success_source=outcome.get('success_source','unknown'),
        legacy_initial_state=json.dumps(recording.initial),task_definition=json.dumps(provenance['scene']['task_definition']),
        scene_configuration=json.dumps(provenance['scene_configuration']))
    return writer.commit(values,attrs,videos,video_staging=recording.video)
