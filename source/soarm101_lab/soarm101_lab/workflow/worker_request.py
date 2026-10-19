"""Shared MuJoCo worker request contract, independent of GUI and engine execution.

Planner and worker validate the same payload. Keep validation order, messages and
option ownership stable: existing callers also import this interface from operations.
"""
import json
from pathlib import Path

from ..learning.minivla.config import MODEL_ROOT
from ..real2sim.workspaces.package import load


MUJOCO_STAGES = dict(environment='Environment', source='Source', replay='Replay',
              prepare='Prepare', convert='Convert', import_='Import', annotate='Annotation',
              datagen='Datagen', train='Train', rollout='Evaluation')
MUJOCO_STAGES['import'] = MUJOCO_STAGES.pop('import_')

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
    if stage in ('environment', 'source', 'replay', 'rollout', 'import'):
        load(workspace)
    if stage in ('prepare', 'convert', 'annotate', 'datagen', 'train', 'replay'):
        if not (Path(input_path) / 'manifest.json').is_file():
            raise ValueError('Input must be a canonical MiniVLA dataset directory')
    if stage == 'import' and not Path(input_path).is_file():
        raise ValueError('HDF5 import requires an explicit recipe JSON')
    if stage == 'train':
        data = json.loads(Path(config).read_text())
        available = json.loads((Path(input_path)/'manifest.json').read_text()).get('cameras',['side'])
        preset = Path(config).resolve().parent == (MODEL_ROOT/'configs').resolve()
        if not preset and data.get('cameras') != available:
            raise ValueError('Model config camera contract differs from dataset: '+str(available))
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
