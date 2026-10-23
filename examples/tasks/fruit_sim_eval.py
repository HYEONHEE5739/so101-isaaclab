"""Edit Scene IDs, intent and run options. Paths are relative to this file.
Geometry/Physics/Camera/Robot and unique pair conditions come from the Workspace.
Source has no success gate. Custom terms need explicit scene.diagnostics for recording.
"""
SCRIPT = {'stage': 'sim_eval',
 'workspace': '../../outputs/workflow_validation/script_sdk/fruit_example/workspaces/builder_b0561aff40a2/v1',
 'scene': {'objects': ['apple', 'banana', 'basket']},
 'options': {'backend': 'mujoco',
             'storage_name': 'fruit_sim_eval',
             'num_envs': 1,
             'device': 'cpu',
             'max_steps': 1000,
             'eval_seed_mode': 'fixed',
             'eval_seed': 1047},
 'checkpoint': 'REPLACE_CHECKPOINT_PATH',
 'evaluations': [{'id': 'fruit_sort',
                  'instruction': 'Place the apple in the basket.',
                  'success': 'placement_success',
                  'episodes': 10}]}

from functools import partial
from soarm101_lab.task_terms.placement import placement
placement_success = partial(placement, object_id='apple', target_id='basket')
