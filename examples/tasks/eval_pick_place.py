"""Edit Scene IDs, intent and run options. Paths are relative to this file.
Geometry/Physics/Camera/Robot and unique pair conditions come from the Workspace.
Source has no success gate. Custom terms need explicit scene.diagnostics for recording.
"""
SCRIPT = {'stage': 'sim_eval',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': ['cube_red',
                       'cube_blue',
                       'cube_green',
                       'cup_a',
                       'cup_b']},
 'options': {'backend': 'mujoco',
             'storage_name': 'eval_pick_place_generated',
             'num_envs': 1,
             'device': 'cpu',
             'max_steps': 1000,
             'eval_seed_mode': 'fixed',
             'eval_seed': 1047},
 'checkpoint': '../../outputs/workflow/models/act/train_cube_red_to_cup_a_20261018_010654_e162/checkpoints/050000/pretrained_model',
 'evaluations': [{'id': 'cube_red_to_cup_a',
                  'instruction': 'Pick up the red block and place it in cup A.',
                  'success': 'placement_success',
                  'episodes': 10}]}

from functools import partial
from soarm101_lab.task_terms.placement import placement
placement_success = partial(placement, object_id='cube_red', target_id='cup_a')
