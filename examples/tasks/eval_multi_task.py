"""Two user tasks share one explicit physical object set. Paths are relative to this file."""
SCRIPT = {'stage': 'sim_eval',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'checkpoint': '../../outputs/workflow/models/act/train_cube_red_to_cup_a_20261018_010654_e162/checkpoints/050000/pretrained_model',
 'scene': {'objects': ['cube_red',
                       'cube_blue',
                       'cube_green',
                       'cup_a',
                       'cup_b']},
 'options': {'backend': 'mujoco',
             'mode': 'operator',
             'renderer': 'egl',
             'storage_name': 'eval_multi_task',
             'num_envs': 2,
             'device': 'cpu',
             'max_steps': 1000,
             'eval_seed_mode': 'fixed',
             'eval_seed': 1047},
 'evaluations': [{'id': 'cube_red_to_cup_a',
                  'instruction': 'Pick up the red block and place it in cup A.',
                  'success': 'placement_success',
                  'episodes': 3},
                 {'id': 'blue_to_b',
                  'instruction': 'Pick up the blue block and place it in cup '
                                 'B.',
                  'success': 'blue_success',
                  'episodes': 2}]}
from functools import partial
from soarm101_lab.task_terms.placement import placement
placement_success = partial(placement, object_id="cube_red", target_id="cup_a")
blue_success = partial(placement, object_id="cube_blue", target_id="cup_b")
