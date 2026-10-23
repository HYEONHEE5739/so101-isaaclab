"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {},
                       'cube_blue': {},
                       'cube_green': {},
                       'cup_a': {},
                       'cup_b': {}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_red_cup_a',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5},
 'annotation': {'diagnostics': {'object_id': 'cube_red', 'target_id': 'cup_a'},
                'events': {'grasp': 'grasp_event', 'place': 'place_event'},
                'signals': ['grasp'],
                'final_events': ['place'],
                'manual_event': 'grasp',
                'accept': 'native_acceptance'},
 'mimic': {'recipe': 'pick_place/1', 'accept': 'native_acceptance'}}

from soarm101_lab.task_terms.diagnostics import grasp_completed as grasp_event
from soarm101_lab.task_terms.annotation import placed as place_event, native_acceptance
