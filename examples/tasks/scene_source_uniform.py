"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {'reset': 'random_red'}, 'cube_blue': {}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_source_uniform',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5}}

from soarm101_lab.task_terms.reset import uniform as random_red
