"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_source_minimal',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5}}
