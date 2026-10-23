"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {},
                       'cube_blue': {'pose': {'xyz': [0.04, -0.2, 0.039]}}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_source_two',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5}}
