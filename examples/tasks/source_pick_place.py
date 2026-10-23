"""Edit Scene IDs, intent and run options. Paths are relative to this file.
Geometry/Physics/Camera/Robot and unique pair conditions come from the Workspace.
Source has no success gate. Custom terms need explicit scene.diagnostics for recording.
"""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': ['cube_red',
                       'cube_blue',
                       'cube_green',
                       'cup_a',
                       'cup_b']},
 'options': {'backend': 'mujoco',
             'storage_name': 'source_pick_place_generated',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5},
 'source': {'id': 'cube_red_to_cup_a',
            'instruction': 'Pick up the red block and place it in cup A.',
            'pick_object': 'cube_red',
            'target_object': 'cup_a'}}
