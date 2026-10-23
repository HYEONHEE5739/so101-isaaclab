"""Record freely; downstream native Annotation/Mimic inherit this exact binding."""
SCRIPT = {
    'stage': 'source',
    'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
    'scene': {'objects': ['cube_red', 'cube_blue', 'cube_green', 'cup_a', 'cup_b']},
    'source': {'id': 'blue_into_b', 'instruction': 'Pick up the blue block and place it in cup B.',
               'pick_object': 'cube_blue', 'target_object': 'cup_b'},
    'options': {'backend': 'mujoco', 'storage_name': 'source_blue_cup_b', 'port': '/dev/so101_leader',
                'num_envs': 1, 'episodes': 3, 'episode_time_s': 20, 'reset_time_s': 5},
}
