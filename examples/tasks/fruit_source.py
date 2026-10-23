"""Edit Scene IDs, intent and run options. Paths are relative to this file.
Geometry/Physics/Camera/Robot and unique pair conditions come from the Workspace.
Source has no success gate. Custom terms need explicit scene.diagnostics for recording.
"""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/workflow_validation/script_sdk/fruit_example/workspaces/builder_b0561aff40a2/v1',
 'scene': {'objects': ['apple', 'banana', 'basket']},
 'options': {'backend': 'mujoco',
             'storage_name': 'fruit_source',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5},
 'source': {'id': 'fruit_sort',
            'instruction': 'Place the apple in the basket.',
            'pick_object': 'apple',
            'target_object': 'basket'}}
