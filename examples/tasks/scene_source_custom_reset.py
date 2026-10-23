"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {'reset': 'reset_red'}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_source_custom_reset',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5}}

from soarm101_lab.task_terms.reset import metadata_pose

def reset_red(ctx, object_id, rng):
    pose=metadata_pose(ctx,object_id,rng)
    pose[:3,3]=[0.02+rng.uniform(-0.005,0.005), -0.20, 0.039]
    return pose
