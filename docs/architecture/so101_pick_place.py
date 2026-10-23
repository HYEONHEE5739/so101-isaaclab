"""Generated Task: edit semantics explicitly; no implicit reset or Mimic support.

Object poses: workspace frame, metres, 4x4 matrix. Cameras below are Metadata
references only. No camera is created or rendered by this Task.
"""
TASK_ID = 'cube_red_to_cup_a'
INSTRUCTION = 'Pick up the red block and place it in cup A.'
OBJECT_IDS = ('cube_red', 'cup_a')
CAMERA_IDS = ('side', 'wrist')
ROBOT_ASSET = {'source': 'model_asset', 'urdf': 'assets/SO101/urdf/so101_isaaclab.urdf', 'usd': 'assets/SO101/usd/so101_isaaclab.usd'}
ROBOT_ASSET_SHA256 = '24c03a0740144fefd1bcaf21a4bec006a558be9f28c891ac281ff406302ccf45'
TASK_DEFINITION = {'schema': 'so101.task-definition/1',
 'task_id': 'cube_red_to_cup_a',
 'pick_object_id': 'cube_red',
 'target_object_id': 'cup_a',
 'language_instruction': 'Pick up the red block and place it in cup A.',
 'success_condition': {'max_speed_m_s': 0.03,
                       'relation': 'contained_and_settled'},
 'randomization_spec': {'path': 'task.json#/reset',
                        'sha256': '031be37ee8e7d1db5b8238301eb1e103c595ee050e1b8b3d77734ad1af396096'}}
TASK_CONTRACTS = {'placement': 'simulation.mujoco_backend.MujocoBackend.success_details',
 'stable_grasp': 'real2sim.workspaces.grasp.StableGrasp',
 'grasp_drop_release': 'workflow.adapters.SimulationAdapter.update_grasp',
 'annotation': 'workflow.annotation.annotate_episode',
 'mimic': 'workflow.mimic.configuration',
 'evaluation': 'so101.placement/1'}

def setup(ctx):
    ctx.require(objects=OBJECT_IDS, cameras=CAMERA_IDS)
    if TASK_DEFINITION is not None and ctx.workspace.get('task_definition') != TASK_DEFINITION:
        raise ValueError('Task definition differs from the configured runtime')


def observe(ctx):
    return {name: ctx.object_pose(name) for name in OBJECT_IDS}


def success(ctx):
    return bool(ctx.placement_details(TASK_DEFINITION)["success"])
