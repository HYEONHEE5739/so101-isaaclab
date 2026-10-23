"""Editable Task example for scene2_builder: workspace-frame X in metres.

This file is trusted Python, executed only by an explicitly requested worker.
No environment, robot connection, state mutation or catalog success predicate.
"""
TASK_ID = 'block_crosses_workspace_x'
INSTRUCTION = 'Move the red block beyond workspace X = 0.5 metres'
OBJECT_IDS = ('cube_red',)
CAMERA_IDS = ()
X_THRESHOLD_M = 0.5


def setup(ctx):
    # Binding is the worker's responsibility even when setup does nothing.
    pass


def observe(ctx):
    return {'block_pose': ctx.object_pose(OBJECT_IDS[0])}


def success(ctx):
    return bool(ctx.object_pose(OBJECT_IDS[0])[0, 3] > X_THRESHOLD_M)
