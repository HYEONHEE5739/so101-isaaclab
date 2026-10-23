"""Explicit Scene recording; edit port/output before hardware use."""
SCRIPT = {'stage': 'source',
 'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
 'scene': {'objects': {'cube_red': {}}},
 'options': {'backend': 'mujoco',
             'storage_name': 'scene_custom_event',
             'num_envs': 1,
             'port': '/dev/so101_leader',
             'episodes': 3,
             'episode_time_s': 20,
             'reset_time_s': 5},
 'annotation': {'events': {'left_start': {'factory': 'motion_event'}},
                'signals': ['left_start'],
                'manual_event': 'left_start',
                'accept': 'accept_motion',
                'failure': 'failure'},
 'mimic': {'subtasks': [{'name': 'approach',
                         'object': 'cube_red',
                         'eef': 'tool0',
                         'end_event': 'left_start'},
                        {'name': 'finish',
                         'object': None,
                         'eef': 'tool0',
                         'end_event': None,
                         'selection': 'random'}],
           'accept': 'accept_motion',
           'failure': 'failure'}}

import numpy as np
from soarm101_lab.task_terms.annotation import event_boundary

class MotionEvent:
    def reset(self,ctx):self.start=ctx.object_pose('cube_red')[:3,3].copy()
    def __call__(self,ctx):return bool(np.linalg.norm(ctx.object_pose('cube_red')[:3,3]-self.start)>0.0001)

def motion_event():return MotionEvent()
def accept_motion(ctx):return event_boundary(ctx,'left_start') is not None
def failure(ctx):return None
