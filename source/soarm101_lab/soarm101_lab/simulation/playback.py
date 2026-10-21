"""Recorded physics playback over a caller-owned adapter and engine clock.

No Stage, UI, task acceptance, recording or engine construction is owned here.
Callbacks borrow the already measured pre-command state until they return.
"""
import numpy as np
from ..utils.transforms import pose_matrix


def restore_episode(adapter,values:dict,attrs:dict)->None:
    """Restore once with the existing adapter, then check the recorded frame.

    values/attrs are borrowed, read-only. The worker owns adapter/environment
    creation, reset implementation and close. Restoration errors propagate;
    they are not converted into a completed replay result.
    """
    adapter.restore(values,attrs)
    first=adapter.state();expected=values['initial_state']['articulation']['robot']['joint_position'][0]
    if not np.allclose(adapter.base_world,pose_matrix(values['initial_state']['articulation']['robot']['root_pose'][0]),atol=1e-5):
        raise ValueError('Initial fixed robot base differs from recorded frame')
    if not np.allclose(first['joint_pos'],expected,atol=1e-5):raise ValueError('Initial joint restoration differs')
    for obj in adapter.workspace['task']['dynamic']:
        expected_obj=values['obs']['datagen_info']['object_pose'][obj][0]
        if not np.allclose(first['object_poses'][obj],expected_obj,atol=1e-5):raise ValueError('Initial object transform differs: '+obj)


def replay_steps(adapter,values:dict,*,before_step=None,on_observation=None,after_step=None):
    """Yield command / post-step barrier; return numeric errors and execution error.

    The worker applies each yielded model-rad command exactly once, then resumes
    this iterator. None is a barrier: finish all vector slots' post-step work
    before restoring any next episode. Throw step ValueError/RuntimeError into
    the iterator to retain the previous failure/error metric semantics.

    before_step(index)->bool may stop before observation/command; the other
    optional hooks are on_observation(index,command,state) and after_step().
    No extra state/RGB copies, render, pacing, task signals or file writes are
    added. The existing detached previous-command snapshot is retained.
    Generator.close/Interrupt propagate; the worker closes owned resources.
    """
    boundary=getattr(adapter,'interrupt_boundary',None)
    joint_errors=[];object_errors=[];error=None
    try:
        for i,command in enumerate(values['joint_targets']):
            if boundary:boundary()
            if before_step is not None and before_step(i) is False:break
            before=adapter.state();joint_errors.append(float(np.linalg.norm(before['joint_pos']-values['obs']['policy']['joint_pos'][i])))
            for obj in adapter.workspace['task']['dynamic']:
                object_errors.append(float(np.linalg.norm(before['object_poses'][obj][:3,3]-values['obs']['datagen_info']['object_pose'][obj][i,:3,3])))
            if on_observation is not None:on_observation(i,command,before)
            yield command
            adapter.previous=np.array(command,copy=True)
            if after_step is not None:after_step()
            yield None
    except (ValueError,RuntimeError) as exc:error=str(exc)
    return dict(joint_l2_max_rad=max(joint_errors,default=0.),
        object_position_error_max_m=max(object_errors,default=0.),simulation_error=error)


def step_targets(adapter,command,actions=None):
    """Apply one recorded command using the worker's existing engine clock.

    Isaac actions is a worker-owned tensor initialized once to measured holds;
    only the active slot is changed. Vector coordinators step their full action
    tensor once per tick themselves. No reset, DT conversion or new hold tensor.
    """
    if hasattr(adapter,'backend'):return adapter.backend.step(command)
    import torch
    actions[adapter.index]=torch.as_tensor(command,device=adapter.native.device,dtype=actions.dtype)
    return adapter.native.step(actions)
