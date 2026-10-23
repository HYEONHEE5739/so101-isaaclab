"""Workspace-metre poses using the existing Backend-owned reset RNG/sampler."""
import numpy as np


def metadata_pose(ctx, object_id, rng):
    """Detached Builder initial pose; never substitute current simulation state."""
    ctx.require(objects=(object_id,))
    profile_id=ctx.workspace['task']['targets'].get(object_id,object_id)
    return np.array(ctx.workspace['profile']['objects'][profile_id]['T_workspace'],dtype=float,copy=True)


def uniform(ctx, object_id, rng):
    """Existing support/margin/separation sampler, restricted to this Body."""
    return _sample(ctx, object_id, rng, {})


def edge_uniform(ctx, object_id, rng):
    """The existing Source zero-margin uniform policy; no new random algorithm."""
    return _sample(ctx, object_id, rng, {'yaw_rad':[0.,0.],'xy_jitter_m':0.,'margin_m':0.})


def _sample(ctx, object_id, rng, overrides):
    sampler=ctx.backend.randomizer
    if rng is not sampler.rng:raise ValueError('Reset must use Backend episode RNG')
    r={**ctx.workspace['task']['reset'],'mode':'uniform',**overrides}
    if any(k not in r for k in ('margin_m','separation_m','clearance_m')):raise ValueError('Workspace lacks an explicit sampling distribution; declare a custom Reset term')
    sample=sampler.sample(object_ids=[object_id],reset=r,fixed_positions={n:ctx.object_pose(n)[:3,3].tolist() for n in ctx.backend.metadata['dynamic_objects'] if n!=object_id})
    pose=metadata_pose(ctx,object_id,rng);pose[:3,3]=sample['cube_positions'][object_id]
    return pose
