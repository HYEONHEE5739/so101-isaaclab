"""Custom-term example over current Workspace-frame pose; not a placement substitute."""
import math


def workspace_x_above(ctx, *, object_id: str, threshold_m: float) -> bool:
    """Return whether the runtime object's X exceeds the explicit metre threshold."""
    if not math.isfinite(threshold_m): raise ValueError('Finite threshold_m required')
    return bool(ctx.object_pose(object_id)[0, 3] > threshold_m)
