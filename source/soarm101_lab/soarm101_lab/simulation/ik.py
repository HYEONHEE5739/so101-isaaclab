"""Shared bounded 5-axis pose solve. FK and its scratch state belong to each adapter."""
import json
import time
import numpy as np


class IKFailure(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('Unreachable target pose: '+json.dumps(details,allow_nan=False))


def solve_pose(forward,goal,seed,limits,position_tolerance=.003,rotation_tolerance=.12,frame='robot_base'):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    started=time.perf_counter();goal=np.asarray(goal);seed=np.asarray(seed,dtype=float);limits=np.asarray(limits)
    if goal.shape!=(4,4) or seed.shape!=(6,) or not np.isfinite(goal).all() or not np.isfinite(seed).all():raise ValueError('Finite pose and six model-radian seed joints required')
    if seed[5]<limits[5,0]-1e-7 or seed[5]>limits[5,1]+1e-7:raise ValueError('Gripper target outside model limits')
    def residual(arm):
        actual=forward(arm)
        position=actual[:3,3]-goal[:3,3]
        rotation=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).as_rotvec()
        return np.r_[position,.04*rotation]
    fitted=least_squares(residual,np.clip(seed[:5],limits[:5,0],limits[:5,1]),bounds=(limits[:5,0],limits[:5,1]),
        max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    error=residual(fitted.x)
    details=dict(position_error_m=float(np.linalg.norm(error[:3])),rotation_error_rad=float(np.linalg.norm(error[3:])/.04),
        position_tolerance_m=position_tolerance,rotation_tolerance_rad=rotation_tolerance,seed_joint_rad=seed.tolist(),
        fitted_joint_rad=fitted.x.tolist(),joint_limits_rad=limits[:5].tolist(),target_pose=goal.tolist(),target_frame=frame,
        evaluations=int(fitted.nfev),solver_status=int(fitted.status),duration_s=time.perf_counter()-started,arm_dof=5,target_dof=6,
        rotation_objective_weight_m=.04)
    if details['position_error_m']>position_tolerance or details['rotation_error_rad']>rotation_tolerance:raise IKFailure(details)
    return np.r_[fitted.x,seed[5]],details
