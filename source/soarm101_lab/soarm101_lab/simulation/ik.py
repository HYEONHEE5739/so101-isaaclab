"""Shared bounded 5-axis pose solve. FK and its scratch state belong to each adapter."""
import json
import time
import numpy as np


class IKFailure(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('Unreachable target pose: '+json.dumps(details,allow_nan=False))


def solve_pose(forward,goal,seed,limits,position_tolerance=.003,rotation_tolerance=.12,frame='robot_base',*,reject_on_error=True):
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
    if reject_on_error and (details['position_error_m']>position_tolerance or details['rotation_error_rad']>rotation_tolerance):raise IKFailure(details)
    return np.r_[fitted.x,seed[5]],details


SOLVERS = ('scipy_ls', 'pinocchio_dls')
DLS_PARAMETERS = dict(position_weight=1.0, orientation_weight=0.01,
                      max_iterations=200, tolerance=1e-3, damping=1e-3, step_size=0.3)


def validate_solver(value='scipy_ls'):
    if value not in SOLVERS:
        raise ValueError(f'Unknown ik_solver {value!r}; expected one of {SOLVERS}')
    return value


FAILURE_POLICIES = ('strict', 'hold', 'best_effort')
MAX_ARM_COMMAND_JUMP_RAD = .5


def validate_failure_policy(value='strict'):
    if value not in FAILURE_POLICIES:
        raise ValueError(f'Unknown ik_failure_policy {value!r}; expected one of {FAILURE_POLICIES}')
    return value


def policy_configuration(policy='strict'):
    return dict(ik_failure_policy=validate_failure_policy(policy),max_arm_command_jump_rad=MAX_ARM_COMMAND_JUMP_RAD,
                jump_reference='previous_command_arm',invalid_candidate_action='abort_attempt',
                jump_enforcement='diagnostic_only' if policy=='strict' else 'reject')


def solver_configuration(solver='scipy_ls',failure_policy='strict'):
    solver = validate_solver(solver);policy=validate_failure_policy(failure_policy)
    return {'ik_solver': solver, 'parameters': dict(DLS_PARAMETERS) if solver == 'pinocchio_dls' else
            dict(max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8, rotation_objective_weight_m=.04),
            'seed_policy': 'measured_joint_position' if solver == 'pinocchio_dls' else 'previous_command',
            'failure_policy': ('apply_valid_candidate' if policy=='best_effort' else 'hold_measured_arm' if policy=='hold' or solver=='pinocchio_dls' else 'abort_attempt'),
            'position_tolerance_m': .003, 'rotation_tolerance_rad': .12}


def pose_error(actual, goal):
    from scipy.spatial.transform import Rotation
    return (float(np.linalg.norm(actual[:3, 3] - goal[:3, 3])),
            float(np.linalg.norm(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).as_rotvec())))


def solve_selected(forward, goal, seed, limits, position_tolerance=.003, rotation_tolerance=.12,
                   frame='robot_base', *, solver='scipy_ls', kinematics=None, base_from_frame=None,
                   failure_policy='strict', measured=None, previous_command=None):
    """Separate numerical convergence from candidate validity and command policy.

    Strict retains the old per-solver behavior. Hold/best_effort enforce the arm
    jump guard; neither changes solver weights, limits, seed policy or gripper.
    """
    solver=validate_solver(solver);policy=validate_failure_policy(failure_policy)
    started=time.perf_counter()
    goal,seed,limits=np.asarray(goal),np.asarray(seed,dtype=float),np.asarray(limits)
    if goal.shape!=(4,4) or seed.shape!=(6,) or not np.isfinite(goal).all() or not np.isfinite(seed).all():
        raise ValueError('Finite pose and six model-radian seed joints required')
    if limits.shape!=(6,2) or not np.isfinite(limits).all():raise ValueError('Six finite joint limits required')
    if seed[5]<limits[5,0]-1e-7 or seed[5]>limits[5,1]+1e-7:raise ValueError('Gripper target outside model limits')
    reference=np.asarray(previous_command if previous_command is not None else seed,dtype=float)
    hold=np.asarray(measured if measured is not None else seed,dtype=float)
    if reference.shape!=(6,) or hold.shape!=(6,) or not np.isfinite(reference).all() or not np.isfinite(hold).all():
        raise ValueError('Finite measured and previous six-joint states required')
    details=dict(position_error_m=None,rotation_error_rad=None,position_tolerance_m=position_tolerance,
        rotation_tolerance_rad=rotation_tolerance,seed_joint_rad=seed.tolist(),target_pose=goal.tolist(),
        target_frame=frame,joint_limits_rad=limits[:5].tolist(),arm_dof=5,target_dof=6)
    if solver=='scipy_ls':
        raw,details=solve_pose(forward,goal,seed,limits,position_tolerance,rotation_tolerance,frame,reject_on_error=False)
        candidate=np.asarray(raw[:5]);converged=details['solver_status']>0;iterations=details['evaluations']
    else:
        if kinematics is None:raise ValueError('Pinocchio DLS requires a worker-owned SO101Kinematics')
        base_goal=goal if base_from_frame is None else np.asarray(base_from_frame)@goal
        candidate,converged,native=kinematics.inverse_kinematics(seed[:5],base_goal,**DLS_PARAMETERS)
        candidate=np.asarray(candidate)
        details['native_result']={k:float(v) if np.isfinite(v) else None for k,v in native.items()}
        iterations=int(native['iterations'])
        if candidate.shape==(5,) and np.isfinite(candidate).all():
            details['position_error_m'],details['rotation_error_rad']=pose_error(forward(candidate),goal)
    shape_valid=candidate.shape==(5,);finite=bool(shape_valid and np.isfinite(candidate).all())
    violation=bool(finite and (np.any(candidate<limits[:5,0]-1e-7) or np.any(candidate>limits[:5,1]+1e-7)))
    jump=float(np.max(np.abs(candidate-reference[:5]))) if finite else None
    excessive=jump is not None and jump>MAX_ARM_COMMAND_JUMP_RAD
    invalid=not finite or violation or (policy!='strict' and excessive)
    reason=('candidate_shape' if not shape_valid else 'candidate_nonfinite' if not finite else
            'joint_limit_violation' if violation else 'candidate_jump' if policy!='strict' and excessive else '')
    position=details['position_error_m'];rotation=details['rotation_error_rad']
    common_pass=position is not None and rotation is not None and position<=position_tolerance and rotation<=rotation_tolerance
    numerical_failure=not converged or not common_pass
    abort=invalid
    held=False;applied=False
    if not invalid:
        if policy=='best_effort':applied=True
        elif policy=='hold':held=numerical_failure;applied=not held
        elif solver=='pinocchio_dls':held=not converged;applied=not held
        else:abort=not common_pass;applied=not abort
    if policy=='strict' and solver=='pinocchio_dls' and finite and violation:
        abort=False;held=True;applied=False
    command=np.r_[candidate,seed[5]] if applied else np.r_[hold[:5],seed[5]]
    if held:
        hold_jump=float(np.max(np.abs(command[:5]-reference[:5])))
        if np.any(command[:5]<limits[:5,0]-1e-7) or np.any(command[:5]>limits[:5,1]+1e-7) or (policy!='strict' and hold_jump>MAX_ARM_COMMAND_JUMP_RAD):
            # Strict Isaac historically holds measured state; MuJoCo retains its
            # additional hard command boundary in MujocoBackend._target.
            if policy!='strict':abort=True;held=False;reason='invalid_measured_hold'
    if not reason:reason='native_nonconvergence' if not converged else 'pose_tolerance_exceeded' if not common_pass else ''
    details.update(fitted_joint_rad=candidate.tolist() if finite else None,ik_solver=solver,
        ik_failure_policy=policy,ik_success=bool(converged),ik_converged=bool(converged),
        ik_numerical_failure=bool(numerical_failure),common_pose_passed=bool(common_pass),
        command_applied=applied and not abort,ik_candidate_applied=applied and not abort,ik_held=held,
        ik_invalid_candidate=bool(invalid),ik_invalid_command=bool(abort and (invalid or reason=='invalid_measured_hold')),joint_limit_violation=violation,ik_candidate_jump_rad=jump,
        ik_excessive_jump=bool(excessive),failure_reason=reason,ik_iterations_or_evaluations=iterations,
        position_error_mm=position*1000 if position is not None else None,
        orientation_error_deg=float(np.degrees(rotation)) if rotation is not None else None,
        ik_position_error_mm=position*1000 if position is not None else None,
        ik_orientation_error_deg=float(np.degrees(rotation)) if rotation is not None else None,
        applied_joint_rad=command.tolist() if not abort else None,measured_joint_rad=hold.tolist(),
        previous_command_rad=reference.tolist(),ik_duration_ms=(time.perf_counter()-started)*1000)
    details['duration_s']=details['ik_duration_ms']/1000
    if abort:raise IKFailure(details)
    return command,details
