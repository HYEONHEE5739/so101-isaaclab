"""Workspace-only stable grasp detection; completion is a latched subtask boundary."""
import numpy as np

# Initial engineering thresholds, not measured/calibrated physical constants.
HOLD_SECONDS = 0.20
RELATIVE_DRIFT_M = 0.006


class StableGrasp:
    def __init__(self, dt, size):
        self.dt = dt
        self.size = size
        self.samples = []
        self.completed = False
        self.last_tick = None
        self.details = {}

    def update(self, tick, relative, relative_quat, ee_position, object_position, left, right):
        if tick == self.last_tick:
            return self.completed
        if self.last_tick is not None and tick != self.last_tick + 1:
            self.samples.clear()
        self.last_tick = tick
        sample = tuple(np.asarray(v, dtype=float).copy() for v in
                       (relative, relative_quat, ee_position, object_position))
        if left and right:
            self.samples.append(sample)
            while self.samples:
                origin = self.samples[0]
                drift = max(np.linalg.norm(p[0] - origin[0]) for p in self.samples)
                angle = max(2*np.arccos(np.clip(abs(np.dot(p[1], origin[1])) /
                            max(np.linalg.norm(p[1])*np.linalg.norm(origin[1]), 1e-12), 0, 1))
                            for p in self.samples)
                if drift <= RELATIVE_DRIFT_M and angle <= np.deg2rad(10):
                    break
                self.samples.pop(0)
        else:
            self.samples.clear()
        duration = max(0, len(self.samples)-1)*self.dt
        ee_motion = object_motion = 0.
        if self.samples:
            ee_motion = np.linalg.norm(sample[2]-self.samples[0][2])
            object_motion = np.linalg.norm(sample[3]-self.samples[0][3])
        moving = ee_motion >= .005 and object_motion >= .005
        stable = bool(left and right and moving and duration + 1e-9 >= HOLD_SECONDS)
        self.completed |= stable
        self.details = dict(left_contact=bool(left), right_contact=bool(right), moving=bool(moving),
                            ee_motion_m=float(ee_motion), object_motion_m=float(object_motion),
                            stable=stable, hold_seconds=duration, required_seconds=HOLD_SECONDS,
                            relative_drift_limit_m=RELATIVE_DRIFT_M)
        # Rolling window; confirmation requires motion inside the sustained-contact window.
        if len(self.samples) > int(np.ceil(HOLD_SECONDS/self.dt))+1:
            self.samples.pop(0)
        return self.completed


def object_stably_grasped(env, robot_cfg, ee_frame_cfg, object_cfg,
                         distance_threshold=0.045, gripper_closed_threshold=0.3,
                         gripper_open_threshold=0.5):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms
    robot, ee, obj = (env.scene[c.name] for c in (robot_cfg, ee_frame_cfg, object_cfg))
    relative, relative_quat = subtract_frame_transforms(ee.data.target_pos_w[:, 0],
                                           ee.data.target_quat_w[:, 0],
                                           obj.data.root_pos_w, obj.data.root_quat_w)
    forces = []
    for name in ('grasp_fixed_contact', 'grasp_moving_contact'):
        matrix = env.scene[name].data.force_matrix_w
        if matrix is None or matrix.shape[1:3] != (1, 1):
            raise RuntimeError(f'{name}: expected one finger body filtered against one target cube')
        forces.append(torch.linalg.vector_norm(matrix[:, 0, 0], dim=-1))
    # Numerical noise floor, not a calibrated grip-strength threshold.
    contacts = [force > 1e-4 for force in forces]
    w = env.cfg.workflow_workspace
    size = w['profile']['objects'][object_cfg.name]['dimensions_m']
    if not hasattr(env, '_stable_grasp_trackers'):
        env._stable_grasp_trackers = [StableGrasp(env.step_dt, size) for _ in range(env.num_envs)]
    tick = env.common_step_counter
    result = [tracker.update(tick, relative[i].detach().cpu().numpy(),
                             relative_quat[i].detach().cpu().numpy(),
                             ee.data.target_pos_w[i, 0].detach().cpu().numpy(),
                             obj.data.root_pos_w[i].detach().cpu().numpy(),
                             bool(contacts[0][i]), bool(contacts[1][i]))
              for i, tracker in enumerate(env._stable_grasp_trackers)]
    env._grasp_completed = torch.tensor(result, dtype=torch.bool, device=env.device)
    env._stable_grasp_diagnostics = [t.details.copy() for t in env._stable_grasp_trackers]
    for i, d in enumerate(env._stable_grasp_diagnostics):
        d.update(left_force_n=float(forces[0][i]), right_force_n=float(forces[1][i]))
    return env._grasp_completed
