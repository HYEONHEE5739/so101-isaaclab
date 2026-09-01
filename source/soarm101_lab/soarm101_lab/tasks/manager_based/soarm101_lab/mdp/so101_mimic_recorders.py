import torch
from isaaclab.managers import RecorderTerm


class PreStepPolicyObservationsCpuRecorder(RecorderTerm):
    """Record policy observations on CPU to reduce GPU memory usage."""
    """ VRAM 부족해서 OOM 발생 문제 해결용"""
    def record_pre_step(self):
        obs = self._env.obs_buf["policy"]
        return "obs", {name: value.detach().cpu() for name, value in obs.items()}

class PostStepJointTargetsRecorder(RecorderTerm):

    def record_post_step(self):
        arm = self._env.action_manager.get_term("arm")
        gripper = self._env.action_manager.get_term("gripper")

        arm_target = arm.joint_position_targets.clone()
        gripper_target = gripper.processed_actions.clone()

        joint_targets = torch.cat(
            [arm_target, gripper_target],
            dim=-1,
        )

        return "joint_targets", joint_targets