import torch
from isaaclab.managers import RecorderTerm


def observation_telemetry(env):
    """Pre-step poses share policy observation/RGB time; never use post-step states here."""
    workspace=getattr(env.cfg,'workflow_workspace',None)
    names=workspace['task']['dynamic'] if workspace else list(env.scene.rigid_objects)
    poses={name:torch.cat([env.scene[name].data.root_pos_w,env.scene[name].data.root_quat_w],-1).detach().cpu().float() for name in names}
    stamp=torch.full((env.num_envs,1),float(env.sim.current_time),dtype=torch.float64)
    events={}
    for group,key,label in [('subtask_terms','grasp','grasp'),('task_state','place_success','place'),('task','success','place')]:
        value=env.obs_buf.get(group,{}).get(key)
        if value is not None:events[label]=value.detach().cpu()
    return {'timestamp':stamp,'object_poses':poses,**({'events':events} if events else {})}


def telemetry_metadata(names):
    return {'object_telemetry':{n:{'path':'obs/object_poses/'+n,'frame':'world','pose_format':'xyz_wxyz','timestamp_path':'obs/timestamp'} for n in names},
            'ee_frame':'robot_base','observation_timestamp_path':'obs/timestamp',
            'success_source':'operator_saved_unverified'}


class PreStepPolicyObservationsCpuRecorder(RecorderTerm):
    """Record policy observations on CPU to reduce GPU memory usage."""
    """ VRAM 부족해서 OOM 발생 문제 해결용"""
    def record_pre_step(self):
        obs = self._env.obs_buf["policy"]
        return "obs", {**{name: value.detach().cpu() for name, value in obs.items()},**observation_telemetry(self._env)}

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
