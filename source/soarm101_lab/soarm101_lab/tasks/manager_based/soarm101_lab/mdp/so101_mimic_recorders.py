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
        from pathlib import Path
        obs=self._env.obs_buf['policy']
        if not hasattr(self,'_video_episodes'):self._video_episodes={}
        for i in range(self._env.num_envs):
            episode=self._env.recorder_manager.get_episode(i)
            previous=self._video_episodes.get(i)
            if previous is not episode and getattr(previous,'_rgb_video',None):previous._rgb_video.abort()
            self._video_episodes[i]=episode
            stream_observation_rgb(episode,{k:v[i] for k,v in obs.items()},
                Path(self._env.cfg.recorders.dataset_export_dir_path),self._env.step_dt**-1)
        return 'obs',{**{name:value.detach().cpu() for name,value in obs.items() if not name.endswith('_cam')},
                      **observation_telemetry(self._env)}

    def reset(self,env_ids=None):
        for i in list(getattr(self,'_video_episodes',{})) if env_ids is None else env_ids:
            old=getattr(self,'_video_episodes',{}).pop(int(i),None)
            if getattr(old,'_rgb_video',None):old._rgb_video.abort()
        return {}

    def __del__(self):self.reset()

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


from isaaclab.utils.datasets import HDF5DatasetFileHandler


class VideoHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Isaac's numeric schema is unchanged; RGB is an episode-level external attribute."""
    def create(self,file_path,env_name=None):
        from pathlib import Path
        file_path=str(file_path)
        target=Path(file_path if file_path.endswith('.hdf5') else file_path+'.hdf5')
        if target.exists():raise FileExistsError('Choose a new dataset output: '+str(target))
        result=super().create(str(target),env_name=env_name)
        self._created_here=target.resolve()
        return result

    def finalize_common(self):
        if not getattr(self,'_created_here',None):raise RuntimeError('Only a new writer-owned recording may be finalized')
        from soarm101_lab.workflow.dataset import finalize_owned_isaac_recording
        return finalize_owned_isaac_recording(self._created_here)

    def write_episode(self,episode):
        from pathlib import Path
        import json,shutil,uuid
        video=getattr(episode,'_rgb_video',None)
        if video is None:return super().write_episode(episode)
        frames=len(episode.data['actions']);parent=Path(self._hdf5_data_group.file.filename).resolve().parent
        folder=parent/(Path(self._hdf5_data_group.file.filename).stem+'.videos')/uuid.uuid4().hex
        count=self._demo_count;total=self._hdf5_data_group.attrs.get('total',0)
        try:
            specs=video.finish(frames)
            folder.parent.mkdir(parents=True,exist_ok=True)
            video.folder.rename(folder)
            specs={name:{**spec,'file':str((folder/spec['file']).relative_to(parent))} for name,spec in specs.items()}
            super().write_episode(episode)
            self._hdf5_data_group[f'demo_{count}'].attrs['external_videos']=json.dumps(specs)
            self.flush()
            episode._rgb_video=None
        except BaseException:
            if f'demo_{count}' in self._hdf5_data_group:del self._hdf5_data_group[f'demo_{count}']
            self._demo_count=count;self._hdf5_data_group.attrs['total']=total
            video.abort()
            if folder.exists():shutil.rmtree(folder)
            raise


def stream_observation_rgb(episode,obs,parent,fps):
    """Attach private writers to the existing EpisodeData, never to its serialized numeric dict."""
    from soarm101_lab.utils.async_video_writer import VideoEpisode
    video=getattr(episode,'_rgb_video',None)
    if video is None:
        import os
        video=VideoEpisode(parent,fps,int(os.environ.get('SO101_VIDEO_CRF','18')))
        episode._rgb_video=video
    images={}
    for name,value in obs.items():
        if not name.endswith('_cam'):continue
        image=value.detach().cpu()
        if image.dtype.is_floating_point:
            image=image*(255 if image.max().item()<=1.5 else 1)
            image=image.clamp(0,255).to(torch.uint8)
        images[name]=image.numpy()[...,:3]
    if images:video.append(images)



def close_video_recorders(env):
    for term in getattr(env.recorder_manager,'_terms',{}).values():
        if isinstance(term,PreStepPolicyObservationsCpuRecorder):term.reset()
