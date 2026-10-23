"""Physics adapters for the common Mimic protocol. No engine import at module load."""
from collections import defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..utils.transforms import as_pose7
from ..data import recording
from ..simulation.control import GenerationStopped
from .mimic import configuration
from ..real2sim.workspaces.grasp import StableGrasp, HOLD_SECONDS, RELATIVE_DRIFT_M
from ..so101_joint_mapping import JOINTS


class _RecorderProtocol:
    def __init__(self,adapter):self.adapter=adapter;self.success=False
    def reset(self,env_ids):self.adapter.buffer.abort();self.success=False
    def set_success_to_episodes(self,ids,flags):self.success=bool(flags[0,0])
    def export_episodes(self,ids):raise RuntimeError('Only the central DatasetWriter may commit episodes')


class SimulationAdapter:
    def __init__(self,workspace,hz,output,seed,stop,require_standard_cameras=True):
        self.workspace=workspace;self.hz=hz;self.device='cpu';self.num_envs=1;self.seed=seed;self.stop=stop
        self.camera_names=workspace['task'].get('observation_cameras',['side','wrist'])
        if require_standard_cameras and self.camera_names!=['side','wrist']:raise ValueError('Common collection requires Side and Wrist')
        self.grasp_object=workspace['task']['pick'];self.place_target=workspace['task']['place'];self.place_condition=None
        self.cfg=configuration(self.grasp_object,self.place_target,seed) if self.grasp_object else None
        self.base_world=np.asarray(workspace['profile']['robot_base']['T_world'])
        self.buffer=recording.EpisodeBuffer(output,hz,self.base_world,self.camera_names);self.recorder_manager=_RecorderProtocol(self)
        self.scene=SimpleNamespace(get_state=lambda is_relative=True:self.initial_state())
        self.obs_buf={};self.reset_count=0;self.step_index=0;self.pending_pose=None;self.metrics=defaultdict(float)
        self.ik_solver="scipy_ls";self.ik_failure_policy="strict";self.last_ik=None;self.ik_attempt=defaultdict(float);self.ik_motion_evidence=[]
        self.grasp_reference=None;self.grasp_command=None;self.release_commanded=False;self.loss_steps=0;self.drop_step=None
        self.tracker=StableGrasp(1/hz,workspace['profile']['objects'][self.grasp_object]['dimensions_m']) if self.grasp_object else None

    def bind_followup(self, runtime):
        """Bind a pure acceptance refinement to this existing episode Adapter only."""
        self.followup_runtime = runtime
        if runtime and runtime['contract'].get('schema')=='so101.scene-terms/1':
            from ..simulation.task_context import TaskContext
            from .mimic import subtask_configuration
            declaration=runtime['contract']['declaration'];spec=declaration['annotation']
            ctx=TaskContext(self.backend,adapter=self)
            objects=list(declaration['scene']['objects'])
            ctx.validate_bindings({'OBJECT_IDS':objects,'CAMERA_IDS':self.camera_names})
            self.processing_context=ctx;self.processing_runtime=runtime
            self.processing_failure=None;self.event_boundaries={};self.event_latches={}
            self.buffer.signal_names=spec.get('signals',[]);self.buffer.final_events=spec.get('final_events',[])
            diagnostics=spec.get('diagnostics')
            if diagnostics:
                previous_object=self.grasp_object
                self.grasp_object=diagnostics['object_id'];self.place_target=diagnostics.get('target_id');self.place_condition=diagnostics.get('condition')
                if previous_object!=self.grasp_object or self.tracker is None:
                    self.tracker=StableGrasp(1/self.hz,self.workspace['profile']['objects'][self.grasp_object]['dimensions_m'])
            else:self.grasp_object=self.place_target=None;self.tracker=None
            self.processing_events={name:fn for name,(fn,factory) in runtime['terms'].items() if not factory}
            mimic=declaration.get('mimic')
            if runtime['stage']=='datagen':
                if not mimic:raise ValueError('DataGen requires an explicit Mimic contract')
                self.cfg=(configuration(self.grasp_object,self.place_target,self.seed) if mimic.get('recipe')=='pick_place/1'
                             else subtask_configuration(mimic['subtasks'],self.seed))
            return
        if hasattr(self,'followup_acceptance'):del self.followup_acceptance
        if runtime and runtime['acceptance']:
            from ..simulation.task_context import TaskContext
            ctx = TaskContext(self.backend, adapter=self)
            ctx.validate_bindings({'OBJECT_IDS': [runtime['contract']['pick_object'], runtime['contract']['target_object']],
                                   'CAMERA_IDS': self.camera_names})
            self.followup_acceptance = (ctx, runtime['acceptance'])

    def accept_followup(self):
        """Pure result-adoption hook; native physics/grasp success stays Adapter-owned."""
        term = getattr(self, 'followup_acceptance', None)
        if term is None:return True
        result = term[1](term[0])
        if not isinstance(result,(bool,np.bool_)):raise ValueError('Annotation acceptance term must return a Boolean')
        return bool(result)

    def _term_value(self, fn, observation=None):
        """One borrowed pre-step observation while invoking a term; no retained cache."""
        ctx=self.processing_context
        ctx._observation=observation
        try:return fn(ctx)
        finally:ctx._observation=None

    def term_boolean(self,fn,observation=None):
        result=self._term_value(fn,observation)
        if not isinstance(result,(bool,np.bool_)):raise ValueError('Event/acceptance terms must return Boolean')
        return bool(result)

    def _check_term_failure(self, observation=None):
        """Latch the first explicit category; a final post-step check can catch drops."""
        fn=self.processing_runtime.get('failure')
        if fn is None:return
        reason=self._term_value(fn,observation)
        if reason is not None and reason is not False and (not isinstance(reason,str) or not reason):
            raise ValueError('Failure term must return None/False or a category string')
        if reason:self.processing_failure=self.processing_failure or reason

    def configure_ik(self,solver,failure_policy="strict"):
        from ..simulation.ik import validate_solver,validate_failure_policy
        self.ik_solver=validate_solver(solver);self.ik_failure_policy=validate_failure_policy(failure_policy)

    def record_ik(self,details):
        self.last_ik=details
        if details is None:return
        increments={'ik_calls':1,'ik_s':details['duration_s'],'ik_residual_samples':int(details['position_error_mm'] is not None),
            'ik_nonconverged':int(not details['ik_success']),
            'ik_pose_failures':int(not details['common_pose_passed']),
            'ik_position_error_mm_sum':details['position_error_mm'] or 0.,
            'ik_candidate_applied':int(details.get('ik_candidate_applied',False)),
            'ik_unconverged_applied':int(details.get('ik_candidate_applied',False) and details.get('ik_numerical_failure',False)),
            'ik_held':int(details.get('ik_held',False)),
            'ik_invalid_candidate':int(details.get('ik_invalid_candidate',False)),
            'ik_invalid_command':int(details.get('ik_invalid_command',False)),
            'ik_orientation_error_deg_sum':details['orientation_error_deg'] or 0.}
        for key,value in increments.items():
            self.metrics[key]+=value;self.ik_attempt[key]+=value

    def _check_stop(self):
        if self.stop.is_set():raise GenerationStopped()

    def _check_step_limit(self):
        if self.step_index>=getattr(self,'max_episode_steps',2000):raise TimeoutError('Episode control step limit exceeded')

    def actions_to_gripper_actions(self,actions):return {'tool0':actions[:,-1:]}
    def get_robot_eef_pose(self,eef_name='tool0',env_ids=None):
        return torch.as_tensor(self.state()['eef_pose'],dtype=torch.float32)[None]
    def get_object_poses(self,env_ids=None):
        return {k:torch.as_tensor(v,dtype=torch.float32)[None] for k,v in self.state()['object_poses'].items()}

    def initial_state(self):
        s=self.state();world={k:as_pose7(self.base_world@v) for k,v in s['object_poses'].items()}
        return {'articulation':{'robot':{'root_pose':as_pose7(self.base_world)[None],'root_velocity':np.zeros((1,6)),
            'joint_position':s['joint_pos'][None],'joint_velocity':s.get('joint_vel',np.zeros(6))[None]}},
            'rigid_object':{k:{'root_pose':v[None],'root_velocity':s.get('object_velocities_world',{}).get(k,np.zeros(6))[None]} for k,v in world.items()}}

    def update_grasp(self):
        from scipy.spatial.transform import Rotation
        s=self.state();ee=s['eef_pose']
        if self.tracker is not None:
            obj=s['object_poses'][self.grasp_object]
            rel=np.linalg.inv(ee)@obj;left,right=self.contacts()
            self.tracker.update(self.step_index,rel[:3,3],Rotation.from_matrix(rel[:3,:3]).as_quat(),ee[:3,3],obj[:3,3],left,right)
            self._update_failure_evidence(rel[:3,3],left,right)
        if self.pending_pose is not None:
            from ..simulation.ik import pose_error
            position,rotation=pose_error(ee,self.pending_pose)
            for key,value in {'physics_tracking_samples':1,'physics_position_error_mm_sum':position*1000,
                              'physics_orientation_error_deg_sum':float(np.degrees(rotation))}.items():
                self.metrics[key]+=value;self.ik_attempt[key]+=value
        if self.last_ik and self.last_ik.get('ik_numerical_failure'):
            movement=float(np.linalg.norm(s['joint_pos'][:5]-np.asarray(self.last_ik['measured_joint_rad'])[:5]))
            for key,value in {'ik_failed_physics_steps':1,'ik_failed_motion_rad_sum':movement}.items():
                self.metrics[key]+=value;self.ik_attempt[key]+=value
            if len(self.ik_motion_evidence)<8:
                self.ik_motion_evidence.append({'step':self.step_index,'candidate_applied':self.last_ik['ik_candidate_applied'],
                    'held':self.last_ik['ik_held'],'candidate':self.last_ik['fitted_joint_rad'],
                    'measured_before':self.last_ik['measured_joint_rad'],'command':self.previous.tolist(),
                    'measured_after':s['joint_pos'].tolist(),'motion_rad':movement})
        self.step_index+=1

    def check_success(self):
        if getattr(self,'processing_runtime',None):
            self._check_term_failure()
            native_recipe=(self.processing_runtime['contract']['declaration'].get('mimic') or {}).get('recipe')=='pick_place/1'
            if native_recipe:
                from ..task_terms.annotation import native_acceptance
                if not self.native_success():return False
                if self.processing_runtime['accept'] is native_acceptance:return not self.processing_failure
            return self.term_boolean(self.processing_runtime['accept']) and not self.processing_failure
        return SimulationAdapter.native_success(self)

    def native_success(self):
        return bool(self.place_success() and self.tracker.completed)

    def _update_failure_evidence(self,relative,left,right):
        # Diagnostics only: do not change the latched grasp or place success gate.
        if not self.tracker.completed:return
        grip=float(self.previous[5])
        if self.grasp_reference is None:
            self.grasp_reference=np.asarray(relative).copy();self.grasp_command=grip
        if grip>self.grasp_command+1e-4:self.release_commanded=True
        separated=not (left or right) and np.linalg.norm(relative-self.grasp_reference)>RELATIVE_DRIFT_M
        self.loss_steps=self.loss_steps+1 if separated and not self.release_commanded else 0
        if self.loss_steps/self.hz>=HOLD_SECONDS and self.drop_step is None:self.drop_step=self.step_index

    def failure_category(self):
        if getattr(self,'processing_runtime',None):
            if self.processing_failure:return self.processing_failure
            if self.tracker is not None and (not self.tracker.completed or self.place_target is not None and not self.place_success()):return self.native_failure_category()
            return 'task_condition_not_met'
        return self.native_failure_category()

    def native_failure_category(self):
        if self.tracker is None:return 'task_condition_not_met'
        if not self.tracker.completed:return 'grasp_failure'
        return 'drop_failure' if self.drop_step is not None else 'place_failure'

    def failure_diagnostics(self):
        result=dict(grasp_confirmed=bool(self.tracker and self.tracker.completed),release_commanded=self.release_commanded,
                    drop_step=self.drop_step,contact_loss_steps=self.loss_steps,grasp=dict(self.tracker.details) if self.tracker else {})
        if getattr(self,'processing_runtime',None):result['events']=dict(self.event_boundaries)
        return result
    def after_reset(self):
        self.reset_count+=1;self.step_index=0;self.pending_pose=None
        self.last_ik=None;self.ik_attempt=defaultdict(float);self.ik_motion_evidence=[]
        self.grasp_reference=None;self.grasp_command=None;self.release_commanded=False;self.loss_steps=0;self.drop_step=None
        self.tracker=StableGrasp(1/self.hz,self.workspace['profile']['objects'][self.grasp_object]['dimensions_m']) if self.grasp_object else None
        self.event_boundaries={};self.event_latches={};self.processing_failure=None
        if getattr(self,'processing_runtime',None):
            import inspect
            for name,(fn,factory) in self.processing_runtime['terms'].items():
                term=fn() if factory else fn
                if not callable(term):raise ValueError('Event factory must return a callable')
                inspect.signature(term).bind(self.processing_context)
                self.processing_events[name]=term
                if hasattr(term,'reset'):term.reset(self.processing_context)
        self.buffer.begin(self.initial_state());self.obs_buf={}
        state=self.state()
        self.initial_configuration={'joint_rad':state['joint_pos'].tolist(),
            'object_poses_base':{k:v.tolist() for k,v in state['object_poses'].items()}}

    def resolve_target(self,target,noise):
        """SO101 reference action noise is applied to command-relative pose deltas."""
        from scipy.spatial.transform import Rotation
        pose=target.detach().cpu().numpy().astype(float)
        if not noise:return pose
        previous=self.pending_pose if self.pending_pose is not None else self.state()['eef_pose']
        delta=np.r_[pose[:3,3]-previous[:3,3],Rotation.from_matrix(pose[:3,:3]@previous[:3,:3].T).as_rotvec()]
        with self.random.active():delta+=torch.randn(6).numpy()*noise
        result=np.eye(4);result[:3,3]=previous[:3,3]+delta[:3]
        result[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()@previous[:3,:3]
        return result

    def record_frame(self,joints,target,observation=None,images=None):
        """Capture the pre-command boundary; the buffer only receives snapshots."""
        t=time.perf_counter();state=self.state() if observation is None else observation
        self.buffer.metrics['observation_s']+=time.perf_counter()-t
        t=time.perf_counter();images=self.images() if images is None else images
        self.buffer.metrics['render_s']+=time.perf_counter()-t
        self.last_images=images
        events=None;measured_place=bool(self.place_success())
        if getattr(self,'processing_runtime',None):
            self.processing_context._native_placement=measured_place
            try:
                events={}
                for name,fn in self.processing_events.items():
                    value=self.term_boolean(fn,state)
                    if value:
                        self.event_latches[name]=True;self.event_boundaries.setdefault(name,len(self.buffer.rows))
                    events[name]=self.event_latches.get(name,False) if name in self.buffer.signal_names else value
                self._check_term_failure(state)
            finally:del self.processing_context._native_placement
        self.buffer.append(joints,target,state,images,grasp=bool(self.tracker and self.tracker.completed),
            place=measured_place,ik=self.last_ik,events=events)

    def close(self):self.buffer.abort()


class MujocoAdapter(SimulationAdapter):
    def __init__(self,backend,hz,output,seed,stop,require_standard_cameras=True):
        super().__init__(backend.workspace,hz,output,seed,stop,require_standard_cameras);self.backend=backend
        self.base_workspace=np.linalg.inv(backend.world_from_workspace)@self.base_world
        self.previous=backend.home.copy();self.rng=np.random.default_rng(seed)

    def reset(self,env_id=0):
        self._check_stop();self.backend.reset(seed=int(self.rng.integers(2**32)))
        if getattr(self,'processing_runtime',None) and self.workspace.get('scene_configuration',{}).get('scene_only'):
            from ..simulation.task_context import reset_objects
            reset_objects(self.backend,self.processing_runtime['contract']['declaration']['scene']['objects'],self.processing_runtime['resetters'])
        self.previous=self.backend.home.copy();self.after_reset()

    def restore(self,values,attrs):
        if attrs.get('simulation_backend','mujoco')=='mujoco' and attrs.get('legacy_initial_state') and self.backend.metadata['model_sha256']==json.loads(attrs['legacy_initial_state'])['model_sha256']:
            self.backend.restore_state(json.loads(attrs['legacy_initial_state']))
        else:
            self.backend.reset();self.backend.set_joint_state(values['initial_state']['articulation']['robot']['joint_position'][0])
            self.backend.data.qvel[self.backend.v_indices]=values['initial_state']['articulation']['robot']['joint_velocity'][0]
            for name in self.backend.metadata['dynamic_objects']:
                pose=values['obs']['datagen_info']['object_pose'][name][0]
                self.backend.set_object_pose(name,self.base_workspace@pose)
                velocity=values['initial_state']['rigid_object'][name]['root_velocity'][0]
                _,index=self.backend.object_indices[name]
                world_pose=self.base_world@pose
                self.backend.data.qvel[index:index+3]=velocity[:3]
                self.backend.data.qvel[index+3:index+6]=world_pose[:3,:3].T@velocity[3:]
            self.backend.mj.mj_forward(self.backend.model,self.backend.data)
        self.previous=self.backend.data.ctrl[self.backend.actuator_ids].copy();self.after_reset()

    def state(self):
        s=self.backend.get_state();s['eef_pose']=np.linalg.inv(self.base_workspace)@s['eef_pose']
        s['object_poses']={k:np.linalg.inv(self.base_workspace)@v for k,v in s['object_poses'].items()}
        velocities={}
        for name in self.backend.metadata['dynamic_objects']:
            body=self.backend._id(self.backend.mj.mjtObj.mjOBJ_BODY,name);twist=np.empty(6)
            self.backend.mj.mj_objectVelocity(self.backend.model,self.backend.data,self.backend.mj.mjtObj.mjOBJ_BODY,body,twist,0)
            velocities[name]=np.r_[twist[3:],twist[:3]]
        s['object_velocities_world']=velocities;return s
    def images(self):return {name:self.backend.get_rgb(name) for name in self.camera_names}
    def contacts(self):return self.backend.finger_contacts(self.grasp_object)
    def place_success(self):
        if getattr(self,'processing_runtime',None):
            if self.place_target is None:return None
            return self.backend.success_details(object_id=self.grasp_object,target_id=self.place_target,condition=self.place_condition)['success']
        return self.backend.is_success()

    def target_eef_pose_to_action(self,target_eef_pose_dict,gripper_action_dict,action_noise_dict=None,env_id=0):
        self._check_stop();self._check_step_limit()
        noise=float((action_noise_dict or {}).get('tool0',0.))
        pose=self.resolve_target(target_eef_pose_dict['tool0'],noise)
        grip=float(gripper_action_dict['tool0'].reshape(-1)[0]);self.pending_pose=pose
        measured=self.backend.data.qpos[self.backend.q_indices].copy()
        arm=measured[:5] if self.ik_solver=='pinocchio_dls' else self.previous[:5]
        self.backend.last_ik=None
        try:command=self.backend.solve_pose(self.base_workspace@pose,np.r_[arm,grip],solver=self.ik_solver,
            failure_policy=self.ik_failure_policy,measured=measured,previous_command=self.previous)
        finally:self.record_ik(self.backend.last_ik)
        return torch.as_tensor(command,dtype=torch.float64)

    def step(self,actions):
        self._check_stop();q=actions[0].detach().cpu().numpy()
        if hasattr(self,'control'):self.control.before_step(self.worker_index,self.stop)
        self.record_frame(q,self.pending_pose)
        t=time.perf_counter();self.backend.step(q);self.metrics['physics_s']+=time.perf_counter()-t
        self.previous=q.copy();self.update_grasp();self.obs_buf={}
        if getattr(self,'preview',None):self.preview(self)
        return self.obs_buf,None,None,None,None

    def close(self):super().close();self.backend.close()


class IsaacAdapter(SimulationAdapter):
    """One independent episode slot in an Isaac vector scene, stepped by its coordinator."""
    def __init__(self,native,index,output,seed,stop,contact_views=None,require_standard_cameras=True):
        super().__init__(native.cfg.workflow_workspace,1/native.step_dt,output,seed,stop,require_standard_cameras)
        self.native=native;self.index=index
        self.contact_views=contact_views
        from ..utils.so101_kinematics import SO101Kinematics
        self.kinematics=SO101Kinematics(str(self.workspace['root']/native.cfg.simulation_configuration['robot_model']['urdf']),
            target_frame_name='tool0',joint_names=list(JOINTS[:5]))
        robot=native.scene['robot'];self.ids=[robot.joint_names.index(n) for n in JOINTS]
        self.previous=np.array([self.workspace['task']['robot_initial_joint_rad'][n] for n in JOINTS])
        self.limits=robot.data.joint_pos_limits[0,self.ids].detach().cpu().numpy()
        self.last_ik=None;self.action_queue=None

    def reset(self,env_id=0):
        self._check_stop()
        others=[i for i in range(self.native.num_envs) if i!=self.index]
        robot=self.native.scene['robot'];before=robot.data.joint_pos[others].clone()
        object_before={n:self.native.scene[n].data.root_state_w[others].clone() for n in self.workspace['task']['dynamic']}
        self.native.reset(env_ids=torch.tensor([self.index],device=self.native.device))
        if others:
            if not torch.equal(before,robot.data.joint_pos[others]):raise RuntimeError('Reset changed another environment joint state')
            if any(not torch.equal(v,self.native.scene[n].data.root_state_w[others]) for n,v in object_before.items()):
                raise RuntimeError('Reset changed another environment rigid-body state')
            self.metrics['independent_reset_checks']+=1
        self.previous=np.array([self.workspace['task']['robot_initial_joint_rad'][n] for n in JOINTS]);self.after_reset()

    def restore(self,values,attrs):
        self.reset();robot=self.native.scene['robot'];device=self.native.device;ids=torch.tensor([self.index],device=device)
        q=torch.as_tensor(values['initial_state']['articulation']['robot']['joint_position'],dtype=torch.float32,device=device)
        velocity=torch.as_tensor(values['initial_state']['articulation']['robot']['joint_velocity'],dtype=torch.float32,device=device)
        robot.write_joint_state_to_sim(q,velocity,joint_ids=self.ids,env_ids=ids);robot.set_joint_position_target(q,joint_ids=self.ids,env_ids=ids)
        for name in self.workspace['task']['dynamic']:
            root=np.array(values['initial_state']['rigid_object'][name]['root_pose'],copy=True)
            root[:,:3]+=self.native.scene.env_origins[self.index].cpu().numpy()
            obj=self.native.scene[name];obj.write_root_pose_to_sim(torch.as_tensor(root,dtype=torch.float32,device=device),env_ids=ids)
            vel=values['initial_state']['rigid_object'][name]['root_velocity']
            obj.write_root_velocity_to_sim(torch.as_tensor(vel,dtype=torch.float32,device=device),env_ids=ids)
        self.native.scene.write_data_to_sim();self.native.sim.forward();self.native.scene.update(self.native.physics_dt)
        self.native.obs_buf=self.native.observation_manager.compute();self.previous=q[0].cpu().numpy().copy();self.after_reset()

    def state(self):
        from ..utils.transforms import pose_matrix
        robot=self.native.scene['robot'];i=self.index
        root=robot.data.root_pose_w[i].detach().cpu().numpy();base=pose_matrix(root)
        tool=robot.body_names.index('tool0');ee=pose_matrix(torch.cat([robot.data.body_link_pos_w[i,tool],robot.data.body_link_quat_w[i,tool]]).cpu().numpy())
        names=self.workspace['task']['dynamic']+list(self.workspace['task']['targets'])
        objects={n:np.linalg.inv(base)@pose_matrix(self.native.scene[n].data.root_pose_w[i].cpu().numpy()) for n in names}
        return {'joint_pos':robot.data.joint_pos[i,self.ids].cpu().numpy().copy(),
            'joint_vel':robot.data.joint_vel[i,self.ids].cpu().numpy().copy(),'eef_pose':np.linalg.inv(base)@ee,'object_poses':objects,
            'object_velocities_world':{n:torch.cat([self.native.scene[n].data.root_lin_vel_w[i],self.native.scene[n].data.root_ang_vel_w[i]]).cpu().numpy().copy() for n in names}}

    def images(self):
        return {n:self.native.scene['camera_'+n+'view'].data.output['rgb'][self.index,...,:3].cpu().numpy().copy() for n in self.camera_names}
    def contacts(self):
        if self.contact_views is not None:
            return [bool(torch.linalg.vector_norm(v.get_contact_force_matrix(self.native.physics_dt))>.01)
                    for v in self.contact_views]
        return [bool(torch.linalg.vector_norm(self.native.scene[n].data.force_matrix_w[self.index,0,0])>1e-4)
                for n in ('grasp_fixed_contact','grasp_moving_contact')]
    def place_success(self):
        from ..real2sim.workspaces.environment import task_success
        return bool(task_success(self.native,self.workspace)[self.index])

    def target_eef_pose_to_action(self,target_eef_pose_dict,gripper_action_dict,action_noise_dict=None,env_id=0):
        self._check_stop();self._check_step_limit();pose=self.resolve_target(target_eef_pose_dict['tool0'],float((action_noise_dict or {}).get('tool0',0.)));self.pending_pose=pose
        from ..simulation.ik import solve_selected,IKFailure
        grip=float(gripper_action_dict['tool0'].reshape(-1)[0])
        measured=self.native.scene['robot'].data.joint_pos[self.index,self.ids].cpu().numpy().copy()
        arm=measured[:5] if self.ik_solver=='pinocchio_dls' else self.previous[:5]
        self.last_ik=None
        try:command,self.last_ik=solve_selected(self.kinematics.forward_kinematics,pose,np.r_[arm,grip],self.limits,solver=self.ik_solver,kinematics=self.kinematics,
            failure_policy=getattr(self,"ik_failure_policy","strict"),measured=measured,previous_command=self.previous)
        except IKFailure as exc:self.last_ik=exc.details;raise
        finally:self.record_ik(self.last_ik)
        return torch.as_tensor(command,dtype=torch.float64)

    def before_batch_step(self,command):self.record_frame(command,self.pending_pose)
    def after_batch_step(self,command):self.previous=np.array(command,copy=True);self.update_grasp();self.obs_buf={}
