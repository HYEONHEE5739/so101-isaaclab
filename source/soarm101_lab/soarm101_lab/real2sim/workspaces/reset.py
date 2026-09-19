"""Workspace-local randomization; never uses legacy pick_place world coordinates."""
import copy
import numpy as np


class WorkspaceRandomizer:
    def __init__(self, workspace):
        self.workspace=workspace
        self.rng=np.random.default_rng(workspace['task']['reset']['seed'])

    def sample(self):
        p,t=self.workspace['profile'],self.workspace['task'];r=t['reset'];support=p['objects'][r['support_object']]
        center=np.array(support['T_workspace'])[:3,3];half=np.array(support['dimensions_m'])/2
        placed={};sizes={}
        for name in t['dynamic']:
            size=np.array(p['objects'][name]['dimensions_m']);limit=half[:2]-size[:2]/2-r['margin_m']
            for _ in range(1000):
                xy=center[:2]+self.rng.uniform(-limit,limit)
                if all(np.any(np.abs(xy-np.array(pos[:2]))>(size[:2]+sizes[n][:2])/2+r['separation_m']) for n,pos in placed.items()):break
            else:raise ValueError('Unable to place non-overlapping objects in workspace')
            placed[name]=[*xy,float(center[2]+half[2]+size[2]/2+r['clearance_m'])];sizes[name]=size
        return {'frame':'workspace','cube_positions':placed,'joint_positions':copy.deepcopy(t['robot_initial_joint_rad'])}


def reset_workspace(env,env_ids,workspace,random_state=None):
    import torch
    from ..profile import pose
    from ..scene import apply_robot_appearance
    from isaaclab.sim import get_current_stage
    if env.num_envs!=1:raise ValueError('Workspace Phase 1 supports one environment')
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    if len(ids)==0:return
    if random_state is None:
        if not hasattr(env,'_workspace_randomizer'):env._workspace_randomizer=WorkspaceRandomizer(workspace)
        random_state=env._workspace_randomizer.sample()
    if not isinstance(random_state,dict) or random_state.get('frame')!='workspace':
        raise ValueError('Expected workspace-local reset state; legacy world reset state is not compatible')
    world=np.array(workspace['profile']['workspace']['T_world']);_,quat=pose(world)
    for name,xyz in random_state['cube_positions'].items():
        obj=env.scene[name];point=world[:3,:3]@np.array(xyz)+world[:3,3]
        state=torch.tensor([[*point,*quat]],device=env.device,dtype=torch.float32)
        state[:,:3]+=env.scene.env_origins[ids];obj.write_root_pose_to_sim(state,env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros((len(ids),6),device=env.device),env_ids=ids)
    robot=env.scene['robot'];q=torch.tensor([[random_state['joint_positions'][n] for n in robot.joint_names]],device=env.device)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids);robot.set_joint_position_target(q,env_ids=ids)
    apply_robot_appearance(get_current_stage(),workspace['profile'])
