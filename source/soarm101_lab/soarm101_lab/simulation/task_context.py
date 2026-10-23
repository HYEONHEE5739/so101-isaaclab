"""Borrow an existing single-environment Backend; no creation, stepping or state cache."""


class TaskContext:
    """Read-only task access to an already configured Backend.

    The caller owns reset/step/close. Poses are 4x4 workspace-frame matrices (metres).
    Runtime support currently requires the existing MuJoCo Backend contract.
    Metadata is borrowed; task code must not mutate it. Python tasks are trusted code.
    """
    def __init__(self, backend, *, adapter=None):
        if backend.metadata.get('backend') != 'mujoco' or backend.metadata.get('env_count') != 1:
            raise NotImplementedError('TaskContext currently supports a single MuJoCo Backend')
        self.adapter = adapter
        self.backend = backend
        self.workspace = backend.workspace

    @property
    def object_ids(self):
        return tuple(self.backend.metadata['dynamic_objects'] + self.backend.metadata['target_objects'])

    @property
    def camera_ids(self):
        # Declared Metadata IDs, not a promise that every sensor is instantiated.
        return tuple(self.workspace['profile']['cameras'])

    @property
    def robot_asset(self):
        return self.workspace['profile']['robot_asset']

    def require(self, *, objects=(), cameras=()):
        missing = set(objects) - set(self.object_ids)
        if missing:raise ValueError('Task objects absent from runtime: '+str(sorted(missing)))
        missing = set(cameras) - set(self.camera_ids)
        if missing:raise ValueError('Unknown Metadata camera IDs: '+str(sorted(missing)))

    def validate_bindings(self, task):
        """Check declarations before user setup, without creating sensors or caching state."""
        self.backend._open()
        self.require(objects=task['OBJECT_IDS'], cameras=task['CAMERA_IDS'])
        for name in task['OBJECT_IDS']:
            self.backend._id(self.backend.mj.mjtObj.mjOBJ_BODY, name)
        for name in task['CAMERA_IDS']:
            self.backend._id(self.backend.mj.mjtObj.mjOBJ_CAMERA, name)
        if task.get('ROBOT_ASSET') is not None and task['ROBOT_ASSET'] != self.robot_asset:
            raise ValueError('Task robot asset reference differs from runtime Workspace')
        digest=task.get('ROBOT_ASSET_SHA256')
        if digest is not None and digest != self.workspace.get('manifest',{}).get('robot_asset_sha256'):
            raise ValueError('Task robot asset identity differs or is unavailable in validated Workspace')
        if task.get('TASK_DEFINITION') is not None and task['TASK_DEFINITION'] != self.workspace.get('task_definition'):
            raise ValueError('Task definition differs from explicitly configured scene')

    def object_pose(self, object_id):
        self.require(objects=(object_id,))
        observation=getattr(self,'_observation',None)
        if observation is not None:
            return self.adapter.base_workspace@observation['object_poses'][object_id]
        return self.backend.get_object_pose(object_id)

    def joint_state(self):
        """Measured model-rad state from existing Backend; no command conversion."""
        observation=getattr(self,'_observation',None)
        if observation is not None:return observation['joint_pos'].copy()
        return self.backend.get_state()['joint_pos']

    def placement_details(self, definition):
        """Existing engine-specific placement predicate; not grasp/annotation success."""
        if definition != self.workspace.get('task_definition') or definition is None:
            raise ValueError('Placement definition differs from explicitly configured scene')
        return self.backend.success_details()

    def placement(self, object_id: str, target_id: str, condition: dict | None = None) -> dict:
        """Explicit bindings in this compiled scene; reuse Backend mathematics."""
        self.require(objects=(object_id, target_id))
        return self.backend.success_details(object_id=object_id, target_id=target_id, condition=condition)


def validate_scene_objects(workspace, objects):
    """Validate explicit dynamic poses against loaded Metadata, without an engine.

    Positions are workspace metres; optional quat_wxyz is normalized. Fixed
    targets can be selected but moved only by a Builder re-export.
    """
    import numpy as np
    if not isinstance(objects,dict) or not objects:raise ValueError('Declare a nonempty Scene object mapping')
    dynamic=set(workspace['task']['dynamic']);targets=set(workspace['task']['targets'])
    if not set(objects)<=dynamic|targets:raise ValueError('Unknown Scene Object IDs: '+str(set(objects)-dynamic-targets))
    for name,spec in objects.items():
        if not isinstance(spec,dict):raise ValueError('Object configuration must be a dictionary')
        if set(spec)-{'pose','reset'} or ('pose' in spec and 'reset' in spec):raise ValueError('Use pose OR reset for '+name)
        if spec and name not in dynamic:raise ValueError('Pose/Reset override requires a dynamic Body: '+name)
        if 'pose' in spec:
            pose=spec['pose']
            if not isinstance(pose,dict) or set(pose)-{'xyz','quat_wxyz','frame'} or pose.get('frame','workspace')!='workspace':
                raise ValueError('Pose requires workspace-frame xyz metres and optional quat_wxyz')
            xyz=np.asarray(pose.get('xyz'),dtype=float)
            if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('Pose xyz requires three finite metres')
            if 'quat_wxyz' in pose:
                q=np.asarray(pose['quat_wxyz'],dtype=float)
                if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1,atol=1e-6):raise ValueError('quat_wxyz must be normalized')
        if 'reset' in spec and (not isinstance(spec['reset'],str) or not spec['reset'].isidentifier()):raise ValueError('Reset must name a callable')


def reset_objects(backend, objects, resets):
    """Apply explicit poses after the owner's normal reset; rollback on error.

    No engine creation/step/RNG ownership. Unspecified bodies remain at the
    normal reset pose. The Backend owns the final native initial-state snapshot.
    """
    import numpy as np
    from ..task_terms.reset import metadata_pose
    from scipy.spatial.transform import Rotation
    if not any(objects.values()):return
    b=backend;ctx=TaskContext(b);snapshot=b.get_initial_state();changed=[]
    try:
        for name,spec in objects.items():
            if not spec:continue
            if 'reset' in spec:pose=np.asarray(resets[name](ctx,name,b.randomizer.rng),dtype=float)
            else:
                pose=metadata_pose(ctx,name,b.randomizer.rng);override=spec['pose'];pose[:3,3]=override['xyz']
                if 'quat_wxyz' in override:
                    q=override['quat_wxyz'];pose[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            if (pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1])
                or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(pose[:3,:3])<.999999):
                raise ValueError('Reset must return finite workspace-frame SE(3): '+name)
            # Reuse the recorded support geometry. Rotated footprint must fit.
            profile=ctx.workspace['profile'];r=ctx.workspace['task']['reset'];support=profile['objects'][r['support_object']]
            support_pose=np.asarray(support['T_workspace']);local=np.linalg.inv(support_pose)@pose
            half=np.asarray(profile['objects'][name]['dimensions_m'])/2
            corners=np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])@local[:3,:3].T+local[:3,3]
            limits=np.asarray(support['dimensions_m'])/2
            if np.any(np.abs(corners[:,:2])>limits[:2]+1e-7) or corners[:,2].min()<limits[2]-1e-6:
                raise ValueError('Reset pose outside/inside support: '+name)
            b.set_object_pose(name,pose);changed.append(name)
        roots={b._id(b.mj.mjtObj.mjOBJ_BODY,n) for n in changed};ids=set(roots)
        for body in range(1,b.model.nbody):
            if int(b.model.body_parentid[body]) in ids:ids.add(body)
        for contact in b.data.contact:
            bodies={int(b.model.geom_bodyid[contact.geom1]),int(b.model.geom_bodyid[contact.geom2])}
            if ids&bodies and contact.dist < -1e-6:raise ValueError('Reset pose penetrates Collision geometry: '+str([b.mj.mj_id2name(b.model,b.mj.mjtObj.mjOBJ_BODY,i) for i in bodies])+' depth='+str(float(contact.dist)))
        b.initial_state=b._snapshot()
    except BaseException:
        b.restore_state(snapshot);raise
