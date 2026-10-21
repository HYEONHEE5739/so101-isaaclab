"""Pure recorded pose transforms shared by readers and simulation."""
import numpy as np


def pose_matrix(value, quaternion_order='wxyz'):
    """Recorded SE(3), without a simulator or SciPy dependency in the UI."""
    value=np.asarray(value,dtype=float)
    if value.shape[-2:]==(4,4):
        if not np.isfinite(value).all():raise ValueError('Nonfinite recorded pose')
        return value.copy()
    if value.shape[-1]!=7:raise ValueError('Expected matrix4 or xyz + quaternion pose')
    q=value[...,3:7]
    if quaternion_order=='xyzw':q=q[...,[3,0,1,2]]
    elif quaternion_order!='wxyz':raise ValueError('Unknown quaternion order')
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(value).all() or np.any(norm<1e-10):raise ValueError('Invalid recorded pose')
    w,x,y,z=np.moveaxis(q/norm,-1,0)
    result=np.broadcast_to(np.eye(4),value.shape[:-1]+(4,4)).copy()
    result[...,0,:3]=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],-1)
    result[...,1,:3]=np.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],-1)
    result[...,2,:3]=np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1)
    result[...,:3,3]=value[...,:3]
    return result



def as_pose7(matrix):
    from scipy.spatial.transform import Rotation
    matrix=np.asarray(matrix)
    q=Rotation.from_matrix(matrix[..., :3, :3].reshape(-1,3,3)).as_quat()
    return np.concatenate([matrix[..., :3,3].reshape(-1,3),q[:,[3,0,1,2]]],-1).reshape(matrix.shape[:-2]+(7,))


def cartesian_actions(targets,initial,gripper):
    from scipy.spatial.transform import Rotation
    previous=np.concatenate([np.asarray(initial)[None],targets[:-1]])
    rotation=Rotation.from_matrix(targets[:,:3,:3]@previous[:,:3,:3].transpose(0,2,1)).as_rotvec()
    return np.column_stack([targets[:,:3,3]-previous[:,:3,3],rotation,gripper])
