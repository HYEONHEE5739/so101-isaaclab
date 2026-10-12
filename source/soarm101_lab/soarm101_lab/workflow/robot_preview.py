"""CPU-only URDF visual preview in robot-base coordinates (not physics replay)."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from ..so101_joint_mapping import JOINTS


def rotation(axis, angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        r,p,y=np.fromstring(element.get('rpy','0 0 0'),sep=' ')
        t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
    return t


class RobotPreview:
    def __init__(self,path):
        path=Path(path);root=ET.parse(path).getroot()
        self.joints=root.findall('joint');self.visuals=[]
        for link in root.findall('link'):
            for visual in link.findall('visual'):
                mesh=visual.find('geometry/mesh')
                if mesh is None:continue
                raw=(path.parent/mesh.get('filename')).read_bytes()
                count=int.from_bytes(raw[80:84],'little')
                if len(raw)!=84+count*50:raise ValueError('Preview requires binary STL')
                dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
                tri=np.frombuffer(raw,dtype=dtype,offset=84,count=count)['vertices'].astype(float)
                tri*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                # Vertex clustering reduces CPU rendering cost; production meshes unchanged.
                flat=tri.reshape(-1,3);cells=np.round(flat/.0045).astype(int)
                _,inverse=np.unique(cells,axis=0,return_inverse=True)
                vertices=np.zeros((inverse.max()+1,3));np.add.at(vertices,inverse,flat)
                vertices/=np.bincount(inverse)[:,None]
                faces=inverse.reshape(-1,3)
                faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])]
                _,idx=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
                tri=vertices[faces[idx]]
                t=origin(visual.find('origin'));tri=tri@t[:3,:3].T+t[:3,3]
                mat=visual.find('material');dark=mat is not None and mat.get('name')=='sts3215'
                self.visuals.append((link.get('name'),tri,dark))

    def transforms(self,q):
        values=dict(zip(JOINTS,q));poses={'base':np.eye(4)};pending=list(self.joints)
        while pending:
            ready=[j for j in pending if j.find('parent').get('link') in poses]
            if not ready:raise ValueError('Disconnected URDF tree')
            for j in ready:
                t=origin(j.find('origin'))
                if j.get('type')!='fixed':
                    axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                    t[:3,:3]=t[:3,:3]@rotation(axis,values[j.get('name')])
                poses[j.find('child').get('link')]=poses[j.find('parent').get('link')]@t
                pending.remove(j)
        return poses

    def triangles(self,q):
        poses=self.transforms(q);tri=[];dark=[]
        for link,vertices,is_dark in self.visuals:
            t=poses[link];tri.append(vertices@t[:3,:3].T+t[:3,3]);dark.extend([is_dark]*len(vertices))
        return np.concatenate(tri),np.asarray(dark)
