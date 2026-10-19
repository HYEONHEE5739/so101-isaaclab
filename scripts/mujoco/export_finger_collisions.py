#!/usr/bin/env python3
"""Offline SO101 collision asset export. CoACD is never imported by a worker.

Use an isolated CoACD tool path for this command. The generated JSON contains
convex parts in the original STL frame; URDF origin and scale remain runtime inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    import coacd
    import numpy as np
    import trimesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,default=Path('assets/SO101/urdf/so101_isaaclab.urdf'))
    parser.add_argument('--output',type=Path,default=Path('assets/SO101/mujoco/finger_convex_parts.json'))
    parser.add_argument('--threshold-m',type=float,default=.00075)
    args=parser.parse_args();urdf=args.urdf.resolve();tree=ET.parse(urdf).getroot()
    config=dict(threshold=args.threshold_m,real_metric=True,seed=0,max_convex_hull=96,
                resolution=2000,mcts_iterations=100,preprocess_mode='off',merge=True,decimate=False)
    coacd.set_log_level('warn');entries=[]
    for link in ('gripper','jaw'):
        for index,node in enumerate(tree.find(f"link[@name='{link}']").findall('collision')):
            path=(urdf.parent/node.find('geometry/mesh').get('filename')).resolve()
            mesh=trimesh.load_mesh(path,process=True)
            if not mesh.is_watertight:raise ValueError('Watertight source required: '+str(path))
            print('Decomposing',link,index,path.name,flush=True)
            parts=coacd.run_coacd(coacd.Mesh(mesh.vertices,mesh.faces),**config)
            if not parts:raise ValueError('Empty convex decomposition')
            entries.append({'link':link,'collision_index':index,'mesh':path.name,
                            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'parts':[{'vertices':np.asarray(v).round(12).tolist(),'faces':np.asarray(f).tolist()} for v,f in parts]})
            print('Completed',link,index,'convex parts:',len(parts),flush=True)
    value={'schema':'so101.finger-convex-parts/1','generator':'CoACD '+importlib.metadata.version('coacd'),
           'configuration':config,'source':'URDF collision STL; original mesh frame; offline export',
           'license':'CoACD: MIT; geometry derived from existing SO101 assets','meshes':entries}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,separators=(',',':'))+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
