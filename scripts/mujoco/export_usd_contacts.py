#!/usr/bin/env python3
"""Offline audit/export of USD-only SO101 pads. Never imported by a MuJoCo worker.

Run using the installed Isaac USD libraries (pxr); this does not start Isaac Sim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    from pxr import Usd, UsdGeom
    p = argparse.ArgumentParser()
    p.add_argument('--asset-root', type=Path, default=Path('assets/SO101'))
    p.add_argument('--output', type=Path, default=Path('assets/SO101/mujoco/usd_contact_pads.json'))
    args = p.parse_args()
    root = args.asset_root.resolve()
    stage = Usd.Stage.Open(str(root/'usd/so101_isaaclab.usd'))
    pads = []
    for link in ('gripper', 'jaw'):
        path = '/so101_calibrated/' + link + '/Cube'
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        pose = matrix.copy(); pose[:3, :3] /= scale
        low, high = points.min(axis=0), points.max(axis=0)
        pose[:3, 3] += matrix[:3, :3] @ ((low + high)/2)
        material = stage.GetPrimAtPath('/so101_calibrated/gripper_collider_material')
        pads.append({'link': link, 'source_prim': path, 'T_link': pose.tolist(),
                     'dimensions_m': ((high-low)*scale).tolist(),
                     'collision': bool(prim.GetAttribute('physics:collisionEnabled').Get()),
                     'dynamic_friction': material.GetAttribute('physics:dynamicFriction').Get(),
                     'static_friction': material.GetAttribute('physics:staticFriction').Get(),
                     'rgba': [.18, .18, .18, 1], 'appearance_source': 'provisional matte gray; USD has no display material'})
    data = {'schema': 'so101.usd-contact-pads/1', 'source': 'offline USD extraction; no worker USD dependency',
            'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted((root/'usd').rglob('*.usd'))}, 'pads': pads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__': main()
