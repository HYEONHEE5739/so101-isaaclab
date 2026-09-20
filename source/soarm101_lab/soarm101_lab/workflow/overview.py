"""Inspection camera framing from workspace bounds and model link lengths.

This camera is not a calibrated observation and never enters policy/recording inputs.
"""
import numpy as np
import xml.etree.ElementTree as ET


def framing(workspace):
    from scipy.spatial.transform import Rotation
    p = workspace['profile']
    base = np.asarray(p['robot_base']['T_world'])[:3, 3]
    transform = np.asarray(p['workspace']['T_world'])
    points = [base, transform[:3, 3]]
    for name, obj in p['objects'].items():
        if 'T_workspace' in obj and not name.startswith('background_'):
            points.append((transform @ np.asarray(obj['T_workspace']))[:3, 3])
    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
    urdf = ET.parse(workspace['root'] / 'assets/SO101/urdf/so101_isaaclab.urdf')
    reach = sum(np.linalg.norm(np.fromstring(o.get('xyz', '0 0 0'), sep=' '))
                for o in urdf.findall('./joint/origin'))
    radius = max(np.linalg.norm(np.asarray(points) - center, axis=1).max(), reach, .1)
    center[2] = max(center[2], base[2] + reach * .3)
    # View from the same open side as the existing camera, not behind a backdrop.
    direction = -np.asarray(p['cameras']['side']['T_parent_camera'])[:3, 2].copy()
    if np.linalg.norm(direction[:2]) < 1e-6:
        direction[:2] = (1., 1.)
    direction[2] = max(direction[2], .5)
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius * 1.5
    forward = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    q = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return tuple(eye), (float(q[3]), *map(float, q[:3]))
