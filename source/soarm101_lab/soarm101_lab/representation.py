"""Explicit model-radian contract; mapping occurs only at hardware boundaries."""
import copy
import hashlib
import json
import numpy as np
from .so101_joint_mapping import JOINTS, DEFAULT_MAPPER

CONTRACT_ID = 'MODEL_JOINT_RAD_V1'


def describe(coordinates):
    """Describe existing coordinates, never relabel unknown data as current."""
    if coordinates.get('provenance_status') == 'legacy_unknown':
        raise ValueError('Unknown coordinate provenance requires explicit migration')
    from .so101_dataset_contract import JOINT_SPACE, MIMIC_SPACE
    if (coordinates.get('action_space') not in (JOINT_SPACE, MIMIC_SPACE)
            or coordinates.get('joint_order') != list(JOINTS)
            or coordinates.get('joint_targets_space') != JOINT_SPACE):
        raise ValueError('Unknown action/joint representation; do not relabel as model radians')
    return {
        'id': CONTRACT_ID, 'version': 1, 'joint_order': list(JOINTS),
        'observation': {'semantic': 'absolute_joint_position', 'units': ['rad'] * 6},
        'action': {'semantic': coordinates['action_space'],
                   'units': ['m'] * 3 + ['rad'] * 4 if 'tcp_delta' in coordinates['action_space'] else ['rad'] * 6},
        'gripper': 'URDF joint rad; hardware RANGE_0_100 at boundary only',
        'mapper': copy.deepcopy(coordinates['mapping']),
        'robot_asset': copy.deepcopy(coordinates['robot_asset']),
        'normalization': 'dataset statistics applied by LeRobot pre/post processors, not by joint mapper',
    }


def identity(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require_compatible(actual, expected, action_space=None):
    for key in ('joint_order', 'mapping', 'robot_asset'):
        if not actual.get(key) or actual[key] != expected.get(key):
            raise ValueError(f'Representation mismatch: {key}')
    if action_space is not None and actual.get('action_space') != action_space:
        raise ValueError('Representation mismatch: action semantic')
    return actual


def require_policy(path, expected):
    from .so101_dataset_contract import read_contract, JOINT_SPACE
    return require_compatible(read_contract(path), expected, JOINT_SPACE)


def real_observation(native, mapper=None):
    """LeRobot .pos degree/% dictionary -> ordered model rad vector."""
    mapper = mapper or DEFAULT_MAPPER
    q = mapper.calibrated_to_urdf({n:native[n + '.pos'] for n in JOINTS})
    return np.asarray([q[n] for n in JOINTS],dtype=np.float32)


def real_action(model_radians, mapper=None):
    """Postprocessed policy rad -> LeRobot degree/% dictionary, once."""
    q = np.asarray(model_radians)
    if q.shape != (6,):
        raise ValueError('Expected one six-joint policy action')
    result = (mapper or DEFAULT_MAPPER).urdf_to_calibrated(dict(zip(JOINTS,q)))
    return {n + '.pos': float(result[n]) for n in JOINTS}
