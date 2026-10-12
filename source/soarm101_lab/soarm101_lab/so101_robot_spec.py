"""Shared SO101 limits and configured actuator values, without engine imports."""
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    'shoulder_pan': (-110., 110.), 'shoulder_lift': (-100., 100.),
    'elbow_flex': (-100., 90.), 'wrist_flex': (-95., 95.),
    'wrist_roll': (-160., 160.), 'gripper': (-10, 100.),
}

# Existing Isaac values, not measured hardware motor parameters.
SO101_POSITION_FORCE_LIMIT = 30.
SO101_POSITION_GAINS = {
    'shoulder_pan': (55., .7), 'shoulder_lift': (30., .8),
    'elbow_flex': (25., .7), 'wrist_flex': (12., .5),
    'wrist_roll': (7., .5), 'gripper': (4., .3),
}


def simulation_joint_offsets(node, names):
    """Optional model-frame zero offsets; hardware and recorded q stay unchanged."""
    import numpy as np
    if node is None:
        return np.zeros(len(names))
    values = np.asarray(node.get('value'), dtype=float)
    if (values.shape != (len(names),) or not np.isfinite(values).all()
            or np.any(abs(values) > .35)
            or node.get('provenance') not in ('estimated', 'optimized')
            or not isinstance(node.get('evidence'), list) or not node['evidence']):
        raise ValueError('Invalid unvalidated simulation joint offsets')
    return values
