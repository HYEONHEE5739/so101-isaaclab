"""Small provenance sidecars; never remap already-model-space dataset actions."""
import hashlib
import json
from pathlib import Path
import warnings
from .so101_joint_mapping import DEFAULT_MAPPER, JOINTS

JOINT_SPACE = 'so101_urdf_joint_targets_rad'
MIMIC_SPACE = 'base_frame_tcp_delta_m_rotvec_rad_plus_gripper_urdf_rad'


def robot_identity():
    root = Path(__file__).resolve().parents[3]
    files = [root / 'assets/SO101/urdf/so101_isaaclab.urdf']
    files += sorted((root / 'assets/SO101/usd').rglob('*.usd'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'scope': 'URDF and composed USD files; external mesh files not included', 'files': hashes,
            'sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def current_contract(action_space):
    return {'schema': 'so101.coordinate-contract/1', 'action_space': action_space,
            'joint_targets_space': JOINT_SPACE, 'joint_order': list(JOINTS),
            'mapping': DEFAULT_MAPPER.metadata(), 'robot_asset': robot_identity(),
            'provenance_status': 'recorded'}


def sidecar(path):
    p = Path(path)
    return Path(str(p) + '.so101.json') if p.suffix in ('.hdf5', '.h5') else p / 'so101_coordinates.json'


def read_contract(path):
    f = sidecar(path)
    if f.exists():
        contract = json.loads(f.read_text())
    elif Path(path).suffix in ('.hdf5', '.h5'):
        import h5py
        with h5py.File(path, 'r') as h:
            contract = json.loads(h['data'].attrs.get('env_args', '{}')).get('so101_coordinates')
    else:
        contract = None
    if contract is None:
        warnings.warn(f'{path}: no mapping/asset provenance; retain legacy unknown, do not relabel as current',
                      UserWarning, stacklevel=2)
        return {'schema': 'so101.coordinate-contract/1', 'provenance_status': 'legacy_unknown',
                'mapping': None, 'robot_asset': None, 'action_space': None}
    if contract.get('schema') != 'so101.coordinate-contract/1':
        raise ValueError('Unsupported SO101 coordinate contract')
    return contract


def write_contract(path, contract):
    f = sidecar(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        if json.loads(f.read_text()) != contract:
            raise ValueError(f'Coordinate contract differs; do not mix datasets: {path}')
        return
    # Exclusive creation prevents silently replacing another run's provenance.
    with f.open('x') as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)


def require_resume_contract(path, expected):
    actual = read_contract(path)
    if actual != expected:
        raise ValueError('Dataset mapping/asset/action provenance differs or is unknown; use a new dataset')


def inherit_contract(source, destination, action_space):
    contract = read_contract(source)
    source_space = contract.get('action_space')
    if source_space is not None and source_space != MIMIC_SPACE:
        raise ValueError(f'Mimic input is not Cartesian source actions: {source_space}')
    contract = {**contract, 'action_space': action_space, 'source_dataset': str(Path(source).resolve())}
    # Keep original mapping and asset identities, including unknowns; never stamp the current mapper.
    write_contract(destination, contract)
    return contract


def require_joint_replay(path):
    contract = read_contract(path)
    space = contract.get('joint_targets_space')
    if space not in (None, JOINT_SPACE):
        raise ValueError('joint_targets are not in the expected URDF coordinate')
    order = contract.get('joint_order')
    if order is not None and order != list(JOINTS):
        raise ValueError('Dataset joint order differs; explicit reorder required')
    return contract
