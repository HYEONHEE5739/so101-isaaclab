"""Shared SO101 limits and configured actuator values, without engine imports."""
from pathlib import Path
from functools import lru_cache
import math
import xml.etree.ElementTree as ET

ROBOT_URDF = Path(__file__).resolve().parents[3] / 'assets/SO101/urdf/so101_isaaclab.urdf'


def joint_limits(urdf=ROBOT_URDF):
    from .so101_joint_mapping import JOINTS
    nodes = {j.get('name'): j.find('limit') for j in ET.parse(urdf).findall('joint')}
    return {n: tuple(float(nodes[n].get(k)) for k in ('lower','upper')) for n in JOINTS}


def model_without_limits(urdf):
    """Compare all model content; only revolute lower/upper may differ."""
    tree = ET.fromstring(urdf) if isinstance(urdf, bytes) else ET.parse(urdf).getroot()
    from .so101_joint_mapping import JOINTS
    for j in tree.findall('joint'):
        if j.get('name') in JOINTS:
            for key in ('lower','upper'): j.find('limit').attrib.pop(key)
    def node(e):
        return (e.tag, sorted(e.attrib.items()), (e.text or '').strip(), [node(c) for c in e])
    return node(tree)


@lru_cache(maxsize=32)
def urdf_bytes(sha256):
    """Read exact hash-matching model evidence; never infer identity from a filename."""
    import hashlib
    import subprocess
    root = ROBOT_URDF.parents[3]
    data = ROBOT_URDF.read_bytes()
    if hashlib.sha256(data).hexdigest() == sha256: return data
    # Saved profiles predate structural fingerprints. Git provides the original
    # complete model, including origin/axis/link topology and non-limit fields.
    relative = 'assets/SO101/urdf/so101_isaaclab.urdf'
    result = subprocess.run(['git', '-C', str(root), 'log', '--format=%H', '--', relative],
                            capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        for revision in result.stdout.splitlines():
            result = subprocess.run(['git', '-C', str(root), 'show', revision + ':' + relative],
                                    capture_output=True, timeout=10)
            if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == sha256:
                return result.stdout
    # Immutable workspace/capture models are evidence too; only exact bytes count.
    for path in (root / 'outputs').rglob('*.urdf'):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha256: return data
    raise ValueError('Precision profile URDF identity conflict: original/current model evidence unavailable')


def limit_usd(source, urdf, output):
    """New USD layer only: no input USD/Workspace mutation or engine-global paths."""
    from pxr import Usd, UsdPhysics
    import os
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists(): raise ValueError('USD limit output must be a new file')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, output.parent)]
    source_stage = Usd.Stage.Open(str(source))
    default = source_stage.GetDefaultPrim()
    if not default: raise ValueError('Source USD has no default prim')
    # defaultPrim is layer metadata and is not inherited from a sublayer.
    stage.SetDefaultPrim(stage.GetPrimAtPath(default.GetPath()))
    limits = joint_limits(urdf); found = set()
    for prim in stage.Traverse():
        if prim.GetName() in limits and prim.IsA(UsdPhysics.RevoluteJoint):
            found.add(prim.GetName()); joint = UsdPhysics.RevoluteJoint(prim)
            lo, hi = limits[prim.GetName()]
            # Preserve unchanged legacy wrist/gripper USD values (URDF decimal
            # rounding differs from their exact degree values by <0.001 degree).
            if abs(joint.GetLowerLimitAttr().Get()-math.degrees(lo)) > .001:
                joint.GetLowerLimitAttr().Set(math.degrees(lo))
            if abs(joint.GetUpperLimitAttr().Get()-math.degrees(hi)) > .001:
                joint.GetUpperLimitAttr().Set(math.degrees(hi))
    if found != set(limits): raise ValueError('USD does not contain the six expected joints')
    stage.GetRootLayer().Save()
    return output


# Historical misspelling is a public import; retain it with one URDF source of truth.
SO101_FOLLOWER_USD_JOINT_LIMLITS = {n: tuple(map(math.degrees, v)) for n,v in joint_limits().items()}

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
    if 'precision_mapping_sha256' in node:
        import re
        if not isinstance(node['precision_mapping_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', node['precision_mapping_sha256']):
            raise ValueError('Invalid Precision mapping binding for simulation joint offsets')
    return values


def require_precision_simulation_binding(node, mapping):
    """A fitted model residual is valid only after the exact recorded mapping."""
    reference = node.get('precision_mapping_sha256')
    if (mapping.get('schema') != 'so101.precision-joint-mapping/1'
            or not reference or mapping.get('sha256') != reference):
        raise ValueError('Scene joint correction requires the Precision mapping used for its fit')
