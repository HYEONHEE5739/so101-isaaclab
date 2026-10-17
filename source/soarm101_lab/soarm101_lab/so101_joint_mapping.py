"""Authoritative calibrated-control <-> model-joint boundary (no hardware/Isaac imports).

The default preserves historical behavior, NOT verified physical correspondence.
Already-URDF actions, replay targets and IK outputs must not pass through this mapper.
"""
from dataclasses import dataclass, asdict
from collections.abc import Mapping
import hashlib
import json
import math
import numpy as np

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
POLICY_UNITS = ('rad',) * 5 + ('fraction',)


@dataclass(frozen=True)
class SO101SimJointMapper:
    arm_sign: tuple = (1, 1, 1, 1, 1)
    arm_zero_offset_rad: tuple = (0., 0., 0., 0., 0.)
    # Separate percent coordinate: preserve the legacy 0% -> 0rad, 100% -> 100deg.
    gripper_rad_per_percent: float = math.pi / 180
    gripper_zero_offset_rad: float = 0.
    evidence: str = 'legacy_unverified'
    version: str = 'so101.sim-joint-mapping/1'

    def __post_init__(self):
        object.__setattr__(self, 'arm_sign', tuple(self.arm_sign))
        object.__setattr__(self, 'arm_zero_offset_rad', tuple(self.arm_zero_offset_rad))
        if len(self.arm_sign) != 5 or any(s not in (-1, 1) for s in self.arm_sign):
            raise ValueError('Exactly five arm signs (+1/-1) required')
        if len(self.arm_zero_offset_rad) != 5 or not np.isfinite(self.arm_zero_offset_rad).all():
            raise ValueError('Exactly five finite arm offsets required')
        if not math.isfinite(self.gripper_rad_per_percent) or self.gripper_rad_per_percent == 0:
            raise ValueError('Gripper percent mapping must be finite and invertible')
        if not math.isfinite(self.gripper_zero_offset_rad):
            raise ValueError('Finite gripper offset required')
        if not self.evidence:
            raise ValueError('Mapping evidence/status required')

    def _input(self, values):
        named = isinstance(values, Mapping)
        if named:
            if set(values) != set(JOINTS):
                raise ValueError('Expected exact SO101 joint names')
            values = [values[n] for n in JOINTS]
        a = np.asarray(values, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != 6 or not np.isfinite(a).all():
            raise ValueError('Expected finite (..., 6) joint coordinates')
        return a, named

    @staticmethod
    def _output(a, named):
        return dict(zip(JOINTS, a.tolist())) if named else a

    def calibrated_to_urdf(self, values):
        a, named = self._input(values)
        out = np.empty_like(a)
        out[..., :5] = a[..., :5] * (math.pi / 180) * self.arm_sign + self.arm_zero_offset_rad
        out[..., 5] = a[..., 5] * self.gripper_rad_per_percent + self.gripper_zero_offset_rad
        return self._output(out, named)

    def urdf_to_calibrated(self, values):
        a, named = self._input(values)
        out = np.empty_like(a)
        out[..., :5] = (a[..., :5] - self.arm_zero_offset_rad) / self.arm_sign * (180 / math.pi)
        out[..., 5] = (a[..., 5] - self.gripper_zero_offset_rad) / self.gripper_rad_per_percent
        return self._output(out, named)

    def metadata(self):
        spec = {**asdict(self), 'joint_order': list(JOINTS),
                'input': 'arm_calibration_relative_degree; gripper_percent', 'output': 'urdf_joint_radian'}
        encoded = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return {**json.loads(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}


# One model-level definition shared by leader, Real2Sim, and recording provenance.
# Do not insert robot-specific calibration JSONs here; no nonidentity correction is established yet.
LEGACY_MAPPER = SO101SimJointMapper()
DEFAULT_MAPPER = LEGACY_MAPPER


def calibrated_to_urdf(values):
    return DEFAULT_MAPPER.calibrated_to_urdf(values)


def urdf_to_calibrated(values):
    return DEFAULT_MAPPER.urdf_to_calibrated(values)


def observed_positions(mapper, native, raw=None, motor_specs=None):
    """Decode real feedback without clipping; preserve raw SDK saturation evidence."""
    convert = getattr(mapper, 'calibrated_to_observation', mapper.calibrated_to_urdf)
    q = convert(native, raw=raw) if raw is not None and hasattr(mapper,'calibrated_to_observation') else convert(native)
    if raw is not None and motor_specs is not None:
        from .real2sim.precision import native_from_raw
        if set(raw) != set(JOINTS) or not np.isfinite(list(raw.values())).all():
            raise ValueError("Invalid raw feedback packet")
        entries=getattr(mapper,'profile',{}).get('joints',{})
        legacy=DEFAULT_MAPPER if entries else mapper
        for n in JOINTS:
            if n in entries: continue
            spec=motor_specs[n];cal=spec['calibration']
            expected=native_from_raw(np.clip(raw[n],cal['range_min'],cal['range_max'])
                if spec['mode']=='RANGE_0_100' else raw[n],spec)
            if abs(expected-native[n])>1e-6: raise ValueError(n+': raw/native packet mismatch')
            unbounded=dict(native);unbounded[n]=native_from_raw(raw[n],spec)
            q[n]=legacy.calibrated_to_urdf(unbounded)[n]
    if set(q) != set(JOINTS) or not np.isfinite(list(q.values())).all():
        raise ValueError('Non-finite or invalid observed joint coordinates')
    return {n:float(q[n]) for n in JOINTS}


def require_capture_mapping(state):
    recorded = state.get('sim_joint_mapping')
    if recorded and recorded.get('schema') == 'so101.precision-joint-mapping/1':
        from .real2sim.precision import mapper_from_metadata
        mapper_from_metadata(recorded)
        return
    expected = DEFAULT_MAPPER.metadata()['sha256']
    if recorded is None:
        if expected != LEGACY_MAPPER.metadata()['sha256']:
            raise ValueError('Legacy capture has no mapping identity; explicit migration required')
    elif recorded.get('sha256') != expected:
        raise ValueError('Capture Sim joint mapping differs; explicit remapping/FK revision required')


class RobotCoordinates:
    def __init__(self, closed_rad=None, open_rad=None, *, measured_gripper_limits=None):
        mapper = DEFAULT_MAPPER.metadata()
        self.closed = mapper['gripper_zero_offset_rad'] if closed_rad is None else float(closed_rad)
        self.open = self.closed + 100 * mapper['gripper_rad_per_percent'] if open_rad is None else float(open_rad)
        if not np.isfinite([self.closed, self.open]).all() or self.open <= self.closed:
            raise ValueError('Explicit finite increasing closed/open gripper endpoints required')
        self.measured_gripper_limits = None
        if measured_gripper_limits is not None:
            bounds=np.asarray(measured_gripper_limits,dtype=float)
            if (bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]
                    or bounds[0]>self.closed+1e-6 or bounds[1]<self.open-1e-6):
                raise ValueError('Measured gripper model limits must enclose canonical command endpoints')
            self.measured_gripper_limits=tuple(float(v) for v in bounds)
        self.saturated_states = 0

    def metadata(self):
        result = {'schema': 'so101.minivla-coordinates/1', 'joint_order': list(JOINTS),
                'model_units': ['rad'] * 6, 'policy_units': list(POLICY_UNITS),
                'gripper_closed_rad': self.closed, 'gripper_open_rad': self.open,
                'source': 'configured endpoints; default is existing hardware-percent mapper, legacy_unverified',
                'measured_gripper_saturation_tolerance_fraction': .02,
                'measured_state_saturations': self.saturated_states}
        if self.measured_gripper_limits is not None:
            result['measured_gripper_limits_rad']=list(self.measured_gripper_limits)
            result['measured_gripper_saturation_tolerance_fraction']=None
        return result

    @staticmethod
    def _array(value):
        q = np.array(value, dtype=np.float64, copy=True)
        if q.ndim < 1 or q.shape[-1] != 6 or not np.isfinite(q).all():
            raise ValueError('Expected finite (...,6) joint vector')
        return q

    def to_policy(self, model_rad, measured=False):
        q = self._array(model_rad)
        if measured and self.measured_gripper_limits is not None:
            low,high=self.measured_gripper_limits
            if np.any(q[...,5]<low-1e-6) or np.any(q[...,5]>high+1e-6):
                raise ValueError(f'Measured gripper outside recorded model limits [{low}, {high}] rad; observed [{q[...,5].min()}, {q[...,5].max()}]')
        q[..., 5] = (q[..., 5] - self.closed) / (self.open - self.closed)
        tolerance = .02 if measured else 1e-6
        if not (measured and self.measured_gripper_limits is not None) and (np.any(q[..., 5] < -tolerance) or np.any(q[..., 5] > 1 + tolerance)):
            raise ValueError('Gripper outside declared canonical endpoints')
        clipped = np.clip(q[..., 5], 0, 1)
        self.saturated_states += int(np.count_nonzero(clipped != q[..., 5])) if measured else 0
        q[..., 5] = clipped
        return q.astype(np.float32)

    def to_model(self, canonical):
        q = self._array(canonical)
        if np.any(q[..., 5] < 0) or np.any(q[..., 5] > 1):
            raise ValueError('Canonical gripper must be fraction [0,1]')
        q[..., 5] = self.closed + q[..., 5] * (self.open - self.closed)
        return q
