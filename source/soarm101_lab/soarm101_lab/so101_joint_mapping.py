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


def require_capture_mapping(state):
    recorded = state.get('sim_joint_mapping')
    expected = DEFAULT_MAPPER.metadata()['sha256']
    if recorded is None:
        if expected != LEGACY_MAPPER.metadata()['sha256']:
            raise ValueError('Legacy capture has no mapping identity; explicit migration required')
    elif recorded.get('sha256') != expected:
        raise ValueError('Capture Sim joint mapping differs; explicit remapping/FK revision required')
