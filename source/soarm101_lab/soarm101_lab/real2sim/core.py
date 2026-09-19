"""Unit conversion, timestamps, and explicit quality gates; no hardware imports."""

from collections import deque
import hashlib
import json
import math
import time
import numpy as np

from soarm101_lab.so101_joint_mapping import JOINTS, calibrated_to_urdf, urdf_to_calibrated
NATIVE_UNITS = {n: ("percent_0_100" if n == "gripper" else "calibration_relative_degree") for n in JOINTS}
CONVENTION = {
    "native_units": NATIVE_UNITS,
    "sim_units": {n: "radian" for n in JOINTS},
    "mapping": "Repo V2: sim_joint = native_value * pi/180 for all six joints; inverse for comparison.",
    "gripper_caveat": "Sim gripper radians are the legacy control coordinate, NOT a measurement of real jaw angle. Compare gripper in normalized percentage points.",
    "zero_and_sign": "No additional sign or zero offsets. Each device uses its own actual calibration file and the audited degree/0..100 normalization.",
    "physical_correspondence": "Matching calibration-relative values does not prove identical physical angles; verify motor calibration and multiple contact poses.",
}


def stamp():
    return {"monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def values(data):
    if set(data) != set(JOINTS):
        raise ValueError(f"Expected exact joints {JOINTS}, got {list(data)}")
    out = {n: float(data[n]) for n in JOINTS}
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("Non-finite joint input; command rejected")
    return out


def to_sim(native):
    return calibrated_to_urdf(values(native))


def from_sim(radians):
    # Measured simulation may overshoot endpoints. Do not clamp or hide errors.
    if set(radians) != set(JOINTS):
        raise ValueError("Sim joints must match all six names")
    out = urdf_to_calibrated(radians)
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("Non-finite simulation state")
    return out


def joint_error(real_native, sim_radians):
    sim = from_sim(sim_radians)
    per = {n: (sim[n] - real_native[n]) * (1 if n == "gripper" else math.pi / 180) for n in JOINTS}
    return {
        "sim_minus_real": per,
        "units": {n: ("percentage_points" if n == "gripper" else "radian") for n in JOINTS},
        "max_arm_error_rad": max(abs(per[n]) for n in JOINTS[:-1]),
        "gripper_error_percentage_points": abs(per["gripper"]),
    }


class Stability:
    def __init__(self, window_s=0.6, arm_span_deg=1.0, gripper_span=2.0, max_gap_s=0.2):
        self.window_s = window_s
        self.arm_span_deg = arm_span_deg
        self.gripper_span = gripper_span
        self.max_gap_s = max_gap_s
        self.history = deque(maxlen=600)

    def add(self, t_ns, native, tip=None):
        if self.history and t_ns <= self.history[-1][0]:
            raise ValueError("History timestamps must increase")
        self.history.append((t_ns, np.array([native[n] for n in JOINTS]), tip))
        while len(self.history) > 2 and (t_ns - self.history[1][0]) / 1e9 > self.window_s * 1.5:
            self.history.popleft()

    def assess(self):
        if len(self.history) < 3:
            return {"stable": False, "reason": "insufficient_history"}
        end = self.history[-1][0]
        start = end - int(self.window_s * 1e9)
        rows = list(self.history)
        before = [i for i, r in enumerate(rows) if r[0] <= start]
        if not before:
            return {"stable": False, "reason": "warming_up"}
        rows = rows[before[-1] :]
        gaps = np.diff([r[0] for r in rows]) / 1e9
        span = np.ptp(np.array([r[1] for r in rows]), axis=0)
        tips = [r[2] for r in rows if r[2] is not None]
        tip_span = max(np.linalg.norm(np.asarray(tips) - np.median(tips, axis=0), axis=1)) * 1000 if tips else None
        stable = bool(
            max(gaps) <= self.max_gap_s and max(span[:-1]) <= self.arm_span_deg and span[-1] <= self.gripper_span
        )
        return {
            "stable": stable,
            "window_start_ns": rows[0][0],
            "window_end_ns": end,
            "max_gap_ms": float(max(gaps) * 1000),
            "arm_span_deg": float(max(span[:-1])),
            "gripper_span_percentage_points": float(span[-1]),
            "sim_tip_spread_mm": float(tip_span) if tip_span is not None else None,
            "thresholds": {
                "window_s": self.window_s,
                "arm_span_deg": self.arm_span_deg,
                "gripper_span": self.gripper_span,
                "max_gap_s": self.max_gap_s,
            },
            "reason": "stable" if stable else "motion_or_sampling_gap",
        }


def assess_pair(
    frames,
    sim_stamp,
    follower_interval,
    real_stability,
    sim_stability,
    error,
    now_ns,
    max_skew_ms=100.0,
    max_age_ms=200.0,
    max_arm_error_rad=0.035,
    max_gripper_error=2.0,
):
    reasons = []
    timing = {}
    stamps = [sim_stamp["monotonic_ns"]]
    for name in ("real_side", "real_wrist"):
        f = frames.get(name)
        if f is None:
            reasons.append("missing_" + name)
            continue
        t = f["read_end"]["monotonic_ns"]
        stamps.append(t)
        timing[name] = {
            "receive_minus_sim_ms": (t - sim_stamp["monotonic_ns"]) / 1e6,
            "age_ms": (now_ns - t) / 1e6,
            "timestamp_kind": "host_read_completion_not_exposure",
            "exposure_timestamp": None,
        }
        if now_ns - t > max_age_ms * 1e6:
            reasons.append("stale_" + name)
        if abs(t - sim_stamp["monotonic_ns"]) > max_skew_ms * 1e6:
            reasons.append("skew_" + name)
        if (
            real_stability.get("window_start_ns", now_ns) > f["read_start"]["monotonic_ns"]
            or real_stability.get("window_end_ns", 0) < t
        ):
            reasons.append("frame_not_covered_by_stability_" + name)
    if follower_interval:
        rt = follower_interval["end"]["monotonic_ns"]
        stamps.append(rt)
        timing["follower_read_minus_sim_ms"] = (rt - sim_stamp["monotonic_ns"]) / 1e6
        timing["follower_read_duration_ms"] = (rt - follower_interval["start"]["monotonic_ns"]) / 1e6
        if now_ns - rt > max_age_ms * 1e6 or timing["follower_read_duration_ms"] > max_skew_ms:
            reasons.append("follower_measurement_stale_or_slow")
    else:
        reasons.append("no_physical_follower_measurement")
    spread = (max(stamps) - min(stamps)) / 1e6
    if spread > max_skew_ms:
        reasons.append("observation_time_spread")
    if now_ns - sim_stamp["monotonic_ns"] > max_age_ms * 1e6:
        reasons.append("stale_sim")
    if not real_stability.get("stable"):
        reasons.append("physical_follower_not_stable")
    if not sim_stability.get("stable"):
        reasons.append("sim_not_stable")
    if error is None:
        reasons.append("joint_error_unavailable")
    elif error["max_arm_error_rad"] > max_arm_error_rad or error["gripper_error_percentage_points"] > max_gripper_error:
        reasons.append("real_sim_joint_error")
    return {
        "hardware_synchronized": False,
        "eligible_for_calibration": not reasons,
        "reasons": reasons,
        "observation_time_spread_ms": spread,
        "timing": timing,
        "thresholds": {
            "max_skew_ms": max_skew_ms,
            "max_age_ms": max_age_ms,
            "max_arm_error_rad": max_arm_error_rad,
            "max_gripper_error_percentage_points": max_gripper_error,
        },
        "limitation": "Software pairing of host receipt times only; USB exposure/buffering latency is unknown. Eligibility is a heuristic, not proof of geometric calibration.",
    }
