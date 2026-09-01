import h5py
import numpy as np

from soarm101_lab.utils.so101_kinematics import SO101Kinematics


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

URDF_PATH = "/home/hyeonhee/soarm_isaaclab/soarm101_lab/assets/SO101/urdf/so101_isaaclab.urdf"
HDF5_PATH = "/home/hyeonhee/soarm_isaaclab/soarm101_lab/datasets/test.hdf5"

DEMO = "demo_0"
STEP = 100


# ============================================================
# 1. HDF5에서 같은 timestep의 q와 EE pose 가져오기
# ============================================================

with h5py.File(HDF5_PATH, "r") as f:

    joint_pos = np.asarray(
        f[f"data/{DEMO}/obs/policy/joint_pos"][STEP],
        dtype=np.float64,
    )

    ee_state = np.asarray(
        f[f"data/{DEMO}/obs/policy/ee_state"][STEP],
        dtype=np.float64,
    )


# joint_pos = 6개
# [arm 5개 + gripper 1개]
q_current = joint_pos[:5]


print("\n===================================")
print(f"HDF5 {DEMO} STEP {STEP}")
print("===================================")

print("joint_pos all:")
print(np.round(joint_pos, 6))

print("\narm q rad:")
print(np.round(q_current, 6))

print("\narm q deg:")
print(np.round(np.rad2deg(q_current), 3))

print("\nHDF5 ee_state:")
print(np.round(ee_state, 6))


# ============================================================
# 2. Pinocchio 생성
# ============================================================

kin = SO101Kinematics(
    urdf_path=URDF_PATH,
    target_frame_name="tool0",
    joint_names=ARM_JOINT_NAMES,
)


# ============================================================
# 3. HDF5 q → Pinocchio FK
# ============================================================

T_pin = kin.forward_kinematics(q_current)

pin_xyz = T_pin[:3, 3]
isaac_xyz = ee_state[:3]


print("\n===================================")
print("FK COMPARISON")
print("===================================")

print("Isaac/HDF5 tool0 xyz:")
print(np.round(isaac_xyz, 6))

print("\nPinocchio tool0 xyz:")
print(np.round(pin_xyz, 6))

fk_error = np.linalg.norm(
    isaac_xyz - pin_xyz
)

print(
    f"\nFK position error: "
    f"{fk_error * 1000:.3f} mm"
)


# ============================================================
# 4. Pinocchio IK 테스트
#
# 현재 위치에서 Z + 2cm
# ============================================================

T_target = T_pin.copy()
T_target[2, 3] += 0.02

print("\n===================================")
print("IK TARGET")
print("===================================")

print("current xyz:")
print(np.round(T_pin[:3, 3], 6))

print("target xyz:")
print(np.round(T_target[:3, 3], 6))


q_target, success, info = kin.inverse_kinematics(
    current_joint_pos_rad=q_current,
    desired_ee_pose=T_target,

    position_weight=1.0,
    orientation_weight=0.0,

    max_iterations=200,
    tolerance=1e-4,
    damping=1e-3,
    step_size=0.3,
)


print("\n===================================")
print("IK RESULT")
print("===================================")

print("success:", success)

print("\nq current rad:")
print(np.round(q_current, 6))

print("\nq target rad:")
print(np.round(q_target, 6))

print("\nq delta deg:")
print(
    np.round(
        np.rad2deg(q_target - q_current),
        3,
    )
)

print("\ninfo:")
print(info)


# ============================================================
# 5. IK 결과를 다시 FK해서 검증
# ============================================================

T_result = kin.forward_kinematics(q_target)

ik_error = np.linalg.norm(
    T_target[:3, 3]
    - T_result[:3, 3]
)


print("\n===================================")
print("IK → FK VERIFICATION")
print("===================================")

print("wanted xyz:")
print(np.round(T_target[:3, 3], 6))

print("solved xyz:")
print(np.round(T_result[:3, 3], 6))

print(
    f"\nIK position error: "
    f"{ik_error * 1000:.3f} mm"
)