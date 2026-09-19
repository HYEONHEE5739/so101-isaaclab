# SO-101 joint coordinate audit (2026-09-18)

## A. CURRENT COORDINATE SYSTEMS

1. Motor register encoder: Feetech Homing_Offset is already applied on the device. Do not apply it again to the measured normalized state.
2. LeRobot calibrated control: five arm joints in calibration-relative degrees; gripper in 0..100 percent.
3. SO-101 model: six named URDF articulation coordinates in radians. Positive rotation is right-handed around each joint's LOCAL Z axis, after its origin transform.

`devices/lerobot/motors/motors_bus.py`: `sync_read(normalize=True)` decodes register values then calls `_normalize`. For STS3215 the implementation uses 4096 - 1 = 4095:

    d = (encoder - (range_min + range_max)/2) * 360 / 4095
    encoder = int(d * 4095 / 360 + (range_min + range_max)/2)

DEGREES is not range-clipped and this branch does not reverse using drive_mode. Percent branches clamp and optionally reverse. All six inspected follower drive_mode values are 0. Integer inverse introduces encoder quantization. Installed LeRobot follower MotorsBus implements the same relevant operations.

The calibration midpoint is a recorded sweep midpoint, not an independently measured URDF zero. The calibration procedure does not certify that the sweep endpoints equal model joint limits.

## B. CURRENT ACTION FLOW

Paths below are relative to `source/soarm101_lab/soarm101_lab/` except `scripts/`.

| Path / function | Input → output and units | Calibration, zero/sign and gripper |
|---|---|---|
| `devices/lerobot/motors/motors_bus.py:sync_read/_normalize` | encoder → arm degree, gripper percent | Device-specific midpoint; DEGREES positive encoder increment; percent range mapping |
| `devices/lerobot/so101_leader.py:advance` | calibrated six values → model six rad | Common mapper; default +1/zero arm, legacy percent-to-rad gripper |
| `scripts/envs/teleoperation/teleop_so101.py` → `base_pick_place_teleop_env_cfg.py` → Isaac `JointPositionAction` | model rad → position target rad | scale=1, default offset disabled; explicit joint order; no hardware calibration |
| `real2sim/hardware.py:Follower` and installed `SO101Follower.send_action` | leader calibrated degree/% → follower encoder goal | Follower's own calibration inverse; no Sim mapping on this hardware path |
| `real2sim/runtime.py:tick/check_target` | leader/sent calibrated state → model target rad | Same common mapper as leader |
| `real2sim/runtime.py:set_measured_pose` → `real2sim/core.py:to_sim` | follower measured degree/% → model state rad | Same mapper; writes state/target, forwards and renders, not a follower command |
| `real2sim/runtime.py:render_dataset` | stored native measured state → model state/render | Validates mapping identity; legacy accepted only under legacy mapping |
| `scripts/envs/teleoperation/record_mimic_dataset.py:pose_delta_action` | measured successive Sim TCP poses + gripper rad → 7 actions | Translation delta m, base-frame rotation-vector delta rad, gripper rad; separate 6-rad joint_targets |
| `scripts/envs/teleoperation/record_lerobot_dataset.py` → `datasets/lerobot_recorder.py` | leader-produced model action → stored 6-rad action | Already mapped; observed joints also model radians |
| `scripts/mimic/annotate_demos.py` | source 7 actions → replay and measured datagen_info | Cartesian actions, no degree conversion |
| IsaacLab Mimic `datagen/data_generator.py`, `waypoint.py`; `so101_mimic_env.py:target_eef_pose_to_action` | transformed target EEF pose → Cartesian delta + gripper rad | Object/EEF pose geometry; no motor calibration |
| `mdp/so101_ik_actions.py:apply_actions` | Cartesian delta → Pinocchio IK model targets rad | Uses URDF kinematics; no second mapping |
| `mdp/so101_mimic_recorders.py:PostStepJointTargetsRecorder` | IK arm targets + gripper processed action → 6-rad joint_targets | Concatenation in canonical order |
| `scripts/tools/replay_joint_targets.py` | stored 6-rad joint_targets → env.step | Metadata/order check only; no remapping |
| `scripts/tools/convert_isaac2lerobot.py` | joint_targets or next_joint_pos → LeRobot dataset actions | Both model rad; explicit mimic_actions option remains Cartesian 7D |
| `scripts/inference/inference_act_sim.py` | policy postprocessed action → env.step | Units inherited from training dataset; ordinary joint-target policy uses model rad |

No corresponding real policy deployment path was identified. A future 6-rad model policy requires the mapper inverse before follower.send_action. A Cartesian 7D policy requires IK first, then inverse. Dataset/policy names alone do not establish units.

Actual HDF5 inspected: `datasets/demo_20ep.hdf5` demo_0 actions (390,7), joint_targets (390,6), joint states (390,6); annotated demo contains EEF datagen poses; `datasets/generated_demo_20ep_30ep.hdf5` actions (360,7), joint_targets (360,6). These files had no mapping/asset provenance and were not rewritten.

## C. PER-JOINT URDF ↔ LEROBOT CORRESPONDENCE TABLE

Follower file: `/home/hyeonhee/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower_arm.json`.
Model: `assets/SO101/urdf/so101_isaaclab.urdf` and composed `assets/SO101/usd/so101_isaaclab.usd` inspected with USD bindings.

| Joint | Encoder min/max; midpoint | Recorded degree range | URDF/USD limit degrees | Sign correspondence | Zero correspondence |
|---|---|---|---|---|---|
| shoulder_pan | 724/3448; 2086 | -119.736..119.736 | -110..110 | Unverified | Unverified |
| shoulder_lift | 943/3242; 2092.5 | -101.055..101.055 | -100..100 | Unverified | Unverified |
| elbow_flex | 964/3053; 2008.5 | -91.824..91.824 | -100..90 | Unverified | Unverified |
| wrist_flex | 970/3223; 2096.5 | -99.033..99.033 | -95..95 | Unverified | Unverified |
| wrist_roll | 0/4095; 2047.5 | -180..180 | -160..160 | Unverified | Unverified |
| gripper | 1906/3349 | 0..100 percent | -10..100 | Percent endpoint correspondence unverified | Unverified |

All axes `(0,0,1)` are LOCAL axes; this cannot establish motor sign. URDF origins (xyz m; rpy rad, rounded):

| Joint | xyz | rpy |
|---|---|---|
| shoulder_pan | .0207909, -.0230745, .0948817 | -pi, 0, pi/2 |
| shoulder_lift | -.0303992, -.0182778, -.0542 | -pi/2, -pi/2, 0 |
| elbow_flex | -.11257, -.028, 0 | 0, 0, pi/2 |
| wrist_flex | -.1349, .0052, 0 | 0, 0, -pi/2 |
| wrist_roll | 0, -.0611, .0181 | pi/2, pi/2, pi |
| gripper | .0202, .0188, -.0234 | pi/2, 0, 0 |

URDF zero means these origin transforms with joint rotation zero. USD limits and local rotations agree with the relevant URDF definitions. Follower homing offsets are respectively 1465, -1957, 94, 1067, -39, 380; these are not Sim zero offsets.

## D. ROOT CAUSE OF CURRENT REAL/SIM POSE MISMATCH

The code establishes a previously unverified identity assumption; it does not establish which physical joints are wrong. Near-zero numeric discrepancy after writing converted measured state is circular evidence and does not validate physical correspondence. Tracking error, camera/base error and coordinate mismatch must be separated.

Two concrete issues were corrected: positional `SO101Leader(cfg)` previously bound cfg to env and silently used defaults; Python shoulder_lift limit constant was -10 instead of the actual USD -100 degrees. Neither proves the observed image mismatch's cause. No guessed offset was installed.

## E. CHOSEN CANONICAL ACTION REPRESENTATION

At the joint-control boundary choose model URDF radians (Option B). Preserve Mimic's separate existing Cartesian delta action representation. Do not convert already recorded model targets or IK results again.

## F. COMMON SO101 SIM JOINT MAPPER DESIGN

`so101_joint_mapping.py`: one immutable definition; five signs restricted to +/-1; five offsets rad; no arm scale. Forward `q = sign * deg2rad(d) + offset`; inverse `d = rad2deg((q-offset)/sign)`. Gripper has separate invertible percent mapping. Input validation, named joints/batches, version/hash. Defaults preserve prior numerical behavior and are marked `legacy_unverified`.

`so101_dataset_contract.py`: small sidecar with action representation, joint order, mapper version/hash and URDF/USD file hashes. External mesh hashes are explicitly out of scope. New HDF source also embeds metadata in env_args. Annotation/datagen inherit source provenance, including unknown; no retroactive stamping. Resume rejects unknown/different provenance. Captures record mapper identity and loading/rerender rejects incompatible mappings.

## G. FILES CHANGED

New: `so101_joint_mapping.py`, `so101_dataset_contract.py`, `tests/real2sim/test_joint_mapping.py`, this report.
Modified: `devices/lerobot/so101_leader.py`; `real2sim/core.py`, `runtime.py`, `calibration.py`; `assets/robots/so101.py`; `tasks/manager_based/soarm101_lab/base_pick_place_teleop_env_cfg.py`; `datasets/lerobot_recorder.py`; `scripts/envs/teleoperation/record_mimic_dataset.py`; `scripts/mimic/annotate_demos.py`, `generate_dataset.py`; `scripts/tools/replay_joint_targets.py`, `convert_isaac2lerobot.py`.

Hardware calibration files, motor homing/limits, physical follower calibration logic, policy architecture, Mimic/IK/datagen algorithms were not modified. Other pre-existing working-tree changes were retained.

## H. SOURCE DEMO COMPATIBILITY

Default mapper is numerically equivalent to legacy conversion. New motion uses the shared mapping before recording; stored data remains model-space. New metadata avoids silent resume across mappings/assets. An existing unknown dataset should be kept as a separate legacy artifact.

## I. MIMIC / DATAGEN COMPATIBILITY

Only provenance hooks changed. Pose transforms, Cartesian integration, IK and generated-target recording retain their representations. Different mapping provenance does not justify remapping model-space source or generated trajectories.

## J. WHETHER OLD DEMOS MUST BE REGENERATED

No mandatory regeneration for this identity-preserving change. Old model trajectories remain valid under the same asset/controller. A future verified nonidentity mapping changes how physical inputs produce new trajectories, not the meaning of old radians. Do not run old rad targets through the new mapper. Recollect/regenerate only if training objectives require the newly established physical correspondence; keep dataset versions separate. Old native captures may support explicit reprocessing with recomputed FK and a new revision; current guards prohibit silent reinterpretation.

## K. GRIPPER FINDING

Physical input is percent, whereas model gripper action is a revolute angle in radians. Legacy code effectively used 0%→0 degrees and 100%→100 degrees. This numerical behavior is retained in an explicitly separate branch. Encoder span corresponds to ~126.857 motor degrees; that is not proof of the modeled jaw angle, transmission relationship, or URDF endpoints. Do not replace it by mapping to -10..100 without measurements.

## L. TEST RESULTS / LIMITATIONS

Non-hardware tests cover identity numerical agreement, synthetic sign/offset, forward/inverse, common leader/Real2Sim boundary, actual measured-state writes with mocked articulation, HDF target roundtrip, Isaac JointPositionAction forwarding, actual Cartesian source conversion/recorder functions, metadata inheritance/resume and capture compatibility, and actual motor normalization/quantization.

AST extraction executes actual function bodies while avoiding AppLauncher and hardware imports. It does not test import-time GPU lifecycle. Replay validation establishes identical target arrays, not identical dynamic physical trajectories. Existing replay script does not restore source initial state, which is an additional condition for full dynamic reproduction. No GPU Isaac run, physical serial connection or motor movement was performed.

## M. CALIBRATION METHOD

Status: **NEEDS_MORE_DATA** for physical sign/zero determination. No numerical physical offset fitter was enabled based on inadequate correspondence evidence.

Endpoint alignment is valid only after mechanical endpoint identity is established. Observed spans differ; aligning endpoints or taking their midpoint difference blindly is unjustified.

Use existing COMPLETE multi-pose captures with native follower measurements and model geometry. Obtain identifiable link landmarks, preferably across multiple links, not only the TCP. Use static metric workspace observations to anchor the scene, but a single planar square does not independently solve arbitrary intrinsics, camera/base pose and joint offsets. Fix a documented gauge: base yaw and shoulder-pan zero are coupled; wrist-roll zero and wrist mount rotation are coupled. Anchor base orientation independently or freeze that gauge rather than claiming both solved.

Recompute FK for each candidate mapping from native measurements. Stored link transforms under the old mapping cannot serve as ground truth for a new mapping. Evaluate discrete sign hypotheses only with informative joint variation; optimize zero offsets using robust landmark reprojection residuals, fixed/anchored camera/base first. Keep entire poses held out. Check Jacobian rank/conditioning and train/validation residuals. Accept only corrections supported across poses. Subsequently refine base/side camera under explicit gauge constraints; fit wrist mount after arm correspondence. Gripper needs its own visible jaw/aperture observations. No arbitrary RGB warp, physical-camera intrinsic calibration requirement, or freely fitted arm scale.

Existing optimizer/capture machinery remains available for measurements; a reported low error from fitting camera and joint offsets together without gauge handling is not sufficient evidence.

## N. NEXT USER ACTION

First annotate usable existing COMPLETE captures: known robot landmarks and static metric reference corners, with clear joint variation and visibility. Do not ask the user for numeric offsets. Additional physical captures are needed only if existing captures lack link visibility, informative variation, or held-out poses. Keep base, side camera, marker fixed during such captures and record measured follower state. No need to recalibrate hardware or touch mechanical stops merely for this audit.
