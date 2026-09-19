# Physical joint correspondence: evidence assessment (2026-09-18)

## Result

**NEEDS_MORE_DATA. No verified mapping candidate or physical sign/zero estimate was produced.** Production side/wrist camera poses, profiles, hardware calibration and active mapper were not changed. No hardware commands were sent. Only diagnostic artifacts and this report were added.

## A. Existing capture sufficiency

Inspected all COMPLETE states and decoded all four images for each: 42 captures, 168 image files.

| Session timestamp | COMPLETE | Quality eligible | Mapping hash present |
|---|---:|---:|---:|
| 144939 | 14 | 14 | 0 |
| 150629 | 10 | 10 | 0 |
| 155502 | 13 | 13 | 0 |
| 161608 | 5 | 5 | 0 |

Session 161423 has no COMPLETE capture. Four distinct embedded profiles exist. All 42 share recorded follower calibration SHA256 `28be25b5eafd203c9f6d3218f214dc26d54f5cdef26733a706f9a3f96a618173` and hardware identity. Mapping hashes are absent; the recorded convention describes legacy degree conversion, but it is not retroactively a verified hash. Host receipt spread is at most 70.113 ms; this does not establish exposure synchronization or physical identifiability.

All 42 Real Side images were visually inspected through session contact sheets. Latest session Real Wrist and both Sim views were also visually inspected. Other wrist/Sim images were decoded and inventoried, not all visually judged. Real Side sees distal gripper/mount, occasional distal arm fragments; proximal base/shoulder/elbow joint centers are not usable. Several poses put the robot entirely/almost entirely outside the image. Wrist view is excluded as primary joint evidence because its mount remains provisional.

## B. Joint pose diversity

| Joint | Observed min..max deg | Span deg | Side evidence / sufficiency |
|---|---:|---:|---|
| shoulder_pan | -35.165..37.363 | 72.527 | Base and shoulder not visible; distal motion only; base yaw gauge unresolved |
| shoulder_lift | -100.176..67.033 | 167.209 | Shoulder center and upper-arm geometry not adequately visible |
| elbow_flex | -85.670..90.769 | 176.440 | Elbow center not adequately visible |
| wrist_flex | 42.418..96.132 | 53.714 | Distal geometry partly visible but clipped/occluded; no verified link correspondence |
| wrist_roll | -56.044..31.253 | 87.297 | Distal asymmetric geometry sometimes visible; large mount occlusion, no verified off-axis landmark |

Numerical excitation exists; missing evidence is not simply insufficient joint range. These are coupled multi-joint poses, not isolated sign tests. A train/validation split is possible by whole capture, but no trusted joint-calibration residual dataset exists yet. Invisible joint centers alone do not mathematically preclude end-effector-based identification; however that requires trustworthy 3D/2D endpoint correspondence and an independently anchored base/camera, which are missing here. No rank claim is made from joint-value diversity alone.

## C. Static gauge / frozen parameters

Known metric references remain 150 mm square, 24 mm cube, cup opening 70 mm and height 65 mm. These lengths do not certify their world poses or camera/base alignment.

Latest embedded profile: workspace pose provisional, calibrated=false; table size/pose provisional; robot_base numerically_fitted but calibrated=false, explicitly conditional on prior joint assumptions; side K/extrinsics numerically_fitted but both calibrated flags false; wrist mount similarly conditional/unverified; distortion zero is an unmeasured seed.

All production parameters remain frozen as existing runtime state, **not relabeled as truth for joint fitting**. Fixing a wrong base yaw and estimating pan zero would produce a biased conditional answer. Do not jointly optimize them. A calibration-only base reference should independently establish the gauge.

Current `real2sim/calibration.py:optimize` only fits scene groups; it requires both intrinsics flags and side/base calibration in sequence. `world_point` uses stored links_base. It is not a joint-offset fitter. Do not turn on flags or reuse stored identity FK to bypass these requirements.

## Repository/model-only determination

Read local URDF joints/origins/axes/transmissions, asset file layout and existing kinematics. URDF header says onshape-to-robot and links an Onshape document. Local assets contain exported URDF, USD layers and STL meshes, but no discovered CAD-to-encoder index/horn datum calibration or export source configuration tying current physical encoder to CAD zero. The CAD document itself was not fetched. SimpleTransmission mechanicalReduction=1 does not encode encoder zero or installed motor polarity.

Installed follower calibration asks the user to place the robot in the middle of its range, then records a sweep. It does not record a metrologically defined CAD pose. Homing is device-side and must not be reapplied in Sim. Different recorded-vs-URDF spans do not justify endpoint alignment or arm scaling.

A deterministic zero-FK calculation is saved in `outputs/real2sim/analysis/joint_correspondence_audit/model_zero_geometry.json`. At zero, joint origins in base coordinates (mm):

| Joint | x | y | z | Axis in base at zero, approximately |
|---|---:|---:|---:|---|
| shoulder_pan | 20.791 | -23.074 | 94.882 | -Z |
| shoulder_lift | 2.513 | -53.474 | 149.082 | +X |
| elbow_flex | 2.514 | -81.473 | 261.652 | +X |
| wrist_flex | 2.515 | -216.373 | 266.852 | +X |
| wrist_roll | 20.615 | -277.473 | 266.852 | +Y |

Thus shoulder-lift→elbow at zero is approximately (0,-28,112.57) mm; elbow→wrist-flex (0,-134.9,5.2) mm. Zero is geometrically expressible but not exactly a naive vertical-upper-arm/horizontal-forearm alignment. Matching a verified physical geometry pose would give `offset=q_reference-sign*deg2rad(measured_native)`, but the physical reference observation is currently absent. One pose cannot determine both sign and offset. Roll also needs a non-axisymmetric orientation reference; a joint-center position alone cannot resolve axial rotation. These are model-derived geometry values, not fitted physical offsets or commands to move the robot to zero.

## D. Landmark / silhouette extraction

Ran deterministic HSV connected-component diagnostic on all Real Side images (OpenCV HSV H115..165, S65..255, V35..255). The largest purple component touches the upper image border in 41/42 captures. This is a clipping diagnostic, not a model landmark or proof of joint visibility; no threshold-derived pixel was promoted to calibrated correspondence.

Existing `runs/capture_iteration_20260918_145943_*/landmarks.json` uses an approximate fixed-finger tip: lower silhouette median of purple component. A silhouette extremum need not track the same material/model point across rotations. Existing later scene measurements include approximate points without sufficient independent joint-center provenance. They are not accepted as final joint calibration evidence.

Reuse UI raw-image click annotation only for identifiable visible model features; it already records frame, point_m, uv and whole-capture split. It cannot create evidence for invisible joints. No VLM pixel guesses were saved.

## E–H. Sign, zero, train and held-out results

All five signs: undetermined. All five physical zero offsets: null/unestimated. Uncertainty: not estimable from an accepted fit. Existing mapper +1/0 stays legacy_unverified and is not a result of this assessment.

No discrete hypotheses were accepted/rejected numerically because trusted observations/gauge were not available. No optimizer was run on an invalid residual dataset. Train RMSE, held-out RMSE, Jacobian rank, conditioning and confidence bounds: not evaluated, not zero. No candidate or verified hash exists.

## I–M. Status and pipeline effect

NEEDS_MORE_DATA. No runtime, mapping, camera/profile, source demo, Mimic, datagen or policy behavior changed. No existing dataset was modified. There is no reason to recollect source demos yet. Future accepted model-level calibration must be keyed by URDF/USD identity and physical follower calibration hash; it must not be stored as side/wrist scene pose. A changed hardware calibration invalidates that correspondence. Sharing the mapper with leader input preserves control-coordinate semantics but does not independently prove the leader's physical pose matches the follower.

Future candidate must include sign/offset, numerically_fitted provenance, source measurements, train/validation metrics, rank/conditioning, asset/calibration hashes and mapper version/hash. Accept only after held-out validation. Already-model-space trajectories must never be remapped.

## N. Minimal additional measurement without moving production cameras

Additional correspondence measurement is needed; repeating the same production framing alone will not expose the missing proximal geometry. Preferred practical route is a **temporary auxiliary camera** that sees the base reference, shoulder, elbow, wrist and asymmetric gripper features. Do not move Side/Wrist, the base, or existing scene anchors. Auxiliary image/intrinsics/extrinsics belong to a calibration-only bundle, never production profile cameras.

Before collecting, establish identifiable geometric references: visible pivot/screw centers whose model locations can be verified, plus a base-fixed asymmetric reference and off-axis gripper reference. Temporary dots may aid tracking but their model coordinates must be measured/registered; arbitrary dots do not supply known 3D coordinates. A temporary metric reference with known relation to the base can anchor auxiliary pose; a floating planar square with unknown base relation is insufficient to establish pan zero. Auxiliary camera fitting can be observation-based; no mandatory manufacturer-level intrinsic calibration.

Acquire a stationary baseline and modest comfortable positive/negative changes of one joint at a time while other joints remain approximately fixed; record actual follower states for every observation. Suggested initial set: baseline + two directions for each of five joints (11 observations), then at least three separate mixed poses held out. This is a collection design, not a guarantee of full rank. Do not drive to mechanical limits or command URDF zero under an unverified mapping. The user need not supply offsets. Record timestamps and settle before images; validate rank/conditioning before accepting a fit.

If an auxiliary camera is unavailable, a measured mechanical alignment fixture/reference implementing the model-derived relative geometry is an alternative. It must establish actual feature orientation relative to the base and include a second known-direction pose for sign. A generic 'middle of travel' pose is not sufficient.

Calibration-only measurements must pair images/reference readings with contemporaneous measured follower values and the unchanged calibration hash. The current four-camera capture UI does not automatically timestamp an auxiliary camera; integration/pairing must be designed explicitly before treating casual photos as synchronized measurements.

## Artifacts and checks

- `capture_inventory.json`: session counts, joint extrema, identity and image existence.
- `capture_evidence.json`: all 42 native states, quality, image hashes/shapes, profile digest, follower hash, mapping-hash absence, segmentation diagnostics.
- Session contact sheets: visual inspection aids only.
- `model_zero_geometry.json`: model-only deterministic FK at zero.
- `result.json`: machine-readable NEEDS_MORE_DATA; null physical estimates and metrics.

All artifacts are under `outputs/real2sim/analysis/joint_correspondence_audit/`. Image decoding and numeric inspection ran offline. No application source was changed, so no new hardware/GPU integration success is claimed.
