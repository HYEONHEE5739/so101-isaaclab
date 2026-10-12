# SO-101 Agentic Real2Sim — Final Prompt

You are responsible for improving the existing SO-101 Real2Sim environment in this repository.

Use the current working scene/profile as the baseline. Reuse existing assets, measurements, robot models, and tools whenever possible. Do not rebuild the project from scratch unless the current representation cannot support the required reconstruction.

## Goal

Build a MuJoCo scene that matches the real SO-101 workspace closely enough for manipulation-policy evaluation.

Priority:

1. camera calibration
2. robot base / workspace alignment
3. object pose and geometry
4. collision / contact geometry
5. visual appearance
6. physics parameters

Photorealism is secondary to camera correspondence and interaction geometry.

## Inputs

Use all available evidence:

- real images / videos
- robot joint states and actions
- known measurements
- SO-101 URDF / MJCF / existing assets
- existing Real2Sim profiles and scene code

Prefer, in order:

measured values → calibration → robot FK → synchronized observations → multi-view/image measurements → numerical optimization → visual estimation.

Do not guess values that can be calculated.

## Workflow

Work through these stages in order.

### 1. Inspect

Inspect the current scene, profiles, tools, real references, robot-state data, and documentation.

Keep what already works.

Identify the largest current mismatch and the minimum reusable render/compare loop needed to improve it.

Then continue automatically.

### 2. Camera & Scale

Calibrate camera intrinsics/extrinsics and establish metric scale using known geometry, robot FK, synchronized joint states, and multi-view evidence.

Render the same robot states as the real observations.

Do not distort scene geometry to compensate for a wrong camera.

When camera/scale correspondence is acceptable, continue to Geometry.

### 3. Geometry

Fit:

- robot base
- table/workspace
- task objects
- target/bin
- interaction-critical geometry

Use simple, physically meaningful collision geometry separately from visual geometry when appropriate.

Change only the parameter group responsible for the current mismatch.

When projected geometry and spatial relationships are acceptable, continue to Visual.

### 4. Visual

Match only the appearance that materially affects the policy camera:

- object color/material
- table appearance
- dominant lighting
- important background structure

Do not spend effort on decorative background details while geometry is still wrong.

When visual correspondence is sufficient, continue to Physics.

### 5. Physics

Replay available real actions or motions in MuJoCo.

Compare real and simulated motion/contact behavior.

Tune only parameters supported by observed mismatch, such as:

- friction
- mass/inertia
- damping
- contact parameters

Do not use physics parameters to hide geometry or timing errors.

When task-relevant dynamics are acceptable, continue to Validation.

### 6. Validation

Validate on observations that were not used for fitting when possible.

For each representative pose:

- set the same robot state
- render directly from MuJoCo
- compare real vs sim
- generate side-by-side / overlay / difference views when useful
- compute available reprojection, silhouette, object-position, or joint-state metrics

Do not rely only on subjective visual similarity.

If validation passes, finish with ACCEPTED.

If it fails, return to the earliest stage responsible for the error and repeat.

## Autonomous Iteration

Do not require the user to repeatedly:

- open the UI
- take simulation screenshots
- manually compare renders
- create new test files
- tell you whether each change looks better

Run the render → compare → diagnose → modify → rerender loop yourself.

Use the UI only as an optional final sanity check.

For video input, extract useful frames yourself and align them with robot states when possible.

## File Discipline

Avoid file proliferation.

Prefer:

- one current profile/config
- reusable render/compare/validation tools
- overwriteable scratch outputs
- Git history for accepted changes

Do not create a new Python file for every experiment.

Create a new reusable tool only when the repository genuinely lacks a required capability.

## Decision Rule

At the end of each stage, explicitly choose one:

- `CONTINUE` — stage is good enough; move to the next stage
- `RETRY` — current stage is still the dominant source of error
- `NEEDS_MORE_DATA` — existing evidence cannot resolve an important ambiguity
- `FAILED` — an architectural limitation blocks progress
- `ACCEPTED` — validation passes and remaining mismatch is not meaningful for the target task

If `NEEDS_MORE_DATA`, request only the minimum necessary evidence and state exactly:

- what is needed
- which camera/view or robot pose is required
- why the current data cannot resolve the ambiguity

Do not request exploratory screenshots.

## Final Output

When finished, report only:

1. final status
2. what changed
3. final profile/config used
4. key validation results
5. remaining meaningful mismatch
6. any additional data still required

Start by inspecting the existing implementation and available real data, then proceed through the workflow autonomously.
