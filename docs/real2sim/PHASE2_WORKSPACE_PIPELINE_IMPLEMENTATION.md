# Phase 2 — Workspace Runtime, Representation Contract, Unified Workflow UI, and Hugging Face Publishing

## Context

The current `workspace_001` flow has been verified through:

```text
Publish Workspace
→ Generic Workspace
→ Source Demo with the physical leader
→ HDF5 dataset saved
→ Replay in Isaac
```

Source-demo and replay work correctly.

The next phase should **not stop at an audit-only report**.
Inspect the current implementation as needed, then **implement the architecture incrementally while preserving existing working behavior**.

The goal is to turn the current collection of CLI workflows into a reusable SO-101 imitation-learning workstation built around:

- Real2Sim workspace packages
- reusable manipulation physics
- a consistent Sim/Real representation contract
- artifact-aware workflow execution
- a unified UI
- Hugging Face dataset/model publishing

Do not rewrite working subsystems unnecessarily. Refactor when that reduces duplication or makes the end-to-end contract clearer.

Existing Mimic/datagen/policy behavior must not regress.

---

# 1. Workspace Physics + Physics Compiler

Inspect how the current Real2Sim workspace becomes an Isaac manipulation environment.

For each semantic object role, trace the existing implementation:

- SO-101 robot
- table
- workspace marker / board
- cube / movable object
- cup / receptacle target
- background
- camera visual proxy
- other visual scene geometry

For each role verify:

- visual geometry
- collision geometry
- rigid body state
- static / dynamic / kinematic behavior
- physics material
- static friction
- dynamic friction
- restitution
- mass
- inertia
- reset behavior
- collision approximation
- provenance of physical parameters

Distinguish values that are:

```text
user_measured
model_asset
derived
numerically_fitted
default_template
provisional
unmeasured
```

## Important cup requirement

A cup is an open-top receptacle.

Do not use collision that behaves like a single solid cylinder and blocks the interior volume.

The collision representation must allow a cube to physically enter the cup while still colliding with the wall and bottom.

---

## Physics Compiler goal

The user should not have to manually author collision and rigid-body settings for every new workspace.

Implement a reusable publish/build-time physics layer conceptually like:

```text
Real2Sim Profile
        +
Task Definition
        ↓
Workspace Physics Compiler
        ↓
Isaac Workspace
```

Use semantic roles rather than workspace-specific handwritten physics wherever possible.

Suggested roles:

```text
STATIC_SUPPORT
STATIC_GEOMETRY
DYNAMIC_MANIPULAND
FIXED_TARGET
VISUAL_ONLY
ROBOT
```

Typical defaults:

```text
table
→ STATIC_SUPPORT
→ collision enabled
→ no rigid body motion

workspace marker / paper
→ VISUAL_ONLY or STATIC_GEOMETRY
→ collision normally disabled unless physically necessary

cube
→ DYNAMIC_MANIPULAND
→ rigid body
→ collision
→ mass/inertia
→ physics material
→ resettable

cup
→ FIXED_TARGET
→ open receptacle collision
→ fixed pose
→ no unnecessary rigid-body dynamics

background
→ VISUAL_ONLY

camera proxy / bracket
→ VISUAL_ONLY unless mechanically relevant

SO-101
→ existing articulation physics
```

Do not visually guess mass, friction, restitution, or other measurable physical values.

Use:

```text
measured value
OR
known asset value
OR
documented default template
```

and record provenance.

Implement the smallest reusable compiler/config layer needed to support this architecture without breaking the working workspace.

---

# 2. Sim / Real Representation Contract

Trace the complete observation/action path:

```text
Physical SO-101
→ Source Demo
→ Annotation
→ Mimic
→ Datagen
→ Isaac dataset
→ LeRobot conversion
→ ACT / SmolVLA training
→ Sim inference
→ Real inference
→ Physical follower
```

The preferred canonical representation is:

```text
URDF / model-space joint radians
```

unless repository evidence shows a stronger reason otherwise.

The policy should not need to know whether it is controlling Sim or Real.

---

## Hardware boundary

Audit the current LeRobot 0.4.1 interface.

Expected hardware-native representation:

```text
arm joints
→ calibrated degrees

gripper
→ calibrated 0–100 %
```

These units belong at the hardware I/O boundary.

Conceptually:

```text
REAL OBSERVATION

raw encoder
→ LeRobot calibration
→ degree / %
→ SO101 representation adapter
→ model-space radians
→ policy
```

and:

```text
REAL ACTION

policy model-space radians
→ inverse SO101 representation adapter
→ degree / %
→ LeRobot follower
```

Sim should remain in model/URDF-space radians.

---

## Audit and correct the following paths

Inspect:

- `SO101JointMapper`
- calibrated hardware coordinates
- URDF/model radians
- source-demo HDF5 observations
- source-demo HDF5 actions
- `joint_targets`
- Mimic Cartesian TCP-delta actions
- IK output units
- datagen output
- replay
- `convert_isaac2lerobot`
- LeRobot dataset feature definitions
- normalization statistics
- ACT preprocess
- ACT postprocess
- SmolVLA preprocess/postprocess where present
- Sim policy inference
- Real policy inference
- follower command conversion
- gripper percent ↔ model-radian mapping

Already-model-radian values, IK output, replay targets, or model-space joint targets must **not be mapped twice**.

---

## Versioned Representation Contract

Implement a reusable representation contract instead of relying on assumptions scattered through scripts.

Preferred conceptual ID:

```text
MODEL_JOINT_RAD_V1
```

The contract should be able to carry or reference:

- contract id
- schema/version
- robot joint names/order
- unit per field
- action semantic
- observation semantic
- gripper convention
- mapper metadata
- mapper version/hash
- robot asset identity/hash
- normalization metadata where applicable

Datasets, policies, and runtime loaders should eventually be able to reject incompatible combinations instead of silently running them.

Preserve existing working formats when possible, but add explicit metadata and adapters where needed.

---

# 3. Unified Workflow UI

Implement a unified UI that allows the user to execute the entire project workflow step by step.

Target workflow:

```text
Real2Sim Capture
→ Fit / Validate
→ Accept Revision
→ Publish Workspace
→ Source Demo
→ Replay
→ Annotation
→ Mimic / Datagen
→ LeRobot Conversion
→ SmolVLA Train
→ Hugging Face Dataset Publish
→ Hugging Face Policy Publish
→ Sim Evaluation
→ Real Evaluation
```

The order may be adjusted when technically necessary, for example dataset upload may happen immediately after conversion and policy upload after training.

---

## Meaning of automation

Do **not** automatically execute the entire pipeline.

Automation means:

```text
user presses Run for one stage
→ stage executes
→ success is verified
→ output artifact is registered
→ output becomes the default input for the next stage
→ compatible parameters are pre-filled
→ next stage becomes READY
```

The user still explicitly presses the next `Run` button.

Example:

```text
Source Demo
[ Run ]

↓ success

output:
workspace_001_source.hdf5

↓ automatically registered

Annotation
input:
workspace_001_source.hdf5

[ Run Annotation ]
```

---

## Independent execution must remain possible

Every stage must also work independently.

Examples:

- manually select another source HDF5 for Annotation
- manually choose another annotated dataset for Datagen
- manually choose another LeRobot dataset for training
- manually choose another policy for Sim Eval
- manually choose a workspace/revision
- manually upload a selected dataset or checkpoint

Therefore support both:

```text
Guided Workflow
+
Independent Tool Execution
```

Do not create separate duplicated backend implementations for these two UI modes.

They should call the same operation/service layer.

---

# 4. Artifact-Aware Workflow

Do not make the UI infer pipeline relationships only from filenames.

Introduce typed artifacts or an equivalent explicit model.

Suggested artifact types:

```text
RevisionArtifact
WorkspaceArtifact
SourceDemoArtifact
ReplayArtifact
AnnotatedDemoArtifact
DatagenArtifact
LeRobotDatasetArtifact
PolicyArtifact
SimEvaluationArtifact
RealEvaluationArtifact
HFDatasetArtifact
HFPolicyArtifact
```

Each artifact should carry useful metadata such as:

```text
artifact id
artifact type
path
workspace id
workspace version
source artifact id(s)
creation time
status
representation contract
task/config identity
relevant hashes
optional remote repository metadata
```

Suggested execution states:

```text
NOT_READY
READY
RUNNING
SUCCEEDED
FAILED
```

The UI should derive readiness from actual artifact availability and compatibility.

---

# 5. File / Folder Selection UX

Fix the current annoying behavior where browse dialogs open in unrelated directories.

Each Browse action must open at the most relevant directory for the artifact type.

Suggested defaults:

```text
Revision
→ outputs/real2sim/revisions/

Workspace
→ outputs/real2sim/environments/

Source Demo
→ actual source-demo output directory

Annotation
→ annotation output directory

Datagen
→ datagen output directory

LeRobot Dataset
→ converted dataset directory

Policy / Checkpoint
→ training output/checkpoint directory
```

Prefer the actual current repository directories if they differ from these examples.

When possible:

```text
current workspace/session
→ most recently generated compatible artifact
→ automatically preselected
```

Manual override must always remain available.

Also provide an `Open Folder` action for artifacts so the user can immediately open the directory containing the selected file.

For revisions specifically, the picker must default to the actual revisions directory rather than an unrelated working directory.

---

# 6. Shared Operation Layer

Avoid a UI that constructs many unrelated shell command strings directly.

Prefer:

```text
UI
↓
Workflow / Operation API
↓
existing project implementation
```

For example:

```text
publish_workspace(...)
record_source_demo(...)
replay_demo(...)
annotate_demo(...)
generate_dataset(...)
convert_to_lerobot(...)
train_policy(...)
evaluate_sim(...)
evaluate_real(...)
publish_hf_dataset(...)
publish_hf_policy(...)
```

Existing CLI scripts may remain as wrappers around the same common APIs.

The purpose is to prevent:

```text
CLI behavior
≠
UI behavior
```

and avoid duplicating business logic.

Refactor existing scripts only where useful and preserve their current CLI functionality.

---

# 7. Hugging Face Dataset Publishing

After a LeRobot dataset is successfully created, allow it to be uploaded to Hugging Face Hub from the UI.

The upload must be an explicit user action:

```text
[ Upload Dataset ]
```

Do not automatically upload after conversion.

Allow configuration of:

- repository id
- public/private visibility
- existing repository vs new repository
- revision/branch/version when supported
- manually selected local dataset artifact

Prefer standard Hugging Face / LeRobot APIs and formats.

Do not invent a custom remote dataset format unnecessarily.

---

## Dataset publishing metadata

Preserve enough provenance to reproduce or understand the dataset.

Include or reference where practical:

- workspace id/version
- task definition/version
- episode count
- source-demo provenance
- annotation provenance
- datagen config
- randomization config
- Real2Sim profile/revision
- representation contract id/version/hash
- robot asset identity/hash
- camera/profile identity
- dataset format/version
- generation timestamp/version
- relevant code/config revision where available

Record successful remote publication information in the local artifact registry:

```text
repo id
repo type
revision
commit/version if available
URL
upload status
```

---

# 8. Hugging Face Policy Publishing

After ACT/SmolVLA training succeeds, allow the trained policy/checkpoint to be uploaded from the UI.

Explicit action:

```text
[ Upload Policy ]
```

Allow:

- repository id
- public/private
- new/existing repo
- revision/version
- checkpoint selection

Prefer standard Hugging Face model repository conventions.

---

## Policy publishing metadata

Include or reference:

- base model
- policy architecture
- training dataset repository/local artifact
- exact dataset revision/version where available
- training configuration
- training steps / checkpoint
- representation contract
- normalization statistics/config
- preprocessing/postprocessing metadata
- robot identity
- workspace/task identity
- code/config version where available

Record resulting Hugging Face repository metadata in `HFPolicyArtifact`.

The policy should then be selectable directly for:

```text
Sim Evaluation
Real Evaluation
```

without forcing the user to browse for files again.

---

# 9. Current Randomization Direction

Do not redesign the full randomization system in this phase unless needed for clean architecture.

Preserve the existing project behavior and prepare it for later generalization.

Current intended first SmolVLA randomization direction:

- keep the existing four representative/anchor object positions concept
- increase XY positional jitter around those anchors
- no cube yaw randomization initially
- reject overlapping cube layouts
- keep objects inside the allowed workspace boundary
- record the seed/layout parameters for reproducibility

The physical workspace is approximately a 150 mm square, but use the actual profile/task data rather than hard-coding this if the current workspace schema already provides it.

Future dataset coverage analysis may decide whether the four-anchor approach should be replaced by broader uniform/stratified sampling.

Do not over-engineer this part during the current phase.

---

# 10. Regression Requirements

The following currently working behavior must remain functional:

```text
workspace publish
generic workspace preview/runtime
physical leader connection
source-demo recording
HDF5 writing
workspace metadata
coordinate contract metadata
replay
```

Existing CLI workflows should continue to work unless there is a compelling architectural reason to change an interface.

If an interface must change:

- keep a compatibility wrapper where reasonable
- document the migration
- update all internal callers
- add tests

Do not silently alter units or semantics.

---

# 11. Implementation Strategy

This phase may modify code.

Do not stop after writing an architecture report.

Use the following pattern:

```text
inspect current path
→ determine smallest coherent architecture
→ implement incrementally
→ run targeted tests
→ run regression tests
→ continue
```

Do not perform a massive speculative rewrite first.

When an ambiguity affects physical correctness, dataset semantics, or backward compatibility, inspect the actual code/data before choosing.

Do not guess measurable physical parameters from images.

---

# 12. Recommended Implementation Order

Use repository evidence to adjust this order if necessary, but prefer approximately:

```text
1. Audit current physics and representation paths

2. Introduce/solidify Representation Contract
   without breaking source-demo/replay

3. Introduce common artifact model / artifact registry

4. Introduce shared workflow operation layer
   around existing working CLI functionality

5. Refactor workspace physics into reusable
   semantic-role-based Physics Compiler/config

6. Wire Source Demo + Replay through common workflow API

7. Integrate Annotation

8. Integrate Mimic / Datagen

9. Integrate LeRobot conversion

10. Integrate training

11. Add Hugging Face Dataset publishing

12. Add Hugging Face Policy publishing

13. Integrate Sim Evaluation

14. Integrate Real Evaluation

15. Build/finish unified guided UI on top of the same APIs
```

UI work can proceed incrementally as each operation becomes available; a full backend rewrite is not required before showing functionality.

---

# 13. Tests

Add tests focused on contracts and regression.

At minimum consider tests for:

- workspace physics role compilation
- cup collision structure validity
- source-demo representation metadata
- replay representation compatibility
- mapper single-application guarantee
- hardware degree/% ↔ model-radian round trip
- gripper mapping round trip
- artifact dependency registration
- workflow readiness transitions
- file-picker default-directory resolution
- CLI and UI/common-operation behavior equivalence where practical
- HF metadata construction without requiring a live upload
- workspace publish/source-demo/replay regression

Do not require Hugging Face network access for normal unit tests.

---

# 14. Documentation

Update project documentation so a user can understand:

```text
Workspace
→ Source Demo
→ Replay
→ Annotation
→ Datagen
→ LeRobot Dataset
→ Train
→ HF Publish
→ Sim Eval
→ Real Eval
```

Document:

- canonical representation
- hardware boundary units
- artifact relationships
- physics role defaults
- how the guided workflow works
- how to run each stage independently
- where files are stored
- how Hugging Face publishing works
- how to choose another file manually

---

# 15. Completion Report

After implementation, report:

1. What the previous implementation did
2. Problems/risks discovered
3. Files changed
4. Files/modules added
5. Physics architecture implemented
6. Representation contract implemented
7. Artifact/workflow architecture implemented
8. UI changes
9. Hugging Face publishing support
10. Backward compatibility decisions
11. Tests executed and exact results
12. Remaining limitations
13. Commands for the user to test the next workflow stage

Do not claim a test passed unless it was actually executed.

If a feature cannot yet be completed because of an external dependency, hardware requirement, or missing information, implement/test the non-hardware parts and clearly identify the remaining manual verification.

---

# Final Goal

The project should move toward:

```text
Physical Environment
        ↓
Agentic Real2Sim
        ↓
Published Workspace
        ↓
Source Demonstrations
        ↓
Annotation
        ↓
Mimic / Datagen
        ↓
LeRobot Dataset
        ↓
SmolVLA / ACT
        ↓
Sim Evaluation
        ↓
Real Evaluation
```

with:

```text
one reusable workspace/runtime architecture
one explicit representation contract
one artifact-aware workflow
one unified UI
optional Hugging Face dataset/model publishing
```

The user should not need to manually hunt for intermediate files or repeatedly reconstruct command-line arguments between stages.
