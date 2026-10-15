# SO-101 Real2Sim Testing Guide

테스트의 목표는 "실행된다"를 넘어서 **변경이 맞다는 증거를 만드는 것**이다.

## 1. 테스트 층

### A. Pure/unit tests
Isaac renderer나 실제 serial 장비 없이 검증 가능한 로직.

예:
- joint mapping degree/% ↔ model rad
- profile schema/provenance
- transform math
- artifact/workflow state
- workspace package metadata

### B. Isaac integration tests
실제 Isaac Lab/Isaac Sim API 경계가 필요한 검증.

예:
- articulation state/target 적용
- camera prim/config
- workspace scene compile
- reset/physics 적용
- render/replay

### C. Hardware integration tests
Leader/follower/camera 실제 장비가 필요한 검증.

예:
- calibration identity
- follower measured state
- leader→follower command
- capture synchronization

### D. End-to-end workflow
Demo → replay → annotation → datagen → conversion → train/eval artifact 흐름.

## 2. 현재 테스트 위치

핵심 regression suite는 `tests/real2sim/`에 있다.

주요 예:
- `test_joint_mapping.py`: hardware/model coordinate 경계
- `test_camera_prim.py`: camera config/prim
- `test_pipeline.py`: file IPC와 USD scene 생성
- `test_snapshots.py`: Snapshot 계약/저장/세션/UI와 독립 Builder import·FK·fit·rollback
- `test_provenance.py`: measured/provisional provenance
- `test_runtime_liveness.py`: UI/runtime IPC와 liveness
- `test_workspaces.py`: workspace package/environment
- `test_workflow.py`, `test_workflow_view.py`: workflow operation/UI
- `test_task_definitions.py`: task definition contract
- `test_provenance.py` (기존 Profile/visual 검사 통합): 현재 기본 Profile/scene geometry와 visual

옛 paired/continuous capture와 Reference/API-agent/fitting 전용 테스트는 제거했다.
MuJoCo는 import 후 GL 모드를 바꿀 수 없으므로 Builder rendering 검사와
`test_mujoco_backend.py`/`test_minivla_integration.py`의 physics-only 검사는 별도
프로세스에서 실행한다. USD를 import하는 tests도 MuJoCo backend와 같은 process에 넣지 않는다.
Renderer/engine guard를 테스트 때문에 우회하지 않는다. 최신 분리 suite 결과 및 실행 목록은
`outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md`를 따른다.

## Active Object Set 검사 (2026-10-15)

`test_active_objects.py`는 imported Builder fixture의 실제 MuJoCo body/geom/joint/state/step,
1/2/3개·비연속 조합·새 Dynamic ID, reset/overlap/seed, 기록 상속·Resume·mixed curation을 검사한다.
기존 Task/UI/Source/Precision/두 엔진 회귀와 함께 실행한다. 상세 실행 로그/테스트 분리 목록,
실제 Isaac GPU 렌더·20-frame HDF5 Replay·HDF5/LeRobot 변환 및 초기 실패 기록은
`outputs/workflow_validation/active_scene_objects/REPORT.md` 참조. 모델 학습·물리 grasp 성공과 구분한다.

## 3. Joint 변경 시 최소 검증

- known degree → expected rad
- rad → hardware representation round trip이 허용 오차 내인지
- joint name/order가 유지되는지
- gripper가 arm joint와 다른 normalization 경로를 유지하는지
- sim action target과 measured state를 혼동하지 않는지
- legacy demo를 새 mapping으로 자동 오인하지 않는지

중요: repository/model 구조만으로 physical zero/sign이 완전히 증명되지 않을 수 있다.
실제 camera/follower 측정 데이터가 부족하면 `NEEDS_MORE_DATA`로 남겨야 한다.

## 4. Camera/Real2Sim 변경 시 최소 검증

가능하면 train/fit data와 held-out validation을 분리한다.

추천 metric:
- landmark/keypoint reprojection error (px)
- silhouette/mask overlap (예: IoU)
- object center/corner reprojection error
- camera calibration reprojection error
- paired-capture joint/state mismatch
- timestamp/receipt spread

결과는 가능하면 before → after 수치로 남긴다.

예:
```text
camera reprojection: 31.4 px -> 6.8 px
robot keypoint:      42.0 px -> 9.1 px
held-out views:      no regression
```

## 5. Optimizer에서 반드시 보는 실패 조건

- camera extrinsic 오차를 robot base transform이 대신 보상
- object pose 오차를 camera pose가 대신 보상
- training frames만 좋아지고 held-out frames가 악화
- wrist/side 중 한 camera만 과적합
- provisional parameter가 verified/measured로 승격
- joint mismatch 상태에서 image metric만 최적화

따라서 parameter group을 나누고, 한 그룹을 바꿀 때 다른 그룹을 고정하거나 prior/constraint를 둔다.

## 6. Dataset/replay 변경 시 최소 검증

```text
record
  → save
  → reload
  → restore initial state
  → replay
  → compare trajectory/state
```

확인 항목:
- field 존재 여부
- dtype/shape
- coordinate contract/version
- initial state
- action semantics
- observation semantics
- replay target field
- task/workspace metadata

## 7. AI/Codex 작업 완료 기준

"테스트 통과" 한 줄이 아니라 다음을 보고해야 한다.

- 실행한 정확한 test/command
- 어떤 실패 조건을 커버하는지
- 결과
- 실행하지 못한 test와 이유
- 실제 hardware/Isaac validation이 필요한 남은 부분

고위험 변경(joint mapping, transform, dataset schema, physics state write)은 사람이 관련 구현과 테스트를 직접 확인한다.

## 공통 Joint Control 검사 (2026-10-12)

`test_joint_control.py` + 기존 precision/calibration/UI/snapshots/launcher 검사를 실행한다.
MuJoCo/Source/Replay 검사와 USD/Workspace 검사는 별도 프로세스에서 실행한다.
설치된 LeRobot SDK의 normalize/integer encoding을 포트 없는 객체로 검사하므로 모터 이동이 아니다.
정확한 명령/최종 결과/Isaac 렌더 범위/실패 이력은
`outputs/workflow_validation/unified_joint_control/REPORT.md`를 따른다.
공통 계약과 구동 heuristic의 한계는 [JOINT_CONTROL.md](JOINT_CONTROL.md)를 따른다.
