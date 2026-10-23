# Metadata 기반 Task / Script SDK — 현재 구조

기준: 2026-10-23 로컬 코드. 공개 문법·명령·예제·지원 한계는 [SCRIPT_GUIDE](../task_sdk/SCRIPT_GUIDE.md) 한 곳에서 관리한다. 이 문서는 내부 경계와 상태 소유권을 설명한다. 이전 Phase 결과는 [Phase 1–3](../../outputs/workflow_validation/task_sdk/REPORT.md), [3.5](../../outputs/workflow_validation/task_sdk_phase35/REPORT.md), [4](../../outputs/workflow_validation/task_sdk_phase4/REPORT.md), [5](../../outputs/workflow_validation/task_sdk_phase5/REPORT.md), [Scene/Terms](../../outputs/workflow_validation/scene_terms_sdk/REPORT.md)에 보존한다.

## Metadata → 실행 경계

| 경계 | 실제 소유자 | 계약 |
|---|---|---|
| Builder export 검증 | workspaces.package.load / builder | 실제 ID·asset·frame·Physics/좌표 증거. 원본 불변 |
| Scene 구성 / 템플릿 | workspaces.tasks.configured / generate_template | 검증된 Metadata의 물체 선택. 객체 이름으로 역할 추측 안 함 |
| 정적 Script | task_scripts.read_script / read_followup / resolve_script | 제한된 bytes·AST literal·Path/SHA·선언 의미. 사용자 코드 실행 없음 |
| 실행 Python | task_scripts.load_execution_script / load_followup | 동일 entry bytes 검증 후 명시 Worker에서만 실행. Reset callable binding 공유 |
| Runtime binding / Event | SimulationAdapter.bind_followup / after_reset / record_frame | 빌린 Context, 환경/Episode별 factory·Tracker·Boundary/실패 latch |
| Dataset | SourceEpisode / EpisodeBuffer → DatasetWriter | 전달된 snapshot/Signal·영상 기록. 물리 실행/판정 수학 없음 |
| 학습 | learning.training → 공식 CLI | Workspace/Task SDK 없이 Native Dataset 사용 |

`SCRIPT`와 `FOLLOWUP`의 문법 검사는 task_scripts에 남는다. 여기서 Subtask는 실행 알고리즘이 아니라 지원 가능한 literal 설정을 검사한다. `.mimic`의 Torch/reference generator를 GUI의 정적 검사에 import하지 않는다. Runtime Term 실행은 Adapter가 소유하며 파서를 역참조하지 않는다. 사용자 Task별 분기·Plugin Registry·Manager·추가 Launcher는 없다.

## TaskContext

기존 단일 MuJoCo Backend/workspace와 선택 Adapter를 빌린다. `object_ids`, `camera_ids`, `robot_asset`, `require`, `validate_bindings`, `object_pose`, `joint_state`, `placement_details`, `placement`를 제공한다. Pose는 workspace-frame 4×4/metre, joint는 기존 measured model-rad다. Camera ID 검증은 RGB 생성 API가 아니다.

Metadata 초기 Transform과 현재 Pose를 구별한다. Context는 Workspace를 다시 읽거나 Engine을 생성/reset/step/close하지 않는다. Callback 중 `_observation`/`_native_placement`는 이미 확보한 pre-step 관측/fact를 잠시 빌리며 finally에서 해제한다. Frame 간 상태 Cache가 아니다. Robot identity는 존재하는 asset 참조와 검증된 manifest hash 범위이며 compiled robot 전체/실물 인증이 아니다. Python은 신뢰한 코드이며 sandbox가 아니다.

## Reset 순서와 소유권

1. Source Worker의 `env.reset(seed + episode)` 또는 Mimic Adapter의 기존 seed stream → Backend.reset.
2. Backend가 RNG를 초기화하고 Legacy는 WorkspaceRandomizer.sample을 먼저 호출한다(Scene-only는 생략). 이후 `mj_resetData` → Robot home/ctrl → 샘플 배치 적용 또는 Builder 배치 유지 → `mj_forward` → native snapshot 순서다. Physics step은 추가하지 않는다.
3. Scene-only 명시 Pose/Reset이 있으면 `reset_objects`가 **기존 Backend RNG**를 전달하여 선언 순서대로 지정 Body만 적용한다. Uniform/Edge는 같은 WorkspaceRandomizer.sample을 사용하며 다른 Body의 현재 위치를 고정 이웃으로 검사한다. 따라서 여러 명시 Reset의 선언 순서와 RNG 소비 순서도 계약이다.
4. Support/SE(3)/침투 검사 → 기존 set_object_pose(해당 Body velocity 0, 전체 qacc warm-start 해제, forward) → Backend 최종 initial snapshot. 오류 시 native state를 복원하고 전파한다. 소비된 RNG는 되돌리지 않으며 실패 Episode를 계속하지 않는다.
5. Adapter.after_reset은 기존 Tracker/진단·Event latch를 초기화하고 환경/Episode마다 factory를 호출한다. 선택 reset(ctx) → Buffer 초기 state 기록. Annotation 재생은 저장된 native/numeric state를 복원하며 Object Reset Term을 재실행하지 않는다.

기존 Legacy random Reset과 새 Builder 기본 Pose는 의도적으로 다르다. Context/Reset Term이 별도 RNG/Engine을 소유하지 않는다. Adapter의 RNG는 기존 Episode seed stream이고 Backend RNG는 실제 배치 샘플러다. Mimic reference의 NumPy/Torch RNG는 기존 RandomStream이 잠금/복원한다.

## 판정과 저장 책임

Placement는 Backend/Isaac workspaces.environment 원본, StableGrasp는 workspaces.grasp.StableGrasp 원본, Drop/Release는 Adapter.update_grasp/_update_failure_evidence 원본이다. Task Term은 이 상태를 조회하며 Tracker를 새로 계산하지 않는다. Native recipe는 기존 Placement AND latched StableGrasp를 유지한다. Generic 채택·실패는 명시 Script 의미이며 Native 성공으로 위조하지 않는다.

Annotation은 Manual Mark/Skip/Pause/Step·채택·최종 Signal을 소유한다. EpisodeBuffer는 Legacy/Term Event 배열을 같은 경로로 조립하고 기존 두 annotation schema/JSON 순서를 유지한다. Mimic은 저장된 Signal을 DataGenInfoPool로 읽으며 Annotation Event를 다시 계산해 과거 Boundary를 덮어쓰지 않는다. 생성 중 새로운 궤적의 Event/채택은 같은 Adapter 실행을 사용한다.

Mimic configuration의 Native recipe와 사용자 Subtask는 하나의 SubTaskConfig 조립을 사용한다. 기존 reference sampling/trajectory/IK/scheduler 수학은 그대로다. Evaluation의 Placement 종료는 Annotation acceptance와 다르며 사용자 성공/물리 진단을 별도로 기록한다.

## 호환성·후속 경계

함수형 Task Python, 목록형 Source, options.tasks, FOLLOWUP, Native Isaac 및 기존 Checkpoint/Reader 계약을 유지한다. `workflow.mujoco.rollout/rollout_lerobot`의 직접 호출 API는 CLI 주 경로는 아니지만 기존 MiniVLA 추론 검사 호출자가 있어 이번에 삭제하지 않는다. Worker의 `rollout` stage는 evaluate_sim으로 dispatch한다.

Isaac TaskContext, 다른 Robot/EEF, 일반적 Task Controller는 미구현이다. Source/Annotation/Mimic/Evaluation을 하나의 성공 Boolean/범용 Runner로 합치지 않는다. UI·Worker 연결 위치는 [WHERE_TO_EDIT](WHERE_TO_EDIT.md), 실제 A/B·기술 부채는 [정리 보고서](../../outputs/workflow_validation/architecture_cleanup/REPORT.md)를 따른다.
