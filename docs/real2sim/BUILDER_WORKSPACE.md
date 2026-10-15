# Builder export를 기존 Real/Sim과 Workflow에서 사용

## Workflow에서 사용할 물체 선택 (2026-10-15)

Workspace Pipeline의 기존 TaskSelector에서 `Scene objects`를 체크한다. 기본은 전체 선택이며
`Active Objects: n / total`로 수를 표시한다. Source와 Sim Eval에서 바꿀 수 있고 실행 중에는 잠긴다.
이 설정은 Workflow 실행에 적용된다. 원본 Workspace나 Live Real/Sim의 영구 Scene을 편집하지 않는다.

1. Workspace와 backend를 고른다. 예: `scene2_builder/v8`.
2. Red만 체크하고 Red→Cup Task를 선택하면 물체 1개 실험이다.
3. Red/Blue 또는 Red/Green을 체크하면 2개, 모두 체크하면 3개 실험이다.
4. Pick은 체크된 물체의 명시적 Task 중에서 고른다. 선택 해제로 기존 Task가 빠지면 경고가 나오며
   사용자가 다른 Preset을 선택해야 한다. 문구만 바꿔 Target Identity를 추측하지 않는다.
5. Source 수집 → Replay → Annotation → DataGen → Convert → Train을 기존대로 실행한다.
   Replay/Annotation/DataGen의 체크박스는 잠기며 입력 자료의 물체 구성을 보여준다.
6. Sim Eval에서는 학습과 다른 물체 조합과 목표도 명시적으로 선택할 수 있다.
   결과의 `scene_composition_distribution_shift`로 구성 차이를 확인한다.

컵/작업대/로봇/카메라는 선택 목록에서 제외하고 유지한다. 체크를 끈 조작 물체는 시각적 Hide가
아니라 body/collision/joint/state/reset에서 빠진다. MuJoCo Full Workspace 배경도 유지한다.
빈 선택·중복·알 수 없는 ID·비활성 Pick·Four Anchors에서 5개 이상 선택은 실행 전에 거절한다.
Workspace별 선택은 기존 Registry `ui.json.object_selections`에 원본 identity를 key로 저장한다.
다른 Workspace에 이전 ID를 무조건 복사하지 않는다. 파생 단계 열람은 Source의 선택을 덮어쓰지 않는다.

새 Builder 물체는 canonical manipulable 역할 및 Workflow dynamic으로 등록하고 dimensions,
collision, mass를 제공한 새 export를 import하면 표시된다. Display Name이 있으면 사용하고
내부 ID와 분리한다. 해당 물체를 distractor로 포함할 때 새 Task는 필요 없으나 Pick으로 쓰려면
Builder의 명시적 `scene.workflow.tasks`가 있어야 한다. 결합 mesh에서 객체를 독립 제거할 근거가
없으면 실행을 거절한다. Published 파일이나 기존 Export를 수정하여 목록을 바꾸지 않는다.

공통 구성은 `workflow.tasks.configured()`의 실행 복사본과 기존 Request의
`options.active_object_ids` / `scene_configuration`으로 전달한다. 데이터 저장 위치와 과거 데이터
복원 규칙은 [DATA_FLOW.md](DATA_FLOW.md), 학습/평가 실험 구분은
[MiniVLA 평가](../minivla/EVALUATION.md)를 참조한다.
검증 근거: `outputs/workflow_validation/active_scene_objects/REPORT.md`.

## Real/Sim에서 바로 Scene 선택 (2026-10-14)

이제 **Real / Sim 탭 → 오른쪽 맨 위 `Scene / 관절 보정 · 다음 Connect`**에서
Scene 목록 선택 또는 `Scene 선택 / Import`를 할 수 있다. 기존 Workflow Registry와
Workspace 선택을 공유하며, Source Demo의 Workspace 선택도 Real/Sim에 동기화된다.
Workflow 탭으로 이동할 필요 없이 저장 관절 보정과 Scene 보정 제외 옵션도 같은 영역에서 바꾼다.
두 옵션은 Capture와 동일한 위젯/값이고 재시작 후에도 선택을 복원한다.
Pair의 저장 위치·Scene 이름·Scene state는 같은 탭 오른쪽 `Snapshot 저장 설정`에 있다.

Scene2 v7 접촉 보정 검토: `scene2_builder/v7` 선택 → `선택한 Scene 사용` ON →
`저장된 관절 보정 사용` ON → `URDF 원형 검사 · Scene 관절 보정 제외` OFF → backend 선택 → Connect.
이미 연결되어 있으면 직접 Disconnect → Connect한다. Scene 선택/탭 전환/옵션 변경은
장치 연결이나 로봇 이동을 실행하지 않는다. 화면 아래 `연결된 Scene/설정`은 현재 worker 값이며
다음 연결 선택값과 구분한다. Workflow 실행 중에는 Scene 설정 변경을 막는다.

검사/화면: `outputs/workflow_validation/real_sim_scene_controls/REPORT.md`.
아래 이전 기록의 Workflow 탭 선택 절차는 계속 사용할 수 있지만 필수는 아니다.

## 최신 Scene2 v7 접촉 보정 검토

현재 UI용 **scene2_builder/v7**은 marker XY/yaw와 캡처 기반 Sim joint residual을 포함한다.
보정 파일은 그대로이며 model residual은 정확한 Precision mapping hash에 binding된다.
UI를 재시작하고 v7 선택/저장 관절 보정 사용 ON/**URDF 원형 검사·Scene 관절 보정 제외 OFF**로
사용자 직접 Connect한다. 원형 옵션이 켜져 있으면 residual을 제외한다.
MuJoCo/Isaac 실제8pose FK/6task smoke 및 Fake 저장 Profile Connect를 확인했다.
실물 접촉/independent calibration은 UNVALIDATED다.
`outputs/workflow_validation/scene2_v7_contact_application/REPORT.md` 참조. 아래v6는 이전 버전이다.

## 최신 Scene2 검토 Workspace — 2026-10-14

**scene2_builder/v6**는 v5의6개 단일 블록→컵 task를 유지하면서 상판과 robot base의
수평·접촉 조건을 수정한 새 package다. UI에서는 v6를 선택한다. 이미 연결했으면 직접 재연결한다.
베이스 매립15.898mm→0, 상판 기울기0.27736°→0. 카메라/보정/원본 v5는 유지한다.
영상 정합은 UNVALIDATED이며 Wrist 잔차와 실제 grasp/slip 문제가 남는다.
근거: `outputs/workflow_validation/scene2_base_table_contact/REPORT.md`.
아래 v5/v4 절차는 각 버전 당시의 기록이다.

## Scene2 작업용 Workspace — 2026-10-14

현재 작업용은 `scene2_builder/v5`: 빨강/초록/파랑 블록 × cup_a/b의6개 단일 pick/place task.
Builder `single_block_workflow` export의 명시적 Metadata를 사용한다. 성공은 Scene1과 같은
컵 내부 포함+속도0.03m/s 미만, Reset은 four_anchors와 XY±20mm다.
UI에서 v5와 task 하나를 선택하고 Source→Annotation→Datagen으로 진행한다.
파생 데이터는 Source task를 유지한다. 아래 v4는 역사적 geometry/physics 검토용이다.
검사 범위·미검증 E2E: `outputs/workflow_validation/scene2_workflow_binding/REPORT.md`.

## 현재 사용법 — 2026-10-13

Scene2 검토 Workspace: `outputs/real2sim/environments/scene2_builder/v4`.
Builder `scene2-visual-physics-refinement-20261013-bbb16a35`의 `mounted_clip_detail_review` export를 사용한다.
재구성은 OPEN/UNVALIDATED이며 실행 검토 준비와 별개다.
실제 검사/잔여 한계는 `outputs/workflow_validation/scene2_visual_physics_refinement/REPORT.md`.

```bash
python scripts/real2sim/workflow.py import-builder /absolute/path/to/export --workspace-id scene2_builder --version 1
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

Import CLI는 package checksum 검사→새 immutable Workspace→Workflow Registry 등록을 수행한다.
같은 package/task는 재사용하고 다른 결과는 새 version이다. 기존 UI 선택은 변경하지 않는다.
UI Workspace Pipeline에서 `scene2_builder/v4` 선택 → Real/Sim의 Pipeline Workspace 사용
→ MuJoCo 또는 Isaac → **Sim 미리보기**. 실제 장치는 사용자가 직접 Connect/Start Teleop한다.
실행 Worker Python은 기존 lerobot-arena를 사용한다. GUI 전용 Python에 simulator를 설치하지 않는다.

### Geometry review와 Task Workspace

명시적인 `scene.workflow` 또는 `--task`가 없으면 `task.json.mode=geometry_review`다.
pick/place/success는 null이다. 명시적 양의 canonical mass가 없는 기존 review는 dynamic/targets가 비어 있다.
명시적 질량을 가진 manipulable이 있으면 기존 엔진의 dynamic object와 열린 cup collision을 사용한다.
Scene2 v4는 Scene1 템플릿의 20g 블록 3개와 고정 컵 2개를 사용한다. 물성 provenance는 task.object_physics에 보존한다.
이 경우도 task를 자동 생성하지 않는다. reset은 저장 initial pose로 되돌리며 물성은 실측 검증 완료가 아니다.
기울어진 표시 영역도 world transform을 그대로 보존한다. reset은 robot 초기 자세와 고정 배치만 제공한다.
가짜 task나 randomization을 만들지 않는다. Source/Annotation/Datagen/Task Eval에는 명시적인 workflow가 필요하다.
기존 task Workspace 파일과 task/reset/success 경로는 유지한다.

### Precision 계약

Adapter는 기존 legacy metadata와 `so101.precision-joint-mapping/1`을 구분한다.
서명, joint order/rad output, Profile coefficient/reference 일관성, reference motor/device/calibration identity,
원본 URDF의 limit-only 호환성을 검증한다. Portable URDF의 joint tree/origin/axis도 현재 모델과 비교한다.
manifest mapper는 실제 입력을 보존하며 DEFAULT로 위장하지 않는다. Sim은 URDF radians를 직접 받는다.
실물은 hardware.attach_precision에서 현재 활성 Profile/device identity를 검사하고 경계에서만 변환한다.
Import는 저장 Profile/active/LeRobot JSON/register를 변경하지 않는다.

Builder 작업 기준: sibling `real2sim-builder/docs/SCENE_REFINEMENT.md`.
패키지/도구 상세는 Builder `docs/CONSUMING_SCENES.md`, `docs/AGENTIC_TOOLKIT.md`를 참조한다.

## 이전 Scene1 실행 및 구현 기록

아래 “최신”, “지금 선택할 장면”은 기록 당시 기준이며 현재 Scene2 절차는 위를 따른다.

2026-10-11 최신 실행용 후보는 **`outputs/real2sim/environments/scene1_builder/v5`**.
v2 컵 배치 기반 Sim 영점/base/Wrist/블록 추정 보정이며 UNVALIDATED 검토용이다.
실제 두 엔진 렌더/FK smoke와 제한은 `outputs/workflow_validation/builder_v2_scene_refinement/REPORT.md`.

이전 기록: `outputs/real2sim/environments/scene1_builder/v3`는 기존 baseline geometry에
표면 색상 개선을 적용했다. Holdout 영상4장 모두 nonregression PASS지만 geometry/physics는
UNVALIDATED다. v2의 변경된 컵 배치는 포함하지 않는다. v1/v2는 비교용으로 보존했다.
`SCENE_REFINEMENT_HANDOFF.md` 최상단과 `surface_refinement_result.json`을 참고한다.

2026-10-10 당시 보정 비교용 Workspace는 `outputs/real2sim/environments/scene1_builder/v2`다.
컵 배치/색상 후보를 UNVALIDATED로 export한 것으로, holdout 검증 성공이 아니다.
기존 v1은 유지된다. 다른 Codex 계정의 재개는 root `SCENE_REFINEMENT_HANDOFF.md`와
`outputs/workflow_validation/paired_refinement_160851/layout_export_result.json`을 먼저 확인한다.

2026-10-09. Builder의 `scene.xml`, `scene.usd`, `scene_manifest.yaml`을 기존
SO-101의 제어 가능한 Workspace에 연결한다. Builder 원본 export는 수정하지 않는다.

## 지금 선택할 장면

`outputs/real2sim/environments/scene1_builder/v5`

UI를 다시 열고 Workspace Pipeline에서 이 Workspace를 선택한다.
Real / Sim에서는 **Pipeline Workspace 사용**을 켜고 Isaac 또는 MuJoCo를 선택한 뒤
Connect를 누른다. 카메라와 실물 장치 연결/Teleop 시작은 기존 버튼으로만 수행한다.
UI 시작이나 Workspace 가져오기는 엔진·실제 로봇을 시작하지 않는다.

Pipeline의 Source → Replay → Annotation → Datagen → Convert → Train / Eval 흐름은
그대로다. Task에서 블록 3개와 컵 2개의 조합을 선택한다. 새로운 장면으로 Source를
수집해야 하며, 이전 Workspace 데이터의 identity를 바꾸어 재사용하지 않는다.

다음 export를 가져올 때는 **Workspace 선택**에서 `scene_manifest.yaml`이 있는
Builder export 폴더를 선택하면 된다. GUI 전용 Python에 simulator/YAML 패키지를
추가할 필요 없이, UI의 **실행 Python**으로 기존 workflow CLI를 호출한다.

```bash
python scripts/real2sim/workflow.py import-builder /path/to/builder/export \
  --workspace-id scene1_builder --version 2
```

자동 ID는 source manifest 해시 기반이다. 같은 export와 task의 재가져오기는 기존
Workspace를 재사용한다. 다른 설정은 새 version을 사용한다. `--task task.json`으로
기존 workspace-task 계약의 pick/place, randomization, physics를 명시할 수 있다.
가져온 immutable Workspace 파일을 직접 수정하면 hash 검증에서 거절된다.

## 실제 연결 방식

```text
Builder export (읽기 전용)
  → workspaces.builder.import_scene
  → 기존 so101.workspace/1 + Builder provenance
  → Operations / Runtime의 기존 Workspace 로더
      ├─ Isaac make_config → 기존 articulation + exported USD visual references
      └─ MuJoCo build_scene → 기존 actuators/contacts + exported MJCF visuals
  → 기존 Task / reset / success / annotation / Mimic / policy 경로
```

- 참조용 로봇 snapshot을 두 번째 고정 로봇으로 넣지 않는다. 기존 SO-101 articulation,
  joint limits, actuator와 contact pad를 사용하고 Builder 로봇 visual은 해당 link에 붙인다.
  가져오기 전에 joint order, mapper, URDF joint tree를 검사한다.
- Table/base/camera pose, K, 크기와 시각 형상은 export/metadata에서 가져온다.
  Wrist 외형과 카메라는 움직이는 gripper frame을 공유한다.
- 동적 물체와 target은 export의 visual/collision geometry를 재사용한다. 구형·복합
  형상과 사각 target도 지원한다. 비원형 target은 명시적인 성공 영역이 필요하다.
  동적 관성은 bounding box 기반 근사이며 실측 관성이 아니다.
- 컵 인쇄는 컵의 자식 visual로 붙어 있다. 독립 장애물이나 꽉 찬 컵 collision으로
  처리하지 않는다. 카메라 장착물 역시 visual-only다.
- `workspace` 영역을 reset 지지 영역으로 사용한다. 기존 seed/four-anchor 방식으로
  블록들이 겹치지 않게 배치한다. Task pick/place ID가 성공 판정과 Mimic 대상에 전달된다.
- 질량 0.02 kg, 마찰 1, restitution 0은 기존 task의 **임시 시뮬레이션 설정**이다.
  Builder가 실측/식별한 물성으로 표시하지 않는다. 각 workspace의 `task.json`,
  `physics.json`, `provenance.json`과 manifest의 `builder`에 출처/상태를 남긴다.
- `READY`는 런타임 package 구성이 갖춰졌다는 뜻이다. Builder의 `UNVALIDATED`와
  물성 미검증 상태는 보존되며, 데이터 artifact에도 `builder_scene`으로 전달된다.
- MuJoCo는 원본 색상에 명시적인 sRGB 변환을 한 번 적용한다. 엔진별 조명과 접촉 solver가
  다르므로 pixel-identical rendering/동역학을 보장하지 않는다.

## 파일 구조

```text
scene1_builder/v1/
  manifest.json             # checksums / runtime identity / Builder status
  profile.json              # 기존 UI와 camera/reset 계약
  task.json                 # pick/place / randomization / provisional physics
  physics.json
  provenance.json
  builder/                  # 원본 geometry/manifest/assets의 필요한 사본
  assets/SO101/             # 기존 제어·충돌용 robot assets
  scene/
    static.usda
    wrist_mount.usda
    robot_visual_<link>.usda
    cube_<color>.usda
    cup_<id>.usda
```

Builder checkout이나 원본 export가 이후 이동돼도 runtime package는 자체 포함된다.
`builder/scene_manifest.yaml`은 원본 출처 기록이다. runtime 사본의 무결성 검사는
최상위 `manifest.json`을 사용한다(원본 export의 validation preview는 복제하지 않는다).

## 검증 범위

- 실제 scene1 package: MuJoCo compile, RGB 두 카메라, 10개 random reset, finite step.
- 6개 Task 모두 컵 내부로 블록을 배치한 후 contained-and-settled 판정 통과.
  이것은 실제 grasp 또는 Mimic 성공 episode를 수행했다는 의미가 아니다.
- Isaac headless 환경 생성/카메라 렌더, Real/Sim의 `apply_profile`, Mimic pick/place
  바인딩 검사. 기존 SO-101 USD의 tool0 빈 visual reference 경고는 남지만 실행은 완료했다.
- 실제 로봇/Leader/카메라 하드웨어, 새 source demo로 annotation/datagen/train을 끝까지
  실행하는 검증은 수행하지 않았다. 이 검증에는 새 장면에서 수집한 실제 source가 필요하다.
- 결과: `outputs/workflow_validation/builder_scene1/report.json` 및 두 엔진 preview.
  이 폴더는 실행 결과 검증용이며 Workspace 입력 데이터가 아니다.

## 2026-10-10 이전 Workspace 정리

- 삭제: `outputs/real2sim/environments/workspace_001/v1`, `v2` (80,223,004 bytes),
  `outputs/real2sim/workspace_tasks/workspace_001.json`.
- 제거한 코드: `workspaces/scene_assets.py`, `package.publish`, CLI `publish`,
  Workflow의 Revision 승인/Publish 단계와 옵션, revision 신규 artifact 등록.
  `check_workspace_physics.py`는 가져온 Workspace를 읽기 전용으로 검사한다.
- 유지: `scene1_builder/v1`, Snapshot, HDF5, 학습 데이터/모델, 공통 reset·physics·Mimic,
  제어 articulation. `so101.workspace/1`은 공통 runtime 계약이므로 버전 숫자 자체를 삭제하지 않는다.
- UI의 Workspace 경로를 `scene1_builder/v1`으로 변경하고 Registry에 등록했다.
  새 export는 Workspace 폴더 선택에서 가져온다. 별도 승인/발행 화면은 없다.
- 이전 Workspace Registry 중 자식 데이터가 없는 3개 레코드를 제거했다.
  기존 데이터의 parent인 2개 레코드는 출처 이력으로 유지한다. 원본 HDF5의 metadata는 변경하지 않았다.
  **이전 환경에 종속된 Replay/Datagen은 삭제된 원래 패키지를 복구해야 재현 가능하다.**
- Capture용 `device_profile.json`에 이전 카메라 값/evidence를 보존했다. 카메라 값은
  수치 fitting/provisional 출처이며 실측 보정 완료를 뜻하지 않는다. Real/Sim 연결은
  Builder Workspace profile을 사용하며 Capture용 선택을 덮어쓰지 않는다.
- Builder MuJoCo에 `multiccd`를 적용했다. 컵 원통 바닥/블록의 단일 접촉점 흔들림을
  여러 접촉점으로 줄였다. 성공 threshold나 질량/크기는 변경하지 않았다.
  6개 조합 모두 80/240 step에서 성공, 240 step 속도 약 0.00056037 m/s.
- Isaac 재검증: Builder robot link visual 교체, Real/Sim apply_profile,
  Mimic config 생성, Side/Wrist 640×480 렌더 통과. 기존 tool0 asset 경고는 남아 있다.

사용 순서: UI를 다시 실행 → Workspace Pipeline의 `scene1_builder/v1` 확인 →
Isaac/MuJoCo 선택 → Real/Sim에서 Pipeline Workspace 사용 후 Connect, 또는
Workflow에서 Task를 선택하고 Source부터 실행. UI 실행만으로 장치를 연결하지 않는다.

최종 회귀 검사: UI/Workspace/Builder/Camera/Task/Presentation/Launcher/Snapshot 140 passed,
1 skipped; `test_mujoco_backend.py`는 깨끗한 별도 프로세스에서 10 passed.
통합 프로세스로 두 엔진 테스트를 섞은 최초 시도는 renderer/import 격리로 6 failed/3 errors가
발생했으며, 런타임 격리 규칙을 약화시키지 않고 분리 실행했다. `git diff --check` 통과.
실제 두 엔진의 이미지/수치는 `outputs/workflow_validation/builder_scene1/report.json`에 기록했다.


## Metadata 기반 연결 — 2026-10-10

Canonical scene.yaml의 선택적인 workflow(또는 manifest의 workflow)를 기존
workspaces/builder.py가 읽는다. 두 정의가 충돌하면 거절한다. 기존 scene1은 role 기반
기존 작업을 유지하며, 새로운 환경은 명시적인 작업을 제공한다.

```yaml
workflow:
  schema: so101.scene-workflow/1
  support_surface: bench
  observation_cameras: [front]
  real_camera_bindings: {front: front_camera}
  object_physics:
    apple: {mass_kg: 0.1, dynamic_friction: 0.8}
  tasks:
    - id: fruit_sort
      pick: apple
      place: basket
      instruction: Place the apple in the basket.
      success:
        relation: inside_region_and_settled
        bounds_m: [[-0.08, -0.07, -0.047], [0.08, 0.07, 0.1]]
        max_speed_m_s: 0.03
  provenance:
    tasks: {source: derived, status: UNVALIDATED, evidence: [user_task_brief]}
    reset: {source: estimated, status: UNVALIDATED, evidence: [candidate_hypothesis]}
    success: {source: estimated, status: UNVALIDATED, evidence: [candidate_hypothesis]}
    physics: {source: unknown, status: UNVALIDATED, evidence: []}
```

영역은 target 로컬 좌표다. 물체 bounding box의 모든 꼭짓점이 내부에 있고 속도
조건을 만족해야 한다. 원형 컵에는 기존 contained-and-settled 계약을 사용한다.
reset/physics도 전달할 수 있다. 입력 geometry만으로 작업 의도나 물성을 추론하지 않는다.

실제 변환 순서:
1. 원본 hash, 단위, robot joint/tree/mapping을 검증한다.
2. geometry/camera를 profile.json, 역할/작업/reset/성공 조건을 task.json으로 변환한다.
3. 물성을 physics.json으로 컴파일하고 USD reference wrapper를 만든다.
4. 원본 scene/XML/USD/assets와 provenance를 runtime package에 보존한다.
5. UI가 camera별 Real/Sim view·장치 필드·preview와 task 목록을 구성한다.
6. Isaac은 articulation + USD reference, MuJoCo는 articulation + exported geometry를 쓴다.
7. Source/변환/정책 입력은 실제 camera ID를 사용한다. 새 학습 preset은 데이터 camera
   목록을 따르고 custom config/resume은 계약 일치를 검사한다.

front 하나이면 front_cam 관측과 front.npy episode가 생성된다. 실물 장치 경로는
사용자가 설정한다. 제어 로봇은 SO-101, 작업 recipe는 기존 pick/place이며 임의 로봇이나
새 작업 알고리즘까지 생성하지 않는다. Camera mount는 world/gripper, Capture는 기존
Side/Wrist snapshot 계약이다. 기존 두 카메라 checkpoint는 다른 입력에 자동 연결하지 않는다.

새 synthetic 검증: front 하나, apple 구형 collision, banana 복합 collision, 다른 질량,
사각 basket 성공 영역에서 MuJoCo finite step과 내부 안정화가 통과했다. 새 환경의 Isaac
실제 렌더·실제 학습·실물 grasp는 아직 검증하지 않았다. 기존 scene1 두 엔진 검증과 구분한다.


## 2026-10-10 코드 재검토와 새 package 실험

이전 상태에서는 Adapter만 workflow를 읽었고 Builder 검증/편집 기능은 없었다.
이제 Builder의 core/scene.py가 workflow schema를 검증하고, core/editing.py의
set_workflow가 기존 candidate edit에 연결된다. exporters/package.py는 canonical과
manifest에 같은 workflow를 export하고 SCENE.md에 intent/provenance를 기록한다.
Codex의 구체적인 작성 절차는 Builder docs/AGENTIC_TOOLKIT.md에 있다. 내부 LLM이
사용자 문장에서 task를 생성하는 기능은 없다. 외부 Codex가 brief에서 의도를 해석하고
명시적인 metadata를 작성한다. tasks/reset/success/physics provenance 그룹이 필수다.

실제 새 Builder project → exporter → checksum 검증 → adapter import를 실행했다.
Scene1의 블록/컵 package를 재사용하지 않았다. 구성은 front 하나, sphere apple,
3-part compound banana, 5-part 열린 rectangular basket, bench와 spawn_region이다.
이 Scene은 실제 과일 환경을 재구성한 결과가 아니라 synthetic integration fixture다.
Export 상태는 UNVALIDATED/SYNTHETIC_ADAPTER_SMOKE/PHYSICS_NOT_VALIDATED로 유지한다.

결과는 outputs/workflow_validation/metadata_adapter 아래에 있다:

- paths.json: 실제 새 project/export/workspace 경로.
- mujoco_report.json, mujoco_front.png: 5 seed reset, RGB 320×240, 질량 0.1/0.06 kg,
  collision 1/3/5개, 150 step 안정화 후 region success 통과. Isaac/omni/pxr import 없음.
- isaac_final/physics_report.json, isaac_front.png: actual headless environment,
  RGB와 front_cam 관측, 실제 PhysX 질량, 120 step 안정화 후 region success 통과.
- mimic_report.json: actual Isaac Mimic environment reset, apple/basket object refs,
  tool0 pose, front_cam와 grasp 관측 초기화 통과. Annotation/Datagen 전체 성공은 아님.
- ui_report.json: 실제 package로 offscreen UI의 camera/task/device 필드 구성 통과.
  UI 시작으로 runtime이나 하드웨어가 연결되지 않음.
- annotation_report.json: 실제 replay annotation 경로가 정상 실행되고 grasp/place 없는
  stationary synthetic episode를 거절(accepted=0). 성공 annotation을 만들었다는 뜻이 아님.
- conversion_report.json: 실제 HDF5→LeRobot converter가 12 frame을 처리하고
  observation.images.front_cam을 생성. synthetic_only; Registry 학습 데이터로 등록하지 않음.
- policy_contract_report.json: front 입력 batch, deterministic/flow head optimizer 1 step
  및 [1,4,6] 출력 확인. feature fixture를 사용했으며 Qwen/정책 전체 학습 검증이 아님.
- report.json: 위 결과와 최초 실패/수정 이력, 검증하지 않은 항목을 모은 보고서.

실제 export를 썼더니 기존 단위 fixture가 놓친 중첩 USD RigidBodyAPI 문제가 드러났다.
Workspace wrapper에서 source의 rigid/mass API를 제거하고 runtime owner 하나만 유지했다.
일반 URDF import의 link_local_pose 누락, workflow provenance dict 보호 검사 오류도
수정했다. Off-center K는 Builder와 같은 MuJoCo principalpixel/USD aperture offset으로
전달한다. 새로운 실제 실험에서 두 엔진 camera 렌더가 통과했다.

처음 synthetic 성공 영역 z 하한 -0.035는 바닥에 놓인 sphere의 AABB를 제외했다.
실험의 추정 metadata를 -0.038로 조정해 다시 export/import했다. 성공 판정 알고리즘을
완화하거나 원본 사용자 Scene을 변경하지 않았다. 이는 실측 success 영역이 아니다.

판정: metadata-driven import/두 엔진 환경/UI/변환은 실제 새 package로 확인했다.
새 환경의 실제 Source teleop, 성공 grasp annotation, 성공 Datagen, 새 Qwen/ACT/SmolVLA
전체 학습과 trained-policy task rollout은 미검증이다. 실제 성공 demo와 독립 validation
데이터가 필요하다. MuJoCo Datagen은 object-relative bounded IK이고 Isaac MimicGen과
같은 알고리즘이 아니다. 사진/Scene package만으로 학습 데이터나 학습된 정책이 생기지 않는다.
기존 Snapshot Capture는 Side/Wrist 계약을 유지한다. Arbitrary tasks/robot/camera mounts,
회전 reset 영역과 distortion rendering은 지원 범위를 넓힌 것으로 주장하지 않는다.

이번 변경 최종 검사: SO-101 clean worker/head 37 passed, Workspace/UI/Task/Presentation/
Snapshot 77 passed/1 skipped, Builder 30 passed/1 skipped. 합계 144 passed/2 skipped.
Structural MJCF/USD parity도 별도 pxr 환경에서 PASS. Export 당시 pxr 미노출로 기록된
UNAVAILABLE은 원본 package에 그대로 남기고 별도 parity_report.json에 실제 검사를 기록했다.
기존 scene1의 같은 export 재import는 기존 immutable package를 재사용함을 확인했다.

## Sim 모델 영점 보정 추가 — 2026-10-11

Builder의 선택적 `robot_model.simulation_joint_offsets_rad`를 Adapter가 소비한다.
`joint_names` 순서의 rad 배열이며 estimated/optimized provenance, evidence가 필수이고
각 값은 ±.35rad 이내다. 원본 observed q와 hardware mapper/calibration은 변경하지 않는다.
MuJoCo는 joint 앞 body origin, Isaac은 별도 `assets/SO101/usd/so101_sim.usda` zero-frame
override와 파생 `urdf/so101_sim.urdf` IK를 사용한다. 원본 USD/URDF 및 contact-pad source
hash를 보존한다. offsets가 없는 Workspace는 기존 모델 경로를 그대로 사용한다.

이 값은 영상 fitting의 **미검증 Sim 가설**이다. 실제 조인트/접촉 calibration으로 인증하지 않는다.
새 review Scene1 Workspace는 `outputs/real2sim/environments/scene1_builder/v5`이며 결과와
UI 사용법은 `outputs/workflow_validation/builder_v2_scene_refinement/REPORT.md`를 참조한다.
v4는 중간 source hash 충돌을 발견한 실패 버전이므로 선택하지 않는다.
