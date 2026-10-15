# SO-101 / Real2Sim 개발 인수인계

## 최신 — MiniVLA Flow-DiT 교체 (2026-10-15)

`docs/minivla/DESIGN.md`, `README.md`, `VALIDATION.md` 및
`docs/real2sim/MINIVLA_INTEGRATION.md` 우선. 기존 패키지를 직접 교체했다.
Qwen NF4 block15 output/letterbox256square/side64image tokens, adapter256, state+40actions,
AdaLN-Zero FlowDiT4, Euler8/replan4. ActionHead7,319,558params. 기본config/default.json.
Deterministic/head선택/UV head/옛4presets 제거. 공용 grounding data/ACT/SmolVLA 유지.
Checkpoint schema so101.minivla.flow-dit/1, 구형 MiniVLA 가중치 거절(원본 보존).
실제 frozen/QLoRA GPU F/B와8~15층 Action gradient 통과. 짧은2step 합성 모델은 작업용 아님.
검증 결과/실패/VRAM/남은 성공률 검증은 outputs/workflow_validation/minivla_flow_dit 참조.


## 최신 — v8 충돌 / 미끄러짐 (2026-10-14)

`outputs/workflow_validation/scene2_v8_contact_physics/REPORT.md` 참조.
정정: v8 finger collision은 USD instance proxy 내부에 존재한다. 앞선 누락 진단 오류.
복원 helper는 instance proxy까지 검사해 중복 생성하지 않도록 수정했다.
후속 Isaac 실제 렌더에서 왼쪽 틈/오른쪽 겹침 재현.
Builder Isaac dt1/240, decimation/render8로 변경해30Hz 제어 유지.
동일 합성 자세의 fixed-pad 평면 최소간격5.51→0.54mm, 실제 grasp 완전 해결 아님.
후속 근거/오진 정정은 같은 폴더 FOLLOWUP.md 우선.
원본 USD/Workspace 수정 없음. Builder isaac_matched MuJoCo noslip_iterations=10.
비관통 합성 유지6/6, 열기 bilateral 해제6/6. 실제 pick/lift/Sim2Real 검증 아님.
Isaac GPU/CPU AppLauncher는 CUDA 초기화 오류로 동적 검사 미완료.
UI 동일v8, Precision ON/Scene보정제외 OFF. 관절 보정/패드 두께/마찰계수 변경 없음.


## 최신 — Scene2 새 접촉 캡처 v8 (2026-10-14)

`outputs/workflow_validation/scene2_final_contact_refinement/REPORT.md` 우선.
새9Pair(3–9 접촉; 사용자 블록 윗면 중앙/컵rim 확인), static001 영상14개와8/9높이 제약.
Builder scene2-final-contact-review-20261014-932c8f2e/final_contact_review export/import→scene2_builder/v8.
평면 tilt2.604°+물체위치 보정; 기존 v7 signed Sim residual/Precision/카메라/원본 보존.
사용자가 base/table 접촉·수평강제를 해제했다. 물리측량된 기울기라고 해석하지 않는다.
contact-height fitting RMS3.95→1.50mm, Side RGB3.2%개선/Wrist0.14%악화, OPEN/UNVALIDATED.
UI v8, Precision ON, Scene보정제외 OFF, 직접 재연결. 실제 grasp/동적응답/Sim2Real 미해결.


## 최신 진행 중 — v7 grasp 대응 진단 (2026-10-14)

`outputs/workflow_validation/scene2_v7_grasp_alignment/REPORT.md` 참조. 실행 개선본은 아직 없다.
새 v7 두 Pair의 실물 grip0.431° 대비 Sim5.859/-9.913°, 동일모델 FK 높이차+6.927/+6.382mm.
MuJoCo full-dynamic18조건과 Isaac GPU2합성 fixture를 실행했다. 초기 관통이 있어 pick 성공 아님.
팔 q 강제 고정한 이전 probe의 낙하는 실제 Runtime 실패로 인용하지 않는다.
그리퍼 offset6/10°는 깊이/접촉 형상과 결합되어 미채택; 새 export/Workspace/Profile 변경 없음.
사용자가 실제 pad는 finger flat surface 크기의 약0.5mm 스티커라고 확인했다(추정 두께).
0.5mm MuJoCo 메모리 실험6조건 후에도 깊이 불일치가 남는다. user_pad_evidence.json 참조.
다음은 flat surface 충돌형상/부착 위치 확인, 비관통 접근/lift 및 Isaac bilateral contact 검사다.

## 최신 UI — Real/Sim Scene·보정 설정 통합 (2026-10-14)

Real/Sim 오른쪽 맨 위 Scene/관절 보정 영역에서 기존 Registry의 Workspace 선택/Import,
저장 Precision 사용 및 identity-Sim 제외 옵션을 설정한다. Workflow와 Workspace 경로를 공유한다.
Capture와 보정 체크박스는 동일 위젯이며 기존 ui_settings.json에 선택을 저장한다.
Pair 저장 설정도 Real/Sim에 표시한다. 설정 변경/탭 전환에는 worker/device 시작이 없다.
현재 worker의 workspace/precision_run_configuration과 다음 Connect 선택을 구분해 표시한다.
현재 연결의 camera tiles는 다음 Scene 선택만으로 바뀌지 않는다. 직접 재연결한다.
사용법: `docs/real2sim/BUILDER_WORKSPACE.md` 맨 위, 검사: `tests/real2sim/test_live_scene_selection.py`,
실제 v7 UI payload/화면: `outputs/workflow_validation/real_sim_scene_controls/REPORT.md`.

## 최신 — Scene2 v7 접촉 보정 실행 연결 (2026-10-14)

UI 검토용은 `scene2_builder/v7`. Builder `scene2-v6-corner-contact-review-20261014-56e6a8bf`
`corner_joint_review`를 실제 재평가·Blender·export/import했고 두 엔진8pose FK/6task smoke 완료.
기존 offsets에 precision_mapping_sha256 binding을 추가해 정확한 Precision 뒤의 Sim residual만 허용한다.
역사적 unbound 중첩 차단은 유지한다. active Profile/raw q/motor target/URDF는 변경하지 않았다.
UI 재시작→v7→저장된 관절 보정 ON→URDF 원형 검사(Scene 보정 제외) OFF→직접 재연결.
model residual을 제외하는 checkbox가 켜지면 v7 보정도 보이지 않는다.
접촉 proxy RMS12.49→3.07mm; RGB MAE는 소폭 악화, 독립 검증/실물 grasp 미완료,OPEN/UNVALIDATED.
보고서/코드 계약/실패/테스트: `outputs/workflow_validation/scene2_v7_contact_application/REPORT.md`.
아래 중첩guard 때문에 export 못했다는 v6 진단은 이 후속 계약 수정 이전의 기록이다.
Workflow Source의 기존 identity-Sim 선택은 유지하여 전 단계가 동일 잔차를 쓰는 E2E는 미검증이다.

## 최신 진단 — Scene2 v6 네 꼭짓점 (2026-10-14)

새 snapshot_20261014_110211_377fb860의8자세는 전부scene2_builder/v6.
사용자가 모두 같은 고정 finger 끝 접촉이라고 확인했다.8장 전부 measured-q 분석/import.
COPIED Builder `scene2-v6-corner-contact-review-20261014-56e6a8bf`: geometry-only REJECTED,
corner_joint_review OPEN draft. 접촉 proxy RMS12.49→3.07mm의 추정 zero+marker 제안이다.
현재 Builder/Runtime가 Precision 관측에 역사적 simulation offsets 중첩을 거절하여 export/import하지 않았다.
Guard/서명/Reference를 우회하지 않았다. transient32렌더는 실제 Candidate acceptance가 아니다.
UI용은 여전히scene2_builder/v6. Calibration/active/원본/URDF를 변경하지 않았다.
계수 일부±2° prior 경계, contact point·pose 결합 및 독립 검증 부족이 남는다.
보고서/제안/재현: `outputs/workflow_validation/scene2_v6_corner_refinement/REPORT.md`.
이전 snapshot_20261014_103225_018883f6의8장은scene1_builder/v1이므로 혼동하지 않는다.

## 최신 — Scene2 베이스·상판 수평 접촉 (2026-10-14)

작업용 검토 Workspace는 `scene2_builder/v6`, Builder 같은 project의 `base_table_contact`다.
v5에서 base CAD 최저점보다 상판이15.898mm 높고0.27736° 기울어진 문제를 수정했다.
상판 normal=robot +Z, 최저 base mesh와 상면 높이 일치. 물체들은 지지면을 따라 이동했다.
로봇/base/q/카메라/보정/물성/6개 workflow는 그대로다. v5/원본은 보존한다.
이후 정합에서도 베이스 매립/상판 기울기로 영상 오차를 숨기지 않는다.
Wrist RGB는 악화했고 Real fingertip 접촉·grasp/slip 해결은 미검증이다. OPEN/UNVALIDATED 유지.
검사/비교/UI: `outputs/workflow_validation/scene2_base_table_contact/REPORT.md`.

## 현재 — Scene2 Workflow 연결 (2026-10-14)

사용자 지정대로 단일 블록→컵6조합을 Builder `single_block_workflow`에 추가했다.
Project `scene2-visual-physics-refinement-20261013-bbb16a35`, 실행 Workspace `scene2_builder/v5`.
Scene1 성공/랜덤 Reset을 재사용한다. v4 geometry_review는 보존한다.
상세 검사와 사용법: `outputs/workflow_validation/scene2_workflow_binding/REPORT.md`.
Source 사전검사/두 엔진 판정·Mimic 연결 완료, 실제 데모→학습 E2E와 그리퍼 slip 해결은 미완료다.

## 현재 — Scene2 외형/물리 후속 (2026-10-13)

최신 검토 후보는 sibling Builder `scene2-visual-physics-refinement-20261013-bbb16a35`의
`mounted_clip_detail_review`, 실행 Workspace는 `scene2_builder/v4`다. 기존 v2/v3는 보존한다.
Scene1 camera mesh의 장착/클립을 수정하고 roller pose와 cup ink pose를 보정했다.
배경색/조명/광학 카메라/q/보정/실측 물체 치수는 유지했다.
명시적 canonical mass가 있는 geometry_review는 기존 physics로 dynamic block와 열린 고정 cup을 만든다.
픽앤플레이스 task/success를 자동 생성하지 않는다. Source/Datagen은 별도 workflow metadata가 필요하다.
상세 증거·불확실성·UI 절차: `outputs/workflow_validation/scene2_visual_physics_refinement/REPORT.md`.
아래 이전 Scene1 기록은 역사적 맥락이다. 엔진 smoke와 실물 grasp 성공을 혼동하지 않는다.

## 최신 — 2026-10-13 / Teleop 모터 범위 초과 처리

사용자 요청으로 Runtime.command_target의 기존 clip에 Follower 모터 명령 가능 범위를 포함했다.
PrecisionMapper.command_limits는 현재 URDF/Sim과 모터 범위의 정수 encoder 교집합을 계산한다.
정상적인 목표 초과는 경계로 제한하고 계속 추종한다. 저장 profile/active/URDF는 그대로다.
모호한 branch/비정상 값/identity/통신 오류는 계속 거절한다. Policy inverse는 strict 유지.
이전 “motor range 부족이면 Teleop 전체 거절” 설명은 현재 정책이 아니다.
실제 로봇 자동 테스트 없음. TELEOP_LATENCY_DIAGNOSIS.md의 후속 결과 참조.


## 후속 확인 — 2026-10-13 / Teleop 지연 및 Start 오류

사용자 재현 로그에서 목표 전송 간격은 약33ms, send0.10–0.15ms였다.
모든 모터 Goal_Velocity338–347이 Connect/Start 전후 유지되었고 사용자의 모터 전원
재시작 후0으로 초기화됐다. 이전 속도 제한 잔류가 지연의 유력 원인이다.
이후 shoulder_lift가 기존 모터 보정 범위를 넘는 목표를 받아 Start에서 거절됐다.
위 Teleop 목표 clip 수정 후 사용자가 정상 동작한다고 확인했다. 외부 센서로
지연/정확도를 실측한 결과는 아니다. 재보정/보정 파일 변경은 하지 않았다.
진단은 기존 runtime 로그/status에 남기며 모터 속도/Phase를 임의로 쓰지 않는다.
관련 최종 검사171 PASS,1 SKIP(Blender 선택 검사); 실제 Profile 1001개 목표 offline 변환 통과.
상세 로컬 기록: `outputs/workflow_validation/unified_joint_control/TELEOP_LATENCY_DIAGNOSIS.md`.
outputs는 Git 제외이므로 코드 사용 계약은 `docs/real2sim/JOINT_CONTROL.md`를 참조한다.

## 최신 — 2026-10-12 / Limit-only Profile 자동 호환

`outputs/workflow_validation/joint_limit_extension/COMPATIBILITY_REPORT.md`와
`docs/real2sim/PRECISION_CALIBRATION.md`를 우선 읽는다. 이번 사용자 지시로 이전 승인형
Limit Migration UI/worker/ProfileStore.rebind_limits를 제거했다. 기존 active Profile을 재저장하거나
재승인하지 않는다. `precision.profile_for_model`은 exact hash에 맞는 Git/보존 URDF를 읽어
lower/upper 외 모델 전체가 동일함을 확인하고 실행 복사본에만 선택 URDF의 제한을 적용한다.
원본 Profile/active/Sign/Scale/Zero/References를 변경하지 않는다. model evidence 부재,
Axis/Origin/geometry/topology 변경 및 모터 identity 변경은 계속 거절한다. 장치 제어는 기존
SDK/Precision strict inverse를 사용한다. 새 컨트롤러/Guard/Migration 설정은 만들지 않는다.
Calibration의 실제 보정 Save/사용자 승인 Activate/Restore는 그대로다.

Pan/Lift/Elbow/Flex ±5°는 이미 적용되어 추가 확장하지 않았다. Roll/Gripper도 그대로다.
원본 USD/Workspace는 보존. 실행 USD는 `assets/SO101/usd/so101_joint_limits.usda`, 새 Workspace는
`scene1_builder/v10`. Real-only 기본은 root URDF, Workspace는 자신의 URDF/USD로 재현한다.
이전 검증 보고서의 Migration 지침은 현재 정책이 아니다. 사용자가 기존 worker를 종료하고
UI 재시작 → Connect하면 된다. 자동 장치 구동/설정 설치/사용자 데이터 수정은 하지 않는다.


## 최신 — 2026-10-12 / 관절 제어 단순화 (현재 정책)

사용자 요청으로 별도 Controller/MotionGuard/자동 복귀 및 전용 UI/IPC를 제거했다.
`docs/real2sim/JOINT_CONTROL.md`와
`outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md`를 우선 읽는다.
기존 Runtime.command_target이 Leader observed_positions → URDF/선택 엔진 clip → Follower
Precision inverse → 실제 SDK 정수 변환 → 같은 q를 Real/Sim에 전달한다. Motor range 부족은
명령 전 거절한다. Start 5°/30°/s/10° Guard나 RETURNING_TO_LIMIT 경로가 없다.
Connect는 읽기 전용이며 유효한 범위 밖 Follower 피드백도 허용한다. Start는 명시 사용자 요청이다.
Precision/보정 승인·복구/모터 identity·유효성/통신/heartbeat/stall/Sim limit 검사 유지.
새 Controller/launcher/config를 만들지 않았다. 관측 계산은 기존 so101_joint_mapping에 통합했다.
새 Snapshot에서 controller metadata를 생성하지 않으며 역사적 applied q reader만 유지한다.
SDK 정수 conversion은 strict inverse의 1-count 부동소수점 문제와 Real/Sim target 동일성을
위해 Hardware에 한 개 유지한다. 이전 guard 설정을 다시 복원하지 않는다.
URDF는 이미 ±5° 확장되어 추가 수정하지 않았다. v10 및 기존 사용자 Profile/Workspace/Snapshot/
HDF5/checkpoint를 보존한다. 실제 보정 변경의 activation만 사용자 승인이다. Builder/GLB 변경 없음.


## 현재 기능 — 2026-10-12 / Joint Precision Calibration

상세 계약/사용법은 `docs/real2sim/PRECISION_CALIBRATION.md`, 검사/변경 이력은
`outputs/workflow_validation/precision_calibration/REPORT.md`와 `docs/real2sim/CHANGELOG_AI.md`를 읽는다.

- UI 네 번째 탭 → 기존 RuntimeLauncher/IPC → `real2sim_sim.py --backend calibration`
  → `precision.Calibration` → `hardware.CalibrationDevice`. 새 launcher/Builder 변경 없음.
- 명시적 read-only Connect와 승인한 선택 joint torque release만 지원한다.
  Goal/Homing/Range/Drive register 쓰기, 자동 장치 연결/로봇 이동은 없다.
- 새 기록은 raw + 손으로 맞춘 URDF 기준각으로 통일한다. split=train/reference_kind=angle은
  worker가 고정하며 validation/open/closed/opening 입력과 예전 precision.validate 실행 명령은 제거했다.
  fit → review(손으로 움직이며 확인) → save → 승인 activate. 검토는 REVIEWED_MANUAL,
  independent_validation=False/physical_verified=False이며 실물 정확도 인증이 아니다.
- `validate_joint`와 legacy_endpoints는 기존 저장 Profile을 읽고 증거를 재검사하는 데 사용한다.
  이 reader를 삭제하면 이전 저장값의 로드/복구가 깨진다. 새 기록 UI/worker 경로로 노출하지 않는다.
- 0°/90°는 fit 기준점이다. raw_domain은 증거이며 사용 한계가 아니다.
  UI/Calibration/Runtime/Compare는 joint_raw_coordinate/joint_angle로 URDF 범위 전체를 계산한다.
  goal은 기존 calibrated motor range 및 유일한 encoder branch도 확인한다.
  Precision gripper는 기록 구간으로 clamp하지 않는다. legacy gripper 제한은 유지한다.
- Real-only Capture의 STOPPED/passive 관측은 calibrated_to_observation으로 범위 밖 측정도 보존한다.
  precision_observation_only/joint_limit_warnings를 기록하고 사진/측정값을 자르지 않는다.
  motion/physics는 calibrated_to_urdf/inverse의 기존 제한을 유지한다. encoder ambiguity는 거절한다.
- raw Present_Position은 firmware Homing 이후 값이다. Homing/LeRobot normalization을 이중 적용하지 않는다.
  live read는 같은 raw packet의 SDK normalization, record는 별도 read/안정성 검사를 유지한다.
- 보정 전은 현재 저장/적용 값이다. 새 Candidate가 없으면 전/후/현재가 같다.
  0.5°는 기록 재현 residual이며 두 점 fit에서는 raw 간격의 정당성을 검증하지 못한다.
- 선택 자세 삭제는 원본 journal을 유지하고 exclusions에 ID/관절/이유/시각을 기록한다.
  해당 Candidate만 초기화하고 다른 joint/기존 active를 유지한다.
- save는 해당 role의 모든 확인된 Candidate와 기존 active를 합친다. partial activate는 다른 joint를 유지한다.
  동일 저장/적용은 파일/pointer 재사용, 실제 변경은 새 immutable 버전이다.
  같은 device/model 재연결은 active reference를 새 세션 표에 복사하고 loaded_profile_id로 출처를 남긴다.
- ProfileStore는 `<UI workspace>/joint_calibration/{leader,follower}`, Fake는 `fake/` namespace다.
  사용자 최신 active는 store를 직접 읽는다. 2026-10-12 현재 Follower profile_1a377745dd6441f0bacee6ff3f0a5690,
  Leader profile_a4ea4642b1da4587943331fb8083c322, 양쪽 6관절이다.
  이전 Follower 복구 Profile: profile_0af8c777427a4c999b47ca67e8e6298a.
  source journal: references_b2834b1d41884b80b281de6a09a24411.json.
  기존 4관절 보존, wrist_flex/gripper만 사용자 원본으로 복구했고 이후 사용자가 보정을 추가했다.
- Real/Sim/Capture는 저장 active가 있으면 다음 Connect에 Precision/원본 URDF 설정을 기본 선택한다.
  Teleop은 두 장치의 같은 보정 joint set과
  URDF/모터 범위 preflight가 필요하다. nonzero Scene Sim offsets는 충돌 검사한다.
  명시적 identity-Sim session은 immutable Scene 수정 없이 MuJoCo session/Isaac 원본 USD를 사용한다.
  hardware.Leader/Follower와 standalone SO101Leader는 Connect 시 승인 active를 로드/identity 검증한다.
  Source는 동일 mapper로 model rad를 기록하며 collection_coordinates로 hardware_mapping/URDF frame을 남긴다.
  원본 LeRobot JSON/register/전역 DEFAULT_MAPPER를 변경하지 않는다.
  새 Source와 후속 Annotation/Datagen/Convert/Train/Sim Eval은 기록된 frame을 이어받고
  기존 데이터는 당시 frame을 유지한다. real_eval은 현재 Follower mapping을 경계에서 사용한다.
  이미 URDF rad인 값에 재보정/자동 데이터 migration은 하지 않는다.
  Calibration이 포트를 소유하면 UI Source/real_eval을 차단한다.
- 이전 코드가 이미 worker에 로드됐으면 사용자가 보정 작업 종료 → UI 재시작 → 명시 Connect한다.
  저장 보정은 재기록/재저장하지 않아도 된다. agent가 사용자 worker를 자동 재시작하지 않는다.
- Fake worker+Qt/수학/실제 MuJoCo FK preview 검사가 있다. 실제 serial/torque/물리각/Teleop/
  Isaac 새 session 경로는 HARDWARE_NOT_TESTED다. 사용자 Profile/Scene/Snapshot/HDF5를 보호한다.


## 최신 — 2026-10-11 / v7 새 2 Pair → v8 Wrist K 보정

`outputs/workflow_validation/v7_wrist_camera_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_222612_a9610eee` 2자세(v7) train, 이전6자세(v6) 회귀 비교.
새 Builder project `scene1-v7-wrist-camera-refinement-20261011-58823cd8`,
선택 `wrist_view_review`: Wrist K만 fx/fy[560.8899,558.3139], cx/cy[326.8775,262.6275].
원본 q/geometry/Workspace축/robot/base/joint zero/mount/Side/task/물성은 그대로다.
새 paper IoU .8290→.9459, .8207→.8938; 이전6도 모두 개선.
영상전체 MAE 및 일부 center 회귀로 공식 nonregression FAIL, OPEN/미 promote/UNVALIDATED.
그리퍼 벌어짐은 여전히 다르다. 영점 probe는 닫힌 자세 개선/열린 자세 악화로 적용하지 않았다.
pose+focal 대안은 실제 camera body 내부 가림으로 제외. 모든 대안은 REJECTED로 닫았다.
review XML/USD export와 sibling 소비 Workspace `outputs/real2sim/environments/scene1_builder/v8` 생성.
MuJoCo 16RGB/raw-q/FK/camera/reset/200step, Blender Wrist3렌더, export USD load/구조 확인.
Isaac v8 runtime/contact/실물 calibration/독립 blind validation 미검증. 장치 접근 없음.
새 실험의 disposable Blender payload/mask 약1.11GB만 제거; source/Scene/PNG/metrics 유지.
기존 v7/원본/canonical/evidence 보존, production/dependency 변경 없음.
UI 선택은 자동 변경하지 않았다. 사용자가 v8 선택 후 직접 재연결하고 Pair version/hash 확인.
상세 실패 로그·보존·실험 recipes/·다음 finger mapping 검증은 위 REPORT.md 참조.


## 최신 — 2026-10-11 / v6 새 6 Pair → v7 Wrist 작업공간 검토 후보

`outputs/workflow_validation/v6_wrist_workspace_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_145450_d2cf22b8`는 전부 v6이며 6자세/12이미지 모두 사용했다.
새 Builder `scene1-v6-wrist-workspace-refinement-20261011-b3a03339`의
`paper_pose_guarded`에서 종이/테두리만 XY[-3.380,+1.292]mm, yaw+1.448° 수정했다.
원본 q, robot/base/zero, camera K/mount, 컵·블록·table의 world pose/치수/물성 유지.
Wrist 종이-only 진단 IoU는 6자세 모두 개선했으나 Side 일부 mask와 RGB 회귀가 남아
공식 nonregression FAIL, **OPEN / 미 promote / UNVALIDATED**다.
자세5/6은 fitting 제외 비교지만 같은 session이며 독립 validation 인증이 아니다.
`paper_pose_wrist_priority`는 Side 회귀로 REJECTED. camera-only 대안은 적용하지 않았다.
review XML/USD export와 `outputs/real2sim/environments/scene1_builder/v7` 생성.
MuJoCo 12RGB/raw-q/EEF/camera/reset/200step, Blender 두 자세4렌더 완료.
v7 Isaac runtime/실물 contact/독립 session validation 미검증. actual hardware 접근 없음.
Adapter의 Workspace 축은 수정한 support yaw를 따르므로 runtime-local 좌표는 달라지지만
나머지 물체 world pose는 동일하다. Source Reset 영역은 새 종이를 따라 이동/회전한다.
원본/canonical/evidence/기존 v6/export 보존, production/dependency 변경 없음.
이번 실험의 disposable Blender payload/mask 약1.39GB만 정리했다. recipes/ 재현 가능.
UI는 자동 변경하지 않았다. 사용자가 v7 선택 후 직접 Disconnect→Connect하고
새 Pair version/hash를 확인한다. 상세 metrics/보존/다음 단계는 위 REPORT.md 참조.


## 최신 보정 — 2026-10-11 / 실제 v5 새 4 Pair → v6 검토 후보

`outputs/workflow_validation/v5_new_capture_refinement/REPORT.md`를 우선 읽는다.
새 `snapshot_20261011_140108_47734fec` 4자세/8이미지는 실제 v5이며 전부 사용했다.
raw q 보존. tracking 차이의 모델 FK 거리 5.42–9.92mm를 scene 오차와 분리했다.
새 Builder `scene1-v5-cup-approach-refinement-20261011-56afacbf`의
`approach_contact_review`에서 작은 Sim zero/base만 보정, 카메라/컵/블록 world geometry 잠금.
추정 컵 rim52.94→45.50px, 이전 추정 landmark 평균 개선이나 개별2px 증가 허용,
Wrist 전체 MAE는 악화했다. **미 promote / UNVALIDATED / NEEDS_MORE_DATA**다.
review XML/USD export와 `outputs/real2sim/environments/scene1_builder/v6` 생성.
MuJoCo 8영상/FK/camera/reset/200step 실제 검사와 Blender 두 자세 렌더 완료,
v6 Isaac runtime/실물 grasp/place/독립 validation 미검증. v5와 원본 모두 보존.
UI/장치 연결은 자동 변경하지 않았다. 다음은 블록 접근·컵 자세의 사용자 정지 Pair와
독립 session 검증이며, Teleop 중 real/Sim plant 차이를 임의 영점에 흡수하지 않는다.

## 최신 읽기 전용 진단 — 2026-10-11

새 7 Pair(`snapshot_20261011_121308_65bb7b1f`)의 저장된 Runtime은 모두 v1이며
v5가 아니다. hash/profile/14 camera transforms로 확인했다. 모든7자세를 같은 measured q로
오프라인 재렌더했지만 calibration 식별/새 독립 holdout/real object annotation 부족으로
NEEDS_MORE_DATA다. 새 scene/export/production 변경 없음. v5는 기존 검토 후보로 보존한다.
`outputs/workflow_validation/v5_root_cause_20261011/REPORT.md`와
`SCENE_REFINEMENT_HANDOFF.md` 최상단을 우선 읽는다. UI Workspace 선택은 Connect 시 적용되므로
v5 선택 후 사용자 재연결과 새 Pair identity 검사가 필요하다. 로봇을 자동 구동하지 않는다.

## 최신 재개 상태 — 2026-10-11

이전 v3 상태보다 이 문단을 우선한다. Builder V2 우선순위 구현 후 새 project에서 v2 컵 배치와
모든12자세/24이미지로 Sim 영점/base/Wrist/블록을 추정 보정했다. 최신 실행용 검토 Workspace는
`outputs/real2sim/environments/scene1_builder/v5`. original q/model/사용자 데이터 보존.
실제 두 엔진 raw-q/FK/RGB smoke 통과, 물리/독립 validation 미완료, 후보 미 promote.
`outputs/workflow_validation/builder_v2_scene_refinement/REPORT.md`와
sibling `docs/V2_IMPLEMENTATION_REPORT.md`, `SCENE_REFINEMENT_HANDOFF.md` 최상단을 먼저 읽는다.
v4는 USD source hash 충돌을 발견한 중간 실패 버전이므로 선택하지 않는다.


기준일 2026-10-10. 실제 source와 결과 JSON을 확인하여 작성. 이 문서는 코드 변경이 아니다.
상대 경로는 이 프로젝트 기준. Builder 인수인계는 `../real2sim-builder/CODEX_HANDOFF.md`.
기존 상세 설계 원칙은 `docs/AGENTS.md`; 현재 root AGENTS.md를 우선하여 작업한다.

## 목적과 실행 구조

SO-101은 실제 Leader/Follower·카메라 제어, 실측 Snapshot 수집, Scene 소비 및 학습 Workflow 프로젝트다.
Builder는 독립 offline Scene reconstruction toolkit이며 실제 장치를 제어하지 않는다.
외부 Codex가 이미지 해석·가설 선택을 담당한다. Builder run은 deterministic 도구이지 내부 LLM이 아니다.

실제 진입점 `scripts/real2sim/real2sim_ui.py` → `real2sim.ui.Window`.
Capture/Real-Sim → `real2sim.launcher.RuntimeLauncher` → `scripts/real2sim/real2sim_sim.py`.
Isaac은 `real2sim.runtime.Runtime`, MuJoCo 구현은 `simulation/`에 있다.
Pipeline → `workflow.ui.WorkflowPanel` → `workflow.operations.Operations`.
공통 CLI `scripts/real2sim/workflow.py`; MuJoCo stage는 `workflow/mujoco.py`.
Isaac source/replay는 workspace.py와 scripts/teleoperation, annotation/datagen은 scripts/mimic,
변환은 scripts/tools/convert_isaac2lerobot.py. 세부 호출은 `docs/real2sim/PROJECT_MAP.md`와 DATA_FLOW.md 참조.

## Builder → Workspace 변환

Builder canonical `real2sim.scene/1` + 선택적 `so101.scene-workflow/1`
→ export scene.yaml/XML/USD/scene_manifest.yaml/SCENE.md
→ `real2sim/workspaces/builder.py: inspect_export, profile_and_task, build_usd_assets, import_scene`
→ immutable `so101.workspace/1`: manifest.json/profile.json/task.json/physics.json/provenance.json,
builder 사본, SO101 제어 assets, USD wrappers
→ package.load checksum 검사 → Real/Sim 또는 Operations.

Geometry/Camera K·pose는 profile, 역할·pick/place·reset·success는 task, 물성은 physics로 연결된다.
두 엔진은 기존 SO101 articulation을 사용하고 Builder reference robot을 두 번째 고정 로봇으로 넣지 않는다.
Isaac은 USD visual reference, MuJoCo는 exported geometry + 기존 actuator/contact를 사용한다.
동일 recipe를 쓰지만 renderer/solver 차이로 영상·동역학 동일성을 보장하지 않는다.
상세 schema·예제·지원 범위는 `docs/real2sim/BUILDER_WORKSPACE.md`.

## 기능과 제한

Metadata camera ID 기반 UI/view·관측·변환·정책 입력, object collision/physics/reset/성공 영역,
Source/Replay/Annotation/Datagen/Convert/Train/Eval 경로가 있다.
범용성은 SO101 + 기존 pick/place recipe 범위다. 임의 로봇/임의 task 알고리즘 자동 생성은 없다.
MuJoCo Datagen bounded IK는 Isaac MimicGen과 동일 알고리즘이 아니다.
MiniVLA 구현은 `source/soarm101_lab/soarm101_lab/learning/minivla/`: 단일 Flow-DiT head(위 최신 문서 참조).
Qwen 이미지 진단은 action policy와 다르다. 전체 새 Scene의 성공 grasp→annotation→datagen→훈련→rollout은 미검증.
Camera mount world/gripper 지원; Snapshot producer는 Side/Wrist 계약이다.
새 camera 입력을 기존 checkpoint에 자동 치환하지 않는다.

## 데이터·사용법

실측: `outputs/real2sim/snapshots/snapshot_<시각>_<id>/`.
Start→Disconnect 한 세션; 공통 robot_model/calibration을 한 번 저장하고 snapshots/<shot>에
side/wrist PNG, state.json, metadata.json, COMPLETE 저장. schema `real2sim.snapshot/1`.
이전 snapshot_dataset_* 원본도 보호한다. COMPLETE와 manifest index/hash를 확인한다.
단위 m/rad, named joint order, measured/command 분리, host timestamp는 exposure 시간이 아니다.
Mapping legacy_unverified를 물리적으로 확인하지 않고 calibrated로 바꾸지 않는다.
Capture Start는 실제 장치만 연결; Teleop은 별도 Start Teleop. Real/Sim은 backend 선택 후 Connect.
오른쪽 연결 설정, 아래 로그. UI 시작과 탭 전환은 자동 연결을 하지 않아야 한다.
사용법·계약: `docs/real2sim/CAPTURE_DATASET.md`.
해당 문서의 “downstream adapter 별도/미지원” 문장은 오래된 상태이며 현재 BUILDER_WORKSPACE.md를 따른다.

현재 실제 Scene workspace: `outputs/real2sim/environments/scene1_builder/v1`.
Builder 제작 프로젝트: `../real2sim-builder/projects/so101-scene1-snapshot-reconstruction-2202744f`.
사진/설명: `../real2sim-builder/inputs/scene1/`.
UI Pipeline에서 workspace 선택; Real/Sim에서 Pipeline Workspace 사용을 켜고 backend 선택.
새 export 폴더를 UI에서 선택하거나:

```bash
python scripts/real2sim/workflow.py import-builder /absolute/export --workspace-id scene_new --version 1
```

새 package는 checksum 고정. 변경은 Builder candidate/new export/new workspace version으로 한다.
Task 선택 후 Source→Annotation→Datagen→Convert→Train/Eval을 각각 실행한다.
Snapshot은 실측 scene fitting 자료이며 학습 Source HDF5와 다르다.
Workflow registry `outputs/workflow/artifacts/*.json`은 데이터 자체가 아니라 identity/parent/path 기록이다.
runs/는 request/state/log/presentation IPC 및 stage 결과. real2sim/ipc는 UI-worker 명령·응답이다.
workflow_validation은 개발 실행의 증거이며 runtime 입력과 혼동하지 않는다.
기존 Workspace v1/v2 producer 삭제와 이력 복구 제한은 BUILDER_WORKSPACE.md 참조.

## 검증 증거와 최근 수정

`outputs/workflow_validation/builder_scene1/report.json`: 실제 scene1 두 엔진/카메라/reset·success smoke.
`outputs/workflow_validation/metadata_adapter/paths.json` 및 report.json: 별도 synthetic 과일 scene.
workspace `.../metadata_adapter/workspaces/metadata_fruit/v2`는 실제 과일 reconstruction이 아니다.
front 1개, sphere apple/compound banana/open basket; MuJoCo·Isaac 실제 환경 렌더와 reset/안정화,
Mimic binding, synthetic HDF5→LeRobot, head forward 검증. 물체를 놓아 성공 영역 검사한 것이지 grasp 성공이 아니다.
기록된 회귀 144 passed/2 skipped. 결과 report에 첫 실패·수정 경위를 보존했다.
실제 Leader/USB/카메라 수집과 새 task 훈련/성공 rollout은 해당 회귀에서 NOT_TESTED.
사용자는 Snapshot을 직접 수집했지만 자동검사에서 하드웨어 통합을 인증한 것은 아니다.

최근 사용자 Source 오류: EGL developer viewer를 먼저 켜고 lazy Renderer 생성 시 EGL_NOT_INITIALIZED.
`simulation/mujoco_backend.py: PolicyEnvironment`에서 RGB context를 viewer보다 먼저 생성,
`workflow/mujoco.py: source`에서 Leader.connect 전 observe preflight 추가.
RTX4060 EGL + developer + device import 60 RGB/physics step(320×240) 실제 통과,
backend/minivla 관련 30 tests 통과. 실제 로봇 Source 수집은 수행하지 않았다.
다른 mode/renderer 비교는 fresh process. OSMesa 라이브러리가 없을 수 있으며 software-egl 선택 가능.

## 이어갈 작업

후속 재개 최신: `scene1_builder/v3`는 표면 색상 후보를 promote/export/import한 버전이다.
holdout MAE0.13643→0.12364, 4장 모두 nonregression PASS. Geometry/contact calibration은
미검증이며 v2 컵 배치를 채택하지 않았다. 상세는 `SCENE_REFINEMENT_HANDOFF.md` 최상단과
`outputs/workflow_validation/paired_refinement_160851/surface_refinement_result.json`.
아래는 v2 인계 당시 기록이다.

2026-10-10 최신 상태는 `SCENE_REFINEMENT_HANDOFF.md`를 먼저 읽는다.
새 Pair `snapshot_20261010_160851_00c94af5` 8쌍은 모두 import/evaluate했다.
영점/카메라 및 컵 배치 후보는 holdout 검사를 통과하지 못했다. 사용자 요청으로
컵 배치/밝은 종이컵 후보를 `UNVALIDATED` 검토용 XML/USD로 export하고
`scene1_builder/v2`로 import했다. v1과 기존 canonical은 보존한다.
v2는 검증 성공이나 physics/실물 grasp 완료가 아니다. 파일/검사 결과는
`outputs/workflow_validation/paired_refinement_160851/layout_export_result.json`.
사용자는 다른 Codex 계정으로 작업을 인계하며, 현재 턴에서는 추가 fitting을 시작하지 않는다.
이후 wrist/side/contact 차이를 기존 도구로 보정하되 measured q는 고정하고,
수치 영점을 controller/URDF의 실제 calibrated 값으로 임의 적용하지 않는다.

## 실행 환경과 검사

기존 `/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python` (Python3.11), RTX4060 Laptop8GB.
GUI venv `.venv-real2sim-ui`; root pyproject는 package installer가 아니라 tooling 설정이다.
root에 pip -e . 하지 않는다. 설치/버전 변경 없이 기존 환경을 사용한다.
SO UI 진입은 `scripts/real2sim/real2sim_ui.py`; GUI dependency는 해당 기존 venv를 확인한다.
Blender `/home/hyeonhee/apps/blender/blender`. Builder 환경 명령은 sibling docs/LOCAL_RUN.md.

```bash
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -B -m pytest tests/real2sim/test_mujoco_backend.py tests/real2sim/test_minivla_integration.py tests/minivla/test_head.py -q -p no:cacheprovider
```

MuJoCo 검사는 pxr/Isaac을 import한 pytest 프로세스와 분리한다.
Qt/USD 검사는 기존 PyQt6/pxr 경로가 필요하며 docs/real2sim/TESTING.md 및 Builder LOCAL_RUN.md 참고.
이전 /tmp/builder_test_modules PyQt symlink는 임시파일이므로 새 계정에서 존재를 가정하지 않는다.
명령별 env를 사용하고 GUI site-packages·USD PYTHONPATH를 worker에 전역 전달하지 않는다.

## Git·보호 범위

작성 전 HEAD `ca923a737e1c5dcafb8f8e8da9f2bf0a2db8b58a`, status 113 entries(문서 작성 전).
README/docs/runtime/workflow 등 수정, bootstrap/reference/구 publish/teleop launcher 등 삭제,
simulation/learning/adapter/snapshot/tests 등 untracked. 이는 진행된 구현·정리이며 복구하지 않는다.
Builder sibling도 현재 Git 저장소다. 미커밋 변경 및 ignored project/inputs/outputs를 별도 인계해야 한다.
outputs는 Git으로 보존되지 않을 수 있다. status/diff/ignored 목록을 확인하고 사용자 데이터도 별도 백업한다.
이 문서는 commit하지 않는다. reset --hard, clean -fd, 전체 checkout 금지.
기존 사용자 photos/snapshots/HDF5/checkpoint/scene1/reference 프로젝트는 변경·삭제하지 않는다.

## 추가 변경: Pair 관측과 프리뷰/Source

Source는 실행별 fresh seed(명시 seed 재현 가능), MuJoCo preview는 기존 RGB 재사용,
Pipeline 전달/Qt polling은 최대30Hz. 실제 FPS는 renderer/기기에 따라 다르다.
Pair Capture는 Runtime cached workspace context + 기존 Snapshot 저장 확장이다.
Builder import-paired와 기존 Candidate의 three-way diagnostics를 사용한다.
현재 사용법·데이터 계약·미검증 범위는 `docs/real2sim/PAIRED_CAPTURE.md`를 먼저 읽는다.
