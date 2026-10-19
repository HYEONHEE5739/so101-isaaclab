# SO-101 관절 제어 — URDF와 기존 Precision Mapping

2026-10-12. 사용자 요청으로 별도 Unified Controller, MotionGuard, 범위 교차 Controller,
자동 복귀와 전용 UI/IPC 설정을 제거했다. 현재 정책은 이 문서이며 이전 구현 보고서는
역사적 검사 기록이다. 최신 결과는
`outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md`를 읽는다.

## Real-only / Real-Sim

```text
Leader raw → 기존 LeRobot degree/percent → Leader Precision/legacy observation mapping
→ Runtime.command_target: URDF/선택 Sim/현재 Follower 명령 가능 범위로 목표 clip
→ Follower Precision inverse → 실제 SDK 정수 encoder conversion
→ 같은 encoded URDF q
   ├ Follower native goal
   └ Isaac / MuJoCo target
```

`so101_joint_mapping.observed_positions`는 기존 native/raw 관측 변환만 담당한다.
Controller, 구동 상태 머신, 시작 자세/속도/추종 Guard는 없다. 이미 URDF rad인 데이터,
Replay, IK, Policy action에 Leader/Follower mapping을 중복 적용하지 않는다.

Leader 요청이 URDF 밖이면 목표만 제한한다. 선택한 옛 Workspace의 엔진 범위가 더 좁으면
그 엔진 범위도 지킨다. 2026-10-13부터 Teleop은 저장된 Follower 모터 범위와의 교집합에서
표현 가능한 정수 목표 경계로 제한하여, 정상 범위 초과만으로 Start/추종을 중단하지 않는다.
범위 안으로 돌아오면 재시작 없이 추종한다. 실제 측정값/보정 계수는 바꾸지 않는다.
공통 범위가 없거나 불연속/모호한 encoder branch이면 명령 전에 거절한다. 하드웨어 range를 늘리거나 먼 목표 대신 자동 복귀 목표를 생성하지 않는다.
SDK encoder 변환은 Hardware에 한 개 남겼다. Precision inverse가 선택한 정수 raw를 SDK
`int()`가 한 count 잘못 절삭하지 않도록 native 표현의 ULP만 보정한다. 변경된 계수가 아니다.
Real/Sim은 이 실제 정수 goal을 forward mapping한 같은 q를 사용한다.

Connect는 통신/calibration identity/피드백을 읽고 카메라를 연결한다. 목표/torque 변경,
Home 이동을 수행하지 않는다. 유효하게 해석되는 범위 밖 Follower 측정값도 보존한다.
명시 Start는 목표 유효성을 확인한 후 기존 hold seed/readback/configure를 수행한다.
5° 시작 차이, 30°/s 변화율, 10° 추종 차이 차단과 자동 경계 복귀는 제거했다.
Stop은 새 목표 전송을 멈추며 기존 마지막 목표의 hold를 유지한다. Disconnect는 기존 API로 닫는다.

NaN/Inf, 잘못된 joint 이름, 통신 오류, 보정 identity/encoder branch 오류, Motor range 밖
명령과 엔진 범위 밖 목표의 직접 전송은 계속 거절한다. Teleop에서만 사전 목표 제한을 적용하며
Policy/Replay의 strict inverse 계약은 그대로다. UI heartbeat loss/0.75s loop stall 정지를 유지한다.

## 측정과 Snapshot

raw/native/measured q를 임의로 clip하지 않는다. 측정과 command를 따로 저장한다.
현재 Sim으로 재현할 수 없는 measured q는 마지막 유효 Sim 자세를 표시하며 mismatch를
Pair 품질에 기록한다. 정상적으로 보이는 것으로 속이지 않는다. 범위 밖이라는 사실만으로
유효한 실제 관측·Capture 연결을 중단하지 않는다. ambiguity/잘못된 packet은 거절한다.
새 Snapshot은 retired controller metadata를 쓰지 않는다. 이전 Snapshot의 applied q를 읽는
호환 코드는 남겨 기존 사용자 데이터를 해석한다. 사진/측정/품질 게이트는 기존 계약을 따른다.

## Source / Policy / 과거 데이터

Isaac Source/독립 Teleop은 기존 SO101Leader.advance에서 Precision 관측 후 선택 URDF로
목표를 제한한다. MuJoCo Source는 기존 worker에서 URDF/backend 및 선언된 policy gripper
범위로 목표를 제한하고 요청/실제 action을 별도로 기록한다.
실물 Policy는 UI의 `Real 평가 시작`(내부 enable_motion 지정) 또는 CLI의 명시 opt-in과 dataset frame 검사, strict policy observation/inverse,
모터 유효성 검사로 실행한다. 삭제된 Teleop Guard는 Policy에도 적용하지 않는다.
Policy ActionGuard와 RobotCoordinates는 기존 별도 표현/rollout 계약으로 유지한다.

과거 dataset/Workspace/Scene offsets를 수정하거나 현재 보정으로 재해석하지 않는다.
2026-10-15부터 `precision.simulation_configuration`이 새 Source/Runtime/파생 단계/Sim Eval의
모델을 결정한다. numeric URDF rad와 identity geometry를 구분한다. 검증된 role mapping에
연결된 Scene 잔차는 Sim URDF/USD geometry에 한 번 적용하며 command 값은 바꾸지 않는다.
Leader 수집 mapping과 Follower fitting binding은 서로 다른 hash를 가질 수 있다.
Replay/Annotation/DataGen/Sim Eval은 원본 계약을 상속하고 Isaac IK/Physics는 대응 모델을 쓴다.
명시 URDF 원형 검사는 유지하며, 과거 URDF-frame/legacy 데이터는 당시 identity/Scene 동작을 유지한다.
Real Eval은 Follower 호환성을 activate 전에 확인하고 Sim 잔차를 hardware inverse에 섞지 않는다.
의사결정표/로그/저장 위치는 [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md#workflow-공통-모델-결정-2026-10-15)
및 [DATA_FLOW.md](DATA_FLOW.md)를 따른다. 기존 encoder/통신/range 검사는 그대로다.

## Model / 보정 보존

URDF의 Pan/Lift/Elbow/Flex만 이전 rad에서 양쪽 radians(5) 확장되어 있다. 다시 늘리지 않는다.
Roll/Gripper, origin/axis/mesh/inertia/effort/velocity는 그대로다. 실행용 USD override는
`assets/SO101/usd/so101_joint_limits.usda`, 새 실행 Workspace는 `scene1_builder/v10`이다.
원본 USD/Builder export/Workspace는 덮어쓰지 않는다. 기존 Workspace는 자신의 모델을 재현한다.

Limit-only 변경은 `precision.profile_for_model`에서 모델 구조를 검증한 뒤 실행 복사본에만
현재 URDF 제한을 적용한다. 기존 Profile/active는 그대로이며 재저장/재승인/Migration UI는 없다.
이전 모델의 정확한 hash에 맞는 Git/보존 URDF 근거가 필요하다. Origin/Axis/이름/geometry 변경과
모터 identity 변경은 거절한다. 실제 보정 계수를 바꾸는 Save/승인 Activate/Restore는 유지한다.
세부 사용법은 [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md), 최신 테스트는
`outputs/workflow_validation/joint_limit_extension/COMPATIBILITY_REPORT.md`를 따른다.

실제 장치의 새로운 범위·대상 jump·접촉/충돌/정확도는 HARDWARE_NOT_TESTED다.
소프트웨어/시뮬레이터 검사와 실제 하드웨어 안전 인증을 구분한다. 이번 검사는 장치 포트를
열거나 torque를 활성화하지 않는다. 기존 worker는 사용자가 종료하고 새 코드로 재실행한다.
