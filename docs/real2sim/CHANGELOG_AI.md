## 2026-10-15 — Workspace 선택형 조작 물체 / Workflow 전달

- 기존 TaskSelector에 Scene objects 체크박스/count/Workspace별 선택 저장. Task 충돌은 명시적으로 처리.
- 기존 tasks.configured의 실행 복사본으로 두 엔진에서 비선택 visual/body/collision/joint/state/reset 제거.
- 기존 Request/Provenance의 active IDs와 Scene hash, Source 상속·Resume 검사, episode별 변환·학습 보존.
- SimEval composition shift 및 전체 distractor/선택·grasp·transport·place metrics. 단일 물체 선택 정확도 N/A.
- Scene/Calibration/MiniVLA architecture 원본 보존. 새 controller/launcher/schema 파일 없음.
- 사용법: BUILDER_WORKSPACE.md / DATA_FLOW.md / ../minivla/EVALUATION.md.
- 실제 검증/한계/초기 실패: outputs/workflow_validation/active_scene_objects/REPORT.md.

## 2026-10-13 — Teleop 목표 포화 처리

- URDF/Sim과 Follower 정수 encoder 명령 범위 교집합으로 기존 Teleop clip 확장.
- 정상 범위 초과 시 Start/추종 거절 대신 경계 목표 유지, 범위 재진입 시 계속 추종.
- Real/Sim 동일 encoded q, 실제 관측/보정/Profile/모터 설정/Workspace 보존.
- 범위 없음/encoder ambiguity/identity/통신 검사는 유지. Policy strict inverse 변경 없음.
- 관련171 PASS,1 SKIP. 실제 로봇 미구동. 상세: outputs/workflow_validation/unified_joint_control/TELEOP_LATENCY_DIAGNOSIS.md.

## 2026-10-12 — URDF Limit-only 자동 호환 / Migration 제거

- 이전 승인형 limit migration 경로를 제거하고 정확한 Git/보존 모델 근거로 lower/upper 외
  동일성을 검사한다. 기존 Profile 계수/파일/active를 유지하며 실행 제한만 선택 URDF에서 읽는다.
- hardware binding, Calibration 표시/부분 보정 저장/복구, MuJoCo Source 모델 선택을 연결했다.
- 새 controller/guard/config/보정 버전을 자동 생성하지 않는다. motor identity/range/encoder 검사 유지.
- 결과: `outputs/workflow_validation/joint_limit_extension/COMPATIBILITY_REPORT.md`.
- 아래 승인형 Limit 재사용 항목은 당시 이력이며 현재 사용법이 아니다.

## 2026-10-12 — Unified Controller 제거 / 기존 URDF 관절 경로 복원

사용자 요청으로 전용 Controller/MotionGuard/자동 복귀와 UI 5°/30°/s/10° 및 IPC 설정을 제거했다.
관측은 기존 mapping, 목표 제한은 기존 Runtime/Source, strict inverse/SDK 변환은 기존 Hardware로
정리했다. 삭제 기능의 테스트는 제거하고 필요한 encoder/관측 검사는 기존 mapping tests에 통합.
Precision/Reference/active/사용자 데이터/Builder 보존. URDF ±5°는 이미 적용되어 재확장하지 않았다.
Sim Eval의 옛 USD 선택을 기존 identity_sim Workspace 경로에 연결했고 Fake real policy loop를 검사했다.
독립 process 회귀 332 PASS / 1 SKIP, MuJoCo 200 step+두 render, Isaac20step+두camera를 확인했다.
초기 combined suite의 engine/renderer 충돌과 sandbox GPU 실패를 숨기지 않고 기록했다.
현재 정책: JOINT_CONTROL.md. 최신 증거: outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md.
아래 Unified 구현 항목은 당시 이력이며 현재 실행 경로가 아니다.

## 2026-10-12 — 새 URDF와 이전 활성 보정의 충돌 진단

사용자 Capture 오류에서 Leader/Follower active는 이전 URDF hash이고 새 Limit용 후보는
현재 hash임을 읽기 전용으로 확인했다. 양쪽 6관절의 계수/reference/motor identity는 보존되어 있다.
hardware.attach_precision은 장치/LeRobot 충돌의 다른 필드와 URDF 버전 충돌을 구분하여 안내한다.
Calibration UI는 보정 소실이나 재측정으로 오해하지 않도록 기존 재사용/승인 버튼을 안내한다.
관련 Fake·Qt·worker·control·launcher 회귀 119 PASS. 추가 migration worker 검사도 PASS:
재기록 없이 재사용, 승인 전 active 보존, 승인 후 모델 오류 해소, activation motor write 없음.
실제 장치 연결/구동이나 사용자 active 변경은 수행하지 않았다.

## 2026-10-12 — URDF 네 Joint Limit ±5° / 승인형 Profile 재사용

- 네 lower/upper만 수정; Roll/Gripper/origin/axis/inertia/effort/velocity 및 원본 USD 보존.
- limit-only 모델 동일성 및 기존 계수/reference/encoder/motor 범위 검증 후 새 immutable Profile 저장.
  활성화는 기존 승인 경로 유지. 기존 active/LeRobot/사용자 데이터 변경 없음.
- 별도 USD Limit layer와 새 Workspace scene1_builder/v10. Source/Policy/두 엔진 경로의 모델 범위 일관성.
- 실제 MuJoCo/Isaac smoke 및 관련 회귀 결과/초기 실패/defaultPrim 수정/실물 미검증은
  `outputs/workflow_validation/joint_limit_extension/REPORT.md`에 기록.

# AI Change Log

## 2026-10-12 — Capture 재시작 시 이전 worker 재사용 수정

사용자 재시도에서 19:10에 실행된 UI가 수정 전 18:53에 시작한 real worker를 재사용하여
StrictLeader abstract 오류를 반복한 것을 PID/시작 시각/로그로 확인했다.
Capture UI 종료는 기존 disconnect를 보내 worker를 정상 종료한다. 명시 Start 시 real ERROR
worker가 있으면 disconnect/종료 확인 뒤 새 worker를 실행한다. UI 시작 자체는 연결하지 않는다.
재시도 순서와 Qt 종료 회귀를 추가했다. 실제 사용자 프로세스 종료/장치 연결은 수행하지 않았다.

## 2026-10-12 — Capture의 StrictLeader abstract class 연결 오류 수정

관절 제어 정리에서 필수 `_get_raw_data()` 구현을 제거하여 실제 SO101Leader/StrictLeader를
생성할 수 없던 회귀를 수정했다. 이 메서드는 normalize=False encoder read를 제공하고
advance가 재사용한다. 읽기 실패/누락을 0 또는 빈 결과로 대체하는 옛 fallback은 복원하지 않았다.
실제 클래스와 StrictLeader를 사용하고 serial bus만 Fake로 교체한 생성/읽기 전용 연결,
advance, 통신 오류/누락, 암묵 calibration 차단 검사를 추가했다.
Control/Precision/Desktop/Launcher/Liveness 회귀 114 passed (18.02s).
실제 장치 연결·구동, 사용자 보정/데이터 변경, dependency 변경은 없다.
기존 worker의 클래스는 메모리에 남으므로 사용자가 UI/worker를 종료하고 다시 실행한다.

## 2026-10-12 — Unified Joint Control / 중복 live target 경로 정리

`so101_joint_control`을 Runtime의 Real-only/Isaac/MuJoCo, Source/독립 Teleop, real_eval에 연결했다.
관측/요청/명령 분리, inverse 전 clip, SDK 정수 encoding preflight, 시작/속도/추종 guard,
read-only Connect, 범위 밖 passive Snapshot 보존과 공통 오른쪽 진단 표시를 구현했다.
기존 target wrappers/LeaderTargets/missing-zero fallback/미사용 device USD limits property를 삭제했다.
기존 사용자 데이터/보정/Workspace/Builder/dependency는 변경하지 않았다.
상세 파일·호환성·한계: `JOINT_CONTROL.md`; 최종 실행 검사는
`outputs/workflow_validation/unified_joint_control/REPORT.md`를 따른다.


## 2026-10-12 — Real-only Capture의 수동 관측과 구동 한계 분리

STOPPED Capture에서도 구동용 URDF guard로 반환해 이미지/측정 갱신이 막히던 문제를 수정했다.
passive 관측은 구별 가능한 reference encoder branch의 계산각을 그대로 기록하며 범위 초과 경고를 저장한다.
Snapshot feedback/history/image 정렬에도 같은 관측 변환을 사용한다. passive 상태의 motor command는 거절한다.
Teleop/physics/goal 제한과 encoder ambiguity 검사는 유지한다. 사용자 보정/원본/장치는 변경하지 않았다.
검사 결과는 precision_calibration/REPORT.md 최신 항목 참조.

## 2026-10-12 — 관절 보정을 Source/Teleop/실물 정책 경계에 통합

hardware 어댑터와 standalone SO101Leader가 Connect 시 승인 active를 검증/로드한다.
Real/Sim UI는 저장값이 있으면 Precision 및 원본 URDF 관절 기준을 기본 선택한다.
새 Source는 보정된 model rad, hardware_mapping, simulation_joint_frame을 기록한다.
두 엔진의 Annotation/Datagen/Convert/Train/Sim Eval은 기록된 기준을 따라가며 원본 rad를 다시 보정하지 않는다.
real_eval은 현재 Follower 보정으로 관측/정책 goal을 변환하고 기존 명시적 motion 활성화·안전을 유지한다.
원본 LeRobot 파일/Workspace/기존 데이터는 변경하지 않으며 보정이 바뀐 Source resume은 거절한다.
검사 결과와 초기 실패/환경 제한은 precision_calibration/REPORT.md 최신 항목 참조.

## 2026-10-12 — 이전 Calibration 실행 경로 정리

현재 UI 호출을 확인한 뒤 미사용 precision.validate worker 명령과 그 reply/오류 안내를 제거했다.
새 record는 수동 기준각만 받아 train/angle로 고정하며 이전 validation/open/closed/opening 입력 생성 경로를 제거했다.
UI 내부 버튼 키를 review로 통일하고 읽히지 않는 last_saved 상태를 삭제했다.
기존 Profile의 validate_joint/legacy_endpoints 증거 reader는 로드·복구에 실제 사용되므로 유지한다.
현재 인수인계의 여러 버전 설명을 하나로 통합하고 상세 이력은 기존 Report/Changelog로 연결했다.
사용자 보정/Profile/원본 데이터와 다른 프로젝트 기능은 변경하지 않았다. 검사 결과는 Calibration Report 참조.

## 2026-10-12 — 보정 기준점과 URDF 사용 범위 분리

0°/90°에서 계산한 보정식을 기록 구간 밖의 음수/90° 이후에도 URDF limit 안에서 사용한다.
UI/Calibration/Runtime/Compare에서 URDF 역 raw 구간의 유일한 encoder branch를 공유한다.
기존 Profile/raw_domain은 증거로 보존하며 재저장하지 않는다. Precision gripper의 기록 구간 clamp도 제거했다.
goal은 기존 모터 calibration 범위/URDF limit/branch 검사를 유지하며 URDF 경계의 정수 raw 양자화를 확인한다.
UI는 선택 joint의 표시 범위를 보여준다. 실제 장치 연결/구동 및 사용자 데이터 변경 없음.
검사 결과와 저장된 6관절 offline 점검은 precision_calibration/REPORT.md 최신 항목 참조.

## 2026-10-12 — 현재 저장값으로 재보정 비교 / 반복 저장 재사용

보정 전 표시와 표 계산의 기준을 현재 active joint로 변경했다. 새 계산이 없으면 전/후/현재가 같다.
raw 간격을 현재 비율의 예상값과 비교하는 진단을 추가하고 0.5°가 fit residual임을 명확히 표시했다.
2점은 그대로 통과하므로 residual로 raw 간격 이상을 검증할 수 없다는 설명과 테스트를 추가했다.
같은 identity/model/entries 저장은 파일 재사용, 같은 active 적용은 이전 pointer 이력을 늘리지 않는다.
실제 변경은 새 버전을 만들고 다른 관절/기존 사용자 파일은 보존한다.
관련 회귀 151 passed/1 skipped. 하드웨어 연결/모터 명령/사용자 active 변경 없음.

## 2026-10-12 — 관절 보정 누적 저장/세션 복원 및 사용자 6관절 복구

save는 현재 role의 모든 확인한 Candidate를 기존 active에 합친다. partial activate로 다른 관절을 버리지 않고
별도의 합친 Profile을 만든다. Connect 시 같은 identity/model의 활성 기준 기록을 새 세션 표에 복사한다.
loaded_profile_id를 남기며 원본/Profile/exclusion 이력을 보존한다.
원본 6관절 기록 중 기존 활성 4관절은 그대로 두고 wrist_flex/gripper만 angle mode로 재계산했다.
사용자 요청으로 새 Follower 6관절 Profile을 software activate했으며 Calibration worker가 읽은 것을 확인했다.
모터 명령/새 hardware 연결 없음. Leader 데이터를 임의로 만들지 않았다.
경로/근거/검사와 복구 결과는 precision_calibration/REPORT.md 및 saved_profile_recovery.json 참조.

## 2026-10-12 — 재시작 후 활성 보정 각도 표시

Calibration worker는 active Profile을 이미 읽고 있었으나 UI가 빈 Candidate만 보정 후 표시로 사용했다.
맨 위 현재 각도는 worker의 적용된 urdf_angle_rad를 표시하고 새 후보가 없을 때 active joint 계산을 보여준다.
기존 보정 전 baseline은 작은 비교 표시로 유지한다. 범위/변환 실패와 해당 joint 미보정은 구분한다.
새 재시작 회귀는 기존 Fake worker를 종료/재연결해 Candidate가 없어도 saved 각도 표시를 확인한다.
사용자 active/저장 파일은 읽기 전용 진단만 수행했다. 실제 하드웨어 연결 없음.
결과·미적용 파일과 활성 Profile의 차이는 precision_calibration/REPORT.md 참조.

## 2026-10-12 — 그리퍼도 기준각으로만 Calibration

PrecisionPanel의 gripper reference kind/opening 입력을 제거하고 모든 관절에서 angle 기준만 기록한다.
fit_joint 기본 angle mode는 gripper endpoint/mm 검사를 요구하지 않는다.
reference_mode를 저장하며 이전 필드 없는 saved Profile의 endpoint 검사는 그대로 유지한다.
SDK RANGE_0_100/native-percent↔raw 계약과 모터 register는 변경하지 않는다.
Fake Leader/Follower gripper raw→각도→native goal 왕복/저장·적용과 기존 worker+Qt 그리퍼 전체 절차를 확인했다.
실제 하드웨어 연결 없음. 결과/검사 수치는 precision_calibration/REPORT.md 참조.

## 2026-10-12 — Calibration 잘못된 자세 제외·재기록

기존 기록 표에 선택 자세 삭제를 추가했다. 안정된 reference ID로 행을 선택하고 승인 후
계산에서 제외하며 원본 measurement는 journal.references, 제외 결정은 journal.exclusions에 보존한다.
해당 미적용 Candidate만 초기화하고 기존 active/다른 관절/기존 사용자 파일은 자동 변경하지 않는다.
삭제 후 올바른 기준을 재기록하고 다시 계산·저장·적용한다. 파일 실패 시 원래 계산 상태 유지.
기존 worker+Qt Fake에서 선택 유지/취소/삭제/재기록/원본 보존/활성 보정 유지, core에서
중복/잘못된 관절/쓰기 실패/빈 기록 계산 거절을 확인했다. 하드웨어 연결·명령은 수행하지 않았다.
검사 수치와 사용법은 precision_calibration/REPORT.md, PRECISION_CALIBRATION.md 참조.

## 2026-10-12 — Calibration 전체 기준 fit / live 수동 자세 확인

사용 목적 선택 제거, 새 기록은 모두 train. 자세 검증은 별도 validation 자료 대신 raw/보정 전·후
현재각과 차이를 손으로 움직이며 확인한다. 기본 허용 오차는 기록 재현 0.5°다.
기존 precision.py에 review_joint와 REVIEWED_MANUAL을 추가하고 수동 검토와 독립 검증을 구분했다.
승인/identity/hash/계수 검사는 유지하며 이전 saved VALIDATED_MANUAL evidence도 계속 읽는다.
이동 중 표시는 CalibrationDevice의 같은 raw packet normalization, Record는 기존 정지 검사를 사용한다.
하단 Calibration 로그는 요약, 상세 JSON은 접힌 영역이다. 원본/active/실제 장비에 자동 쓰기 없음.
전체 관련 회귀 **145 passed / 1 skipped**; 실제 장비 연결 없이 Fake worker+Qt/core live 검증.
세부 결과·한계·변경 경로는 precision_calibration/REPORT.md와 PRECISION_CALIBRATION.md 참조.

## 2026-10-12 — Calibration 현재각·검증 결과 표시 개선

기존 PrecisionPanel에서 현재 적용각과 미적용 Candidate 현재각을 degree로 크게 표시하고,
기록된 기준각/계산각/차이를 표로 보여준다. 검증 결과·허용값·다음 행동과 번호/한국어 버튼을 추가했다.
gripper 전용 필드는 해당 joint에서만 보이며 JSON/Fake/복구·허용오차 설정은 상세 영역으로 접었다.
원본 사용자 기록의 validation +90°와 사용자 설명 −90°의 차이를 확인했으나 기록/active/모터 상태를 변경하지 않았다.
기존 worker+Fake와 관련 회귀 **143 passed, 1 skipped**. 실제 장치 검증은 수행하지 않았다.
후속 REPORT.md 및 PRECISION_CALIBRATION.md에 사용법·새 세션 재기록 제한을 기록했다.

## 2026-10-12 — Joint Precision Calibration UI / software coordinate profiles

기존 PyQt UI에 네 번째 Calibration 탭을 추가했다. 기존 launcher/worker의 calibration mode를
재사용하며 raw-register single-joint reference, sign/scale/offset/wrap, 별도 validation,
immutable candidate/approved software activation/restore를 구현했다. LeRobot의 range-mid degree
정규화와 firmware Homing 이후 raw를 구분한다. motor calibration/Scene/사용자 데이터를 고치지 않았다.

실제 reference 계산은 real2sim/precision.py, UI는 precision_ui.py, device 읽기/torque release는
hardware.py다. Real/Sim/Capture Runtime에 opt-in common coordinates와 원본 mapping metadata를
연결하고 nonzero Scene offsets는 명시 session-only validation 설정 없이는 거절한다.
기존 standalone Source/real_eval coordinate 계약은 유지한다. actual robot은 연결하지 않았다.

Fake SDK/실제 worker+offscreen Qt 전 과정과 MuJoCo FK preview, 관련 기존 회귀 **142 passed, 1 skipped**.
실물 encoder/torque/지그/serial exclusivity/추종 및 Isaac 새 session 설정은 **HARDWARE_NOT_TESTED**.
사용법/정확한 경로/실패 수정/한계: docs/real2sim/PRECISION_CALIBRATION.md와
outputs/workflow_validation/precision_calibration/REPORT.md. dependency 설치·버전 변경 없음.

AI/Codex가 큰 변경을 만들 때 사람이 전체 diff를 처음부터 읽지 않아도 변경의 의도와 검증 근거를 추적할 수 있도록 남기는 로그다.

## 작성 규칙

- 사소한 typo/UI 문구 변경은 생략 가능하다.
- 구조/데이터/robot/camera/dataset/workflow 의미가 바뀌는 작업은 기록한다.
- 새 항목은 위에 추가한다.
- "완료"가 아니라 **무엇을 증명했고 무엇은 아직 모르는지**를 적는다.

## 템플릿

```md
## YYYY-MM-DD — <change title>

### Problem
무엇을 해결하려 했는가.

### Approach
어떤 구조로 해결했고 왜 이 방법을 선택했는가.

### Changed
- path/to/file.py — 변경 이유
- ...

### Data-flow impact
입출력/단위/좌표계/schema가 바뀌었는가.

### Validation
- command/test:
- result:
- before/after metric:

### Not verified
실제 hardware, Isaac runtime, held-out data 등 확인하지 못한 부분.

### Risks / failure modes
어디에서 틀릴 수 있는가.

### Human-review hotspots
사람이 직접 볼 파일/함수와 이유.

### Future edit point
다음에 같은 기능을 수정할 때 어디부터 보면 되는가.
```

## 2026-09-21 — Documentation consolidation

### Problem
Real2Sim 문서가 Phase 작업지시, audit 보고서, 현재 사용법으로 분산되어 현재 구조와 수정 위치를 찾기 어려웠다.

### Approach
현재 코드 중심의 living documentation으로 통합했다.

### Changed
- `README.md` — 문서 진입점과 현재 실행 개요
- `ARCHITECTURE.md` — 모듈 책임과 시스템 구조
- `DATA_FLOW.md` — joint/camera/dataset/workspace 데이터 흐름
- `PROJECT_MAP.md` — 수정 목적별 코드 위치
- `INVARIANTS.md` — 깨뜨리면 안 되는 계약
- `TESTING.md` — 증거 중심 테스트 전략
- `CHANGELOG_AI.md` — 이후 AI 변경의 검증 기록 형식

과거 Phase 구현지시/audit/status MD의 핵심 현재 규칙은 위 문서들에 흡수하고 중복 문서는 제거한다.

### Validation
문서가 참조하는 현재 주요 source/workflow/workspace/test 경로를 main branch에서 확인했다.

### Not verified
이 문서 정리 자체는 runtime code를 변경하지 않으므로 Isaac/hardware test를 실행하지 않았다.
