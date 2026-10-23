# Engineering Principles

## 1. 프로젝트 목표

본 프로젝트는 Real2Sim Builder와 LeRobot 생태계를 연결하는 로봇 학습 Workflow 플랫폼이다.

지원하려는 사용 흐름은 다음과 같다.

- Builder 기반 Simulation 환경 구성
- Metadata 기반 사용자 Task 개발
- Demonstration 수집 및 Mimic DataGen
- Dataset Curation 및 LeRobot 변환
- 공식 LeRobot 학습·평가·배포
- 외부 LeRobot Dataset 및 Policy 재사용

SO-101은 첫 번째 지원 로봇이며, 플랫폼 전체의 필수 전제는 아니다.

모든 로봇·Task의 완전 자동 생성을 보장하지 않는다.

## 2. 모듈별 책임

| 영역 | 책임 |
|---|---|
| Builder | Robot, Object, Camera, Frame, Scene Assets 및 Metadata 생성 |
| Workspace | Builder Export 로드·검증 및 Scene Definition 관리 |
| Task | 사용자 작업 정의, 성공 조건, Reset, Mimic Subtask |
| Simulation | Isaac Lab/MuJoCo 환경 구성, 상태 조회, 물리 실행 |
| Data | Episode Reader/Writer, 저장 형식 및 무결성 |
| Workflow | Stage 실행, 요청 검증, Process·Lock·Cancel·로그 |
| UI | 입력 선택, 옵션 표시, 실행 요청, 결과 표시 |
| Learning | 공식 LeRobot Trainer 및 Policy 인터페이스 |
| Evaluation | Policy 실행 및 Task 기반 결과 평가 |

## 3. 의존성 원칙

권장 실행 방향:

UI / CLI
→ Workflow Operations
→ Worker
→ Task + Simulation + Data + Learning

이것은 실행 흐름이다. Python import 방향과 자원 소유권은 별도로 검사한다.

핵심 규칙:

- Workspace의 순수 Metadata 계약은 UI와 Workflow에 의존하지 않는다.
- Task의 정의·검증 코드는 Engine 없이 import할 수 있어야 한다.
- Simulation은 Workflow의 UI·실행 정책을 import하지 않는다.
- Data Reader/Writer는 Simulation이나 UI를 실행하지 않는다.
- Learning은 사용자 정의 Simulation Task 없이 독립적으로 실행할 수 있다.
- Stage 간 직접 호출보다 기존 공통 기능을 이용한다.
- 순환 import가 필요한 설계라면 책임과 데이터 전달 경계를 먼저 재검토한다.

## 4. 상태와 자원 소유권

- Workspace Loader는 검증된 Workspace 정보를 제공한다.
- Worker는 실행 세션과 Engine 수명을 소유한다.
- Engine은 Runtime State를 소유한다.
- TaskContext는 Engine 상태를 참조하지만 별도의 Engine이 아니다.
- Dataset Writer는 저장 및 Commit 수명을 소유한다.
- Operations는 실행 요청·프로세스·취소·출력 예약을 소유한다.
- UI는 Engine/Robot/Writer 자원을 직접 소유하지 않는다.

동일한 상태를 여러 객체가 독립적으로 소유하거나 동기화하는 구조를 피한다.

## 5. Metadata 계약

Metadata는 실제 Scene을 구성하기 위한 입력이면서 사용자 Task가 참조할 객체 정보의 출처다.

- 검증된 Object/Robot/Camera/Frame ID를 보존한다.
- 동일한 Scene Definition을 Environment와 TaskContext가 사용한다.
- Metadata의 초기 Pose와 Runtime Pose를 구분한다.
- 좌표계와 단위는 명시적으로 관리한다.
- 확인되지 않은 Calibration·물성·Mapping을 검증 완료로 취급하지 않는다.
- 원본 Builder Export와 기존 Workspace는 임의로 수정하지 않는다.

## 6. 확장 기준

### 새로운 Task

가능하면 사용자 Task Python 파일 추가와 설정 선택만으로 확장한다.

공통 Workflow에 Task 이름별 분기를 추가하지 않는다.

### 새로운 Robot

Robot Asset, Joint/Action 계약, Control/Observation Adapter 등 필요한 구현을 명시적으로 추가한다.

지원되지 않는 로봇을 기존 SO-101 Mapping으로 암묵적으로 처리하지 않는다.

### 새로운 Policy

공식 LeRobot이 지원하는 Policy라면 기존 Trainer·Config·Checkpoint 인터페이스를 우선 활용한다.

MiniVLA 같은 연구용 모델은 독립적인 Policy 확장으로 관리한다.

### 새로운 Simulator

공통으로 제공 가능한 상태·실행 계약을 우선 활용한다.

Engine별 물리·Batch·Reset 차이를 억지로 숨기지 않는다.

## 7. 추상화 도입 기준

새로운 Class/Interface/Module은 다음 조건을 검토한 후 도입한다.

- 실제로 책임이 다른가?
- 실제 호출자가 필요한 인터페이스인가?
- 기존 모듈에서 해결할 수 없는가?
- 새로운 의존성이나 상태 소유자를 만들지는 않는가?
- 테스트 가능한 명확한 입력·출력 계약이 있는가?
- 새로운 기능을 추가할 때 기존 코드를 덜 수정하게 되는가?

단순히 코드 줄 수를 줄이거나 미래 가능성만을 이유로 추상화를 추가하지 않는다.

## 8. 금지할 Anti-pattern

- 새로운 Registry나 중앙 상태 관리자 도입
- 동일 Metadata를 여러 영역에서 독립적으로 재해석
- Stage 또는 Task마다 Environment 생성 로직 복제
- Workflow에서 Task별 성공 조건 하드코딩
- Simulation에서 UI·학습·데이터 저장 정책 직접 실행
- 단순 전달만 하는 다층 Wrapper
- 기능 하나를 위해 다수의 불필요한 Service/Manager 추가
- 검증 없이 Robot/Action 좌표계를 변환
- 기존 기능을 대체하기 위해 공식 LeRobot 기능을 중복 구현

## 9. 변경 완료 기준

기능을 수정할 때 다음을 확인한다.

1. 실제 호출 경로와 책임 변경이 명확한가?
2. 의존성 방향이 유지되는가?
3. 기존 자원 소유권이 보존되는가?
4. 기존 SO-101 실행의 회귀 검사가 가능한가?
5. 새로운 Task/Robot/Policy 확장에 불필요한 공통 코드 수정이 없는가?
6. 실제 검증과 미검증 범위를 구분했는가?

기능 구현뿐 아니라 코드 구조를 유지하는 것까지 완료 조건에 포함한다.
## 10. Phase 1–3 구현 경계 — 2026-10-22

이번 수직 경로는 `Workspace Loader → generate_template → 사용자 Python → 기존 검사 Worker → TaskContext → 기존 Backend`다. 새 Production 모듈은 `simulation/task_context.py` 하나다. 기존 `SimBackend`는 엔진 제어/수명 계약이므로 Task의 객체 ID 검증·빌린 상태 접근 책임을 넣지 않았다. Context 하나만 만들고 Robot/Object별 Wrapper나 새 추상 Task 클래스는 만들지 않는다. 사용자 Task는 함수형 모듈이다.

`TaskContext.workspace`는 Backend가 실제 구성한 Workspace를 참조한다. Metadata는 읽기 전용으로 취급하는 빌린 dict이며 Python sandbox/강제 불변 객체는 아니다. Context가 같은 파일을 다시 읽거나 상태를 동기화하지 않는다. 생성기는 기존 workspaces/tasks.py, 실행은 기존 scripts/mujoco/check_env.py에 붙인다.

현재 Backend의 성공 수학은 이전하지 않았다. Pick & Place 예제만 기존 판정을 호출한다. 신규 Task는 자신의 success를 구현할 수 있지만 UI/Mimic/Evaluation으로 자동 연결되지는 않는다. Learning/Data/UI를 이번 API에 의존시키지 않는다. 기존 패키지 전체의 오래된 역참조까지 제거했다는 뜻은 아니다.

## 11. Scene 기록과 재사용 Term의 경계

Source의 Scene 생성/Recording과 후속 Annotation/Mimic의 작업 의미를 구분한다. 매핑형 Scene은 Builder Pose가 기본이며 Task ID나 Pick/Target을 만들지 않는다. Metadata의 물리 classification/질량/Collision이 없는 Body를 추정해 dynamic으로 바꾸지 않는다. 명시 Reset은 기존 Backend reset 경계/RNG를 쓰고 변경할 Body만 적용한다.

Event는 기존 pre-step 관측을 빌린다. Context의 일시적인 관측/fact는 callback 수명 안에서만 존재하며 다음 step의 상태 cache가 아니다. 상태가 필요한 사용자 Term은 환경/Episode별 factory로 생성한다. StableGrasp는 기존 Adapter 소유 tracker를 조회하고 새 tracker/physics loop를 만들지 않는다. Named Signal·최종 수동 Mark·채택·파일 게시의 소유자는 기존 Annotation/Writer다. 새 Mimic 선언은 기존 reference Generator가 지원하는 설정으로 한정하고 Native recipe의 원래 성공 gate를 유지한다.
