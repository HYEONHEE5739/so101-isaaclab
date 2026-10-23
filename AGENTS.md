# Codex 작업 규칙

- 먼저 `CODEX_HANDOFF.md`와 해당 영역의 문서를 읽고 실제 호출 경로를 확인한다.
- SO-101은 로봇 제어·Snapshot·Workflow 소비자이고, Scene 제작은 sibling `../real2sim-builder`가 담당한다. Builder 원본 export를 수정하지 않는다.
- 기존 Teleoperation, 장치 안전, Snapshot, 두 엔진 경로를 보존한다. UI 시작/탭 전환만으로 장치를 연결하거나 로봇을 움직이지 않는다. 하드웨어 검증은 별도 명시적 요청이 있을 때 수행한다.
- 사용자 Snapshot/사진/HDF5/학습 모델/immutable Workspace를 삭제하거나 metadata를 임의로 바꾸지 않는다. 정리 요청은 명시된 범위만 수행한다.
- 기존 모듈을 확장하고 중복 launcher·임시 버전·새 시스템을 만들지 않는다. 엔진은 별도 worker에서 실행하며 MuJoCo worker에 isaaclab/omni/pxr를 import하지 않는다.
- measured joint와 commanded target을 구분한다. legacy_unverified mapping 및 추정 물성/카메라를 검증 완료로 표시하지 않는다.
- conda dependency 버전을 임의로 변경하거나 설치하지 않는다. GUI·USD 경로를 전체 shell 환경에 섞지 않는다.
- 미커밋 작업이 많다. reset/clean/일괄 checkout/자동 commit을 하지 않는다. 변경 전 상태를 확인한다.
- 기능 변경 후 관련 회귀 검사와 가능한 실제 실행 검사를 수행한다. synthetic/단위 검사와 실물 End-to-End 성공을 구분하여 보고한다.

## 코드 작업 완료 보고

기본 완료 보고서는 개발자가 구조와 핵심 코드를 학습할 수 있는 1~2페이지 요약으로 작성한다.
테스트 개수·파일 나열보다 무엇을 왜 바꿨고 코드가 어떻게 동작하는지 설명한다.

1. **작업 결과 요약**: 최대 5줄로 구현 내용, 변경 전후 동작, 해결한 문제, 실제 검증 범위, 남은 위험·미검증 사항을 먼저 설명한다.
2. **변경된 파일**: 실제 경로를 사용한 `파일 | 구분(신규/수정) | 변경 내용` 표를 중요도 순으로 작성한다. 핵심 5~8개를 우선하고 나머지는 경로만 간략히 적는다. 가능한 경우 에디터에서 바로 열 수 있는 파일·라인 링크를 제공한다.
3. **핵심 구현 코드 설명**: 중요한 함수·로직 1~3개를 선정한다. 각 항목에 수행 기능, 실제 위치·함수명, 변경 이유, 핵심 실제 코드 5~15줄, 입력→처리→반환 흐름, 설계 이유, 주의점을 포함한다. 완료 시점의 Repository에서 확인한 코드만 인용하고 생략 부분을 명시한다. 설계 의도와 코드에서 확인한 사실을 구분한다.
4. **데이터 및 실행 흐름**: 변경 전후 흐름을 간단히 표시하고 각 단계의 파일·함수 및 기존 시스템과 연결되는 지점을 설명한다.
5. **테스트 결과**: 핵심 검증 내용, 실제 데이터/Synthetic Fixture 구분, 사용 환경·실행 명령, 실패·미실행 검사, 실제 하드웨어·정책 성공 검증 여부를 간결히 적는다. PASS 숫자만으로 완료를 설명하지 않는다.
6. 보고서 마지막은 반드시 **내가 직접 확인할 것**으로 마친다: (1) UI/CLI에서 실행할 기능, (2) 직접 살펴볼 핵심 함수, (3) 확인할 출력·로그·예상 결과.

로봇 작업의 핵심 코드는 Joint State/Action 변환, rad/deg/gripper 정규화, Precision/Simulation
Calibration, 데이터 저장·DataLoader, 모델 Tensor 입출력, Loss·학습 과정, Replay/Rollout 실행
경로 중 이번 변경과 관련된 부분을 우선한다.

같은 내용을 반복하거나 수십 개 테스트·파일을 본문에 열거하지 않는다. 전체 SHA256, 긴 로그,
보존 인벤토리는 핵심 판단에 필요한 경우에만 싣고 상세 증거는 경로로 남긴다. 실행하지 않은
검증을 통과했다고 표현하지 않으며 Fixture·실제 학습·실물 성공을 명확히 구분한다.

## 플랫폼 설계 및 확장 원칙

본 프로젝트는 SO-101 전용 Pick & Place 도구가 아니라, Real2Sim 환경 구성부터 데이터 생성·LeRobot 학습·평가·배포를 연결하는 확장 가능한 Workflow 플랫폼을 지향한다.

SO-101과 기존 Pick & Place는 최초 지원 구현 및 회귀 검증 대상으로 유지한다.

모든 코드 변경 시 다음 원칙을 따른다.

1. **책임 분리**: Builder는 환경 정보를, Task는 작업 의미를, Simulation은 실행 상태를, Workflow는 단계별 실행을, LeRobot은 표준 학습을 담당한다.
2. **단방향 의존성**: 하위 모듈이 UI·Workflow·Training을 역참조하지 않도록 한다. 순환 의존성을 새로 도입하지 않는다.
3. **최소 추상화**: 실제 사용 사례가 없는 범용 Framework, 중복 Manager, Registry, Wrapper, Factory를 만들지 않는다.
4. **기존 기능 재사용**: 공식 Isaac Lab, MuJoCo, Mimic, LeRobot 기능과 현재 구현된 Adapter·Reader·Writer를 우선 활용한다.
5. **Metadata 단일 해석**: 검증된 Workspace 정보를 Environment와 Task가 함께 사용하며, 각 단계에서 임의로 다시 해석하지 않는다.
6. **Task 확장성**: 새로운 Task를 추가할 때 공통 Workflow·UI의 Task별 조건문을 수정하지 않는 구조를 지향한다.
7. **Path 기반 실행**: 기존 Path 입력, 독립 Worker, Lock·Cancel·저장 수명 계약을 유지한다.
8. **안전한 확장**: 기존 Robot Mapping, 좌표계, 물리, Calibration, 저장 데이터 및 장치 안전 계약을 임의로 변경하지 않는다.
9. **Training 독립성**: LeRobot Dataset을 이용하는 일반 학습은 Workspace·Simulation·사용자 Task 파일을 필수로 요구하지 않는다.
10. **변경 최소화**: 기능을 추가하기 위해 프로젝트 전체를 재구성하지 않는다. 실제 문제와 필요한 경계가 확인된 부분만 수정한다.

구현 전에는 다음 문서를 확인한다.

- `docs/architecture/ENGINEERING_PRINCIPLES.md`: 모듈 책임, 의존성 방향, 확장 원칙
- `docs/architecture/TASK_SDK_DESIGN.md`: Metadata 기반 Task SDK 설계 및 지원 범위
- `docs/architecture/SYSTEM_FLOW.md`: 실제 구현된 실행 흐름
- `docs/architecture/WHERE_TO_EDIT.md`: 기능별 수정 시작점

설계 문서와 실제 코드가 다르면 실제 코드를 확인하고 차이를 보고한다. 계획된 기능을 구현된 기능처럼 취급하지 않는다.

### Task SDK 최소 구현 경계 (2026-10-22)

- TaskContext에는 이미 생성된 Backend만 전달한다. Context에서 Workspace 재load·Engine 생성/reset/step/close·관측 캐시를 추가하지 않는다.
- Task Template은 검증된 Workspace의 실제 ID와 명시적 기존 Task Definition만 사용한다. 이름으로 물체 역할·Robot ID·성공 조건을 추측하지 않는다.
- 사용자 Python은 신뢰한 코드이며 명시 Worker 실행에서만 읽어 실행한다. 목록/템플릿 생성/탭 전환에서 import하지 않는다.
- 새 Task의 기본 success/Mimic hook은 임의 성공으로 대체하지 않는다. 미지원은 명시적으로 실패시킨다.
- 현재 TaskContext는 단일 MuJoCo만 지원한다. 기존 Backend 성공은 기하학적 배치 판정이며 Annotation의 StableGrasp/Place·Mimic 전체 계약으로 승격하지 않는다.

## 현재 Script / Terms 경계 — 2026-10-23

- 공개 문법·지원 범위는 `docs/task_sdk/SCRIPT_GUIDE.md`, 내부 Reset/상태/API는 `docs/architecture/TASK_SDK_DESIGN.md`를 따른다. 과거 outputs 보고서는 당시 검증 이력이며 현재 지원 범위를 대체하지 않는다.
- `task_scripts.read_script/read_followup/followup_context/load_followup`은 선언/identity/실행 경계다. 실제 binding·Term 호출·Tracker/Event 상태는 기존 `SimulationAdapter.bind_followup/after_reset/record_frame`이 소유하며 Adapter가 파서를 역참조하지 않는다.
- Scene-only Source는 성공 Gate 없이 Builder Pose/명시 Pose·Reset을 기록한다. 기존 목록형 Source/FOLLOWUP은 호환 계약을 유지한다. Backend가 Physics/Reset RNG, Adapter가 Episode seed stream·Tracker, factory 반환 객체가 환경/Episode별 사용자 상태를 소유한다. TaskContext는 이를 빌린다.
- Annotation은 저장된 초기 상태/Signal·Manual 채택을, Mimic은 기존 reference 알고리즘/IK/생성을 소유한다. Native recipe의 Placement AND latched StableGrasp를 우회하거나 Blue strict IK 실패를 완화하지 않는다.
- Python은 UI 목록/탭 조회에서 실행하지 않는다. 명시 Worker의 entry bytes SHA 확인만 보장하며 외부 import pin/sandbox를 주장하지 않는다. Native LeRobot 학습/Hub는 이 SDK와 독립이다.
