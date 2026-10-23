## Native LeRobot 통합 — 2026-10-23

설치 LeRobot0.4.1/Torch2.7.0+cu128/HF Hub0.35.3/draccus0.10.0의 실제 소스·CLI를 사용한다. 새 의존성/업그레이드 없음. UI는 factory 소스만 읽고 Model을 초기화하지 않는다. Learning은 stdlib/HF/공식 LeRobot 및 기존 순수 contract/utils에 의존하며 Workflow/UI/Simulation을 import하지 않는다. Sim 전용 feature 검사는 Learning의 metadata 함수이며 엔진을 import하지 않는다. 설치 factory의 다른 Policy는 optional dependency가 별도로 필요할 수 있으며 목록 항목을 실동작 검증으로 취급하지 않는다.

# 현재 의존 경계

기준: 2026-10-22.

- UI → Paths/Operations → Worker → 기능 함수. UI가 목록 표시를 위해 모델을 로딩하지 않는다.
- Training/Validation → Native learning 입력 검사 → 공식 LeRobot/PyTorch. SO-101 Sim/Real 호환 검사는 workflow/policy.py에 별도로 남는다.
- Annotation → simulation/playback/control 및 기존 Adapter/I/O. 독립 Replay Stage 의존은 없다.
- Simulation 환경 생성 → 명시 Workspace/Scene/카메라/Physics 요구. Presentation.current를 조회하지 않는다. AppLauncher/환경 수명은 Worker/Runtime이 소유한다.
- Curation → 기존 Reader/Writer. 과거 Partition 역할은 선별/재분할 허가 조건이 아니다.
- Registry 모듈/등록/UUID 조회는 삭제됐다. artifact_identity는 실제 내용 fingerprint이므로 남는다.

남은 구조: workflow/adapters.py의 실제 엔진 Adapter, workflow/dataset.py의 형식 검증, workflow/episodes.py의 Conversion/역사 정보, workflow/mujoco.py의 Source/dispatch가 있다. Workspace Source Resume 및 Legacy 준비 helper의 Workflow 참조도 남는다. 전체 패키지가 완전한 단방향 그래프라는 뜻은 아니다.

물리·IK·Mimic·좌표·측정/명령 시점은 이번 Native 학습 호환 작업에서 바꾸지 않았다. [실행 흐름](SYSTEM_FLOW.md)을 기준으로 실제 수정 영역을 판단한다.


Script SDK 추가 edge: UI/Operations → `workflow.task_scripts`(AST/기존 Workspace 검사), Worker/Evaluation → `TaskContext`, 사용자 Script → `task_terms` → 전달된 Context/기존 Backend. 하위 Simulation이 Workflow/UI를 import하지 않는다. `TaskContext.adapter`는 상위에서 빌린 기존 Episode Adapter이며 import/생성/상태 update는 하지 않는다. StableGrasp·Annotation·Mimic은 원래 소유자 그대로다.

## Scene/Terms 정리 후 실제 경계 — 2026-10-23

Adapter/Annotation/DataGen → task_scripts 런타임 import를 제거했다. 파서는 AST/identity/Worker entry를, 기존 Adapter 메서드는 binding·Term 호출과 Episode 상태를 소유한다. Native/사용자 Mimic의 cfg 조립은 같은 subtask_configuration이며 sampling/IK/reference는 원본이다. 자세한 소유권과 Reset 순서는 [SDK 설계](TASK_SDK_DESIGN.md)를 따른다.

현재 AST 조사(함수 내부/조건부 import 포함)는 패키지148모듈,480→474edge, 최대 SCC34→30·순환 묶음4개 유지다. 동적 문자열 import/runtime call graph 전체를 복원한 수치가 아니다. 전체 순환 해소를 주장하지 않는다. 실제 변경/검증·남은 연결은 [정리 보고서](../../outputs/workflow_validation/architecture_cleanup/REPORT.md)와 structure.json을 따른다.
