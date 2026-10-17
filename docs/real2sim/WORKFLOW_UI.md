# Workflow 작업 화면

## 리팩터링 기준과 기존 계약

기준은 `52c4e06c15ff363a300ed31b346b0a34b5740031`이다. 시작 시 remote main과 차이가 없었고,
미추적 `MUJOCO_LOG.TXT`, `docs/minivla/test-output.txt`는 사용자 파일로 보존한다.

아래 표는 변경 전 `operations.STAGES`, `plan/run`, `stages`, 각 worker를 조사한 실행 계약이다.
Navigation 순서와 별개로 이 계약을 유지한다.

| 작업 | 입력 type | 주요 옵션 | Backend / worker | 등록 출력 | 진행 / 제어 |
|---|---|---|---|---|---|
| Source | workspace | port, episodes, episode/reset time, resume_data, identity, device | Isaac workspace CLI / MuJoCo worker | source; sources | presentation/status.json, save/discard/quit |
| Replay | source/annotated/datagen | renderer, device | 공통 unified, Isaac/MuJoCo | replay; evaluations | presentation/status.json, resume/pause/step |
| Annotation | source | auto, renderer, device | 공통 unified, Isaac/MuJoCo | annotated; annotations | resume/pause, manual mark/skip |
| DataGen | annotated | target, num_envs, solver, failure policy, seed, attempts, resume | 공통 unified, Isaac/MuJoCo | datagen; datagen | status/env/failure/timing, resume/pause/step |
| Conversion | source/annotated/datagen | repo_id, include_failed | 기존 단일 Converter | dataset; datasets | process.log |
| Training | dataset | architecture, steps/batch/save, resume; MiniVLA val/config/schedule | policy.train, 기존 trainer | policy; models | process.log, MiniVLA metrics.jsonl |
| Sim Rollout | policy/hf_policy | tasks, episodes/max_steps, seed, checkpoint | evaluation / MuJoCo worker | sim_eval; evaluations | presentation/status.json, resume/pause/step |
| Real Rollout | policy/hf_policy | port/ID, cameras, enable_motion, episodes | 기존 real evaluation worker | real_eval; evaluations | presentation/status.json, resume/pause; 사람 검토 |
| Dataset 업로드 | dataset | repo_id, private, create_repo, revision/branch | hub.publish | hf_dataset; evaluations | process.log / receipt |
| Policy 업로드 | policy | repo_id, private, create_repo, revision/branch | hub.publish | hf_policy; evaluations | process.log / receipt |
| Curation | source/annotated/datagen | 선별, 원본 group 분할 | 기존 CurationDialog | 별도 filtered/splits, Registry | 자체 비동기 분석/저장 |
| 결과물 관리 | Registry | 검색/상세/계보/단계 이동 | 읽기 전용 | 등록물 변경 없음 | checkpoint는 policy 하위 파일 |

기존 Window의 workspace/python/mode/backend/task_selector/service/busy 및 browse/select workspace 호출을
확인했다. 상위 Window는 공유 Context를 사용하며 얇은 호환 getter는 유지한다.

## 구조와 수정 위치

`WorkflowPanel`은 1083줄에서 217줄로 줄었다. 화면 전환·생명주기·상위 Window 연결만 담당한다.
Stage 옵션을 수집하거나 실행/영상/로그를 직접 구현하지 않는다.

| 파일 / 컴포넌트 | 책임 |
|---|---|
| `workflow/ui.py` | Navigation, 화면 factory/cache, 명시적 Artifact ID 전달, Window 호환 getter |
| `ui_components/context.py` | Workspace/Backend/Python/Mode의 단일 소유자, 기존 ui.json 복원, 실행 중 잠금 |
| `ui_components/execution.py` | plain-data snapshot, debounce/세대 토큰/백그라운드 준비 검사, 중복 실행 방지, 취소/완료 |
| `ui_components/navigation.py` | 5개 그룹. Operations stage/type 정의와 독립적인 화면 순서 |
| `ui_components/screens/` | 각 작업의 입력·기본/고급 설정·전용 제어. Training은 모델별 독립 인스턴스 |
| `ui_components/artifact_picker.py` | STAGES 기반 공통 필터, 선택 ID 보존, Metadata/Lineage, 기존 등록/import 경로 |
| `ui_components/monitor.py`, `preview.py`, `training_metrics.py` | run별 상태/로그/결과, Scene 카메라, 기존 실제 Loss reader/plot |
| `screens/artifacts.py`, `screens/curation.py` | 읽기 전용 Registry 탐색, 기존 CurationDialog 임베딩 |

실행 흐름은 `StageScreen.snapshot → Controller.execute → Operations.run → Operations.plan → 기존 Worker → Registry.register`
이다. Controller에서 엔진·학습·업로드 알고리즘을 실행하지 않는다. `readiness`와 MiniVLA Validation 후보 검사는
한 개의 백그라운드 executor를 사용한다. 대기 요청은 최신 snapshot 하나만 유지하고, 화면/입력 변경 또는 Run 시작 때
generation을 올려 이전 결과를 폐기한다. 활성 Run의 deep copy는 이후 위젯 변경에 영향받지 않는다.

`SettingsForm` 하나가 옵션 editor를 소유한다. 기본/고급 영역으로 나눠도 같은 값을 중복 입력받지 않는다.
합성 순서는 schema/모델 기본값 → 사용자가 체크한 SettingsForm 값 → 전용 입력(Task/Port/Resume 등) → 공유 Context다.
전용 Architecture/Validation/Checkpoint/저장 이름은 각 Screen에서 최종 합성한다. MiniVLA trainer 내부 우선순위는
기존의 체크된 일정 값 > 선택 JSON > 기본 JSON이다. Resume은 기존 저장 config의 Batch/Accumulation 검증을 거친다.

## 화면 접근과 기본/고급 설정

왼쪽 Navigation에서 작업을 직접 고른다. 진입·완료만으로 다음 작업을 자동 실행하지 않는다.
상단 Workspace/Backend는 Real/Sim Window와 공유한다. Python/Mode는 `고급 실행 환경`에서 펼친다.
각 입력의 ID/partition 요약을 확인하고 `입력 Metadata / 경로 / Lineage 상세`에서 전체 경로를 확인한다.

| 그룹 | 기본 작업 | 접힌 고급 영역 |
|---|---|---|
| 01 데이터 준비 / Source | Workspace, Task/Active Objects, Leader port, Episodes/시간, 저장 이름 | Task 조합/문장, Resume Data, identity/device/renderer/output |
| Replay / Annotation | 입력·상속 Task, 영상/재생 제어, Auto/Manual | device/renderer/output. Manual만 Mark/Skip 생성 |
| DataGen | 성공 목표, 환경 1/2/4/8, 두 IK solver, Strict/Hold/Best Effort, 저장 이름 | Resume, seed, attempts, renderer/device/output |
| Curation | 입력 선택 후 `선택 데이터 열기`, 목록/동기 영상/포함·제외·저장 | 기존 궤적 분석/품질/Source Group 분할 및 수동 배정 |
| LeRobot 변환 | 입력, repo_id, 저장 이름/예상 위치, 진행 로그 | include_failed/output |
| 02 모델 학습 | 모델별 독립 Train/Batch/Steps/Save, Resume 체크 및 선택, 저장 이름 | ACT/SmolVLA base/workers/map/options; MiniVLA JSON/Patience/Legacy |
| MiniVLA | 독립 Val, Batch8, Steps50000, Val1000, Save2500, Early Stop OFF | Accumulation UI 없음. 신규 1, 옛 Resume 저장값 유지 |
| 03 모델 평가 | Policy, Workspace/Task, Episodes/Max Steps; Sim의 정확한 Best/Last/Step 및 Seed | renderer/device/output; Real calibration/FPS |
| Real Rollout | port/ID/camera mapping/Enable Motion OFF 및 별도 승인 | 기존 worker의 calibration/frame/finite/freshness/encoded goal 검사 유지 |
| 04 업로드 | Artifact/Repo ID/공개 범위/신규 또는 기존 Repo 확인 | Revision/Create Branch/Output |
| 05 결과물 관리 | 이름·ID·Type·Task·Backend·Workspace·Partition·Status 검색, 상세/계보/폴더 | Policy 내부 checkpoint 목록과 명시적 ID 기반 작업 이동 |

Train/Convert/Upload에는 CameraView를 만들지 않는다. Loss는 MiniVLA Training에만 있으며,
새 학습을 실행하면 첫 metrics 기록을 기다린다. 실행 전 자동 연결된 이전 학습 기록은 기존 기록임을 표시한다.
ACT/SmolVLA는 실제 trainer 로그를 표시한다. 새 데이터가 생기기 전의 Loss/Val을 숫자로 채우지 않는다.

Camera는 Source/Replay의 Scene camera, Annotation/DataGen/Sim의 Overall + Scene camera,
Real의 실제 camera를 사용한다. 표준 Scene은 기존 Env0 Side/Wrist와 Overall이며 환경별 4분할은 없다.

실행 중 Navigation/공유 Context/화면 설정을 잠근다. Stop은 기존 cancel/SIGINT 경계를 사용한다.
worker의 FINISHED는 UI에서 결과 검증 중으로 표현하고, Operations 완료 뒤 SUCCEEDED/FAILED/STOPPED를 표시한다.
원본 로그/결과 버튼은 해당 run의 파일/결과가 있을 때만 활성화한다.

결과물 관리의 Checkpoint는 Registry ID를 새로 만들지 않는 하위 파일이다. MiniVLA 파일 선택은 정확한 경로를
Sim 평가 옵션으로 전달한다. LeRobot 중간 checkpoint 디렉터리는 평가 화면의 기존 폴더 등록/import로 검증해야 한다.
Explorer에는 파일 이동/삭제/Registry 수정 기능이 없다.

## 제거·유지 근거

`rg`로 실제 코드·상위 Window·테스트 호출을 확인했다. 숨긴 `real_enabled`, `real_side/wrist`, `options_help`,
과거 section/stage combo/list 동기화, 대량 `configure_stage` show/hide, 공통 화면의 recording/graph 버튼을 제거했다.
상위 Window는 이 숨긴 위젯을 사용하지 않았다. 기존 테스트의 실질 계약은 새 Screen/Monitor 기준으로 옮겼다.
Retired objectName만 가리키던 stylesheet도 제거했다. 옵션 설명 schema `option_help`는 SettingsForm이 사용하므로 유지한다.
CameraView/LogTail, CurationDialog, TaskSelector, TrainingLossPlot 본체와 실제 옵션/등록/import 기능은 재사용한다.
TaskSelector는 기본 API를 유지하며 compact 모드에서 선택적인 Task 상세만 접는다.

Window의 workspace/python/mode/backend/task_selector getter와 browse/select workspace, service/busy는 유지했다.
Window의 실제 호출은 Context 및 `scene_tasks()`로 옮겼다. 상위 Backend의 programmatic 변경도 실행 중 잠금을 우회하지 못한다.

## 검증 및 범위

명령과 로그는 [검증 가이드](TESTING.md#workflow-화면-분리-2026-10-17) 및
`outputs/workflow_validation/ui_architecture/`에 있다. 화면/Curation149, USD/Window75, 공통 계약83,
MiniVLA CPU38개가 통과했다. 합성 Registry/Mock Worker/임시 Workspace를 사용했다.
기존 Registry63개를 읽은 화면11개 캡처는 worker 실행0이며 Registry JSON/UI 환경설정 byte 보존을 검사했다.
이는 신규 GPU 학습·실제 Isaac DataGen·Hub 업로드·실물 로봇 검증이 아니다. 해당 작업은 이번 요청에서 실행하지 않았다.

초기 테스트의 USD 경로 누락은 process 환경을 분리해 해결했다. 기존 모드 비교 테스트에는 고정 저장 이름을 지정했고,
Capture profile 라우팅 fixture는 실제 Task 검증 fixture와 분리했다. 초기 캡처의 QWidget.scroll 동명 메서드 처리도
검증 runner에서 고쳤다. 실패 로그는 보존했다. Production 모델/물리/IK/Mimic/Converter/Registry 코드는 변경하지 않았다.

## 내가 직접 확인할 것

1. GUI를 다시 열어 5개 그룹에서 DataGen·MiniVLA·Curation·Real Rollout·결과물 관리로 이동한다. 기본/고급 설정과 입력 ID를 확인한다.
2. `StageScreen.snapshot`, `WorkflowExecutionController.request_ready/execute`, `ArtifactPicker.select`를 읽는다.
3. MiniVLA의 Batch8/독립 Val, DataGen의 IK Policy, Checkpoint→Sim 이동의 정확한 파일, 실행 중 잠금과 run별 로그를 확인한다.
