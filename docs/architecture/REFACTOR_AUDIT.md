# SO-101 전체 구조 감사

2026-10-18 · 기준 `7d02d9cf029c27aac3b2c03767e85961bfb650cb` (`main`). 시작 시 remote main도 동일했다.
최초 작업 트리는 미추적 `MUJOCO_LOG.TXT`, `docs/minivla/test-output.txt`뿐이었다. 이 둘은 작업 대상이 아니다.

## 감사 방법과 범위

Phase 0에서 production 코드를 수정하지 않고 진입점→호출 함수→데이터/자원 소유자를 조사했다.
Git 추적 Python 189개(설치 패키지 모듈 122개)의 AST inventory, 함수 내부 import까지 포함한 edge와
최근 30 commit의 파일별 변경 빈도는 `outputs/workflow_validation/codebase_architecture/dependencies_before.json`에 있다.
정적 그래프는 실행 가능성이나 실행 순서를 증명하지 않는다. 조건부·lazy import를 포함하며 부모 package의
암묵적 초기화, `importlib` 문자열 및 외부 라이브러리 내부 edge는 완전하게 모델링하지 않는다.
CLI/subprocess와 IPC 흐름은 별도로 코드를 따라 확인했다. 전체 함수의 수학적 정확성 감사는 아니다.

필수 문서와 실제 코드를 함께 읽었다. `ARCHITECTURE.md`의 “기존 runs 결과는 이동하지 않는다” 같은 과거 설명은
현재 `DATA_FLOW.md`의 명시적 기존 자료 이동 기록과 구분한다. 현재 저장 계약은 `artifacts.output_location`이다.
Matt Pocock의 [codebase-design](https://github.com/mattpocock/skills/blob/main/skills/engineering/codebase-design/SKILL.md),
[improve-codebase-architecture](https://github.com/mattpocock/skills/blob/main/skills/engineering/improve-codebase-architecture/SKILL.md)는
작은 interface, locality, 실제로 필요한 seam, interface를 통한 검증의 참고 자료로 사용했다.
패키지 설치·새 UI·자료의 별도 후보 선택 절차는 도입하지 않았다. 사용자가 승인한 단계별 작업이 우선이다.

## 전체 책임과 좋은 구조

| 영역 | 실제 책임·소유자 | 호출 및 검증 지점 |
|---|---|---|
| `real2sim/ui.py`, `launcher.py` | Window/화면 설정, 별도 Runtime process 시작·중단 | UI 시작은 장치 연결이 아니다. file Bus가 명령을 전달 |
| `runtime.py` | 장치/엔진/카메라 연결 세션, 제어 tick, frozen Snapshot과 비동기 저장 수명 | `connect/start/stop/command/run`; 장치 fake와 IPC subprocess 검사 |
| `hardware.py`, `cameras.py` | SDK encoder/보정 identity/실측, RGB read interval·thread | `Follower.encode_goal/send`, `CameraReader.start/nearest/close` |
| `precision.py`, `profile.py` | 모터 raw↔URDF 보정, 승인 profile·simulation model 결정; scene profile 검증 | Mapper/Store/Calibration이 이미 구분돼 있음. 변환을 재작성하지 않음 |
| `capture_dataset.py`, `storage.py` | 측정/영상 timing 검증, Snapshot 원자 저장; durable JSON/file IPC | `prepare_snapshot`, `SnapshotDataset.save`, `Bus` |
| `real2sim/workspaces/` | Builder 소비, immutable package, physics/reset·Isaac cfg | `builder.import_scene`, `package.load`, `environment.make_config` |
| `simulation/` | MuJoCo scene/physics/IK, 공통 backend interface | `MujocoBackend`, `PolicyEnvironment`, `ik.solve_selected` |
| `workflow/operations.py`, `artifacts.py` | 계획/worker 취소/결과 검증·등록; Registry/저장 경로·입력 조사 | `Operations.plan/run`, `execute_request`, `Registry.register/verify` |
| `dataset.py`, `episodes.py`, `curation.py` | 공통 HDF5+영상, Legacy/Source 계보·분할, 검토/선별 | `DatasetWriter.commit`, `split_episodes`, `EpisodeReader`, `export_subset` |
| `unified.py`, `mimic.py`, `datagen.py`, `adapters.py` | 공통 dispatch, 공식 생성/RNG, 성공 슬롯/실행 조정, 엔진별 관측·IK | 공통 알고리즘을 엔진별로 다시 구현하지 않는 실제 seam |
| `policy.py`, `evaluation.py`, `hub.py` | 기존 학습/추론/출판 interface 연결 | GPU·장치·네트워크 실제 작업은 worker와 명시 실행에서만 |
| `learning/minivla/` | Qwen/Flow-DiT, 데이터 sample·통계, trainer/checkpoint/rollout | CPU tensor·Resume/bytes/partition 검사와 실제 GPU 학습은 구분 |
| 루트 joint/contract/spec, `representation.py` | joint 순서/단위/명령 의미·로봇 모델 증거 | 카메라나 Registry가 임의로 좌표를 다시 해석하지 않음 |
| `devices/`, `tasks/`, `assets/`, `utils/` | 기존 Isaac device/task/robot cfg, video·kinematics 등 구현 | 엔진 process에서만 Isaac 등록을 사용; 공통 video writer는 central commit |
| `scripts/`, `tests/` | 의도적인 CLI/진단 진입점, 계약·UI·엔진별 검증 | Workspace CLI는 production 경로, `--task` 고정 씬 도구는 명시 호환 경로 |

실제 Runtime·Worker·학습 흐름과 입출력은 [SYSTEM_FLOW](SYSTEM_FLOW.md)에, 최종 파일별 지도는
[PROJECT_STRUCTURE](PROJECT_STRUCTURE.md)에 정리한다. 미구현 기능을 실제 경로처럼 표시하지 않는다.

## 확인된 의존성 문제와 보류 후보

위치는 기준 commit의 함수 기준이다. 최종 위치는 연결된 파일에서 함수명으로 찾는다.
우선순위는 개선 가치, 위험은 동작을 보존하며 옮기는 비용이다. 높은 우선순위가 즉시 변경을 뜻하지 않는다.

### 1. High / 낮은 위험 — worker 요청 검증의 역방향 import

- 근거: `workflow/mujoco.py:13`은 `operations.mujoco_request/MODEL_ROOT/ROOT`를 import한다.
  `operations.execute_request:627`는 다시 `.mujoco.execute`를 lazy import한다. 실제 양방향 edge다.
- 전파: worker protocol/renderer/입력 검증 수정 때 GUI의 orchestration 파일과 engine worker를 함께 알아야 한다.
- 제안 interface: 엔진/Qt 없는 `workflow.worker_request.mujoco_request(...)`와 기존 stage 이름 표.
  기존 검증 본문을 이동하고 `operations`는 동일 함수 객체를 re-export한다. worker 상수는 원래 소유자인
  MiniVLA config/Workspace package에서 직접 읽는다. 단순 전달 wrapper는 만들지 않는다.
- 필요한 이유: planner와 worker가 동일 검증을 실제로 호출한다. worker implementation을 GUI에 import하는
  대안은 엔진 분리 계약을 위반한다. orchestration을 통째로 새 Service로 감싸는 대안도 이득이 없다.
- 보존: args/반환 dict/옵션 객체/예외 종류·문구·검사 순서/CLI choices와 출력을 모두 유지.
- 검증: 기존 planner/worker/UI tests, 독립 요청 계약·CLI help, 새 process의 engine/Qt 미import.

### 2. High / 낮음~중간 위험 — 파일 identity 계산이 Registry에 묶임

- 근거: `learning/minivla/data.py:45`의 `load_manifest`가 `workflow.artifacts.fingerprint`를 import한다.
  해당 함수(`artifacts.py:124`)는 파일/폴더·HDF5 외부 영상 bytes를 읽는 규칙이며 Registry ID를 사용하지 않는다.
- 전파: 학습 데이터 bytes 검사를 이해하려면 Registry·Workspace·artifact inspection 모듈까지 따라간다.
- 제안: `utils.artifact_identity.fingerprint(path)`에 구현을 단 한 번 두고 기존 artifact 경로는 re-export.
  실사용 학습 호출자는 낮은 모듈을 사용한다. 순수 JSON 보정 fingerprint와는 의미가 달라 합치지 않는다.
- 필요한 이유: 학습과 Registry가 동일 bytes 계약을 사용한다. `datasets/__init__.py`는 LeRobotRecorder를
  eager import하므로 그 아래에 넣으면 단순 hash에 LeRobot 의존이 추가된다. `storage.py`의 IPC/원자 쓰기에
  HDF5 episode 지식을 넣는 대안도 책임이 다르다. 기존 `utils` 안에 책임이 명시된 모듈을 둔다.
- 보존: 파일명/상대경로 포함 순서, `.cache` 제외, 1MiB chunk, 중복 영상 제거, 오류/빈 입력 동작과 SHA.
  Registry 저장/경로 및 학습 Dataset/정규화 계약은 바꾸지 않는다.
- 검증: 기존 bytes 변조/Resume/MP4 tests, 경로·영상 tamper 특성화, 원래 함수와 고정 입력 결과 일치.

### 3. High / 높음 — Workspace 구성 규칙과 Workflow의 양방향 의존

- 근거: `simulation/mjcf.py:build_scene`→`workflow.tasks.configured`; Workspace reset/demo/environment도 같은
  task 함수를 호출한다. `tasks.configured`는 다시 Workspace metadata/physics를 사용한다.
- 전파: active object/reset 규칙 추가 시 compiler·Runtime·Mimic·Source·Curation·eval까지 영향 가능.
- 후보: workspace task semantics(선택/검증/configured)와 artifact task stamping/평가를 구분하는 interface.
  그러나 `SO101_TASK_SELECTION` 환경, recorded reset·hash·object 제외가 한 계약을 이룬다.
- 보존/검증: scene SHA·task/active order·원본 copy·RNG 소비·두 엔진 cfg/Replay와 과거 데이터가 동일해야 한다.
- 결정: **추가 검증 필요, 이번 이동 보류**. 실제 두 엔진 재생을 하지 않고 전체 semantic 소유권을 옮기지 않는다.
  현재 `tasks.py`의 Python-only 규칙은 이미 Qt 없이 테스트할 수 있으므로 분리 자체의 이득도 제한된다.

### 4. High / 높음 — Runtime lifecycle와 Snapshot 비동기 상태 집중

- 근거: `Runtime.__init__/disconnect_hardware/connect/command/run`이 장치·thread pool·save future·최종화를 소유한다.
  `command`에서 state/image copy를 freeze하고 `save_snapshot_requests`가 single writer에 순차 제출한다.
  Disconnect는 finish를 같은 writer 뒤에 예약하고 `run.finally`가 shutdown을 기다린다.
- 전파: 카메라 종류, capture 정책, connect/close 변경이 이 자원 수명과 연결된다.
- 후보: Snapshot 저장 세션의 interface로만 시작/요청/결과/종료를 다루는 내부 책임. 하지만 장치 state와
  승인·타이밍을 분리할 충분한 실제 장치 회귀 근거가 없으며 새 manager는 상태 소유자를 늘릴 수 있다.
- 결정: **보존**. fake/IPC 검사를 실시하되 수명·thread/signal·명령 순서를 이동하지 않는다.

### 5. Medium / 높음 — Precision 순수 계산·승인 I/O 공존

- 근거: `precision.py`는 `fit_joint/review_joint/PrecisionMapper`, `ProfileStore`, `Calibration`을 포함한다.
  mapping compatibility가 URDF/Git 증거와 연결되고 root mapping이 metadata 해석에서 precision을 lazy import한다.
- 후보: 수학과 Store를 나눌 수 있지만 기존 공개 함수/metadata·calibration 승인이 결합된 이유가 있다.
- 전파/보존: raw branch/SDK int encoding/관측 overshoot/URDF residual binding/옛 profile identity 전부 유지.
- 결정: **보존**. 계산과 I/O는 이미 함수/class seam으로 검사 가능. 실제 장치 없이 수학 책임을 옮기는 이득이 작다.

### 6. Medium / 중간~높음 — UI 상태/실행 준비 집중

- 근거: `WorkflowPanel.refresh/ready/execute/stage_options/configure_stage`가 공용 widget와 stage별 기억 상태를 연결.
  stage별 folder/선택 ID를 공유 combo의 이전 값과 구분하는 현재 계약이 중요하다.
- 전파: 새 Stage는 operations/stages/option_help/UI/presentation/worker를 각각 검토해야 한다(서로 다른 책임).
- 후보: 비시각 준비 snapshot을 분리할 여지는 있지만 generation token·thread callback·widget visibility가 연결된다.
- 결정: **UI 코드 무변경**. 과거 Screen 분할을 반복하지 않는다. Qt interface 검사와 동일 캡처로 보존 확인.

### 7. Medium / 중간 — Episode/Registry 기능 집중

- 근거: `artifacts.Registry`가 inspect/register/verify/collection discover/origin 복사/promote rollback을,
  `episodes.py`가 기록/partition/lineage/학습 prepare를 담당한다. `dataset.py`/`curation.py`와 읽기 edge가 왕복한다.
- 후보: 계보와 partition을 별도 도메인으로 옮길 수 있으나 UNKNOWN 처리·Source 증거·저장 경로 규칙은 서로 다르다.
- 보존/검증: Registry IDs/bytes/부모, 누수 거절, 불완전 save rollback/영상 경로, 기존 manifest identity.
- 결정: 파일 identity만 2번에서 좁게 이동. 저장 경로/partition의 소유권 이동은 **보류**.

### 8. Medium / 높음 — DataGen 두 실행 방식의 상태 소유

- 근거: `run_mujoco`는 독립 MjData thread, `_isaac_loop`는 coroutine mailbox와 한 batch physics tick이다.
  둘 다 같은 `MimicGenerator`, `GenerationScheduler.finish`, `DatasetWriter`를 쓴다.
- 전파: 실패 진단은 adapter evidence→attempt→scheduler/presentation, 새 IK는 solver→adapter 연결.
- 결정: **유지**. 같은 while 문처럼 보여도 다른 수명/동시성이다. 통합 executor는 실제 seam을 지우고 RNG/정확한
  성공 수·중지/Resume를 바꿀 위험이 있다. lock/예약 실패/타이밍 특성화는 기존 tests에서 확인한다.

### 9. Low / 중간 — UI의 CameraView 소유권

- 근거: `real2sim.ui`가 `workflow.view.CameraView`를 재사용한다. view 자체는 QLabel/QPixmap이며 engine/worker를 소유하지 않는다.
- 후보: 공통 UI package로 이동할 수 있으나 현재 단일 widget를 위해 import 호환 계층/새 폴더가 생긴다.
- 결정: **유지**. 단일 구현 재사용의 이점이 있고 UI 회귀 위험 대비 이동 가치가 작다.

## 디렉터리 평가

`real2sim`(장치/보정 세션), `simulation`(엔진 구현), `workflow`(artifact 작업), `learning/minivla`(모델)은
기능을 찾는 데 유용한 구분이다. `real2sim/workspaces`는 독립 Builder의 소비자이므로 scene 제작으로 바꾸지 않는다.
`assets`는 Python engine cfg, 최상위 `assets/SO101`는 실제 URDF/USD/mesh라 서로 다른 역할이다.
`datasets` Python package는 recorder adapter이고 최상위 사용자 datasets 폴더와 다르다.
`utils`에는 video/voice/kinematics가 있으므로 범용 모음의 과성장을 막고 새 기능은 이름과 실제 공유 근거를 요구한다.

이번에는 1·2번의 **함수 구현 소유권 이동**만 검증 후 적용한다. 패키지 통째 rename/reparent, 데이터 폴더 이동,
CLI 위치 변경, 새 launcher·DI 계층은 필요하지 않다. 그 밖의 디렉터리는 위치가 문제임을 입증하지 못했다.
추가 요구의 폴더 이동 허용을 이동 의무로 해석하지 않는다.

## 변경 빈도와 테스트 표면

최근 30 commit에서 `workflow/ui.py`, `scripts/mimic/generate_dataset.py`, `workspaces/environment.py`는 각11회,
`operations.py`, converter는10회, `mujoco.py`와 workflow test는9회 변경됐다. 큰 commit의 횟수이며 버그 빈도가 아니다.
변경이 잦아도 UI/수학을 먼저 옮기지 않고 실제 역방향 import를 우선한다.

기존 tests는 fake device/calibration/IPC, Snapshot timing·원자성, Builder hash/물성, MuJoCo CPU physics,
Mimic parity/worker 독립성/저장 예약·rollback, Train/Val 누수/Resume/통계, Qt 표시/선택/잠금에 집중되어 있다.
Qt tests 중 private widget 접근은 화면의 확정된 동작을 고정하는 성격도 있으므로 무조건 제거하지 않는다.
새 검사는 worker request와 bytes identity라는 호출자 interface를 통해 수행하고, AST 검사는 import 방향만 담당한다.
실제 모터/노출 timestamp/GPU 학습/성공률/장시간 엔진 품질은 CPU/Qt tests로 증명할 수 없다.

## 별도 관찰 사항

- Python 정적 그래프에는 lazy import를 포함한 52-module SCC와 Isaac action cfg↔class의 2-module SCC가 있다.
  모두 즉시 import 실패라는 뜻은 아니다. 1·2번 수정만으로 전체 cycle이 사라진다고 주장하지 않는다.
- `test_provenance.py::test_ui_is_api_free_and_loads_profile`의 자식 process는 기본 Registry를 사용할 수 있다.
  기준선에서는 보류했고, Phase 4에서 자식 process에 임시 Registry를 명시한 뒤 기존 assertion으로 통과했다.
- Snapshot Window test에도 기본 Registry를 생성하는 경우가 있어 해당 test에 임시 Registry를 명시했다. production UI는 바꾸지 않았다.
- 문서의 옛 `B=1 기본`, 제거된 joint_control test 등은 현재 코드와 차이가 있다. 이 감사는 알고리즘 변경으로 해결하지 않는다.

최종 적용/보류 결과와 검증 한계는 [REFACTOR_VALIDATION](REFACTOR_VALIDATION.md),
구체적인 전후 import와 대안은 [REFACTOR_BEFORE_AFTER](REFACTOR_BEFORE_AFTER.md)에 기록한다.
