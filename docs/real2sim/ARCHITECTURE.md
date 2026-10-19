# SO-101 Real2Sim Architecture

이 문서는 현재 `so101-isaaclab`의 Real2Sim/Workspace/Workflow 구조를 빠르게 이해하기 위한 상위 수준 지도다.
세부 구현보다 **어떤 책임이 어느 모듈에 있는지**를 우선 설명한다.

전체 Repository의 파일별 책임은 [PROJECT_STRUCTURE](../architecture/PROJECT_STRUCTURE.md),
함수·입출력을 따라가는 설명은 [SYSTEM_FLOW](../architecture/SYSTEM_FLOW.md),
실제 import와 남아 있는 역참조는 [DEPENDENCIES](../architecture/DEPENDENCIES.md)를 본다.

## 1. 전체 구조

```text
Physical SO-101
  ├─ Leader / Follower
  └─ Real cameras (side, wrist)
           │
           ▼
source/.../real2sim/
  ├─ hardware.py / cameras.py
  ├─ runtime.py
  ├─ capture_dataset.py
  ├─ profile.py
  ├─ scene.py
  └─ storage.py
           │
           ├─ REAL Snapshot → 독립 Builder 보정/비교/export
           ├─ Isaac/MuJoCo Real/Sim 프리뷰와 Teleop
           └─ Profile/Workspace 로드
                         │
                         ▼
source/.../real2sim/workspaces/
  ├─ package.py
  ├─ builder.py
  ├─ environment.py
  ├─ physics.py
  ├─ reset.py
  ├─ demo.py
  └─ mimic.py
                         │
                         ▼
Published Workspace Environment
                         │
                         ▼
source/.../workflow/
  ├─ artifacts.py / operations.py / stages.py
  ├─ dataset.py / unified.py
  ├─ mimic.py / mimic_reference/ / datagen.py / adapters.py
  ├─ mujoco.py / episodes.py
  ├─ tasks.py / task_ui.py
  ├─ policy.py / evaluation.py / hub.py
  └─ ui.py / view.py / presentation.py
                         │
                         ▼
Demo → Annotation → Datagen → Convert → Train → Evaluate
  │                │
  ▼                ▼
source/.../simulation/          source/.../learning/minivla/
  mujoco_backend.py              config.py / data.py / train.py
  mjcf.py / workspace_scene.py   backbone.py / head.py / policy.py
```

Workflows의 공통 worker·Builder workspace import·HDF5 import는
`scripts/real2sim/workflow.py`의 subcommand로 실행한다. 실물 상주 Runtime,
Isaac AppLauncher가 필요한 Mimic/teleoperation, 독립 물리 검사는 서로 다른
실행 경계라 별도 진입점을 유지한다.

## 2. Real2Sim core

경로: `source/soarm101_lab/soarm101_lab/real2sim/`

- `runtime.py`: real-only 또는 Isaac/MuJoCo와 실제 장비/카메라를 연결한다. Teleop, 실측 pose 프리뷰, snapshot 요청과 연결 세션을 담당한다.
- `hardware.py`: follower 등 실제 SO-101 하드웨어 경계.
- `cameras.py`: 실제 카메라 frame 획득과 capture timestamp 경계.
- `core.py`: Real/Sim 공통 수치 처리, pose/joint 관련 핵심 helper.
- `profile.py`: camera, workspace, robot/object/scene 파라미터와 provenance를 가진 Real2Sim profile의 검증/로드.
- `capture_dataset.py`: 원본 REAL 두 이미지/실측 관절/타이밍을 검증하고 세션의 공유 모델·보정 및 개별 snapshot을 원자적으로 저장한다.
- `scene.py`: profile을 Isaac Sim scene/USD 표현으로 적용하는 계층.
- `storage.py`: atomic 저장, 디렉터리 fsync, 파일 IPC와 runtime 소유권 잠금.
- `ui.py`: Real2Sim UI. Isaac runtime과 별도 프로세스로 동작하며 workspace의 file IPC를 사용한다.

진입점:
- `scripts/real2sim/real2sim_ui.py`
- `scripts/real2sim/real2sim_sim.py`

옛 paired capture/continuous recorder/변환/Reference import/API agent/수치 fitting은 제거했다.
scene 재구성·landmark fitting·render 비교·dual export는 독립 `real2sim-builder`에서 수행한다.

## 3. Workspace Environment

경로: `source/soarm101_lab/soarm101_lab/real2sim/workspaces/`

Real2Sim으로 맞춘 한 실제 작업공간을 재사용 가능한 environment package로 만드는 계층이다.

- `package.py`: versioned workspace package 로드/메타데이터.
- `builder.py`: Builder export 검증 및 기존 런타임용 asset 참조 구성.
- `environment.py`: workspace package를 Isaac Lab environment 설정에 연결.
- `physics.py`: mass/friction/collision 등 simulation physics 설정.
- `reset.py`: workspace/object reset 정책.
- `demo.py`: source demo와 workspace representation contract 연결.
- `mimic.py`: 개발용 native Mimic 설정/물성 비교. Production 생성은 `workflow.mimic`을 사용한다.

기존 pick-place 환경을 통째로 대체하는 것이 아니라, **기존 Isaac Lab 구조를 유지하면서 workspace를 선택적으로 적용**하는 것이 원칙이다.

## 4. Unified workflow

경로: `source/soarm101_lab/soarm101_lab/workflow/`

이 계층은 기존 teleop/Mimic/datagen/LeRobot 흐름을 하나의 UI/operation layer로 묶는다.

- artifact 추적: `artifacts.py`
- 실행 operation: `operations.py`, `stages.py`
- task 정의/UI: `tasks.py`, `task_ui.py`
- policy train: `policy.py`
- evaluation: `evaluation.py`
- Hugging Face: `hub.py`
- UI: `ui.py`, `view.py`, `presentation.py`

두 backend는 같은 WorkflowPanel, Operations, Registry, Presentation을 사용한다. `operations.plan`은 명령 계획,
`artifacts.output_location`은 UI/worker의 공통 저장 경로 계약이다. 실행 로그는 `runs/<Run ID>`, 실제 새 결과는
sources/annotations/curation/datagen/datasets/models/evaluations 아래 사용자 이름으로 분리한다.
Registry ID/Parent/Path/SHA256은 유지하고 사람에게 보이는 `display_name`만 별도로 기록한다.
기존 결과의 명시적 관리 경로 이동 이력과 현재 경로·Resume·탐색 계약은
[DATA_FLOW](DATA_FLOW.md#단계별-저장-경로입력-선택--2026-10-17)에 있다. 새 실행이 기존 결과를 자동 이동하지 않는다.
`execute_request`는 worker dispatch다. Workspace Replay/Annotation/DataGen은 형식과 엔진에 관계없이 `unified.execute`로 간다.
`dataset.EpisodeDataset`가 공통/과거 형식을 읽고, 과거 입력만 `migrate`로 별도 공통 파일에 Import한다.
유효한 공통 입력은 그대로 사용한다. `unified.replay`는 관절 목표의 실제 재생과 집기/배치를 검사한다.

`mimic.py`는 설정·시도별 RNG·성공 슬롯 예약, `mimic_reference/`는 공식 알고리즘,
`datagen.py`는 스레드/vector 실행 조정, `adapters.py`는 엔진 상태·IK·물리·관측 경계다.
Worker마다 읽기 전용 Source Pool을 한 번 로드하고 시도별 Generator를 만든다. 상태 lock과 Writer lock을 분리한다.
`dataset.DatasetWriter`만 공통 episode를 commit한다. native Isaac Source도 메모리의 EpisodeData를 직접 이 Writer에 넘긴다.

`mujoco.py`에는 Source·정책 학습·rollout·명시 Import가 남으며 별도 Replay/Annotation/DataGen 구현은 없다.
MuJoCo 요청 검증은 `worker_request.py`를 planner/worker가 공유하며, 파일·영상 bytes identity는
`utils/artifact_identity.py`를 Registry/데이터/학습이 공유한다. 기존 import 이름은 동일 객체로 유지한다.
`episodes.py`는 Source 기록·과거 NPY 읽기/학습 호환·Source partition/lineage를 담당한다.
LeRobot 수치/영상 mapping은 `scripts/tools/convert_isaac2lerobot.py` 한곳에서 수행한다.
`learning/minivla/`는 모델·학습·추론을, `policy.py`/`evaluation.py`는 ACT/SmolVLA/MiniVLA의 기존 인터페이스를 연결한다.

`RuntimeLauncher → real2sim_sim.py`는 Real/Sim과 Snapshot Capture가 공유하는 장치 연결이다. real backend는 Sim 없이 실제 장치를 연결하며, 비교 시 선택한 Isaac/MuJoCo 엔진을 별도 프로세스로 실행한다. Inspect UI는 제거했고 Profile/device 설정은 유지하고 scene fitting은 독립 Builder에 둔다. Workflow stage들은 각자 엔진 worker를 실행하며 이 상주 runtime을 사용하지 않는다. MuJoCo worker는 Isaac/omni/pxr를 import하지 않는다.

두 엔진의 Workspace DataGen은 같은 공식 Mimic1.0.15 생성 본문을 사용한다. SO101 단일 arm/두 subtask/SkillGenfalse 범위다.
CLI의 `--workspace`는 UI와 같은 core를 호출한다. `--task`만 사용하는 기존 고정 씬 Isaac 예제는 명시적 개발/호환 경로이며,
Workspace 검증·새 공통 Production 경로의 대체 경로로 선택되지 않는다. 과거 Checkpoint/NPY reader와 실물 안전 경계는 유지한다.

| 수정할 기능 | 핵심 구현 |
|---|---|
| Source | `workflow/mujoco.py:source`, `record_mimic_dataset.py`, `so101_mimic_recorders.py` |
| Dataset/Legacy Import | `workflow/dataset.py` |
| 명시 Replay/Annotation·개발용 재생 캐시 / DataGen 직접 분기 | `workflow/unified.py` |
| Mimic 생성·Pool/RNG·성공 수 | `workflow/mimic.py`, `mimic_reference/`, `datagen.py` |
| Simulation/IK | `workflow/adapters.py`, `simulation/mujoco_backend.py`, `simulation/ik.py`, `workspaces/environment.py` |
| Source 계보/Train·Val | `workflow/episodes.py` |
| Curation | `workflow/curation.py`, `curation_ui.py` |
| LeRobot·Train·Eval | `convert_isaac2lerobot.py`, `workflow/policy.py`, `workflow/evaluation.py`, `learning/minivla/` |

실험 설계와 과거 NPY/MuJoCo 자체 생성 설명은 [archive](../archive/README.md)에 보존한다.

## 5. 중요한 경계

1. **Hardware boundary**: calibrated hardware degree/% ↔ model rad 변환.
2. **Isaac boundary**: Isaac articulation/action은 model-space radians를 사용.
3. **Dataset boundary**: HDF5에 어떤 field가 어떤 의미와 좌표계를 갖는지 contract로 고정.
4. **Real2Sim boundary**: image에서 숫자를 임의 추측하지 않고 measured/calibrated/optimized/configured/assumed provenance를 구분.
5. **Workspace boundary**: Builder export를 checksummed runtime package로 가져오고 task/physics/engine 실행을 연결.

세부 데이터 단위와 흐름은 [DATA_FLOW.md](DATA_FLOW.md), 절대 깨지면 안 되는 규칙은 [INVARIANTS.md](INVARIANTS.md)를 본다.
