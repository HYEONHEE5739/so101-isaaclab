# SO-101 Real2Sim Architecture

이 문서는 현재 `so101-isaaclab`의 Real2Sim/Workspace/Workflow 구조를 빠르게 이해하기 위한 상위 수준 지도다.
세부 구현보다 **어떤 책임이 어느 모듈에 있는지**를 우선 설명한다.

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
  ├─ calibration.py
  ├─ profile.py
  ├─ scene.py
  └─ storage.py
           │
           ├─ paired Real/Sim capture + metrics
           ├─ profile revisions
           └─ accepted Real2Sim profile
                         │
                         ▼
source/.../real2sim/workspaces/
  ├─ package.py
  ├─ scene_assets.py
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
  ├─ tasks.py / task_ui.py
  ├─ policy.py / evaluation.py / hub.py
  └─ ui.py / view.py / presentation.py
                         │
                         ▼
Demo → Annotation → Datagen → Convert → Train → Evaluate
```

## 2. Real2Sim core

경로: `source/soarm101_lab/soarm101_lab/real2sim/`

- `runtime.py`: Isaac runtime과 실제 장비/카메라를 연결하는 핵심 실행 계층. pose 적용, render, capture와 runtime 상태를 담당한다.
- `hardware.py`: follower 등 실제 SO-101 하드웨어 경계.
- `cameras.py`: 실제 카메라 frame 획득과 capture timestamp 경계.
- `core.py`: Real/Sim 공통 수치 처리, pose/joint 관련 핵심 helper.
- `profile.py`: camera, workspace, robot/object/scene 파라미터와 provenance를 가진 Real2Sim profile의 검증/로드.
- `calibration.py`: intrinsics/extrinsics, landmark, image metric 등 측정/최적화 로직.
- `scene.py`: profile을 Isaac Sim scene/USD 표현으로 적용하는 계층.
- `references.py`: 실제 reference 입력 관리.
- `storage.py`: revision, capture, IPC 상태 등 파일 기반 저장.
- `bootstrap.py`: 초기 scene/profile bootstrap 지원.
- `agent.py`: 외부 agent orchestration용 보조 계층. 수치 파라미터를 임의 추측하는 용도가 아니다.
- `ui.py`: Real2Sim UI. Isaac runtime과 별도 프로세스로 동작하며 workspace의 file IPC를 사용한다.

진입점:
- `scripts/real2sim/real2sim_ui.py`
- `scripts/real2sim/real2sim_sim.py`

## 3. Workspace Environment

경로: `source/soarm101_lab/soarm101_lab/real2sim/workspaces/`

Real2Sim으로 맞춘 한 실제 작업공간을 재사용 가능한 environment package로 만드는 계층이다.

- `package.py`: versioned workspace package 로드/메타데이터.
- `scene_assets.py`: workspace의 scene asset 구성.
- `environment.py`: workspace package를 Isaac Lab environment 설정에 연결.
- `physics.py`: mass/friction/collision 등 simulation physics 설정.
- `reset.py`: workspace/object reset 정책.
- `demo.py`: source demo와 workspace representation contract 연결.
- `mimic.py`: 기존 Mimic 흐름에 workspace 정보를 연결하는 adapter.

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

Real2Sim이 Mimic/datagen/policy 알고리즘을 재작성하는 구조가 아니다.

## 5. 중요한 경계

1. **Hardware boundary**: calibrated hardware degree/% ↔ model rad 변환.
2. **Isaac boundary**: Isaac articulation/action은 model-space radians를 사용.
3. **Dataset boundary**: HDF5에 어떤 field가 어떤 의미와 좌표계를 갖는지 contract로 고정.
4. **Real2Sim boundary**: image에서 숫자를 임의 추측하지 않고 measured/calibrated/optimized/configured/assumed provenance를 구분.
5. **Workspace boundary**: accepted profile을 versioned package로 publish하고 기존 task code와 분리.

세부 데이터 단위와 흐름은 [DATA_FLOW.md](DATA_FLOW.md), 절대 깨지면 안 되는 규칙은 [INVARIANTS.md](INVARIANTS.md)를 본다.
