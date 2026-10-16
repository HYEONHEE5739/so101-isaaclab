# SO-101 Real2Sim Documentation

2026-10-16: Workspace Source/Annotation/DataGen은 두 엔진 모두 공통 HDF5 + Side/Wrist640×480 H.264
형식이며, DataGen은 공식 Isaac Lab Mimic1.0.15 공통 core를 사용한다. Backend를 바꾸면 자동 Import와
물리 replay 검증을 거친다. 환경 수1/2/4/8, 실패 종류별 통계, 완료 데이터 중지/Resume을 지원한다.
계약·실행 흐름·이전 형식 지원은 [DATA_FLOW.md](DATA_FLOW.md#통합-episode-계약--2026-10-16)를 참조한다.


이 폴더는 현재 `so101-isaaclab`의 Real2Sim / Workspace / imitation-learning workflow를 설명하는 **living documentation**이다.

현재 실행 구조는 [ARCHITECTURE.md](ARCHITECTURE.md), 데이터 의미는 [DATA_FLOW.md](DATA_FLOW.md)가 기준이다.
과거 Phase 설계와 전환 전 데이터 설명은 [archive](../archive/README.md)에 보존한다.

## 먼저 읽을 문서

1. [ARCHITECTURE.md](ARCHITECTURE.md)  
   전체 구조와 각 모듈의 책임.

2. [DATA_FLOW.md](DATA_FLOW.md)  
   Leader/Follower → model radians → Isaac → HDF5/replay까지 값의 단위와 흐름.

3. [PROJECT_MAP.md](PROJECT_MAP.md)  
   "카메라/랜더마이제이션/HDF5/joint mapping/UI를 어디서 수정하지?"에 대한 파일 지도.

4. [INVARIANTS.md](INVARIANTS.md)  
   Codex/agent가 변경할 때도 깨뜨리면 안 되는 robot, coordinate, dataset, Real2Sim 규칙.

5. [TESTING.md](TESTING.md)  
   실행 성공이 아니라 correctness를 증명하기 위한 테스트/metric 기준.

6. [CAPTURE_DATASET.md](CAPTURE_DATASET.md)
   Start → Disconnect 세션의 Snapshot 수집과 독립 Builder import 절차.

7. [CHANGELOG_AI.md](CHANGELOG_AI.md)  
   큰 AI-generated 변경에서 문제, 접근, 변경 파일, 테스트, 미검증 영역, 위험을 기록.

## 실행 진입점

### Real2Sim UI

GUI는 `lerobot-arena`와 별도인 `.venv-real2sim-ui` 패키지를 사용한다.
Dataset Curation의 Builder 모델 계약 확인에는 PyYAML, MP4 재생에는 PyAV(`av`)가 필요하다.
GUI 의존성을 설치·보완할 때는 아래 Python을 명시한다. conda 환경에 설치해도 GUI 환경에는 적용되지 않는다.

```bash
.venv-real2sim-ui/bin/python -m pip install -r scripts/real2sim/requirements-ui.txt
```

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

UI와 선택한 Isaac/MuJoCo runtime은 별도 프로세스이며 같은 `--workspace`의 file IPC를 사용한다. Real/Sim 4분할과 수동 단계 실행·Annotation 저장 정책은 [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md) 및 [DATA_FLOW.md](DATA_FLOW.md)에 정리했다.
**Connect** 또는 **Sim 미리보기 (장비 연결 없음)** 를 누르면 runtime이 없을 때
UI가 자동 실행하고 준비 완료 후 연결한다. 별도 터미널 실행은 선택 사항이다.
자동 실행에는 `3 · Workspace Pipeline`의 `실행 Python (Isaac / LeRobot)` 경로를
사용한다. GUI 전용 venv가 아닌 Isaac 환경의 Python을 지정해야 한다.
같은 탭의 실행 모드가 Operator이면 headless로 시작하고, Developer이면 Isaac 창을 연다.
두 모드는 동일 runtime과 Side/Wrist 카메라를 사용한다. 이미 실행 중인 runtime의 모드를
바꾸려면 Disconnect 후 다시 Connect한다.
시작 로그는 선택 workspace의 `runtime_startup.log`에 저장된다.
시작 중 Stop 또는 UI 종료는 예약된 Connect를 취소하며 Isaac 프로세스 자체는 유지한다.
`Disconnect / Sim 종료`는 하드웨어 연결과 environment를 정리한 뒤 runtime을 종료한다.
자동 시작 대기 중에 누르면 예약된 연결을 취소하고 UI가 시작한 프로세스만 종료한다.
Connect 이후 실제 조작은 기존처럼 Start로 시작한다.

### Snapshot Capture와 연결 설정

Inspect 탭/JSON editor/수동 fitting UI는 제거했다. 필수 Profile 선택/적용 및 장치
calibration 설정은 Real/Sim의 **연결 설정**에 있다. Profile/장치 보정 로드는 유지한다.
현재 수집은 Snapshot만 사용하며 옛 paired/continuous 캡처와 내부 scene fitting/API agent는 제거했다.
처음에는 대기하며 Capture 탭의 **Start · 카메라/로봇 연결**을 눌러야
동일 runtime이 실제 장치만 연결한다. Sim은 실행하지 않는다.
Capture 탭의 **현재 상태 캡처**는 원본 Side/Wrist + measured q + timestamp를 하나씩
자동 저장한다. Teleop Start/Stop은 별도 안전 제어이며 Record/Event/Save UI는 없다.
자세별 dataset/Builder import와 실행 환경은 [CAPTURE_DATASET.md](CAPTURE_DATASET.md)를 참고한다.

### Isaac runtime

Isaac Lab/Isaac Sim이 설치된 Python 환경에서:

```bash
python scripts/real2sim/real2sim_sim.py
```

### Workspace / workflow

주요 entry point:

```text
scripts/real2sim/workflow.py import-builder
scripts/real2sim/workspace.py
scripts/real2sim/workflow.py {list,register,run,ready,worker,import-builder,migrate,import-hdf5}
scripts/real2sim/check_workspace_physics.py
```

자세한 코드 위치는 [PROJECT_MAP.md](PROJECT_MAP.md)를 본다.

## 현재 Real2Sim 원칙

목표는 실제 SO-101 환경의 관측과 동일 robot configuration의 Isaac Sim 관측을 정량적으로 맞추는 것이다.

우선 보정 대상:

- camera intrinsics / extrinsics
- robot base alignment
- object pose
- table / scene alignment

LLM은 측정하거나 최적화할 수 있는 camera/object/physics 숫자를 이미지에서 임의 추측하지 않는다.
값은 가능한 한 다음 출처 중 하나를 명시한다.

- measured
- calibrated
- optimized
- configured
- provisional / assumed

기존 SO-101 environment, teleoperation, Mimic, datagen, dataset, policy 구조는 가능한 유지한다.

## Scene 제작과 실행 입력

Scene 제작/보정은 독립 `real2sim-builder`에서 수행한다. SO-101은 Snapshot 수집과
선택한 Profile/Workspace의 엔진 실행을 담당한다. UI나 MuJoCo worker가 옛 bootstrap
scene을 자동으로 생성하지 않는다. 새 UI에는 Profile을 명시적으로 선택해야 한다.
현재 저장된 Workspace/profile 선택은 그대로 유지한다.

검증 전용 profile 입력은 `tests/real2sim/fixtures/`에 있다. production 코드에서는
이 테스트 입력을 읽지 않는다. `VALIDATION.json`은 기존 검증 기록이다.

## AI/Codex 작업 규칙

non-trivial 변경 후에는 최소한 다음을 남긴다.

1. 해결하려던 문제
2. 접근 방법과 선택 이유
3. 변경 파일과 이유
4. data-flow / unit / coordinate 영향
5. 실행한 테스트와 결과
6. 검증하지 못한 부분
7. 가능한 실패 조건
8. 사람이 직접 볼 고위험 코드
9. 다음 수정 시 시작할 위치

큰 변경은 [CHANGELOG_AI.md](CHANGELOG_AI.md)에 기록한다.

프로그램이 실행되었다는 사실만으로 correctness를 주장하지 않는다.

### Annotation 판정 로그와 접촉 허용오차

Workspace 컵 배치 판정은 8개 꼭짓점의 반경 초과량에 **1mm 이하**의 접촉 허용오차를
적용한다(`workspaces/environment.py: WALL_TOLERANCE_M`). 높이와 속도 기준은 유지한다.
공통 성공 함수이므로 Annotation/Datagen/Sim 평가에 동일하게 적용되며, 이전 버전에서
생성된 판정 결과를 자동으로 다시 계산하지 않는다.
UI 기본 로그는 episode 진행과 저장/제외 결과를 표시한다. `[DIAGNOSTIC]` 상세 진단
(Grasp, 높이, 벽, 속도, frame 진행)은 `원본 로그 열기` 또는 `전체 로그`로 확인한다.

### 실행 결과 폴더 이름

새 workflow 실행은 `outputs/workflow/runs/YYYYMMDD_HHMMSS_단계_task_ID/`에 저장한다.
시간은 실행 컴퓨터의 로컬 시간이며 마지막 8자리 ID로 같은 초의 실행을 구분한다.
예: `20260922_143025_annotation_cube_red_to_cup_a_a1b2c3d4/annotated.hdf5`.
여러 task는 `multi_N_tasks`, 전체 평가는 `all_tasks`, task가 없는 작업은 `no_task`로 표시한다.
성공/실패 상태는 기존처럼 `state.json`에 기록한다. 기존 UUID 폴더와 artifact 참조는
변경하지 않으며, 직접 지정한 `output` 경로도 그대로 사용한다.

### Datagen 성공 데이터 목표

Workspace Datagen의 `num_successful_demos`는 **성공 episode 목표 개수**다. 예를 들어
`{"num_successful_demos": 100}`은 실패를 제외하고 성공 데이터 100개가 생성될 때 종료한다.
기존 Mimic의 `generation_guarantee=True`를 사용하며 실패 데이터는 출력 HDF5에 저장하지 않는다.
성공이 계속 나오지 않으면 반복이 이어지므로 UI의 중지 버튼으로 중단할 수 있다.
기존 non-workspace Datagen의 설정은 변경하지 않는다. 실행 중인 작업에는 소급 적용되지 않는다.

Datagen UI는 성공/목표, 완료 시도 수, 실패 수, 성공률(성공 ÷ 완료 시도)을 표시한다.
아직 완료한 시도가 없으면 성공률은 `—`이다. 기존 `trials`는 호환 별칭으로 지원하지만,
새 설정에는 `num_successful_demos`를 사용한다. 두 값을 다르게 지정하면 오류로 안내한다.

### Datagen 랜덤 seed 분리

Workspace Datagen은 실행마다 새 seed를 생성하여 상자 reset과 Mimic의 난수 설정에
함께 적용한다. Source Demo와 published workspace의 reset seed는 변경하지 않는다.
사용 seed는 시작 로그, 출력 옆 `datagen.generation.json`, 성공적으로 닫힌 HDF5의
`data.attrs.generation_provenance`에 기록한다. 재현하려면 실행 JSON의 선택 항목
`generation_seed`에 그 값을 지정한다. 생략하면 다음 실행은 새 seed를 사용한다.
Workspace reset seed와 동일한 명시 seed는 거부한다. 네 위치/±2cm 범위와
겹침 검사는 유지되므로 위치가 우연히 비슷할 수는 있으나 같은 난수열 재사용은 피한다.

### 학습 체크포인트에서 이어서 학습

Workspace Pipeline의 학습 단계에서 **체크포인트 선택**으로
`checkpoints/040000` 같은 폴더를 선택하고 **이어서 학습**을 켭니다.
원래 dataset을 등록된 입력으로 선택해야 합니다. 모델 종류와 목표 steps는
체크포인트 선택 시 불러오며, JSON의 `steps`는 추가 횟수가 아니라 최종 목표입니다.
예: 40,000에서 `steps: 100000`이면 60,000회 추가 학습합니다.

LeRobot의 resume으로 가중치, 전처리기, optimizer, scheduler, RNG와 step을
복원합니다. Resume에서는 학습 설정을 저장된 train_config에서 가져오며
UI의 새 학습용 batch/base_model/rename_map 등은 적용하지 않습니다.
결과는 새 run/output에 저장하여 원본 checkpoint를 보존합니다.
가중치만 있는 `pretrained_model` 배포본은 training_state가 없으므로 재개할 수 없습니다.

### Source Demo / Datagen 중지 및 이어 수집

UI 중지는 완료된 episode를 보존하고 HDF5를 검증한 뒤 입력 artifact로 등록합니다.
진행 중인 미완성 episode는 포함하지 않습니다. 완료 episode가 하나도 없거나
파일 검증이 실패하면 등록하지 않습니다. 저장을 보호하기 위해 이 두 단계에서는
중지 후 짧은 timeout으로 강제 kill하지 않고 안전한 simulation 경계를 기다립니다.
OS 강제 종료/전원 차단까지 안전 저장을 보장하는 기능은 아닙니다.

같은 단계에서 **이어 수집할 HDF5 선택**을 사용하고 총 목표 수를 입력하세요.
기존 20개에 목표 50개면 30개를 새로 수집합니다. Datagen은 같은 annotation 입력,
workspace/task를 사용해야 합니다. 기존 파일은 읽기 전용으로 보존하고 새 run의
HDF5에 이전 완료 episode와 새 episode를 합칩니다. 병합에는 추가 디스크 공간이
필요합니다. 재개 파일 입력란을 비우면 새 수집입니다.
재개는 episode 단위의 계속 수집이며 중단된 simulation/RNG/frame 복원이 아닙니다.

### Workspace 안정 grasp annotation

Workspace Mimic/annotation만 `workspaces/grasp.py`의 안정 grasp 판정을 사용합니다.
기존 TCP 거리 45mm/그리퍼 0.3rad 기준을 유지하되, 초기 높이 대비 큐브 높이의
절반 이상 들린 상태에서 EE 좌표계 물체 중심이 0.20초 동안 6mm 이내로 유지돼야
완료됩니다. 열림 이력은 요구하지 않습니다. 0.20초/6mm는 초기 엔지니어링 기준이며
검증된 물리 캘리브레이션 값이 아닙니다. 동일 simulation step의 중복 호출은
시간으로 중복 계산하지 않습니다. 완료 시점이 Mimic grasp 경계로 latch됩니다.
이후 떨어짐/정상 release를 자동으로 구분하는 추가 성공 조건은 도입하지 않았습니다.
컵 내부/정지 조건은 기존대로 유지합니다. 기존 annotated HDF5는 바뀌지 않으므로
새 경계를 사용하려면 원본 Source Demo로 annotation을 다시 수행하세요.

#### 양쪽 collision 접촉 기반 grasp로 변경

앞의 거리/닫힘/들림 기반 초기 판정은 Workspace에서 교체되었습니다.
Robot/gripper와 Robot/jaw 각각의 ContactSensor를 선택한 동일 큐브로 필터링합니다.
두 body 모두 0.0001N 초과 접촉 힘을 보이며, EE와 물체가 각각 5mm 이상 이동하고,
EE 기준 상대 위치 6mm/회전 10도 이내를 0.20초 유지하면 grasp를 확정합니다.
수치는 초기 엔지니어링 기준이며 물리 캘리브레이션 결과가 아닙니다.
열림 이력/닫힘 관절각/TCP 거리/들림 높이는 필수 조건에서 제외했습니다.
고정 손가락은 서보 등과 동일 rigid body이므로 손끝 면만 분리한 센서는 아닙니다.
접촉 데이터가 없으면 거리 조건으로 대체하지 않고 오류를 보고합니다.
Physics offset과 collision geometry는 변경하지 않았습니다.
기존 annotation 파일은 자동 변경되지 않으므로 다시 annotation해야 합니다.

### Data Curation

Workspace Pipeline 왼쪽의 Data Curation에서 완료된 Source/Datagen/Annotation
HDF5를 엽니다. 초기 목표 물체 위치를 workspace 좌표로 표시하며 점과 목록에서
episode를 선택하면 관절 목표값 그래프 및 마우스로 회전하는 EE 3D 경로를 봅니다.
EE 경로는 기록된 ee_state 좌표계입니다. 이미지 재생이나 물리 replay는 아닙니다.

확인 필요 표시는 task별 median ± 3 robust sigma(MAD, 최소 median의 10%) 기준입니다.
5개 미만인 task는 통계적 이상치를 단정하지 않습니다. 시간 이상, EE 속도 2mm/s
미만 비율, 모든 관절 목표 변화 1deg/s 미만 비율, arm 명령 가속도 RMS를 사용합니다.
정지는 정상적인 grasp/place 대기일 수도 있으며 jitter는 실패의 확정 근거가 아닙니다.

제외/복원으로 검토한 뒤 목표 개수를 지정하고 분포 유지 자동 선별을 실행합니다.
남은 데이터의 task × XY 4x4 구간 빈도 비율을 largest-remainder로 배분해 정확한
개수를 선택하고, 구간 내에서는 flag 수와 jitter가 작은 episode를 우선합니다.
균등 분포를 새로 만들거나 학습 성능 향상을 보장하는 기능은 아닙니다.
선택 데이터셋 저장은 원본 전체 episode 필드를 새 HDF5에 복사하며 원본은 보존합니다.
좌표 contract, task metadata와 원본 hash/episode 대응을 남기고 workflow에 등록합니다.
이후 LeRobot 변환에서 새 artifact를 선택하세요. 원본 영상도 복사하므로 디스크 공간이
추가로 필요합니다. 분석/저장은 백그라운드에서 실행됩니다.

### Curation 로봇/궤적 재생

Data Curation에서 HDF5와 에피소드를 선택하면 workspace에 포함된 SO-101
URDF visual mesh를 기록된 `obs/joint_pos`(radian)로 재생합니다.
EE 궤적과 로봇은 모두 robot-base 좌표계입니다. 재생/일시정지, 시간 슬라이더,
0.25/0.5/1/2배속, 휠 줌, 드래그 회전, XY/XZ/YZ 시점을 지원합니다.
흰 점은 현재 EE 위치이고 관절 그래프의 흰 세로선도 같은 시간입니다.
메시는 CPU 미리보기용으로 단순화하며 원본 asset/dataset은 변경하지 않습니다.
이는 기록된 로봇 자세 검토용으로, 물체/접촉 물리를 재계산하는 Isaac Replay는 아닙니다.
URDF 또는 기록된 관절 상태가 없으면 로봇 표시 없이 경로만 확인합니다.
# 커스텀 MiniVLA / MuJoCo

모델 코드는 기존 `source/soarm101_lab/soarm101_lab/learning/minivla/`에 통합했다.
모델 단독 명령·설계·notebook은 [모델 문서](../minivla/README.md), outputs의 삭제
판단 기준은 [DATA_FLOW.md](DATA_FLOW.md#13-outputsminivla-mujoco-real2sim을-삭제해도-되는가-2026-10-07)를 본다.

기존 Isaac workflow와 함께 사용할 커스텀 정책의 수집·학습·sim 평가 경로는
[MiniVLA 통합 안내](MINIVLA_INTEGRATION.md)를 참고한다. Real2Sim 데스크톱의
기존 `Workspace Pipeline` 상단에서 `Isaac Sim / MuJoCo`를 선택한다. Source → Replay → Annotation → Datagen → Convert → Train → Eval → Upload와 실행·중지·로그·검수 화면은 공통이다. 모델 선택(ACT/SmolVLA/MiniVLA)은 엔진과 별개이며, 엔진·모델은 worker에서 로드한다. 공통 `Operations`가 모든 단계의 실행과 등록을 관리한다.


실측 데이터 수집 → 독립 Builder: [Real2Sim Capture 계약과 운영](CAPTURE_DATASET.md).

Builder에서 만든 scene을 기존 UI에서 사용하는 방법: [Builder Workspace 연결](BUILDER_WORKSPACE.md).
