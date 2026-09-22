# SO-101 Real2Sim Documentation

이 폴더는 현재 `so101-isaaclab`의 Real2Sim / Workspace / imitation-learning workflow를 설명하는 **living documentation**이다.

과거 Phase별 작업지시서와 audit 보고서는 현재 구조에 흡수했다. 세부 과거 기록이 필요하면 Git history를 본다.

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

6. [REAL2SIM_ITERATION.md](REAL2SIM_ITERATION.md)  
   Codex/agent가 bootstrap → paired capture → fitting → validation → 다음 iteration으로 scene을 반복 개선하는 실제 작업 프로토콜.

7. [CHANGELOG_AI.md](CHANGELOG_AI.md)  
   큰 AI-generated 변경에서 문제, 접근, 변경 파일, 테스트, 미검증 영역, 위험을 기록.

## 실행 진입점

### Real2Sim UI

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

UI와 Isaac runtime은 별도 프로세스이며 같은 `--workspace`의 file IPC를 사용한다.
**Connect** 또는 **Sim 미리보기 (장비 연결 없음)** 를 누르면 runtime이 없을 때
UI가 자동 실행하고 준비 완료 후 연결한다. 별도 터미널 실행은 선택 사항이다.
자동 실행에는 `4 · Workspace Pipeline`의 `실행 Python (Isaac / LeRobot)` 경로를
사용한다. GUI 전용 venv가 아닌 Isaac 환경의 Python을 지정해야 한다.
같은 탭의 실행 모드가 Operator이면 headless로 시작하고, Developer이면 Isaac 창을 연다.
두 모드는 동일 runtime과 Side/Wrist 카메라를 사용한다. 이미 실행 중인 runtime의 모드를
바꾸려면 Disconnect 후 다시 Connect한다.
시작 로그는 선택 workspace의 `runtime_startup.log`에 저장된다.
시작 중 Stop 또는 UI 종료는 예약된 Connect를 취소하며 Isaac 프로세스 자체는 유지한다.
`Disconnect / Sim 종료`는 하드웨어 연결과 environment를 정리한 뒤 runtime을 종료한다.
자동 시작 대기 중에 누르면 예약된 연결을 취소하고 UI가 시작한 프로세스만 종료한다.
Connect 이후 실제 조작은 기존처럼 Start로 시작한다.

### Inspect의 역할

`3 · Inspect`는 Real2Sim profile을 검사하고 수동으로 적용하는 고급 도구다.
선택 revision과 runtime 적용 hash를 확인하고, Profile 열기/적용, USD + Profile
내보내기, JSON 새 revision 저장을 제공한다. 선택적 측정 도구로 landmark 지정,
정량 오차 계산, parameter 그룹 최적화 및 재렌더를 실행할 수도 있다.
일반 Connect/Capture나 publish된 workspace에서 Source Demo를 수행할 때 필수 단계는 아니다.
USD 내보내기와 Workspace Publish는 서로 다른 기능이다.

### Isaac runtime

Isaac Lab/Isaac Sim이 설치된 Python 환경에서:

```bash
python scripts/real2sim/real2sim_sim.py
```

### Workspace / workflow

주요 entry point:

```text
scripts/real2sim/publish_workspace.py
scripts/real2sim/workspace.py
scripts/real2sim/workflow.py
scripts/real2sim/workflow_stage.py
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

## Repository에서 유지하는 데이터/예제

다음은 문서가 아니라 코드/테스트가 실제 사용하는 자료이므로 유지한다.

- `measurements.example.json`
- `VALIDATION.json`
- `bootstrap/`

이 파일들은 단순 설명 문서처럼 정리/삭제하면 안 된다.

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
