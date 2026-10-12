# MuJoCo / SO-101 MiniVLA 통합

커스텀 모델과 프로젝트 adapter는
`source/soarm101_lab/soarm101_lab/learning/minivla/`로 통합했다. 기존 Isaac 실행 경로와
dataset 계약을 유지하며, MiniVLA 데이터는 독립적인 canonical dataset으로 저장한다.
모델 성능을 보장하는 구현이 아니라 실제 데이터로 학습·평가할 수 있는 실행 구조다.

## 실행 환경

기존 `lerobot-arena` 환경을 사용한다. 통합 작업에서는 pip install이나 버전 변경을
하지 않았다. 별도 MiniVLA editable 설치는 사용하지 않는다.
worker는 프로젝트 source 경로를 직접 추가한다. torch, transformers, bitsandbytes,
peft, accelerate, numpy, scipy, Pillow, psutil, h5py, mujoco가 필요하다.
GUI는 별도 PyQt6 환경에서 실행한다.

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

**기존 `4 · Workspace Pipeline` 오른쪽 상단에서 `Isaac Sim / Isaac Lab` 또는
`MuJoCo`를 선택한다. 별도 MuJoCo 탭은 제거했다.** Workspace, Task preset, Leader port,
Episode/Reset Time, 저장/폐기 버튼, →/← 단축키, 영상과 로그는 같은 화면을 사용한다.
선택했던 workspace와 현재 포트 값은 backend를 바꿔도 유지한다.
엔진별 Python 경로는 따로 기억하며 기본값은 기존 lerobot-arena 환경이다.
GPU 엔진과 모델은
GUI 프로세스에서 import하지 않는다. 각 실행의 request, log, report, scene, RGB,
dataset/checkpoint는 공통 `outputs/workflow/runs/` 아래에 보존된다. 독립 CLI 실행은 `--output` 경로를 사용한다.

바로 Source Demo를 시작하는 순서:

1. 기존 Workspace Pipeline에서 **MuJoCo** 선택.
2. 기존과 동일하게 Workspace·Task·Leader port·Episodes·Episode Time을 확인.
3. **Source Demo → 데모 기록 시작**. Side/Wrist 영상과 저장/폐기/종료 버튼을 사용한다.

Source/Eval은 VRAM 사전 검증 없이 선택한 단계를 바로 실행한다. backend 변경은 실행 중 잠긴다.

## 독립 실행과 검증 순서

아래 명령은 프로젝트 루트에서 실행한다. `--output`은 새 폴더여야 한다.
workspace 예시는 실제 사용할 published version 경로로 바꾼다.

```bash
python scripts/real2sim/workflow.py worker --stage environment \
  --workspace outputs/real2sim/environments/workspace_001/v2 \
  --renderer software-egl --options '{"preview_cameras":true}' --output outputs/workflow/runs/my_mujoco_env

```

CPU physics / env count=1이며 published workspace를 자동 변환한다. 현재 v2에서는
SO-101, table, **큐브 3개·컵 2개**, 마커/테두리, 배경, 손목 카메라 장착물을 포함한다.
robot base/home pose, 모든 물체 치수/pose, physics 역할, reset seed/분포,
Side/Wrist camera transform과 스케일된 intrinsics를 재사용한다. 컵은 기존과 같은
열린 collision과 96분할 tapered visual, 동일 이미지/UV의 프린트 텍스처를 사용한다.
로봇의 보라색 printed-part material도 profile에서 가져온다. workspace를 다시 만들 필요 없다.

Side/Wrist와 inspection용 Overview를 렌더링한다. **MiniVLA 정책/학습 입력은 side 한 장**으로
유지한다. 원본 RGB와 UI 미리보기는 기본 **640×480 / 4× MSAA**이며 미리보기는
최대 10Hz다. Qwen 입력은 모델 config의 이미지 크기로 별도 전처리한다.
`rgb.png`, `sim_side.png`, `sim_wrist.png`, `sim_overview.png`로 실제 장면을 확인한다.
`report.json`에서 finite state, GL renderer, forbidden_imports를 확인한다.
physics-only 검사는 `--renderer none`으로 실행할 수 있다.
MuJoCo 접촉·actuator와 조명/material 응답은 Isaac과 동일하다고 보장하지 않는다.
레이아웃·좌표·초기화 재사용과 실제 grasp 품질 검증을 구분한다.

## Source 오류와 화면 수정 (2026-10-07)

`Joint target exceeds URDF limits`는 Leader mapper가 만든 목표가 MuJoCo의 엄격한
URDF 범위를 벗어나면 Source 전체가 종료되던 문제다. 당시 메시지에는 관절 이름이
없어서 원래 실패한 관절을 특정할 수 없다. 이제 Source의 `LeaderTargets`가
정책 좌표 범위와 정확한 URDF 범위의 교집합으로 목표를 제한한다. 범위를 벗어난
관절 이름은 UI 상태에, 요청값/적용값은 로그에 표시하고 횟수/최대 초과량을 기록한다.
NaN이나 잘못된 차원의 입력은 계속 거절하며 backend 자체의 엄격한 검사는 유지한다.
캘리브레이션이나 관절 부호/offset을 자동으로 바꾸지는 않는다.

데이터에는 `requested_action.npy`에 원래 요청한 model rad 값을,
`raw_action.npy`에 실제 적용한 model rad 값을 저장한다. 정책용 `action.npy`와
replay는 적용된 목표를 사용한다. 따라서 제한된 입력을 원래 목표로 잘못 학습하지 않는다.

화면은 프로젝트의 실제 Side/Wrist 사진과 기존 Isaac 렌더를 함께 비교해 수정했다.

- 컵 texture의 V축 방향이 뒤집혀 흰 컵 대신 사진의 다른 색 영역이 보이던 문제를 수정했다.
- 주변광/주광, 재질 응답과 출력 sRGB 전달함수를 조정했다. UI·저장 영상·정책 입력에
  같은 RGB 경로를 사용하며 render profile은 `so101.mujoco-rgb/2`로 기록한다.
- 640×480과 4× MSAA로 계단 현상을 줄이고 거친 자기 그림자는 비활성화했다.
  이 조명은 기존 Isaac 영상에 맞춘 근사이며 물리적으로 보정된 조명은 아니다.
- URDF에는 없고 원래 Isaac USD에만 있던 손가락 접촉 패드 2개를 원래 위치/치수로
  복구했다. `assets/SO101/mujoco/usd_contact_pads.json`에 USD hash와 출처를 기록한다.
  회색 외관은 임시값이며 접촉 형상과 마찰값은 USD에서 가져왔다.

패드는 `scripts/mujoco/export_usd_contacts.py`에서 오프라인으로 추출했다. 이 도구만
USD reader를 사용하며 MuJoCo worker는 JSON을 읽고 원본 hash를 검증한다.
worker에서는 `isaaclab`, `omni`, `pxr`를 import하지 않는다. published workspace나
기존 Isaac asset은 변경하지 않았다. 패드 추가로 model hash가 달라졌으므로 이전
model의 raw state replay는 호환성 검사에서 거절된다.

비교에 사용한 실제 사진은
`outputs/real2sim/captures/session_20260919_121858_8945750b43f24e6e8ccc79854fd84182`,
Isaac 기준 영상은 `outputs/real2sim/runs/contact_iteration_20260919/renders/balanced`다.
색상 응답은 capture_000001로 조정하고 000007/000009에서 별도 확인했다.
이 두 자세의 물체별 RGB MAE 평균(0–255)은 Side **81.68 → 7.99**,
Wrist **52.76 → 15.68**이다. 이는 Isaac 영상에 대한 색상 오차이며 실제 공간의
위치 오차나 grasp 성공률이 아니다. 카메라/배치/물체 크기는 기존 profile을 유지했다.

기존 비교 이미지와 보고서는 사용자 요청으로 `outputs/minivla`와 함께 삭제했다.

UI를 재시작한 뒤 기존 Workspace Pipeline → MuJoCo → Source/Replay 영상에서
흰 컵의 빨간 인쇄, 보라색 로봇, 회색 손가락 패드를 확인한다. Source에서 Leader를
움직였을 때 제한이 발생하면 UI에 관절 이름, 로그에 적용값이 표시된다. 같은 관절에서 계속
제한된다면 저장된 요청값과 기존 Leader calibration을 대조해야 한다.

## 데이터 수집 → 학습 → 평가

기존 파이프라인에서 다음 순서로 진행한다. 일반 사용에는 JSON 입력이 필요 없다.

1. **Source Demo**: 기존 Leader port, Episodes, Episode Time, Reset Time 입력란을 사용한다.
   실제 calibration 파일과 workspace task instruction을 확인한다. 기존 Leader와
   mapper를 재사용하고 follower에는 명령을 보내지 않는다. 저장/폐기/종료 버튼을 지원한다.
2. **Replay**: 저장된 dataset이 입력 목록에 자동 표시된다. 원래 MuJoCo state/model identity와
   control rate를 복구하고 raw joint replay 오차를 검사한다. imported HDF5는 raw state가
   없으므로 이 replay 대상이 아니다.
3. **Annotation**: 동일한 자동/수동 선택을 사용한다. MuJoCo는 같은 모델의 state/action을 재생하고 양 손가락 접촉·상대 pose 안정성으로 grasp를 판정한다. 수동에서는 grasp 지점을 한 번 표시하고 마지막 물리적 placement를 검증한다. 순서가 맞지 않거나 placement가 실패하면 등록하지 않는다.
4. **Datagen**: 공통 목표 성공 수·seed·일시정지/step/중지 UI를 사용한다. MuJoCo는 source EEF target의 object-relative 위치를 바꾸고 bounded 5-axis IK로 joint target을 구한다. 실제 physics에서 grasp/placement를 충족한 episode만 저장한다. 이 구현은 NVIDIA Isaac MimicGen 자체가 아니며, SO-101 실제 성공률은 별도 검증해야 한다.
5. **Convert**: 기존 LeRobot converter를 재사용하여 6D model-radian action/state, Side/Wrist 영상, Task Definition을 내보낸다. MuJoCo 결과는 canonical dataset과 `lerobot/`를 함께 갖는다. MiniVLA 통계는 train group만 사용하고 generated variant는 source group을 유지한다. 한 source group도 LeRobot 변환은 가능하지만 MiniVLA 학습에는 독립적인 train/val group이 필요하다.
6. **Train**: 엔진과 별개로 ACT / SmolVLA / MiniVLA를 공통 모델 드롭다운에서 고른다. MuJoCo canonical 입력도 ACT/SmolVLA에는 변환된 LeRobot 데이터를 전달한다. Isaac LeRobot 입력으로 MiniVLA를 학습할 때는 canonical adapter를 사용한다. generated LeRobot의 독립 source group이 불명확하면 임의 분할하지 않고 거절한다. MiniVLA는 기본 `configs/frozen_flow.json`을 사용한다. UI의 `MiniVLA head · Frozen`에서 Flow matching / Deterministic을 선택하거나 `model_config`로 직접 지정할 수 있다. 재개는 `last.pt`와 같은 폴더의 `best.pt`, 같은 모델 설정/statistics가 필요하다. `steps`만 더 큰 총 목표로 늘릴 수 있다. 저장 경로를 비우면 기존 학습 폴더에 저장한다.
7. **Sim Eval**: 현재 선택한 엔진에서 checkpoint를 평가한다. ACT/SmolVLA는 기존 inference loader를 사용하고 MiniVLA는 model-radian adapter를 사용한다. MuJoCo는 action limit/속도 guard와 성공 여부를 기록한다. 선택 task 또는 전체 task 평가를 지원한다. 추론 중 physics를 멈추는 동기식 평가다.
8. **Upload**: 기존 Dataset/Policy Hub 단계를 공유한다. MuJoCo dataset은 `lerobot/` 내용을 repository 루트에 올리고 MiniVLA policy는 best/last/config/statistics/provenance/좌표 계약을 포함한다. 업로드는 사용자가 해당 단계를 실행할 때만 발생한다.

렌더 선택은 공통 실행 설정 `renderer`에 `egl`, `software-egl`, `osmesa`를 입력한다. MuJoCo physics는 CPU이고 `device`는 정책 계산 장치다. Developer 모드는 MuJoCo passive viewer 또는 Isaac 창을 함께 띄우며 디스플레이가 필요하다. Operator 모드는 기존 UI 프리뷰를 사용한다.

데이터 검수(Curation)는 HDF5와 canonical 폴더를 같은 분석/선별 화면에서 읽는다. 선택본은 원본을 수정하지 않고 새 artifact로 저장한다.

공통 CLI는 `--stage`, `--workspace`, `--input`, `--config`, `--checkpoint`,
`--renderer`, `--options '{...}'`, `--output`을 제공한다.

정책의 6차원 좌표는 **arm 5개 rad + gripper fraction [0,1]**이다. MuJoCo는 6개 모두
model rad이므로 변환을 runtime adapter에서 수행한다. default gripper endpoint는
기존 mapper의 closed=0, open=100도이며 필요하면 두 endpoint를 명시한다.
관측은 action 적용 직전 상태/영상이다. measured state와 commanded target을 별도
저장하고 replay에는 raw radian action과 integration state를 사용한다.

CLI의 HDF5 Import는 명시적인 recipe만 허용하고 각 episode의 success label을 요구한다.
기존 Isaac/LeRobot 데이터를 이름만 바꿔 호환으로 처리하지 않는다.
CLI에서 영상 Subtask 경계는 `episode`와 `boundaries` 옵션으로 기록한다. 경계는
approach/grasp/transport/place 순서이며 이것은 MimicGen annotation 계약이 아니다.
Isaac HDF5와 MuJoCo canonical raw state는 서로 직접 restore하지 않는다. raw demo 단계는 선택 backend와 데이터 backend를 검증하며, 학습·변환·업로드는 공통 모델/데이터 계약을 따른다. CLI의 dataset-only 네 경계와 workflow grasp/place annotation은 다른 schema다.

## 통합 구조와 변경 파일

```text
real2sim_ui.py → real2sim.ui.Window
  ├─ Real/Sim 연결 → RuntimeLauncher → real2sim_sim.py → 선택 backend Runtime (Teleop·4-view·Snapshot)
  └─ Workspace Pipeline → WorkflowPanel → Operations → Registry / Presentation
      ├─ Isaac physics stages → 기존 workspace / mimic scripts
      ├─ MuJoCo physics stages → scripts/real2sim/workflow.py worker → workflow.mujoco / episodes → MujocoBackend
      └─ Convert / Train / Eval / Upload → 기존 converter / policy / hub와 format adapter
```

`real2sim_sim.py`는 모든 workflow 단계가 공유하는 headless 엔진 서버가 아니다. Workflow는 단계마다 자체 worker를 실행한다.

- `workflow/ui.py`: 동일 단계·설정·실행/중지·키보드·프리뷰, backend/Python 선택.
- `workflow/operations.py`: 양쪽 worker의 계획·실행·로그·취소·artifact 등록을 단독 관리.
- `workflow/artifacts.py`: canonical/LeRobot/MiniVLA policy 계약과 lineage 검증.
- `workflow/policy.py`: 공통 모델 학습/추론 및 MiniVLA model-radian 경계.
- `workflow/curation.py`, `curation_ui.py`, `hub.py`: 기존 검수/업로드를 두 저장 형식으로 확장.
- `workflow/mujoco.py`: 엔진 worker dispatch·source/replay/train/eval.
- `workflow/episodes.py`: 녹화·resume·annotation·datagen·공통 변환 adapter.
- `simulation/mujoco_backend.py`: 접촉·FK/IK·RGB·physics/state.
- 삭제/통합: `workflow/mujoco_ui.py`와 `MiniVLAService` 실행 관리자. 별도 UI/launcher/config는 추가하지 않았다.
- 모델 구현은 `learning/minivla/`, 엔진은 `simulation/`, 단계별 실행·데이터 처리는 `workflow/`에 배치했다. 상세 모델 문서는 [MiniVLA README](../minivla/README.md)에 있다. 데이터·asset·published workspace를 삭제하지 않았다.

## 검증과 한계

```bash
PYTHONPATH=source/soarm101_lab python -m pytest -q \
  tests/real2sim/test_minivla_integration.py tests/real2sim/test_mujoco_backend.py \
  tests/minivla/test_head.py tests/minivla/test_data_and_runtime.py
```

테스트는 좌표 변환, 실제 CPU physics, state replay, dataset 정렬/분할/통계,
invalid annotation 거절, optional import 경계, model head/runtime 계약을 검증한다.
실제 Leader 수집, 긴 학습, 학습된 모델의 pick/place 성공률은 별도 실험이 필요하다.
CPU 테스트 fixture의 성공 label은 학습에 사용할 demonstration의 성공 증거가 아니다.

현재 공통 UI·MuJoCo·모델 head/data·검수·설정 검사 **61개**를 통과했다. 기존 Isaac workflow·학습/수집 재개·presentation·Task Definition·평가 로그 검사 **42개**를 기존 설치의 USD 라이브러리 경로로 실행했다. 패키지 버전은 바꾸지 않았다.

실행 검증 결과:
- 전체 실행 검증: `outputs/workflow_validation/unified_pipeline/report.json`. 공통 UI 화면: `pipeline_isaac.png`, `pipeline_mujoco.png`.
- `outputs/workflow_validation/unified_pipeline/registry/runs/`의 replay: 실제 CPU physics와 RGB, 공통 Operations subprocess/등록, 최대 joint error 0 rad, forbidden engine import 없음.
- `outputs/workflow_validation/unified_pipeline/conversion/dataset/lerobot/`: 기존 converter로 3 episode × 5 frame, Side/Wrist 영상·parquet·좌표 계약 생성.
- `outputs/workflow_validation/unified_pipeline/training_adapter_report.json`: 실제 변환 영상 decoding과 MiniVLA train/val 입력 준비 통과. 모델/optimizer 실행은 하지 않았다.

실제 Leader 연결, Developer viewer의 데스크톱 실행, 물리적 grasp/Datagen 성공률, 긴 학습, 학습 모델의 task 성공률과 실제 Hub 업로드는 이번 자동 검증에 포함하지 않았다. Fixture 성공 라벨은 실제 조작 성공 증거가 아니며 검증 Registry는 사용자 기본 Registry와 분리했다.


## Real/Sim 4분할과 접촉 설정 (2026-10-07)

기존 `2 · Real / Sim` 화면에 Isaac/MuJoCo 선택을 추가했다. `4 · Workspace Pipeline`
선택과 동기화되며, 별도 launcher나 MuJoCo 전용 창은 만들지 않았다. 기존
`real2sim_sim.py`가 backend를 먼저 읽고 선택한 엔진만 import한다. MuJoCo 경로에서는
Isaac AppLauncher를 시작하지 않는다. Connect/Start/Stop/Capture/Disconnect와
REAL SIDE / SIM SIDE / REAL WRIST / SIM WRIST의 4개 배치는 공유한다.

사용 순서:

1. 기존 UI를 재시작한다: `.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py`.
2. Workspace Pipeline에서 원하는 published Workspace와 Task를 선택한다.
3. Real/Sim 탭에서 backend를 선택하고 `Pipeline에서 선택한 Workspace 환경 사용`을 유지한다.
   이 경우 Isaac도 기존 Workspace factory, MuJoCo도 기존 full-workspace MJCF compiler를 사용한다.
   두 live 화면은 reset randomization 대신 profile에 저장된 물체 배치를 표시한다.
4. 먼저 `Sim 미리보기`로 side/wrist를 확인한다. REAL 영상도 보려면 camera 장치 경로를
   입력하고 `Sim 미리보기에서도 REAL 카메라 연결`을 체크한다. 이 옵션은 모터 포트를 열지 않는다.
5. 실제 Leader/Follower 비교는 기존 hardware/calibration 입력 후 Connect → Start를 사용한다.
   Stop에서는 follower 측정 자세를 Sim에 적용해 같은 자세의 영상으로 비교한다.
6. backend를 바꾸거나 Workflow의 simulation 단계를 실행하려면 먼저 Disconnect한다.
   실행 Python은 Pipeline에 저장된 backend별 경로를 사용한다.

MuJoCo live 비교는 현재 side/wrist가 같은 해상도이고 distortion이 없는 profile을 요구한다.
EGL/OSMesa/software-egl 선택은 별도 worker에 적용된다. REAL 카메라·실물 follower를
연결하지 않은 미리보기에서는 실제 물리 로봇과의 오차를 검증했다고 표시하지 않는다.
Capture의 기존 네 프레임·정지 상태·timestamp·mapping 검사는 유지한다. GUI 이미지
위젯은 pixmap 때문에 창의 최소 크기가 계속 증가하지 않도록 size policy를 지정했다.

새 MuJoCo Source/Datagen/live scene의 physics profile은 `isaac_matched`다. 위치 actuator의
kp/kv는 유지하고 force limit을 기존 Isaac의 30에 맞췄다. USD에서 가져온 손가락 패드의
크기·위치·sliding friction 1.0을 유지하며, 접촉 priority 1, condim 4, elliptic friction cone과
solver 설정을 사용한다. 물체를 붙이는 weld나 가짜 grasp attachment는 사용하지 않는다.
설정 근거: [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#contact-parameters).
이 값들은 실제 모터·마찰 측정치가 아니라 기존 설정과의 정합을 위한 구성값이다.

통제된 CPU 물리 시험에서 24 mm cube를 두 패드 사이에 놓고 close → wrist 이동 → hold를
150 control step(30 Hz, 총 5초) 실행했다. 이전 설정은 cube를 떨어뜨렸고, 새 설정은
양쪽 접촉을 유지하며 초기 위치보다 약 24 mm 높은 곳에서 유지했다. 이 시험은 손가락에
물체를 정렬하고 시작한 fixture다. 실제 Leader의 approach/pick/place 및 다양한 자세의
성공률은 따로 확인해야 한다. 테이블/큐브/컵 치수와 패드 geometry는 바꾸지 않았다.

기존 녹화의 Replay/Annotation은 metadata의 physics profile을 복원한다. 필드가 없는
과거 데이터는 `legacy_urdf`(URDF effort 10 + 기존 solver)로 재구성한다. 기존
`rgb_v2_gate_egl/scene`의 scene hash와 재생성한 legacy scene hash가 일치하는 것을
확인했다. 다른 physics profile의 integration state는 계속 거절한다. 오래된 렌더/asset
세대까지 무조건 호환된다고 보장하지 않으며 model/version 검사에 실패하면 중단한다.

새로운 Source에는 개선된 접촉 설정이 적용되지만, 과거 녹화에 있던 미끄러짐을
annotation으로 고쳐 주지는 않는다. 기존 Source/Datagen을 resume하면 기록 당시의
physics profile을 유지한다. scene 치수·layout 재사용과 PhysX/MuJoCo 동역학·색·조명의
완전한 일치는 구분해야 한다.

수동 단계 실행, annotation 대체, resume 저장 위치와 `artifacts`/`workflow_validation`
내용·용량은 [DATA_FLOW.md](DATA_FLOW.md#9-수동-단계-실행과-저장-정책-2026-10-07)에 정리했다.

이번 변경의 회귀 검사는 MuJoCo/UI/runtime/resume/압축/integration 82개,
Isaac USD/workspace/계약/presentation/camera 61개로, 총 143개가 통과했다.
처음 추가한 테스트 중 3건은 fixture에 Task instruction 또는 DomeLight가 빠져
실패했고, 실제 입력 계약에 맞게 fixture를 보완한 뒤 재검사했다. MuJoCo live
worker는 별도 프로세스에서 software-EGL로 Side/Wrist 프레임 생성과 Disconnect를
실행했으며 종료 시 `isaaclab`, `omni`, `pxr`가 import되지 않았음을 검사했다.
Isaac 전체 GPU worker, 실물 Leader/Follower·카메라 연결, GPU 학습 resume와
실제 pick/place 성공률은 이번 회귀 검사의 검증 범위에 포함되지 않는다.

기존 `real2sim_sim.py`와 Runtime/Launcher/UI를 확장했으며 별도 launcher나 창을
추가하지 않았다. 접촉 변경은 기존 MJCF compiler/backend에, 데이터 단계 연결은
기존 WorkflowPanel/Operations에, 교체·이력은 Registry에, 압축은 data_resume에,
학습 재개는 기존 policy/MiniVLA train adapter에 넣었다. 공통 actuator 설정을
robot_spec에서 읽으며 Isaac의 기존 숫자 값은 유지했다. 기존 설치 환경과 과거
데이터를 일괄 변경하거나 삭제하지 않았다.

## 모델 source 통합 검증 (2026-10-07)

최상위 `so101_minivla/`는 모델·설정·문서·테스트를 이동/통합한 뒤 제거했다.
새 모델 package는 `soarm101_lab.learning.minivla`이며 이전 `python -m minivla.…`
명령 대신 이 namespace를 사용한다. UI의 학습/평가 명령도 새 경로를 사용한다.
checkpoint config/state_dict schema와 Qwen/head 구조는 유지했다.

모델·MuJoCo·UI/runtime·resume 통합 검사와 Isaac workflow/계약/settings 검사는
총 **114개 통과**했다. Qwen processor 검사는 처음에 경로 미지정으로 skip됐지만
기존 로컬 cache 경로를 지정한 재검사에서 통과했다. 실제 Qwen 2B/NF4 GPU forward는
재실행하지 않았다. 이전 단독 adapter 테스트가 collection 때 EGL을 import해 CPU
테스트와 충돌하던 문제는 렌더 검사를 별도 software-EGL 프로세스로 분리해 해결했다.

기존 standalone 폴더를 제거한 상태에서 `profile/train/rollout --help`를 실행했고,
현재 published v2 환경으로 독립 MuJoCo worker의 CPU reset/step을 실행해
`passed=true`, `forbidden_imports=[]`를 확인했다. 7개 config/template JSON의
package-data 포함도 확인했다. 출력 데이터 삭제나 패키지 설치/버전 변경은 하지 않았다.

폴더별 삭제 판단 기준은 [DATA_FLOW.md](DATA_FLOW.md#13-outputsminivla-mujoco-real2sim을-삭제해도-되는가-2026-10-07),
모델의 새 실행 명령은 [모델 README](../minivla/README.md)에 정리했다.

## VRAM gate / 자동 단계 진행 제거 (2026-10-07)

사용자 요청으로 VRAM gate stage·보고서 검색·Source/Eval 사전 검증과
`learning/minivla/feasibility.py`를 제거했다. `outputs/minivla/`도 삭제했다.
Source 수집은 Qwen 모델을 검증 목적으로 로드하지 않는다. 모델 단독 `profile.py`는
명시적으로 호출하는 개발 진단 도구이며 UI나 Source/Eval이 실행하지 않는다.

자동 Source → Annotation → Datagen → Convert 체크박스·예약 실행·저장 설정을
제거했다. 각 단계는 사용자가 직접 실행한다. Annotation 전체 검증 성공 시에는
기존 source 경로의 데이터를 annotated 데이터로 교체해 영상 사본을 둘 다 보관하지
않는다. 별도 교체 체크박스도 없앴다. Datagen은 별도 결과다. 실패 또는 일부만
통과한 경우에는 원본을 보존한다.

MuJoCo 시작 중 이전 Isaac의 CLOSED 상태가 backend 선택을 되돌릴 수 있는 경로를
수정했다. 새 worker가 READY를 알리기 전에는 요청한 backend를 유지한다.

이번 제거/수정의 회귀 검사 78개가 통과했다. 현재 published v2 환경에서 실제
RuntimeLauncher의 EGL worker 연결을 검사했고, 별도 offscreen Qt Window에서도
MuJoCo 선택 → Connect → Side/Wrist 픽스맵 표시 → Disconnect를 확인했다.
실물 카메라·Leader/Follower는 연결하지 않았으며 사용자에게 발생한 실패와 같은
원인인지는 아직 확인되지 않았다. 변경 적용에는 실행 중인 UI를 종료하고 다시 연다.


## Real/Sim 관절 제한과 모델별 실행 옵션 (2026-10-08)

실제 MuJoCo IPC 정지 기록에서 `wrist_flex=1.672447 rad`가 URDF 상한
`1.658060 rad`를 넘었다. 기존 Real/Sim 경로는 이 유효한 Leader 목표도
예외로 처리해 전체 제어를 STOPPED로 전환했다.

이제 Runtime은 MuJoCo의 유효한 Leader 목표를 URDF 한계와 gripper의
calibrated 0–100% 범위로 제한한 뒤 같은 목표를 Real Follower와 Sim에 전달한다.
범위 초과 관절과 요청/적용 rad를 상태 메시지와 `leader_target_limits`에 남기며,
Leader가 범위 안으로 돌아오면 별도 Start 없이 추종한다. NaN/잘못된 관절 입력은
여전히 전송 전에 정지한다. 측정 pose는 보정하거나 숨기지 않으며 범위 밖 측정의
capture 거절과 backend의 엄격한 검증은 유지한다. Isaac 명령 경로는 유지한다.

Train에서 MiniVLA를 고르면 기본 Frozen Qwen + Flow matching과 Deterministic
head를 선택할 수 있다. 직접 지정한 model_config 또는 Resume checkpoint의
설정이 있을 때는 이 기본 head 선택을 숨기고 해당 설정을 따른다.
MiniVLA에는 model_config/source_groups/steps/output만 표시하고,
ACT/SmolVLA에는 해당 LeRobot trainer가 받는 batch_size/num_workers/base_model
등의 옵션을 표시한다. 각 모델의 입력값은 모델 전환 시 별도로 유지하며,
다른 모델의 숨겨진 옵션은 실행 요청에 전달하지 않는다.

관련 Runtime/UI/설정/Resume 회귀 검사 **49개 통과**. 관절 제한, 범위 복귀,
NaN 전송 차단은 가짜 Leader/Follower로 검사했다. 실제 연결된 로봇은 조작하지
않았다. 변경 적용에는 기존 Real/Sim을 Disconnect하고 UI를 종료한 뒤 다시 연다.
