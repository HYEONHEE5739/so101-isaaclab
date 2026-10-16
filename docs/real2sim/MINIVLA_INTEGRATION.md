# MiniVLA — 기존 Workspace Pipeline 통합

2026-10-16 통합 Workflow: 새 Source/Annotation/DataGen은 엔진 공통 HDF5+Side/Wrist MP4다.
Legacy 입력은 자동 Import하고, 다른 엔진의 Source는 실제 replay를 검증한다. 공통 Mimic 생성·Source
Partition 계보가 기존 LeRobot→ACT/MiniVLA에 연결된다. 현재 MiniVLA Side-only 고정 모델/H40/6D와
train-only 통계는 유지한다. 자세한 키/단위/시점은 [DATA_FLOW.md](DATA_FLOW.md#통합-episode-계약--2026-10-16).


2026-10-15 현재 구조. 과거 Deterministic/Flow head 선택·별도 UV head 설명을 대체한다.
MiniVLA 모델 package는 기존 `soarm101_lab.learning.minivla` 하나다.
모델 상세는 [DESIGN.md](../minivla/DESIGN.md), [검증](../minivla/VALIDATION.md),
[평가](../minivla/EVALUATION.md), 공통 데이터는 [DATA_FLOW.md](DATA_FLOW.md)를 참조한다.

기존 UI를 사용한다:

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

1. Workspace Pipeline에서 Workspace/Task/backend와 기존 엔진별 Python을 선택한다.
2. Source를 사용자 조작으로 기록한다. 새 Scene의 측정/관절 좌표 계약과30Hz를 유지한다.
   새 Source는 작업영역 전체의 연속 XY에서 회전 없이 배치한다. 물체 크기·가장자리 여유·물체 간 간격을
   고려하며 서로 겹치면 다시 뽑는다. 과거 Source 이어 수집은 당시 배치 정책을 유지한다.
   MuJoCo도 기존 음성 알림을 사용한다. `egl`은 GPU 렌더 경로이고 `software-egl`은 소프트웨어 렌더다.
   MuJoCo 물리 계산은 CPU다. 실제 렌더 장치는 기록의 `scene.render_context`로 확인한다.
3. Replay로 영상/목표를 확인한다. Annotation으로 grasp/place 경계를 생성·검토한다.
4. Datagen/Curation에서 성공한 Source의 변형과 source group을 보존한다.
5. Convert에서 기존 joint-radian/LeRobot 데이터로 변환한다. RGB 원본과 calibration은 보존한다.
6. Train의 학습 모델에서 **MiniVLA · Qwen3-VL + Flow-DiT** 하나를 선택한다.
   head 선택창은 없다. 기본 configs/default.json은 Frozen Qwen/256square/4blocks/H40/flow8/replan4다.
   기존 데이터셋/steps/output/model_config/source_groups 선택과 Resume/Stop은 유지한다.
   QLoRA는 같은 JSON의 adaptation을 qlora로 지정한다. 별도 preset 모델은 없다.
   LeRobot artifact는 Parquet/MP4에서 직접 읽으며 학습 준비에 RGB NPY 복사본을 만들지 않는다.
   원본 Source Group이 명시돼야 하며 UNKNOWN 생성 계보를 독립 train/val로 나누지 않는다.
7. 새 policy artifact를 선택하고 Sim Eval을 실행한다. 기존 best.pt를 실제 MiniVLA로 로드한다.
   새로운 task를 고르면 객체 identity와 성공 판정도 바뀐다. Red 학습/Blue 평가는 평가 문서 참조.

이 단계들은 기존 개별 실행/Registry를 사용한다. 이전에 제거한 자동 단계 예약이나 VRAM gate는
추가하지 않았다. Source는 Qwen을 로드하지 않는다. 실제 장치는 UI 시작/탭 전환으로 연결하지 않는다.
Source/Replay/Annotation/Datagen의 Isaac/MuJoCo 분리도 유지한다. MuJoCo worker에 pxr/Isaac import 없음.
모델 추론·학습은 기존 별도 프로세스에서 실행한다. ACT/SmolVLA 비교 기능은 유지했다.

호출 경로:

```text
WorkflowPanel → Operations → 기존 stage worker
Train → workflow.policy.train / workflow.mujoco.train_from_lerobot 또는 train
      → learning.minivla.train → QwenBackbone + ActionHead
MuJoCo Sim Eval → workflow.mujoco.rollout → MiniVLA.predict → 기존 ActionGuard
Isaac Sim Eval → workflow.evaluation.evaluate → MiniVLAJointPolicy → 같은 MiniVLA.predict
```

Canonical 데이터의 state/action 정규화와 gripper fraction을 유지하고, model rad 변환은 기존
RobotCoordinates/Runtime boundary에서만 적용한다. Precision을 데이터에 중복 적용하지 않는다.
새 기본 모델은 Side만 소비하며 기존 Dataset/환경의 Wrist 영상은 유지한다.

체크포인트 architecture_schema=`so101.minivla.flow-dit/1`. 구형 가중치는 명확히 거절하고
사용자 checkpoint를 바꾸지 않는다. 같은 구조의 Resume은 config/stats/control_hz 및 dataset byte identity
검사 후 optimizer/step/best를 복원한다. 과거 Canonical Resume은 원래 데이터 경로와 형식을 유지한다.
실제 MiniVLA config 변경은 학습 단계의 기존 JSON 입력을 쓴다.

새 Source와 Sim Eval의 시작 로그에서 Hardware Precision / Simulation Residual / Robot /
Calibration Compatibility를 확인한다. 새 Source는 검증된 Scene 잔차 binding을 사용하며,
Replay/Annotation/DataGen/Sim Eval은 기록된 모델을 상속한다. 원형 모델 비교는 기존 Source JSON의
`identity_sim_for_validation: true` 또는 Real/Sim의 URDF 원형 검사 옵션을 사용한다.
원본 데이터의 모델은 현재 설정으로 바꾸지 않는다. [보정 정책](PRECISION_CALIBRATION.md)을 참조한다.

검증용 2-step 합성 checkpoint를 두 엔진의 기존 Sim Eval 경로에서 실행했다. 이는 경로 검사이며
새 Scene의 Source→실물 성공 데모→학습→성공 rollout 전체가 검증됐다는 의미는 아니다.
결과와 실패 기록: `outputs/workflow_validation/minivla_flow_dit/`.

## 새 Source의 가장자리 배치

새 Source 수집은 uniform XY / yaw 0 / 추가 가장자리 여백 0으로 배치한다. 물체 반크기는 계속 빼므로
150×150 mm 지지면과 24×24 mm 물체의 중심 범위는 126×126 mm다(이전 10 mm 여백에서는 106×106 mm).
물체 면이 지지면 가장자리에 닿는 위치까지 허용하며, 다른 물체와의 separation은 유지한다.
연속 균일 추출이므로 모든 episode를 가장자리에 붙이는 방식은 아니다.

Workspace 원본을 바꾸지 않고 새 Source의 `scene_configuration.reset.margin_m=0`을 기록한다.
이 Source의 Replay/Annotation/DataGen은 기록 정책을 상속한다. 이전 Source/생성 결과를 이어 수집할 때는
기록 당시 정책을 유지한다. 과거 데이터의 Curation 경계도 당시 범위로 보인다.

## Data Curation에서 동작 검수

자세한 기능 설명은 **데이터셋 구성 → 설명**을 누르면 팝업으로 확인한다. 기본 화면에는 조작과 실제 결과만 표시한다.
저장은 하단 버튼 하나를 사용한다. 미리보기가 있으면 **Train / Val 저장**, 없으면 **데이터 저장**으로 표시된다.

### Source 기반 Train / Validation 분리

1. Source 수집 직후 Data Curation을 열면 방금 완료된 **원본 Source**가 자동으로 열린다.
   Annotation/DataGen까지 완료했다면 그 최신 결과가 열리므로, Source 분할에는 **데이터 열기**에서 원본 Source를 선택한다.
   **데이터셋 구성**에서 `학습 비율 (Train)` 기본 80%, `분할 랜덤값 (Seed)` 기본 0으로
   **Train/Val 배치 미리보기**를 누른다. 분포 그림은 Train 초록 / Val 파랑, 확인 필요는 주황 테두리다.
   미리보기는 아직 저장 전 계획이다. 같은 데이터·같은 Seed는 같은 분할을 만들고, 값 변경은 다른 조합을 시도한다.
   Seed는 에피소드 개수가 아니다. 선택한 점/목록을 다른 쪽에 넣으려면 학습/검증을 고르고
   **선택 원본 그룹을 이쪽으로 이동**을 누른다. 같은 원본 그룹은 함께 이동한다.
2. 하단 **Train / Val 저장**으로 새 폴더를 만든다. train.hdf5 / val.hdf5(또는 Canonical 폴더)와
   source_split.json을 생성하고 기존 Registry에 두 Source를 등록한다. 원본은 유지한다.
   30개의 독립 Source이면 24/6이다. 중복 group을 묶어야 하면 실제 비율이 달라질 수 있다.
3. 두 Source를 각각 선택해 기존 Annotation→DataGen을 실행한다. 예: Train `num_successful_demos=80`,
   Val `20`. 이 값과 generation_seed는 기존 설정이다. Source split seed는 이 생성 seed와 별개다.
   Partition Source의 Annotation 결과는 별도 artifact이며 원본 Source 파일을 교체하지 않는다.
4. 각 생성 결과를 Curation에서 검토·선별한다. 저장된 split은 잠기고, 생성 episode 수와 원래 Source
   Pool 크기를 구분해 표시한다. Mimic의 정확한 subtask 원본은 UNKNOWN이어도 실제 입력 Source Pool이
   검증되면 Partition 계보를 유지한다. UNKNOWN 문자열을 임의 group 이름으로 바꿔 인정하지 않는다.
5. Train/Val을 각각 기존 Convert로 LeRobot artifact로 만든다. 예: repo_id `local/scene2-train`,
   `local/scene2-val`. MiniVLA Train에서 주 입력은 Train, **Validation Dataset (MiniVLA)**에는 Val을 고른다.
   같은 Source를 공유하면 학습 전 오류다. 데이터셋 MP4를 합치거나 NPY로 다시 저장하지 않는다.
6. 로그의 `Source lineage verified`, Source 수/episode 수, `leakage NONE`과 training/validation loss를
   확인한다. 정규화는 Train만, Best Checkpoint는 기존 Validation Loss로 선택한다. 별도 Validation 선택이
   없고 명시 Partition도 없는 과거 자료는 기존 Source Group 자동 분리를 그대로 사용한다.

Python에서는 기존 `workflow.episodes.source_split()` / `export_source_splits()`로 같은 분할을 쓸 수 있다.
동일한 내용의 Source는 파일 복사·재번호·다른 파일 편입 뒤에도 payload ID로 식별한다. 알려진 group
별칭도 함께 검사한다. Curation에서 Source를 줄이면 허용 Pool만 축소하고 split은 유지한다.
원본이 불명확한 과거 생성 데이터에는 새 Partition 증거를 추측해 붙이지 않는다.

실제 검증 범위·핵심 코드·사용자가 확인할 것:
`outputs/workflow_validation/source_partition_training/REPORT.md`.

1. 기존 Workspace Pipeline의 **Data Curation**을 열면 현재 Workspace의 Source/Annotation/DataGen 중
   가장 최근 완료·등록된 데이터를 자동으로 연다. 단계 우선순위 없이 완료 순서를 사용한다. 다른 데이터는 **데이터 열기** 하나에서
   HDF5 파일 또는 canonical 데이터 폴더를 선택한다. 상단의 Source/Annotation/DataGen 표시와 경로 툴팁으로 입력을 확인한다.
2. Episode를 선택한다. URDF 로봇, EE/Target/Distractor 경로, 관절 목표 그래프와 RGB를 같은 frame으로 본다.
   기본 **영상 · Side / Wrist**에서 두 카메라를 동시에 보고, **로봇 경로**에서 정사각형 뷰를 본다.
   `3D / XY / XZ / YZ`, 마우스 회전·휠 확대, 공통 slider seek, 0.25×/0.5×/1×/2×를 쓴다.
   저장된 Annotation/이벤트는 재생 막대의 색상 마커와 **집기·놓기 등 시점 버튼**에 초·프레임으로 표시한다.
   마커 또는 버튼을 클릭하면 재생을 멈추고 그 프레임의 두 영상·로봇·관절을 함께 확인한다.
   시간은 episode 시작부터의 경과 시간, 프레임 번호는 1부터다. 현재 위치가 어느 주석 전후인지도 표시한다.
   기록된 항목만 표시하며, 예를 들어 grasp만 저장된 Isaac 데이터에서 place 시점을 추측하지 않는다.
3. **에피소드 필터**는 현재 데이터의 Task/물체 구성/목표/Scene/원본 그룹/성공 라벨/Annotation 상태로
   표시할 목록만 좁힌다. 다른 데이터 선택이나 포함/제외 변경은 아니다.
   **상세 분석**의 초기 분포는 위에서 본 Workspace XY다. 90도 단위로만 방향을 맞춰 테두리를 반듯하게 표시한다.
   보라색 네모는 같은 배율·좌표계의 실제 로봇 기준점이다(네모 크기는 표시용). 방향 화살표는 사용하지 않는다.
   점선은 지지면, 실선은 목표 물체 중심 배치 경계(지지면 반크기 − 목표 반크기 − margin)다.
   다른 물체와의 separation 조건도 적용되며, four-anchor 기록은 외곽 안의 네 anchor 주변에서만 배치된다.
   필터를 바꿔도 배율/경계는 고정된다. 기록 Workspace가 없으면 경계를 추측하지 않는다.
4. 확인 필요는 같은 작업·물체 구성(5개 이상)에서 시간/정지 비율/명령 변화가 상대적으로 튀는 항목이다.
   목록에 사유를 표시하고 선택하면 영상 위와 **상세 분석**에 실제 값·비교 기준·중앙값을 표시한다.
   자동 실패 판정이나 자동 제외가 아니다. **상세 분석**에서 Target 이동·Lift·EE 거리·관절 목표·품질 지표를 검토한다.
   Source Group/Scene hash, 객체별 접근, Grasp 전후 이동은 품질 지표 툴팁에서 확인한다. 이벤트는 저장된 annotation/
   signal만 표시한다. `기록 라벨`은 파일 라벨이고 `물리 판정: 판정 근거 없음`은 물리 판정 근거가 없다는 뜻이다.
   가까운 접근 자체로 grasp를 인정하지 않는다. 초기 위치밖에 없으면 `Trajectory unavailable`이다.
5. **데이터셋 구성**의 `사용할 데이터 수`를 입력하고 **적용**하면
   현재 포함된 후보에서 지정한 개수만 사용하고 나머지는 제외한다. Task/물체 구성/초기 XY 4×4 영역의 비율을 최대한 유지하고 원본 그룹 쏠림을 줄인다.
   사용할 데이터 수가 현재 포함 수와 같으면 변화가 없다. Train/Val 배정 기능과는 별개다.
   **제외 취소 · 모두 포함**은 수동/자동 제외를 모두 취소한다. 파일·로봇 상태는 변경하지 않는다.
   포함/제외나 분할 숫자를 바꾸면 기존 미리보기를 지우고 다시 만들도록 하며, **미리보기 지우기**는 포함 상태를 유지한다.
6. 하단 **데이터 저장**으로 포함된 에피소드를 새 경로에 Export/Registry 등록한다. 기록된 분할은 유지한다.
   분할 미리보기가 있으면 같은 버튼이 **Train / Val 저장**으로 바뀌어 두 데이터를 저장한다. 기존 Convert에서 해당 artifact를 고르고
   MiniVLA Train으로 진행한다. 원본과 Source Group은 유지하며 동일 group을 train/validation에 나누지 않는다.

실제 미기록 object 동작을 복원하지 않으며 Curation에서 physics를 다시 실행하지 않는다.
추가 telemetry 형식은 [DATA_FLOW.md](DATA_FLOW.md), 학습 계약은
[DATA_CONTRACT.md](../minivla/DATA_CONTRACT.md)를 참조한다. 새 수집부터 모든 active pose가 기록된다.
저장된 Workspace를 찾을 수 없으면 geometry/등록 근거가 부족하다는 표시를 확인한다.
실행 검증과 제한: `outputs/workflow_validation/curation_object_playback/REPORT.md`.

## 새 Source 영상 저장

UI를 재시작하고 기존과 같은 Source Demo 수집을 시작한다. 두 엔진 모두 숫자 HDF5와
같은 위치의 `.videos` 폴더를 함께 보존한다. 과거 MuJoCo 숫자 NPY+MP4도 Legacy Reader로 계속 읽는다.
Curation/Replay/Annotation/DataGen/Convert는 저장 형식을 자동으로 읽는다. 기존 Dataset을 변경하지 않는다.
새 MuJoCo Convert 결과는 LeRobot Dataset 자체이며 RGB를 복제한 Canonical 폴더를 추가하지 않는다.
품질을 바꾸려면 UI 실행 환경의 `SO101_VIDEO_CRF`를 지정한다(기본18). 해상도/FPS는 바꾸지 않는다.
새 외부영상 Annotation은 원본 Source를 유지하는 별도 artifact다. `.videos`만 삭제하면 해당 영상은 읽을 수 없다.
자세한 구현/측정: `outputs/workflow_validation/rgb_video_storage/REPORT.md`.

## 설정 입력 반응성

설정 입력·체크 변경 후 실행 가능 여부는 입력이200ms 멈추면 백그라운드에서 검사한다.
`CHECKING` 중에는 입력·편집을 계속할 수 있고 실행 버튼은 검사가 끝나면 활성화된다.
이전 설정의 검사 결과는 새 설정에 적용하지 않는다. 실제 실행에서는 최신 설정과 원본 파일을 다시 검증한다.
UI 재시작 후 적용된다. 모델·보정·Workspace 제한이나 장치 연결 동작은 바꾸지 않았다.
