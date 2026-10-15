# MiniVLA — 기존 Workspace Pipeline 통합

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

## Data Curation에서 동작 검수

### Source 기반 Train / Validation 분리

1. Source 수집 후 Data Curation에서 **원본 Source**를 연다. `Source Train` 기본 80%, `Seed` 기본 0으로
   **Source 분할 미리보기**를 누른다. 목록의 train/val과 실제 개수를 확인하고 필요하면 선택 Source의
   소속을 바꾼다. 같은 원본·동일 내용의 복제 episode는 서로 다른 split으로 나눌 수 없다.
2. **Train / Validation 저장**으로 새 폴더를 만든다. train.hdf5 / val.hdf5(또는 Canonical 폴더)와
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

1. 기존 Workspace Pipeline의 **Data Curation**을 열고 Source/Datagen HDF5 또는 canonical 폴더를 선택한다.
2. Episode를 선택한다. URDF 로봇, EE/Target/Distractor 경로, 관절 목표 그래프와 RGB를 같은 frame으로 본다.
   `3D / XY / XZ / YZ`, 마우스 회전·휠 확대, Side/Wrist 선택, slider seek, 0.25×/0.5×/1×/2×를 쓴다.
3. Task/Active count·IDs/Target/Scene hash/Source group/Recorded success/Annotation 필터로 확인한다.
   초기 Scatter는 workspace XY이고 오른쪽 동작은 robot-base 좌표다. 위치 숫자를 직접 섞지 않는다.
4. Target 이동·Lift·EE 거리·Grasp 전후 이동과 객체별 접근을 검토한다. 이벤트는 저장된 annotation/
   signal만 표시한다. `Recorded success`는 파일 라벨이고 `Physical UNKNOWN`은 물리 판정 근거가 없다는 뜻이다.
   가까운 접근 자체로 grasp를 인정하지 않는다. 초기 위치밖에 없으면 `Trajectory unavailable`이다.
5. 제외/복원 후 최종 개수와 **분포 유지 자동 선별**을 사용한다. 포함 후보 전체에서 Task/active 구성별
   초기 XY 비율을 보존하고 Source Group 쏠림을 줄인다. 필터는 표시용이며 불량을 자동 확정하지 않는다.
6. **선택 데이터셋 저장**으로 새 경로에 Export/Registry 등록한다. 기존 Convert에서 해당 artifact를 고르고
   MiniVLA Train으로 진행한다. 원본과 Source Group은 유지하며 동일 group을 train/validation에 나누지 않는다.

실제 미기록 object 동작을 복원하지 않으며 Curation에서 physics를 다시 실행하지 않는다.
추가 telemetry 형식은 [DATA_FLOW.md](DATA_FLOW.md), 학습 계약은
[DATA_CONTRACT.md](../minivla/DATA_CONTRACT.md)를 참조한다. 새 수집부터 모든 active pose가 기록된다.
저장된 Workspace를 찾을 수 없으면 geometry/등록 근거가 부족하다는 표시를 확인한다.
실행 검증과 제한: `outputs/workflow_validation/curation_object_playback/REPORT.md`.

## 새 Source 영상 저장

UI를 재시작하고 기존과 같은 Source Demo 수집을 시작한다. MuJoCo는 Episode 폴더의 `side.mp4`,
`wrist.mp4`와 기존 숫자 NPY를 저장한다. Isaac은 HDF5와 같은 위치의 `.videos` 폴더를 함께 보존한다.
Curation/Replay/Annotation/DataGen/Convert는 저장 형식을 자동으로 읽는다. 기존 Dataset을 변경하지 않는다.
새 MuJoCo Convert 결과는 LeRobot Dataset 자체이며 RGB를 복제한 Canonical 폴더를 추가하지 않는다.
품질을 바꾸려면 UI 실행 환경의 `SO101_VIDEO_CRF`를 지정한다(기본18). 해상도/FPS는 바꾸지 않는다.
새 외부영상 Annotation은 원본 Source를 유지하는 별도 artifact다. `.videos`만 삭제하면 해당 영상은 읽을 수 없다.
자세한 구현/측정: `outputs/workflow_validation/rgb_video_storage/REPORT.md`.
