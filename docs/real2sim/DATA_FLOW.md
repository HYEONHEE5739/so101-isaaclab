# SO-101 Real2Sim Data Flow

## 새 RGB 저장 계약 (2026-10-15)

새 MuJoCo Source/DataGen은 숫자 NPY와 `side.mp4`/`wrist.mp4`를 저장한다. `EpisodeWriter.append`
→ 기존 `AsyncVideoWriter`의 bounded queue(카메라별 8 frame) → H264 CRF18. 프레임 드롭 없이
backpressure를 사용하므로 느린 CPU에서는 append가 기다릴 수 있다. 해상도/FPS/카메라 수는 기존 실행 설정 그대로다.
`SO101_VIDEO_CRF` 환경변수(0~51, 낮을수록 고품질)로 새 수집 품질을 바꿀 수 있다.

Isaac Source 및 Workspace Annotation, Mimic DataGen은 기존 `EpisodeData`에 숫자만 쌓는다.
RGB는 스트리밍하고 `VideoHDF5DatasetFileHandler`가 확정한 뒤 HDF5 episode의 `external_videos`
JSON attribute에 상대경로·shape·FPS·frame 수·codec을 기록한다. 영상은 `<dataset>.videos/<고유폴더>/`
안에 있으며 HDF5와 함께 이동한다. UUID는 subset/Resume 재번호 충돌을 방지한다. 관측 시각은 기존 `obs/timestamp`다.
설치된 Isaac Reader는 숫자 dataset을 재귀적으로 읽고 video attribute를 무시한다. Mimic Pool은
`obs/datagen_info`와 actions를 사용하므로 Source RGB 텐서가 없어도 된다. upstream 코드를 수정하지 않는다.
비 Workspace legacy Annotation의 기본 HDF5 recorder는 기존 형식을 유지한다.

Canonical `VideoFrames`는 PTS로 요청 frame을 검증하고 RGB uint8 HWC 하나만 반환한다.
열린 decoder는 최대2개이며 Curation의 8frame/16MiB LRU와 background seek는 그대로다.
기존 RGB NPY/내부 HDF5도 읽는다. 숫자 timestamp 검사는 유지하고 MP4 frame 번호를 대응시킨다.
MP4 영상의 PTS는 episode-relative CFR이며 실제 장치 노출 timestamp의 새 측정값이 아니다.

새 Episode는 숨긴 staging에서 인코딩한 뒤 frame 수를 검사하고 numeric/manifest를 확정한다.
정상 Stop/취소/예외는 미완성 영상을 폐기한다. 강제 SIGKILL은 staging을 남길 수 있지만 정식 Episode에 등록되지 않는다.
Resume/subset은 이전 데이터와 영상을 덮어쓰지 않고 필요한 영상만 새 결과에 복사한다.
외부 영상 Annotation은 Source 교체 대신 별도 annotated artifact로 등록하여 참조 경로를 보존한다.
artifact/source payload hash는 외부 MP4 bytes까지 포함한다. 기존 내부 HDF5/NPY identity는 유지한다.

MuJoCo Convert는 Source를 read-only로 선택하고 숫자+영상참조만 작은 임시 HDF5에 전달한다.
기존 Converter는 HDF5/NPY/MP4를 frame 단위로 읽어 표준 LeRobot API에 전달한다.
새 결과는 LeRobot MP4/Parquet 자체로 등록하며 Canonical RGB 복사본을 만들지 않는다.
원본 MP4를 무조건 복사하지 않고 기존 LeRobot codec/metadata 계약으로 재인코딩한다.
기존 LeRobot Direct/MiniVLA H40/6D/Source Partition/Train-only 통계/Resume 계약은 그대로다.
검증과 용량 비교: `outputs/workflow_validation/rgb_video_storage/REPORT.md`.

## Source Partition → Mimic → MiniVLA (2026-10-15)

`source_split()`은 원본 읽기만 하고, `export_source_splits()`는 기존 `export_subset()`으로 두 새
Source를 만든다. HDF5 env_args / Canonical provenance의 `source_partition`, episode의
`source_lineage`가 payload ID와 원본 Dataset/episode를 보존한다. source_split.json은 전체 계획이다.
동일 payload 또는 알려진 group 별칭은 함께 배정하며 seed로 재현한다.

Annotation은 성공적으로 export한 원본 이름의 순서를 전달해 실패·건너뛰기 후 재번호도 추적한다.
이 매핑이 없거나 출력 개수와 다르면 Partition 계보 저장을 거절한다. Mimic DataGen은 `validate_input()`에서 실제 입력 Partition을
검사하고 기존 `generation.py`가 그 한 파일만 Pool에 로드한다. 완료 후 `preserve_metadata()`가
입력 hash/실제 Source 집합을 기록하고 정확한 subtask 원본은 UNKNOWN으로 남긴다.
분할된 Source는 Annotation 성공 뒤에도 원본 경로를 교체하지 않는다. 다른 Partition의 Resume 병합은 거절한다.

Curation은 split을 유지하며 Source subset은 Pool을 좁히고 생성 subset은 원래 입력 Pool을 보존한다.
기존 Converter가 partition/input evidence/episode lineage/Review와 좌표 계약을 LeRobot provenance로
전달한다. 별도 Train/Val Registry artifact는 MiniVLA manifest에서 두 root를 참조하고 각각 lazy decode한다.
`split_episodes()`가 원본 ID 교집합을 검사하고, Train-only 통계 및 기존 Validation Loss/Best 선택으로
이어진다. 알려지지 않은 계보를 새 증거로 재라벨하지 않는다. 실제 모터/성공 rollout은 별도 검증이다.

## Direct 학습과 Calibration 계약 (2026-10-15)

Isaac은 HDF5 Source→Annotation→DataGen→Curation→기존 LeRobot Convert를 유지한다.
MiniVLA는 Convert의 MP4/Parquet를 `Episodes`에서 직접 읽는다. 학습 준비는 작은 index manifest와
train-only statistics뿐이며 전체 RGB NPY를 생성하지 않는다. MuJoCo Canonical 학습도 유지한다.
기존 ACT/SmolVLA LeRobot 경로는 바꾸지 않는다. 로더/Resume 세부 계약은
[DATA_CONTRACT.md](../minivla/DATA_CONTRACT.md#lerobot-direct-2026-10-15)를 따른다.

새 Source의 coordinate contract에는 `simulation_model`이 들어간다. 공통 resolver
`precision.simulation_configuration`은 numeric URDF joint frame과 geometry residual을 분리한다.
승인된 role mapping과 Builder의 residual binding 근거로 모델을 결정한다. Scene2 v8은 Follower
binding이며 Source Leader mapping과 hash가 다를 수 있다. 명시 identity 검사는 기존 옵션을 쓴다.

```text
Leader raw → Leader Precision → numeric URDF state/target
                          └ Source contract: role mappings + bound Sim model
Source → Replay/Annotation → DataGen → Curation → Convert → Train/checkpoint
        기록 계약 상속          같은 IK/Physics 모델                  └ Sim Eval: 같은 모델
                                                               └ Real Eval: Follower inverse
```

`simulation_model`은 Workspace/content·URDF·Precision role/hash·residual/hash·mode/applied·실행
URDF/USD·backend·Scene 구성 및 self hash를 가진다. HDF5는 `so101_coordinates`/env_args,
Canonical은 project_provenance/coordinate sidecar, LeRobot은 sidecar/workflow_provenance,
checkpoint는 기존 project_provenance/coordinate sidecar로 전달한다. 학습 manifest의 storage는
원본 LeRobot 전체 byte hash도 보존한다. 새 checkpoint의 dataset_identity는 Resume의 입력 byte를 검사한다.

파생 단계는 현재 UI 체크박스나 활성 Profile로 원본 모델을 바꾸지 않는다. 다른 active object 구성의
Sim Eval도 robot model을 유지한다. Real Eval은 승인 Follower의 URDF 변환을 먼저 확인하며 Sim 잔차를
하드웨어 target에 추가하지 않는다. 과거 identity/legacy 동작은 원래 계약으로 복원하고 증거 없는 값은
UNKNOWN으로 남긴다. 자세한 정책은 [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md#workflow-공통-모델-결정-2026-10-15).

## Curation Object Telemetry (2026-10-15)

`obs[t]`의 관절 상태·EE·RGB·물체 pose를 같은 **명령 적용 전** 시점에 기록한다.
기존 action/tensor/좌표 계약을 바꾸지 않고 다음 선택적 telemetry를 추가한다.

| 입력 | 저장 및 해석 |
|---|---|
| MuJoCo canonical | 기존 Target `object_pose.npy` [T,4,4] 재사용. 나머지 active만 `object_poses.npy` float32 [T,N,7], xyz+wxyz, workspace 좌표 |
| Canonical manifest episode | `object_telemetry`: ID→file/index/pose_format/frame/timestamp_file. `timestamp.npy`를 공유하고 RGB/action을 복제하지 않음 |
| Isaac Source / Mimic DataGen | `obs/object_poses/<id>` float32 [T,7], xyz+wxyz, world 좌표. `obs/timestamp` float64 [T,1] simulation time |
| Isaac env_args | `object_telemetry`: ID→path/pose_format/frame/timestamp_path, `observation_timestamp_path`, `ee_frame=robot_base` |
| 이벤트 / outcome | 실제 존재하는 grasp/place signal 및 annotation만 표시. `success_source`로 operator 저장과 simulation 판정 근거 구별 |

기록 Workspace path/identity/checksum으로 당시 Profile을 읽고, canonical EE/objects는
`inv(T_world_base) @ T_world_workspace`, Isaac world objects는 `inv(T_world_base)`로 변환한다.
Isaac EE는 기존 robot-base 좌표를 유지한다. 로봇은 기록 좌표 계약의 원형/Scene 보정 URDF를 쓴다.
근거가 없는 world/base 등록은 추측하지 않고 해당 overlay를 비활성화한다.
객체 geometry/color는 Profile에서 읽는다. box는 실제 치수, 외부 mesh/cup 등은 치수 기반 preview proxy다.

구형 canonical의 target-only 기록은 target 경로만 사용한다. 과거 HDF5의 초기 pose만으로
동작을 추정하지 않는다(`Trajectory unavailable`). 실제 post-step 시계열과 initial state가 모두
있으면 frame t를 `initial[0] + post_state[:-1]`로 복원한다. 이 경우 timestamp는 기록 control_dt의
시간격자다. 명시된 quaternion 순서와 기록 robot root가 있으면 두 root가 공유하는 기록 좌표계에서
직접 base로 변환한다. Isaac의 environment-relative state를 world pose로 잘못 해석하지 않는다.
하드웨어 노출시각을 실측한 것으로 표시하지 않는다. 명시된 timestamp 불일치,
관측보다 미래인 RGB, 2주기 이상 오래된 RGB는 거절한다.
과거 `obs/camera/<view>` RGB도 lazy frame으로 읽는다. Workspace가 유실된 파일은 기록된
target만 검토하고 Active Set이나 현재 URDF를 임의로 대입하지 않는다. 모델/등록 근거가 없는
overlay와 객체 동작은 unavailable이며, RGB·EE·관절 목표 그래프 검토는 계속 가능하다.

분석은 telemetry를 2048-frame chunk로 읽으며 RGB 전체를 읽지 않는다. 선택 episode plot은
최대1500 sample, frame 재생은 원본 정확한 index를 사용한다. RGB는 background lazy read와
최대8frame/16MiB LRU다. 초기 summary 분석은 전체 frame 수에 비례하며 UI worker에서 실행한다.

Curation Export는 기존 subset 복사/Registry를 사용한다. Task/Scene/Source Group/annotation/
성공 라벨은 보존한다. Converter의 `episode_review`는 성공 근거·annotation 경로와 원본 pose
reference를 유지하며 학습 tensor에 추가하지 않는다. HDF5 pose telemetry는 원본 HDF5에 남고
LeRobot로 중복 복사하지 않는다. canonical prepared subset은 기존 포맷 그대로 pose 파일도 보존한다.
Annotation은 Source Group을 유지한다. 원본 데모 계보를 특정할 수 없는 Isaac 생성 결과는
공통 `UNKNOWN:<source file>` 그룹이며 서로 독립인 train/validation 자료로 나누지 않는다.
기존 source-group split 검사 및 MiniVLA H40/6D 입력은 그대로다.

## 실행별 Scene 구성 (2026-10-15)

Published Workspace identity와 실행별 Scene 구성을 구분한다. `workflow.tasks.configured()`는
원본을 복사하고 비선택 dynamic을 profile/physics roles에서도 제거한다. 객체 순서는 Workspace
순서로 정규화한다. 기존 Request/Artifact에 다음 `scene_configuration`을 넣는다.

```text
workspace: 원본 package metadata/hash
active_object_ids: 정규화한 조작 물체 ID 목록
pick_object_id / place_target_id / task_definition
reset: 기존 Workspace reset 설정 (mode/margin/separation/seed 포함)
backend: isaac 또는 mujoco
sha256: 위 payload의 정렬된 canonical JSON SHA256
```

실행 hash는 원본 Workspace hash를 대체하지 않는다. MuJoCo model XML/hash/state/physics 검사는
추가로 유지한다. Isaac은 구성의 dynamic만 Spawn하고 reset은 그 ID와 정확히 같은 위치 집합을 받는다.
원본 measured q, Precision Profile 및 RobotCoordinates 변환에는 영향을 주지 않는다.

| 경로 | 저장·복원 위치 |
|---|---|
| Isaac Source / Annotation / DataGen | HDF5 `data.env_args.scene_configuration`, episode `scene_configuration`/`task_definition` attrs |
| MuJoCo Source | `project_provenance.json.scene_configuration`, `scene.scene_configuration`, manifest episode의 같은 필드 |
| Replay / Annotation / DataGen | 입력의 저장 구성으로 복원. 현재 UI 선택으로 원본을 재라벨하지 않음 |
| Resume | 원본 workspace/task/physics/model/좌표 계약 검사와 active set 동일성 검사 |
| Curation / Convert | episode 속성 보존, 구성별 `scene_configurations`, `episode_scene_configurations` |
| Train / checkpoint | 기존 provenance에 episode→구성, task, source group을 보존 |
| Sim Eval | 학습 set과 평가 set을 각각 기록. 구성 차이 자체는 거절하지 않음 |

혼합 조합 데이터는 episode별 구성과 Task를 명시하여 기존 데이터 형식으로 결합해 학습할 수 있다.
Source group 분리는 그대로다. 한 환경으로 Replay/Annotation/DataGen할 때는 동일 구성의 Source
subset을 선택한다. 섞인 구성을 하나로 추측하여 재생하지 않는다.
과거 MuJoCo의 명시적 `scene.dynamic_objects`, 또는 동일 Workspace의 HDF5 initial rigid-object
state에서 확정 가능한 구성은 읽기 전용으로 복원한다. 근거가 없으면 `UNKNOWN`이며 전체 3개로
가정하지 않는다. 구형 Dataset/Workspace를 새 구성으로 덮어쓰지 않는다.

## 현재 실시간 관절 경로 (2026-10-12)

Leader raw/SDK native → Precision/legacy mapping → requested URDF q
→ 기존 Runtime의 URDF/Sim 목표 제한 → strict Follower inverse → SDK integer conversion
→ 같은 encoded q를 Follower와 Isaac/MuJoCo에 전달.
모터 범위에서 표현할 수 없으면 명령 전에 거절한다. 별도 Controller/시작 자세/속도/추종
Guard/자동 복귀가 없다. Follower raw/native → 잘리지 않은 measured q → Snapshot.
설명·제거 범위·허용/거절 정책은 [JOINT_CONTROL.md](JOINT_CONTROL.md)를 따른다.
과거 Dataset 좌표계와 Snapshot reader는 보존한다.


이 문서는 값이 **어디서 와서, 어떤 단위/좌표계로 바뀌고, 어디에 저장되는지**를 추적하기 위한 문서다.

## 1. Joint representation

현재 핵심 표현은 다음처럼 구분한다.

| 경계 | 표현 | 단위 |
|---|---|---|
| Motor register | encoder/raw | device unit |
| LeRobot calibrated arm | calibration-relative joint | degree |
| LeRobot gripper | normalized gripper | 0..100 |
| SO-101 model / Isaac Lab | articulation joint | radian |
| Isaac JointPositionAction target | model joint target | radian |

핵심 mapper는 hardware 값을 Isaac model 좌표로 변환하는 단일 경계를 유지해야 한다.
motor homing/calibration과 model joint zero/sign을 같은 것으로 취급하지 않는다.

## 2. Teleoperation

```text
SO-101 Leader
  │ encoder
  ▼
LeRobot calibration
  │ arm degree / gripper %
  ▼
SO-101 joint mapping
  │ model radians
  ▼
Isaac Lab JointPositionAction
  │ rad target
  ▼
Sim robot
```

Follower hardware 경로는 follower 자신의 calibration inverse를 사용한다.
Leader calibration 파일을 follower calibration 대신 사용하면 안 된다.

## 3. Measured Real ↔ Sim pose comparison

```text
Follower measured state
  │ calibrated degree / %
  ▼
common SO-101 mapper
  │ model radians
  ▼
runtime.set_measured_pose(...)
  │
  ├─ 선택 backend의 articulation/joint state 적용 (Isaac 또는 MuJoCo)
  └─ forward/render
          │
          ▼
Sim side/wrist image
```

이 경로의 목적은 follower를 다시 명령하는 것이 아니라, **실제 측정 pose와 동일한 model configuration을 render**하는 것이다.

## 4. Camera / image flow

```text
Real side/wrist camera + Follower measured q
          │
          ▼
현재 상태 캡처 → 연결 세션의 SnapshotDataset
          ├─ 공통 robot_model/calibration (한 번)
          └─ 샷별 원본 PNG + 실측 state + timestamp/quality
          │
          ▼
독립 real2sim-builder import-snapshots
          │
          ▼
candidate fitting / render / compare / validate / dual export
```

SO-101 UI의 Real/Sim 4-view는 실시간 확인용이다. 옛 paired 캡처와 UI 내부 보정 루프는
제거했고, 재구성·수치 metric 계산은 독립 Builder가 수행한다.

## 5. Real2Sim profile

`profile.py`가 관리하는 값은 최소한 아래 provenance를 구분해야 한다.

- `measured`: 실제 측정치
- `calibrated`: calibration 절차로 얻은 값
- `optimized`: objective/metric을 최소화하여 얻은 값
- `configured`: 사용자가 명시적으로 설정한 값
- `assumed` / `provisional`: 아직 검증되지 않은 값

camera intrinsics/extrinsics, robot base alignment, object pose, table/scene alignment은 가능한 경우 measurement/calibration/optimization으로 결정한다.

## 6. HDF5 / demo contract

Dataset에서 이름이 비슷한 값도 의미를 구분한다.

- observation/joint state: 실제 또는 sim에서 관측한 state
- action: 해당 step에 적용한 command/target
- joint target: controller/action path에서 사용한 model target
- initial state: replay 시작 시 복원해야 하는 articulation/object 초기 상태
- timestamp/metadata: real/sim correspondence 검증에 필요한 시간/출처 정보

현재 source demo/replay 경로는 workspace contract를 함께 검증하며, replay 가능한 dataset인지 확인한 뒤 state를 복원한다.
세부 저장 schema를 변경할 때는 `so101_dataset_contract.py`, recorder/converter/replay 경계를 함께 검사한다.

## 7. Workspace → workflow

```text
Accepted Real2Sim profile
        ▼
Publish workspace version
        ▼
Workspace package
        ▼
Source demo / replay
        ▼
Annotation
        ▼
Mimic datagen
        ▼
LeRobot conversion
        ▼
Train
        ▼
Sim / Real evaluation
```

Real2Sim 보정 때문에 Mimic/datagen/policy의 의미를 바꾸지 않는다. Workspace는 기존 파이프라인에 **환경/표현 계약을 제공하는 입력**이다.

## 8. MuJoCo / MiniVLA 공통 workflow

UI → Operations → backend worker → Registry 순서는 공통이다. MuJoCo는 canonical 폴더에 관측 직전 state/RGB와 적용 target을 기록한다. `raw_state/raw_action`은 6개 모두 model rad이고 MiniVLA `state/action`은 arm 5 rad + gripper fraction이다. requested target과 URDF limit 적용 후 target은 별도로 보존한다. Replay에는 같은 model hash의 integration state를 사용한다.

Annotation은 ordered grasp/place + 물리적 placement를 검사한다. Datagen은 새로운 physics rollout과 RGB를 기록하고 원 source group을 유지한다. Convert는 기존 HDF5→LeRobot converter에 run-local adapter를 제공한다. ACT/SmolVLA는 LeRobot의 model-radian action을, MiniVLA는 LeRobot Direct 또는 기존 Canonical 입력의 group split/train-only statistics를 사용한다. MiniVLA를 기존 Isaac 평가에 연결할 때 model-radian facade가 단위를 변환한다.

공통 Hub dataset upload는 MuJoCo의 `lerobot/` 내용을 repository 루트에 게시한다. 자세한 실행 흐름과 한계는 [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md)를 참조한다.

## 9. 수동 단계 실행과 저장 정책 (2026-10-07)

Source, Annotation, Datagen, Convert, Train/Eval/Upload는 각각 사용자가 실행한다.
자동 다음 단계 실행 기능은 제거했다. 단계별 설정은 화면을 이동해도 유지한다.

Annotation은 별도 작업 경로에서 결과를 검증한 후, 전체 episode가 통과하면 **원래
source 경로의 내용과 좌표 sidecar를 annotation 결과로 대체**한다. 이름이
`source.hdf5`여도 Registry 유형은 `annotated`이며, Source와 Annotated 영상 파일을
둘 다 남기지 않는다. MuJoCo canonical 폴더에도 같은 규칙을 적용한다.
기존 source 등록은 `SUPERSEDED` 상태로 이력만 남는다. Datagen은 별도 데이터다.
Annotated 데이터도 Replay/직접 Convert 입력으로 선택할 수 있다.

실패하면 원본을 보존한다. 일부 episode만 annotation을 통과하면 원본을 대체하지
않고 통과한 별도 annotated 결과를 등록한다. 원본의 실패 episode를 잃지 않기 위한
예외다. 교체·등록 중 예외에는 원본과 sidecar를 복구한다. 예기치 않은 프로세스 종료로
`.annotation_backup`이 남았다면 복구용 원본이므로 확인 없이 삭제하지 않는다.

ACT/SmolVLA 학습에서 Resume checkpoint를 고르고 **저장 경로를 비우면 기존
`checkpoints/`의 상위 학습 폴더**에 저장한다. MiniVLA는 기존 `last.pt/best.pt`
폴더에 저장한다. 실행 로그/request/state는 새 run 폴더에 남지만 모델을 새 학습
폴더에 복제하지 않는다. MiniVLA는 기존 준비 dataset과 통계를 재사용하며 모델
설정은 같아야 한다. `steps`는 저장 step보다 큰 총 목표로 늘릴 수 있다.

## 10. outputs/workflow의 파일은 무엇인가?

| 위치 | 생성 시점 / 역할 |
|---|---|
| `artifacts/<UUID>.json` | Workspace/데이터/모델을 가져오거나 단계 결과 검증에 성공하면 생성. 실제 데이터 위치, 유형, workspace/task/backend, checksum, 부모 artifact ID를 기록하는 인덱스 |
| `runs/<시간>_<단계>_<task>_<ID>/state.json` | 실행 시작/성공/실패/중지 상태와 명령·출처 |
| 같은 폴더의 `process.log` | worker stdout/stderr. 오류 원인과 진행 로그 |
| `request.json`, `tasks.json` | worker에 전달할 실행 설정·Task Definition. script 경로에 따라 request가 없는 run도 있음 |
| `presentation/` | UI용 JPEG 프리뷰, 상태와 조작 명령. 학습용 원본 RGB와 별개 |
| `source.hdf5`, `annotated.hdf5`, `datagen.hdf5`, `dataset/`, `policy/` 등 | 실제 데이터/모델. Backend와 단계에 따라 HDF5 또는 canonical/LeRobot 폴더 |
| `ui.json` | 최근 workspace, backend, backend별 Python, 실행 모드 설정 |

UUID JSON은 모델이나 영상의 복사본이 아니다. `parents`는 입력의 ID이고 `path`는
실제 파일 위치다. 같은 파일을 다시 가져올 때 내용과 유형이 같으면 기존 등록을
재사용한다. UI 입력 목록은 존재하지 않는 경로와 대체된 기록을 숨긴다. 기록을
지워도 원본 데이터가 지워지는 것은 아니며, 반대로 부모 기록만 삭제하면 연결을
추적하기 어려워진다. 전체 목록과 실제 경로·사용 관계는 다음으로 확인한다.

```bash
python scripts/real2sim/workflow.py list --summary
```

2026-10-07 점검 당시 artifact는 **39개 / 약 0.97 MiB**, 실제 경로는 37종,
그중 **23개 기록의 경로가 이미 없는 상태**였다. 이 과거 기록은 삭제하지 않았다.
`fc712b…`, `c5d126…`는 Sim Eval 보고서 등록이고 해당 보고서 경로가 현재 없다.
`0ccd21…`는 Source 등록이고 원본 경로가 없다. `8a4edd…`는 아래 validation의
`lerobot_phase2_v2`를 가리키는 Dataset 등록이며 실제 폴더가 있다.

## 11. outputs/workflow_validation은 무엇인가?

일반 UI가 매번 자동으로 만드는 폴더가 아니라, 개발 중 각 기능을 따로 검증할 때
지정한 출력 경로다. 일반 workflow 실행은 `outputs/workflow/runs`로 저장된다.

| 폴더 | 현재 들어 있는 검증 결과 |
|---|---|
| `annotation_ui/` | Annotation HDF5, 좌표 sidecar, 실행 로그·검증 JSON. 약 904.5 MiB |
| `act_smoke_v2/` | ACT 1-step 학습 smoke checkpoint, optimizer/RNG 등 training state. 약 158.1 MiB; 성능 검증용 학습 모델이 아님 |
| `datagen_30_comparison/` | `positions_mm.csv`와 `report.json`: 생성 위치 비교 수치 |
| `physics_compiler/` | 검증용 workspace package/asset 복사와 physics report. 약 38.3 MiB |
| `lerobot_phase2/`, `lerobot_phase2_v2/` | HDF5→LeRobot 변환 시험의 parquet, MP4, 메타데이터 |
| `operator_replay/`, `stage_ui/` | Replay/Eval 보고서, UI screenshot, presentation 프레임과 상태 |
| `task_selection/` | Task 선택·변환·평가 입력과 검증 결과 |
| `unified_pipeline/` | backend 공통 단계 연결을 시험한 작은 데이터·보고서·등록 기록 |
| 루트의 `annotated_*`, `replay_*.json`, `sim_eval_smoke.json` | 별도 annotation/replay/eval 진단 결과 |

합계 약 **1.1 GiB**다. 검증 경로에 있더라도 Registry로 가져온 데이터는 실제
학습 입력이 될 수 있다. 특히 `lerobot_phase2_v2`는 등록된 Dataset이므로 validation
폴더 전체를 단순 임시 파일로 보고 삭제하면 안 된다. 이번 변경에서 과거 데이터,
checkpoint, artifact 기록은 일괄 삭제하지 않았다.

## 12. HDF5 용량

640×480 RGB uint8 한 장은 약 0.879 MiB다. 카메라 2개 × 30 Hz × 20초라면
episode 하나의 RGB만 압축 전 약 **1.03 GiB**다. HDF5에는 영상뿐 아니라 replay
state, joint target, FK/물체 pose와 annotation 정보도 들어간다.

현재 가장 큰 `20260922_003428_datagen_…/datagen.hdf5`는 약 **8.06 GiB**이고
배열의 논리 크기는 약 **35.7 GiB**다. `obs/side_cam`, `obs/wrist_cam`이 큰 비중을
차지하며 **이미 gzip 압축**돼 있다. 압축을 켜는 것만으로 더 크게 줄어들지는 않는다.
그 다음 annotation HDF5는 약 4.39 GiB다. Source/Annotation/Datagen을 각각 보관하면
같은 길이의 영상이 여러 번 들어가므로 Annotation의 source 대체가 중복을 줄인다.

새 Workflow HDF5는 완료 후 압축되지 않은 큰 숫자/RGB 배열을 블록 단위로 무손실
gzip 압축한다. dtype·값·메타데이터와 hard link를 유지하고, 이미 압축된 배열은
재압축하지 않는다. 기존 등록된 HDF5는 일괄 재작성하지 않는다. MuJoCo→LeRobot
변환의 임시 HDF5는 성공 후 제거한다. Canonical `side.npy/wrist.npy`는 여전히
비압축 RGB다. 영상 FPS/해상도를 낮추거나 MP4 중심 schema로 바꾸는 작업은 이번에
수행하지 않았으며, 그런 변경은 학습·replay 계약도 함께 검증해야 한다.

## 13. outputs/minivla, mujoco, real2sim을 삭제해도 되는가? (2026-10-07)

폴더 이름은 backend 코드가 아니라 실행 결과의 저장 위치다. 실행 코드는
`source/soarm101_lab/soarm101_lab/`에 있으므로 검증 결과를 지운다고 MuJoCo 코드가
지워지지는 않는다. 하지만 `outputs`에는 검증 결과와 실제 입력 자료가 함께 있다.
이번 요청으로 `outputs/minivla`를 삭제했다. 다른 outputs 데이터와 등록 기록은 보존했다.

| 폴더 / 점검 용량 | 실제 역할 | 삭제 판단 |
|---|---|---|
| `outputs/mujoco` / 436 KiB | Phase 2 compile·physics·EGL·OSMesa 검증의 run-local scene, RGB, report | 현재 실행 입력/등록 artifact 참조를 찾지 못했다. 검증 증거가 필요 없다면 정리 가능 |
| `outputs/minivla` / 삭제됨 | 과거 VRAM gate·화면 비교 결과 | VRAM gate 기능을 제거하고 사용자 요청으로 삭제했다 |
| `outputs/real2sim` / 293 MiB | 실측 사진·capture·profile revision·published workspace·texture asset·UI 설정과 runtime IPC | 전체 삭제 금지. 현재 환경과 보정 자료가 여기에 있다 |
| `outputs/workflow_validation` / 1.1 GiB | 개발 검증 결과, 일부 실제 등록 데이터와 checkpoint | 이름만 보고 전체 삭제하지 않는다. 아래 등록 항목을 먼저 확인한다 |

### MuJoCo / MiniVLA 결과

`outputs/minivla`와 이를 찾던 VRAM gate 실행/검색 코드를 제거했다. Source/Eval은
해당 보고서 없이 실행되며 삭제한 폴더를 gate 용도로 다시 만들지 않는다.
과거 artifact의 execution 기록에 남은 gate 경로는 당시 실행의 이력일 뿐이다.
`outputs/mujoco`의 Phase 2 검증 결과는 이번 요청의 삭제 대상에 포함하지 않았다.

### Real2Sim의 보관 항목

- `environments/scene1_builder/v1`: 현재 선택한 Builder 기반 runtime Workspace.
  `builder/`의 원본 geometry와 `scene/`의 Isaac 참조 레이어, 제어 로봇 asset, task/physics를 함께 사용한다.
- 기존 `workspace_001/v1`, `v2` 및 `workspace_tasks/workspace_001.json`은 삭제했다.
  이전 HDF5와 출처 기록은 보존했다. 삭제된 환경의 Replay/Datagen은 원래 패키지 복구 없이는
  재현할 수 없으며, 새 Scene으로 metadata를 바꾸어 호환 처리하지 않는다.
- `device_profile.json`: Capture용 카메라 값과 기존 evidence를 보존한 설정이다.
  물리 카메라 보정 완료를 뜻하지 않는다. Real/Sim은 선택한 Builder Workspace profile을 별도로 사용한다.
- `snapshots`: 현재 Snapshot 수집 데이터. Start→Disconnect마다 공유 모델/calibration과 샷을 묶는다.
- `ipc`, `runtime_startup.log`: 실시간 프리뷰·명령·잠금과 worker 로그.
  엔진 임시 파일은 `ipc/scene_<backend>_<id>/`에서 생성하고 정상 Disconnect 때 제거한다.
- `bootstrap_inputs`, `references`, `revisions`, 옛 `runs`, `generated`는 이전 정리에서 삭제했다.
  과거 artifact의 출처 문자열은 당시 이력이며 scene 생성 코드가 아니다.
