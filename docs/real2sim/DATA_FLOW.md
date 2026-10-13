# SO-101 Real2Sim Data Flow

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

Annotation은 ordered grasp/place + 물리적 placement를 검사한다. Datagen은 새로운 physics rollout과 RGB를 기록하고 원 source group을 유지한다. Convert는 기존 HDF5→LeRobot converter에 run-local adapter를 제공한다. ACT/SmolVLA는 LeRobot의 model-radian action을, MiniVLA는 canonical split/statistics를 사용한다. MiniVLA를 기존 Isaac 평가에 연결할 때 model-radian facade가 단위를 변환한다.

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
