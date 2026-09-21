# SO-101 Real2Sim Data Flow

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
  ├─ Isaac articulation state/target 적용
  └─ forward/render
          │
          ▼
Sim side/wrist image
```

이 경로의 목적은 follower를 다시 명령하는 것이 아니라, **실제 측정 pose와 동일한 model configuration을 render**하는 것이다.

## 4. Camera / image flow

```text
Real side camera ─┐
Real wrist camera ├─ capture state + timestamps
                  │
Sim side camera  ─┤
Sim wrist camera ─┘
          │
          ▼
paired capture
          │
          ├─ raw images
          ├─ camera/profile metadata
          ├─ robot state
          └─ capture timing metadata
          │
          ▼
calibration / metrics / optimization
          │
          ▼
profile revision
```

Real/Sim 비교는 가능하면 동일 robot configuration에서 수행한다.
단순한 시각적 인상 대신 reprojection/image/landmark 등 수치 metric을 사용한다.

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
