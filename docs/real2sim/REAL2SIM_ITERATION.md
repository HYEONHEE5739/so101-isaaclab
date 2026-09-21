# SO-101 Real2Sim Bootstrap & Iteration Protocol

이 문서는 Codex/agent가 **실제 SO-101 환경과 Isaac Sim 관측을 반복적으로 맞춰 가는 작업 절차**를 정의한다.

이 문서는 시스템 구조 설명이 아니다. 실제로 scene/profile을 개선할 때 따라야 하는 운영 프로토콜이다.

관련 문서:
- [ARCHITECTURE.md](ARCHITECTURE.md): 전체 구조
- [DATA_FLOW.md](DATA_FLOW.md): 단위/좌표/데이터 흐름
- [INVARIANTS.md](INVARIANTS.md): 절대 깨뜨리면 안 되는 규칙
- [TESTING.md](TESTING.md): 검증 기준
- [PROJECT_MAP.md](PROJECT_MAP.md): 수정 위치

---

## 1. Goal

최우선 목표는 다음 관측을 실제 환경과 최대한 일치시키는 것이다.

- Real Side Camera ≈ Sim Side Camera
- Real Wrist Camera ≈ Sim Wrist Camera

목표는 보기 좋은 3D scene이 아니라, 기존 SO-101 Isaac Lab 환경에서 그대로 사용할 수 있는 **측정 가능하고 재현 가능한 Real2Sim profile/workspace**를 만드는 것이다.

우선 보정 대상:

- camera intrinsics / extrinsics
- robot base alignment
- object pose
- table / workspace / scene alignment

후순위:

- object visual geometry
- material / texture / color
- lighting / background

기존 teleoperation, Mimic, datagen, dataset, policy, training/evaluation semantics는 Real2Sim fitting 때문에 불필요하게 바꾸지 않는다.

---

## 2. Source of truth

Real2Sim의 source of truth는 **versioned profile revision**이다.

```text
Real references / paired captures
        ↓
measurement / calibration / optimization
        ↓
candidate profile revision
        ↓
deterministic scene builder
        ↓
Isaac Sim render
        ↓
Real vs Sim metrics
        ↓
accept / reject / needs-more-data
```

환경별 숫자를 Python 코드나 USD에 숨겨 hardcode하지 않는다.

Accepted profile을 직접 파괴적으로 덮어쓰지 않는다. 개선안은 새 revision으로 만든다.

---

## 3. Provenance

숫자 값은 출처를 명시한다.

- `measured`: 사용자가 실제로 측정한 값
- `calibrated`: calibration 절차로 얻은 값
- `optimized`: 명시적 objective/metric으로 fitting한 값
- `configured`: 사용자가 명시적으로 입력한 값
- `provisional` / `assumed`: bootstrap 또는 임시 초기값

사진을 보고 정한 camera pose, object pose, table pose를 measured/calibrated로 기록하지 않는다.

측정하거나 최적화할 수 있는 값을 LLM이 숫자로 임의 추측해서 확정하지 않는다.

---

## 4. Input types

### Registered Real References

초기 bootstrap용 정적 실제 이미지/영상.

예:
- side
- wrist
- overview
- auxiliary

이 입력은 scene 구조와 초기 composition을 파악하는 데 쓴다.

### COMPLETE Paired Captures

iteration과 quantitative validation에 사용하는 동기화된 capture.

가능하면 포함:
- Real Side
- Sim Side
- Real Wrist
- Sim Wrist
- measured real robot joint state
- sim joint state
- timestamps
- profile/revision id
- capture quality/status

Reference와 paired capture를 같은 의미로 취급하지 않는다.

---

## 5. Frozen vs Tunable

각 iteration은 반드시 **고정할 것과 수정할 것을 구분**한다.

### 보통 frozen

- user/measured geometry
- SO-101 robot kinematics/model asset
- 비교 대상 capture의 measured joint state
- 현재 단계와 무관한 parameter groups
- 이미 충분히 검증된 accepted 값
- dataset/action contract

### 보통 tunable

- side camera intrinsics/extrinsics
- wrist camera intrinsics/extrinsics
- robot base alignment
- workspace/table alignment
- object pose
- provisional object visual geometry
- material / texture / color
- lighting / background
- gripper closing correspondence

한 번에 너무 많은 parameter group을 동시에 열지 않는다.
서로 다른 오차가 서로를 보상하면 원인을 잃는다.

---

## 6. Bootstrap Mode

Bootstrap의 목적은 **첫 usable profile**을 만드는 것이다.

### Step 1 — Real references 확인

reference manifest와 실제 이미지를 직접 확인한다.
파일명만 보고 view 의미를 추측하지 않는다.

### Step 2 — Known measurements / assets 확인

우선순위:

1. 실제 측정값
2. SO-101 URDF/USD/model asset
3. camera calibration data
4. deterministic image measurements
5. provisional visual estimate

실측된 geometry는 visual matching 때문에 임의 변경하지 않는다.

### Step 3 — Scene structure 파악

다음을 확인한다.

- SO-101 base 위치/방향
- Side camera 구도
- Wrist camera 구도
- robot과 workspace의 상대 위치
- table/work surface
- object 종류와 배치
- 큰 background 구조
- 주요 occlusion
- 주요 visual appearance

### Step 4 — Geometry-first initial profile

일반 우선순위:

```text
camera / robot / workspace alignment
        ↓
object pose
        ↓
object shape
        ↓
material / texture
        ↓
lighting / background
```

큰 geometry 오류를 material/lighting으로 숨기지 않는다.

### Step 5 — Initial render

초기 profile로 Side/Wrist render를 만든다.

이 단계에서는 synchronized joint state가 없을 수 있으므로 정밀 calibration 완료라고 주장하지 않는다.

### Step 6 — Bootstrap iteration

한 번 렌더하고 끝내지 않는다.

```text
Real references
    ↓
provisional profile
    ↓
Isaac render
    ↓
largest mismatch
    ↓
one/few parameter groups 수정
    ↓
rerender
```

첫 bootstrap의 목표는 완벽한 scene이 아니라 **다음 paired-capture iteration이 가능한 수준의 scene**이다.

---

## 7. Iteration Mode

Iteration의 목적은 accepted/current profile을 실제 paired capture로 정량 개선하는 것이다.

### Step 1 — Current state inventory

자동으로 확인:

- current / accepted profile
- COMPLETE captures
- measured robot states
- current renders
- available metrics
- capture quality
- parameter provenance

### Step 2 — Representative captures 선택

처음부터 모든 capture를 fitting에 사용하지 않는다.

대표 pose를 4~8개 정도 선택한다.

선택 기준:
- pose 다양성
- side/wrist image quality
- capture completeness
- gripper opening 다양성
- hold-out validation 가능성

### Step 3 — Fit / validation split

가능하면 대표 capture 일부를 fitting용, 일부를 validation용으로 분리한다.

같은 capture로 fitting하고 같은 capture만 보고 성공을 선언하지 않는다.

### Step 4 — Root-cause classification

가장 큰 mismatch가 어디서 오는지 구분한다.

- side camera
- wrist camera / mount
- robot base
- workspace/table
- object pose
- object geometry
- gripper correspondence
- material/color
- lighting/background
- joint mapping/data mismatch

### Step 5 — Parameter group 선택

이번 iteration에서 바꿀 parameter group을 명시한다.

예:

```text
Tunable:
- wrist camera extrinsic
- wrist focal length

Frozen:
- robot base
- side camera
- workspace
- objects
- joint mapper
```

### Step 6 — 계산 가능한 것은 계산

우선순위:

1. measured data
2. calibrated data
3. SO-101 FK/model
4. synchronized joint state
5. deterministic image measurement
6. numerical optimization
7. provisional visual reasoning

LLM은 optimizer 대신 숫자를 찍지 않는다.

### Step 7 — Candidate revision 생성

Accepted profile을 수정하지 않고 새 candidate revision을 만든다.

### Step 8 — Same-pose rerender

Real capture가 robot state `Q`에서 찍혔다면 Sim도 동일한 `Q`로 render한다.

candidate에 맞추기 위해 robot pose 자체를 임의 변경하지 않는다.

### Step 9 — Sparse preview

대표 capture만 빠르게 rerender하고 metric을 확인한다.

추천 metric:
- landmark/keypoint reprojection error
- silhouette/mask overlap
- object center/corner reprojection error
- image diagnostics
- joint discrepancy
- timestamp/capture quality

하나의 RGB metric만 좋아졌다고 전체 scene이 좋아졌다고 판단하지 않는다.

### Step 10 — Broader validation

Sparse 결과가 개선되면 더 많은 capture 또는 hold-out set에서 다시 검증한다.

대표 capture에서만 좋아지고 broader set에서 악화되면 reject한다.

### Step 11 — Decision

결과는 다음 중 하나로 명시한다.

- `ACCEPTED`
- `REJECTED`
- `NEEDS_MORE_DATA`
- `FAILED`

데이터가 부족하면 억지로 숫자를 추측하지 않는다.

---

## 8. Wrist-specific fitting

현재 Side가 대체로 맞고 Wrist mismatch가 큰 경우 wrist부터 분리해서 본다.

### Camera fitting 먼저

고정된 gripper body / wrist structure를 기준으로:
- image 내 크기
- 좌우/상하 위치
- image edge와의 거리
- silhouette
- effective FOV

를 사용해 wrist camera pose/FOV를 먼저 맞춘다.

기본 frozen:
- arm joint mapper
- robot base
- side camera
- workspace
- object poses
- table
- lighting

기본 tunable:
- wrist camera extrinsic
- wrist focal/FOV
- 필요한 경우 mount visual geometry

### Gripper correspondence는 별도

Real gripper와 Sim gripper opening이 다르면 arm joint calibration으로 보상하지 않는다.

순서:

1. fixed gripper body를 이용해 wrist camera pose/FOV fitting
2. camera 값을 고정
3. 여러 opening 상태로 gripper correspondence fitting
4. hold-out opening state validation

camera pose와 gripper mapping을 동시에 무제한 최적화하지 않는다.

---

## 9. Robot base vs camera compensation

화면 mismatch를 camera로 전부 보상하지 않는다.

예:
- robot이 실제보다 90° 돌아가 있음 → robot base alignment 문제
- workspace가 robot 기준으로 틀어져 있음 → workspace/table transform 문제
- object 하나만 틀림 → object pose 문제

parameter의 물리적 의미를 유지한다.

---

## 10. Code change rule

일반적인 iteration에서 주로 바뀌는 것은:

- profile revision
- generated scene/visual assets
- workspace version

매 iteration마다 Python source를 수정하지 않는다.

코드 수정은 현재 시스템이 필요한 표현/측정/검증 기능을 제공하지 못할 때만 한다.

예:
- 필요한 camera mount geometry 표현 기능이 없음
- 필요한 metric이 없음
- representative capture selection 기능이 없음
- current profile schema가 필요한 parameter를 표현하지 못함

새 dependency나 큰 구조 변경이 필요하면 먼저:
- 왜 필요한지
- 어느 파일이 바뀌는지
- 기존 기능으로 왜 해결 안 되는지
- Mimic/datagen/policy 등에 어떤 영향이 있는지

를 설명한다.

---

## 11. Prohibited actions

하지 말 것:

- 기존 Sim screenshot을 Real ground truth로 사용
- robot/base가 틀렸는데 camera를 왜곡해 보상
- 사진에서 본 숫자를 measured 값처럼 기록
- measured geometry를 visual matching 때문에 임의 변경
- arbitrary 2D warp로 차이를 숨김
- synchronized data가 있는데 계속 눈대중으로 pose 수정
- iteration마다 scene-specific 숫자를 Python에 hardcode
- profile에 기록하지 않고 USD만 직접 수정
- reference와 paired capture의 의미를 섞음
- Mimic/datagen/policy를 Real2Sim fitting 때문에 변경
- 불필요한 별도 reconstruction framework 추가
- 명확한 evidence 없이 arm sign/zero calibration부터 수행
- 여러 parameter group을 동시에 열어 원인 분리를 불가능하게 만듦

---

## 12. Bootstrap report

Bootstrap 후 최소 보고:

- 사용한 real references
- 사용한 measured/configured 값
- 생성한 profile revision
- 수정/생성한 scene 요소
- Real Side vs Sim Side 주요 차이
- Real Wrist vs Sim Wrist 주요 차이
- provisional인 값
- 다음 paired capture에서 검증할 항목
- 추가 데이터가 필요한 항목

---

## 13. Iteration report

각 iteration 후 최소 보고:

- baseline profile/revision
- fitting captures
- validation/hold-out captures
- 가장 큰 mismatch
- 이번에 수정한 root cause
- tunable fields
- frozen fields
- measurement/calibration/optimization 방법
- 생성한 candidate revision
- before → after metrics
- sparse preview 결과
- broader validation 결과
- ACCEPTED / REJECTED / NEEDS_MORE_DATA / FAILED
- 아직 가장 크게 남은 mismatch
- 다음 iteration에서 열 parameter group

이 보고는 큰 변경이면 [CHANGELOG_AI.md](CHANGELOG_AI.md)에도 남긴다.

---

## 14. Codex invocation examples

### Bootstrap

```text
docs/real2sim/REAL2SIM_ITERATION.md와 INVARIANTS.md를 따라
현재 SO-101 실제 환경의 Real2Sim bootstrap을 수행해.

registered real references와 repository의 기존 asset/profile/runtime을 직접 확인하고,
측정 가능한 값을 임의 추측하지 마.

Side/Wrist observation matching을 우선하고,
결과는 새 profile revision으로 남겨.
```

### One iteration

```text
docs/real2sim/REAL2SIM_ITERATION.md와 TESTING.md를 따라
현재 accepted Real2Sim scene을 한 iteration 개선해.

COMPLETE paired captures에서 representative fitting set과 validation set을 선택하고,
가장 큰 mismatch의 root cause를 하나 또는 소수 parameter group으로 좁혀라.

동일 robot state로 rerender하고 before/after metric을 제시한 뒤
ACCEPTED / REJECTED / NEEDS_MORE_DATA / FAILED 중 하나로 결론을 남겨.
```

## 15. Stopping condition

"완벽해 보일 때"가 아니라 다음 기준으로 iteration을 멈춘다.

- 핵심 metric이 목표 허용 범위에 들어옴
- hold-out capture에서 regression이 없음
- 남은 mismatch가 현재 policy/task 성능에 의미 있는 영향을 주지 않음
- 또는 추가 개선에 필요한 measurement/data가 없어 `NEEDS_MORE_DATA` 상태임

즉, 반복의 목표는 무한한 시각적 미세조정이 아니라 **검증 가능한 충분한 correspondence**다.
