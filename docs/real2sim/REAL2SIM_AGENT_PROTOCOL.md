# SO-101 Real2Sim Agent Protocol

## 1. Goal

이 프로젝트의 목표는 실제 SO-101 작업환경과 Isaac Sim 환경의 관측을 최대한 일치시키는 것이다.

최우선 목표는 다음 두 관측이다.

- Real Side Camera ≈ Sim Side Camera
- Real Wrist Camera ≈ Sim Wrist Camera

목표는 보기 좋은 3D scene이 아니라,
기존 SO-101 Isaac Lab 구조에서 그대로 사용할 수 있는 Real2Sim scene/profile을 만드는 것이다.

다음 기존 구조는 가능한 유지한다.

- SO-101 robot implementation
- leader/follower teleoperation
- Mimic
- datagen
- dataset
- policy
- training/evaluation


---

## 2. Core Principle

Real2Sim의 source of truth는 profile revision이다.

```text
Real data
    ↓
measurement / reasoning / optimization
    ↓
profile revision
    ↓
deterministic scene builder
    ↓
Isaac Sim
    ↓
Real vs Sim comparison

USD나 Python 코드 안에 환경별 scene 값을 직접 숨겨서 관리하지 않는다.

환경이 달라지면 새 profile revision을 만든다.

accepted profile을 파괴적으로 덮어쓰지 않는다.

3. Provenance Rules

숫자는 반드시 출처를 구분한다.

user_measured
사용자가 직접 측정한 값
model_asset
SO-101 URDF/USD/model에서 얻은 값
derived
측정값에서 계산한 값
provisional
reference image를 보고 bootstrap용으로 추정한 초기값
numerically_fitted
실제 측정 데이터와 optimization으로 구한 값

사진을 보고 추정한 camera pose, object pose, table pose 등을
측정값처럼 기록하지 않는다.

사용자가 채팅으로 제공한 실제 측정값은
그 환경의 user_measured 값으로 기록한다.

4. Data Types

이 프로토콜은 다음 두 종류의 입력을 구분한다.

4.1 Registered Real References

Bootstrap용 정적 reference 이미지/영상.

예:

side
wrist
overview
auxiliary

이들은 scene을 처음 세팅할 때 사용한다.

4.2 COMPLETE Captures

Iteration용 paired capture 데이터.

가능한 경우 포함:

Real Side
Sim Side
Real Wrist
Sim Wrist
measured real robot joints
Sim joints
timestamps
profile/revision 정보
capture quality

Reference와 Capture의 의미를 섞지 않는다.

5. Frozen vs Tunable Fields

각 bootstrap / iteration 단계에서
무엇을 고정하고 무엇을 수정할 수 있는지 명확히 구분한다.

5.1 Frozen Fields

현재 단계에서 변경하지 않는 값.

예시:

accepted profile의 확정값
user_measured geometry
SO-101 robot kinematics/model asset
measured robot state
현재 iteration의 비교 대상 capture
arm joint mapper (기본값: identity 유지)
명시적으로 freeze한 parameter group
5.2 Tunable Fields

현재 단계에서 candidate revision을 위해 수정 가능한 값.

예시:

side camera extrinsics / FOV
wrist camera extrinsics / FOV
robot base alignment
workspace pose
object pose
object provisional visual geometry
material / texture / color
lighting / background
gripper closing correspondence

하나의 단계에서 너무 많은 parameter group을 동시에 열지 않는다.

원인 분리가 가능하도록 필요한 최소 group만 tunable로 둔다.

6. Default Robot Policy
6.1 Arm Joint Mapping

기본적으로 arm 5축 mapper는 기존 identity를 유지한다.

q_urdf = deg2rad(q_lerobot)

또는 현재 repository의 기존 identity-equivalent 경계를 유지한다.

명확한 증거가 없는 한,
arm sign / zero_offset calibration을 먼저 하지 않는다.

6.2 Gripper Mapping

gripper는 arm과 별도로 다룬다.

Real gripper는 percent 계열 좌표이고,
Sim gripper는 URDF articulation 좌표일 수 있다.

육안상 Real과 Sim의 gripper closing amount가 다르면,
gripper correspondence를 별도의 tunable group으로 다룬다.

6.3 Wrist Camera Priority

현재 상태에서 Real/Sim 차이가 wrist camera view에서 더 크게 보인다면,
arm joint calibration보다 먼저 다음을 우선한다.

wrist camera / mount
gripper visual alignment
gripper closing correspondence
7. Bootstrap Mode

사용자가 다음과 같이 요청하면 bootstrap을 수행한다.

docs/real2sim/REAL2SIM_AGENT_PROTOCOL.md를 따라
현재 환경의 Real2Sim bootstrap을 실행해.

이번 환경에서 내가 직접 측정한 값:
- ...
- ...
- ...

나머지는 registered real references와 repository를 직접 확인해.
Side/Wrist observation matching을 최우선으로 해.
7.1 Bootstrap Inputs
현재 환경의 registered real references
사용자가 제공한 실제 측정값
기존 SO-101 robot asset
현재 repository의 Real2Sim profile/scene/runtime 코드

기존 Sim screenshot은 bootstrap 입력으로 요구하지 않는다.

기존 task scene의 물체 배치를 실제 환경의 정답으로 사용하지 않는다.

8. Bootstrap Workflow
Step 1 — Real References 확인

reference manifest를 읽고 실제 이미지를 직접 확인한다.

가능한 view:

side
wrist
overview
auxiliary

파일명만 보고 의미를 추측하지 않는다.

Step 2 — Scene Structure 파악

reference를 보고 다음을 확인한다.

SO-101이 놓인 위치와 방향
Side camera의 구도
Wrist camera의 구도
robot과 workspace의 상대 위치
table/work surface
object 종류와 배치
background의 큰 구조
주요 색상과 재질
눈에 띄는 texture/pattern
Step 3 — Observation 우선 matching

가장 먼저 다음을 최대한 비슷하게 만든다.

REAL SIDE ↔ SIM SIDE
REAL WRIST ↔ SIM WRIST

주요 확인 항목:

viewing direction
camera position/orientation
effective FOV
robot 크기와 위치
workspace 크기와 위치
object 위치
table edge
occlusion
background composition

화면이 다르다고 camera만 무조건 움직이지 않는다.

예:
robot이 실제보다 90° 돌아가 있으면
camera로 보상하지 말고 robot base alignment를 수정한다.

Step 4 — Robot Alignment

SO-101 자체 형상과 kinematics는 기존 model asset을 사용한다.

profile의 robot base alignment를 수정해
실제 환경의 robot 방향과 위치를 최대한 맞춘다.

특히 다음과 같은 큰 오류를 먼저 해결한다.

90° rotation
잘못된 table-facing direction
잘못된 base position
workspace와 robot의 잘못된 상대 위치

Bootstrap 단계에서는 이러한 값은 provisional이다.

Step 5 — Wrist Camera Geometry

실제 robot에 wrist camera와 bracket이 있지만
SO-101 asset에 해당 형상이 없으면 Real2Sim scene에 추가한다.

구조는 다음처럼 유지한다.

SO-101 wrist link
    ↓
camera bracket / mount
    ↓
camera body
    ↓
camera optical frame

실측되지 않은 mount 크기와 transform은 provisional이다.

SO-101 원본 asset 자체를 환경별로 파괴적으로 수정하지 않는다.

Step 6 — Object Shape / Material

실제 observation에서 중요한 형상은 가능한 한 재현한다.

예:
종이컵이 tapered shape이면 단순 cylinder만 유지하지 않는다.

사용자가 측정한 값은 그대로 유지한다.

예:

높이
직경
두께

사진에서 추정한 값은 provisional visual geometry로 관리한다.

예:

taper
bottom diameter
rim shape

Bootstrap부터 observation에 크게 보이는 appearance는 재현한다.

예:

컵의 printed pattern
cube 색상
workspace border
table 색상
camera body
background
주요 material
Step 7 — Bootstrap Iteration

Bootstrap도 한 번 렌더하고 끝내지 않는다.

가능한 경우 다음을 반복한다.

Real references
    ↓
provisional profile
    ↓
Isaac render
    ↓
Real Side/Wrist vs Sim Side/Wrist
    ↓
가장 큰 mismatch 확인
    ↓
profile 또는 visual asset 수정
    ↓
rerender

reference와 robot joint state가 synchronized되어 있지 않다면
정밀 calibration이나 numerical fitting이 완료됐다고 주장하지 않는다.

9. Iteration Mode

사용자가 다음과 같이 요청하면 iteration을 수행한다.

docs/real2sim/REAL2SIM_AGENT_PROTOCOL.md를 따라
현재 Real2Sim scene을 한 번 개선해.

현재 accepted profile과 COMPLETE captures를 직접 찾아서 사용하고,
Real Side/Sim Side와 Real Wrist/Sim Wrist의 차이를 분석한 뒤
가장 큰 원인을 개선해.

내가 별도로 문제 위치를 지정하지 않아도
capture와 repository data를 기반으로 스스로 판단해.
10. Iteration Workflow
Step 1 — 현재 상태 자동 확인

자동으로 확인한다.

current / accepted profile
COMPLETE captures
measured robot states
current renders
available metrics
capture quality
Step 2 — Representative Capture Selection

모든 capture를 바로 쓰지 않는다.

먼저 다양한 robot pose를 대표하는 소수의 capture를 선택한다.

예:

4 ~ 8개의 representative captures

selection 기준 예시:

pose 다양성
side/wrist view quality
capture completeness
gripper opening 다양성
validation에 쓸 수 있는 hold-out 가능성
Step 3 — Sparse Preview

대표 capture들만 사용하여 빠르게 candidate를 평가한다.

representative captures
    ↓
candidate profile
    ↓
sparse rerender
    ↓
comparison / diagnostics

여기서 통과하지 못하면 전체 capture validation으로 가지 않는다.

Step 4 — Root Cause 판단

여러 robot pose에서 다음 mismatch의 원인을 구분한다.

Side camera
Wrist camera / mount
robot base
workspace
table
object pose
object geometry
material / color
lighting
gripper closing correspondence
Step 5 — 계산 가능한 값은 계산

우선순위:

user measurement
SO-101 model / FK
synchronized measured robot state
deterministic image measurement
numerical optimization
provisional visual reasoning

측정하거나 optimization 가능한 값을
눈대중으로 숫자 추측해서 기록하지 않는다.

Step 6 — 새 Candidate Revision 생성

accepted profile을 직접 덮어쓰지 않는다.

새 revision을 만든다.

Step 7 — 동일 robot pose로 rerender

Real capture가 robot state Q에서 촬영됐다면,
candidate Sim도 동일한 Q로 render한다.

candidate에 맞추려고 robot pose를 바꾸지 않는다.

Step 8 — Sparse Comparison

기존 repository에서 사용할 수 있는 metric을 우선 활용한다.

예:

landmark reprojection
edge/silhouette difference
image diagnostics
joint discrepancy
validation error

하나의 RGB metric만 좋아졌다고 전체 scene이 좋아졌다고 판단하지 않는다.

Step 9 — Full Validation

Sparse preview가 충분히 좋아지면
더 많은 capture 또는 전체 validation set에서 다시 확인한다.

candidate profile
    ↓
full rerender on broader capture set
    ↓
validation metrics / comparison

대표 capture에서만 좋아지고 나머지에서 망가지면 reject한다.

Step 10 — 결과 결정

결과는 다음 중 하나로 기록한다.

ACCEPTED
REJECTED
NEEDS_MORE_DATA
FAILED

데이터가 부족하면 숫자를 추측하지 않는다.

필요한 추가 capture가 있다면 구체적으로 제안한다.

11. Wrist-First Rule

현재 evidence상 Side robot pose가 대체로 맞고
Wrist mismatch가 더 크게 보이면,
다음 순서로 접근한다.

11.1 Wrist camera self-view 우선

먼저 Real Wrist와 Sim Wrist에서
scene object보다 robot self appearance를 우선 비교한다.

주요 evidence:

fixed gripper body
left/right finger visible extent
gripper silhouette
wrist camera mount/body
image edge에서 gripper까지의 거리
gripper가 image에서 차지하는 size
gripper의 좌우/상하 위치
11.2 Wrist camera fitting 단계의 frozen fields

이 단계에서는 다음을 freeze하는 것이 기본이다.

arm joint mapper
robot base
side camera
workspace
object poses
table
lighting

이 단계의 주요 tunable fields:

wrist T_gripper_camera
wrist focal/FOV
필요 시 wrist visual mount geometry
11.3 Gripper closing correspondence

Real이 Sim보다 더 많이 닫히거나 덜 닫히면,
arm calibration으로 보상하지 않는다.

gripper는 별도의 tunable group으로 처리한다.

가능하면 다음과 같은 evidence를 사용한다.

left/right finger edge distance
jaw opening width in pixels
finger tip distance
gripper silhouette width

gripper mapping은 여러 capture의 다른 opening 상태를 이용해
fitting set / validation set으로 나눠 검증한다.

camera pose와 gripper mapping을 동시에 무제한 fitting하지 않는다.

순서:

A. fixed gripper body를 이용해 wrist camera pose/FOV를 먼저 맞춘다.
B. camera pose를 고정한다.
C. gripper opening correspondence를 맞춘다.

12. Geometry Before Appearance

큰 geometry 오류가 있는 상태에서
lighting이나 color만 바꿔 차이를 숨기지 않는다.

일반 우선순위:

camera / robot / workspace alignment
        ↓
object pose
        ↓
object shape
        ↓
material / texture / color
        ↓
lighting / background

단 실제 evidence가 다른 원인을 보여주면
그 원인을 우선한다.

13. Code Change Rules

일반적인 Bootstrap/Iteration에서 주로 변경되는 것은:

profile revision
필요한 generated visual asset

Python 코드는 매 iteration마다 수정하지 않는다.

코드 수정은 현재 시스템이 필요한 기능 자체를 표현하지 못할 때만 한다.

예:

wrist camera bracket을 표현할 기능이 없음
tapered object를 표현할 기능이 없음
representative capture / sparse validation 기능이 없음

반면 다음은 profile 변경으로 처리한다.

camera 위치 수정
robot base transform 수정
object pose 수정
material / lighting 수정

새 dependency나 큰 구조 변경이 필요하면 구현 전에 먼저 보고한다.

왜 필요한지
어느 파일이 바뀌는지
기존 기능으로 왜 해결할 수 없는지
기존 pipeline에 어떤 영향이 있는지
14. What Must Not Change

Real2Sim 작업 때문에 다음을 불필요하게 수정하지 않는다.

SO-101 robot implementation
leader/follower teleoperation
Mimic
datagen
dataset
policy
training/evaluation

목표 구조:

Real references / captures
        ↓
Real2Sim profile
        ↓
Real2Sim scene
        ↓
existing SO-101 Isaac Lab environment
        ↓
teleop / Mimic / datagen / policy
15. Prohibited Actions

하지 말 것:

기존 Sim screenshot을 Real ground truth로 사용
robot이 틀렸는데 camera를 왜곡해서 보상
사진에서 본 metric 값을 측정값처럼 기록
user-measured geometry를 visual matching 때문에 임의 변경
arbitrary 2D image warp로 차이를 숨김
synchronized data가 있는데 계속 눈대중으로 pose 수정
iteration마다 scene-specific 숫자를 Python에 hardcode
profile에 기록하지 않고 USD만 직접 수정
Capture와 free reference의 의미를 섞음
Mimic/datagen/policy를 Real2Sim fitting 때문에 변경
불필요한 새로운 reconstruction framework 추가
arm joint sign/zero calibration을 명확한 증거 없이 먼저 수행
여러 tunable group을 동시에 열어 원인 분리를 불가능하게 만듦
16. Bootstrap Report

Bootstrap 후 간단히 보고한다.

사용한 real references
사용한 user measurements
생성한 profile revision
수정/생성한 scene 요소
Real Side vs Sim Side의 주요 차이
Real Wrist vs Sim Wrist의 주요 차이
아직 provisional인 값
추가 데이터가 필요한 항목
17. Iteration Report

Iteration 후 간단히 보고한다.

baseline profile
사용한 representative captures
validation captures
가장 큰 mismatch
이번에 수정한 원인 / parameter group
frozen fields
tunable fields
계산 / optimization 방법
생성한 candidate revision
sparse preview 결과
full validation 결과
ACCEPTED / REJECTED / NEEDS_MORE_DATA / FAILED
아직 가장 크게 남은 mismatch


현재 evidence가 부족한 경우 arm 5축 joint correspondence calibration은 기본적으로 수행하지 않으며,
명확한 불일치 증거가 있을 때만 별도 단계로 연다