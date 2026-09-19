현재 Real2Sim scene fitting 단계는 일단 종료하고,
다음 단계인 "Real2Sim Workspace Environment 통합"을 구현해라.

중요:
이번 작업에서는 Blender를 추가하거나 사용하지 마라.
Blender integration은 후속 작업이다.

docs/real2sim/REAL2SIM_AGENT_PROTOCOL.md의 원칙을 유지한다.


==================================================
1. 최종 목표
==================================================

내가 원하는 구조는 기존 pick_place.py의 scene 좌표를
Real2Sim profile 값으로 덮어쓰는 방식이 아니다.

기존 SO-101 environment와 기존 pick_place.py는 그대로 보존한다.

대신 Real2Sim으로 만든 하나의 실제 작업환경을
독립적인 Workspace Environment Package로 publish하고,

공통 Generic Real2Sim Environment가 이 workspace를 읽어서

- SO-101 robot
- cameras
- scene
- dynamic objects
- reset/randomization
- task
- observation/action

을 구성하도록 한다.


최종적으로는 새로운 실제 환경마다:

REAL references
    ↓
Real2Sim bootstrap / iteration
    ↓
accepted revision
    ↓
Publish Workspace
    ↓
workspace_001
    ↓
Source Demo
    ↓
Annotation
    ↓
Datagen
    ↓
Dataset
    ↓
LeRobot Train
    ↓
Evaluate / Rollout

흐름을 반복할 수 있어야 한다.


==================================================
2. 이번 구현 범위
==================================================

이번 작업에서는 다음까지만 완성한다.

PHASE 1

A. 현재 repository와 기존 workflow audit
B. Workspace Environment Package schema 설계
C. accepted Real2Sim revision → Publish Workspace
D. Generic Real2Sim Environment loader
E. workspace를 Isaac Lab에서 정상 spawn
F. 기존 teleoperation을 workspace environment에서 실행
G. 기존 Source Demo recorder를 workspace environment에서 실행
H. 짧은 demo record / replay 검증

이번에는 다음은 구현하지 마라.

- Annotation UI 통합
- Datagen UI 통합
- Dataset conversion UI 통합
- LeRobot Train UI 통합
- Eval/Real rollout UI 통합
- Blender
- 새로운 reconstruction framework
- Mimic algorithm 변경
- policy 변경

단 이후 확장 가능하도록 interface만 고려한다.


==================================================
3. 기존 코드 보존
==================================================

다음은 불필요하게 수정하지 마라.

- 기존 pick_place.py
- 기존 SO-101 robot implementation
- leader/follower teleoperation logic
- Mimic algorithm
- datagen algorithm
- dataset semantics
- policy/training code
- 기존 Real2Sim fitting/profile/revision 구조

기존 코드를 새 workspace 전용 코드로 복사해서
workspace_001.py, workspace_002.py처럼 늘리는 방식도 피한다.

scene-specific 값은 package/data로 관리하고,
공통 동작은 generic code로 유지한다.


==================================================
4. 핵심 구조
==================================================

원하는 구조는 개념적으로 다음과 같다.

existing SO-101 infrastructure
        ↑
Generic Real2Sim Environment
        ↑
Workspace Environment Package
        ↑
accepted Real2Sim revision


예시 구조는 참고만 하고,
현재 repository 구조에 맞게 최소 변경으로 설계해라.

real2sim/
├── environments/
│   ├── workspace_001/
│   │   ├── manifest.json
│   │   ├── profile.json
│   │   ├── task.json
│   │   ├── provenance.json
│   │   ├── scene/
│   │   │   └── ...
│   │   └── assets/
│   │       └── ...
│   └── ...
│
├── environment_loader.py
└── publish_workspace.py

실제 경로/파일명은 repository audit 후 결정해라.


==================================================
5. Workspace Package 책임
==================================================

Workspace Package는 최소한 다음 정보를 가져야 한다.

1. immutable workspace/environment ID
2. workspace version
3. source Real2Sim revision ID
4. source profile hash
5. robot asset/version reference
6. robot base transform
7. Side camera configuration
8. Wrist camera configuration
9. static scene geometry/assets
10. dynamic object definitions
11. collision / rigid-body requirements
12. workspace region
13. reset/randomization rules
14. task semantics
15. success condition
16. joint mapper identity/hash
17. provenance

모든 값이 하나의 JSON에 들어갈 필요는 없다.

profile / manifest / task 등 역할에 따라 나눌 수 있다.


==================================================
6. Profile과 Workspace의 역할 구분
==================================================

Real2Sim profile은 계속 observation/scene reconstruction의
source of truth다.

profile에는 예를 들어:

- camera pose/FOV
- robot base alignment
- object/table/workspace alignment
- visual parameters

등이 존재한다.

하지만 source demo/datagen을 수행하려면 이것만으로 부족하다.

Workspace Package에는 추가적으로:

- 어떤 object가 dynamic인가
- 어떤 object가 fixed인가
- reset 시 object가 어디에 spawn되는가
- randomization 영역
- target object
- pick/place task 의미
- success condition
- physics/collision 설정

이 필요하다.

Real2Sim profile과 manipulation task definition을
하나의 개념으로 섞지 마라.


==================================================
7. USD 처리 원칙
==================================================

현재 UI의 "USD + Profile 내보내기" 결과를
새 environment의 source of truth로 삼지 마라.

USD는 필요하면 Workspace Package의 generated/static asset으로 사용한다.

예:

accepted profile
    ↓
deterministic scene build
    ↓
static_scene.usd

그러나 최종 truth는:

- accepted profile
- workspace/task manifest
- referenced assets

의 조합이어야 한다.

USD를 직접 수동 수정해야만 environment가 유지되는 구조를 만들지 마라.


==================================================
8. Dynamic Object 처리
==================================================

Real2Sim render용 scene에 보이는 cube/object를 그대로
static USD object로 두고,
기존 task용 RigidObject를 또 spawn해서 중복시키지 마라.

object마다 명시적으로 분류한다.

예:

STATIC
- table
- wall/background
- fixed workspace marker
- fixed camera body visual

DYNAMIC
- pickable cubes
- movable task object

TARGET / FIXED TASK GEOMETRY
- cup
- bin
- tray

실제 task interaction이 필요한 object는
Isaac Lab의 적절한 physics/collision object로 생성되어야 한다.

visual matching용 mesh와 physics object가 필요하면
하나의 logical object로 연결해서 관리한다.


==================================================
9. Task Definition
==================================================

Real2Sim 사진만 보고 task 의미를 추측하지 마라.

현재 기존 SO-101 pick/place workflow에서 재사용할 수 있는
task semantics를 audit하고,
첫 workspace에서는 현재 수행하려는 task를 명시적인 task config로 정의한다.

예:

- movable objects
- target receptacles
- allowed workspace
- randomization bounds
- success relation

기존 pick_place.py의 scene 좌표를 복사하지 마라.

기존 코드에서 재사용해야 할 것은
task/action/observation implementation이지
기존 환경의 hardcoded 위치가 아니다.


==================================================
10. Generic Environment Loader
==================================================

새 Generic Real2Sim Environment는
workspace ID/path를 받아 environment를 구성해야 한다.

개념:

load_workspace("workspace_001")

        ↓

load profile
load manifest
load task
load assets

        ↓

spawn:
- SO-101
- robot base transform
- Side/Wrist cameras
- static environment
- dynamic objects
- target objects

        ↓

attach:
- existing action manager
- existing observation manager
- teleop
- reset/randomization
- task success checks


workspace-specific Python 파일을 새로 만들지 않고
data-driven하게 동작하는 것을 우선한다.


==================================================
11. Publish Workspace
==================================================

현재 selected/accepted Real2Sim revision을
Workspace Package로 publish하는 명확한 command/API를 만든다.

예:

publish_workspace(
    revision=...,
    workspace_id=...
)

실제 인터페이스는 repository 구조에 맞게 결정한다.


Publish는 다음을 해야 한다.

- source revision/profile 검증
- profile 복사 또는 immutable reference
- asset dependency 수집
- workspace manifest 생성
- provenance/hash 기록
- task config 생성 또는 연결
- workspace package validation
- READY / INVALID 상태 기록

기존 accepted revision을 수정하지 마라.


==================================================
12. 현재 revision을 첫 workspace로 사용
==================================================

현재 가장 최신이며 사용자가 만족한 Real2Sim revision을
첫 번째 workspace 후보로 사용한다.

현재 UI에서 선택된 revision과
실제 최신/accepted revision 상태를 코드에서 확인하고,
사용자가 명시적으로 선택한 revision을 우선한다.

임의로 다른 revision을 선택하지 마라.

첫 environment ID는 예를 들어:

workspace_001

또는 repository naming convention에 맞는 이름을 사용한다.


==================================================
13. Source Demo 연결
==================================================

첫 번째 실제 통합 목표는 Source Demo다.

기존 source demo recorder의:

- robot control
- teleoperation
- action recording
- observation recording
- Mimic-compatible output

을 가능한 그대로 사용한다.

다만 scene/environment source만
Generic Real2Sim Workspace Environment로 선택할 수 있게 한다.


최종적으로 사용자가 다음과 같은 복잡한 CLI path를
직접 입력하지 않아도 되는 구조를 준비한다.

개념:

environment = workspace_001
        ↓
start source demo


Source Demo metadata에는 최소한 다음을 기록한다.

- workspace ID
- workspace version
- Real2Sim revision ID
- profile hash
- asset hash if available
- mapper hash
- task definition/version

그래야 이후 annotation/datagen이
동일 environment를 재현할 수 있다.


==================================================
14. Record / Replay 검증
==================================================

Source Demo 통합이 끝나면
바로 전체 pipeline으로 넘어가지 마라.

먼저 작은 smoke test를 수행한다.

1. workspace_001 launch
2. robot/camera/object spawn 확인
3. teleop 가능 확인
4. 짧은 source demo 1개 기록
5. demo artifact 저장 확인
6. 동일 workspace에서 replay
7. robot/object/camera state가 예상대로 재현되는지 확인

GPU/hardware를 실제 사용할 수 없는 실행환경이면
실행하지 못한 부분을 명확하게 구분해서 보고한다.

실행하지 않은 것을 성공했다고 주장하지 마라.


==================================================
15. Randomization
==================================================

기존 randomization의 hardcoded world coordinate를
그대로 사용하지 마라.

새 workspace의:

- workspace polygon/region
- table height
- object size
- valid spawn margin

에서 reset/randomization bounds를 계산하거나 정의한다.

첫 구현에서는 복잡한 자동 추론보다
명시적인 workspace task config를 사용해도 된다.

중요한 것은 환경마다 Python 코드를 수정하지 않는 것이다.


==================================================
16. UI는 이번 단계에서 최소화
==================================================

최종적으로는 하나의 UI에서:

Real2Sim
→ Publish Workspace
→ Source Demo
→ Annotation
→ Datagen
→ Dataset
→ Train
→ Evaluate / Rollout

을 버튼으로 실행할 예정이다.

하지만 이번 Phase에서는 대규모 UI를 만들지 마라.

필요하면 현재 Real2Sim UI에 최소한 다음 정도만 연결할 수 있다.

- current revision 표시
- Publish Workspace
- published workspace 표시
- Launch Source Demo

단 기존 UI 구조를 대규모로 재작성하지 마라.

우선 backend/interface를 안정화한다.


==================================================
17. Blender
==================================================

이번 작업에서는 Blender dependency, Blender scripts,
Blender-generated asset pipeline을 추가하지 마라.

향후 optional visual asset backend로 추가할 예정이다.

따라서 Workspace Package는 나중에:

- Blender-generated USD
- Blender-generated mesh

도 일반 asset처럼 참조할 수 있는 구조이면 충분하다.

현재 구현은 Isaac/현재 asset pipeline만 사용한다.


==================================================
18. 구현 진행 방식
==================================================

먼저 repository를 audit한다.

다음을 정확히 찾아라.

A. 기존 pick_place scene/environment entrypoint
B. robot spawn 코드
C. camera spawn 코드
D. dynamic cube/object 코드
E. reset/randomization 코드
F. observation/action config
G. source demo recorder entrypoint
H. replay entrypoint
I. Mimic annotation이 기대하는 source demo 형식
J. 현재 Real2Sim revision/profile/scene builder 구조
K. 현재 USD+Profile export의 실제 역할


Audit 후 다음을 구분한다.

REUSABLE
- 그대로 재사용 가능한 코드

SCENE-SPECIFIC
- 기존 pick_place 환경에 종속된 값/구조

NEW
- Workspace Package / generic loader에 필요한 최소 코드


큰 dependency나 구조 변경이 필요하지 않다면
audit 후 Phase 1 구현까지 진행해라.

큰 구조 변경이 필요하면 구현 전에:
- 이유
- 변경 파일
- 영향
- 기존 구조로 해결할 수 없는 이유
를 먼저 보고하고 멈춰라.


==================================================
19. 테스트
==================================================

기존 테스트를 깨뜨리지 마라.

새로 최소한 다음을 테스트한다.

- workspace manifest/schema validation
- publish deterministic result
- invalid/missing asset detection
- workspace ID/version
- profile hash preservation
- dynamic/static object classification
- randomization bounds
- generic loader config generation
- source demo metadata provenance
- existing environment unaffected

가능한 non-hardware tests를 실행한다.


==================================================
20. 최종 보고 형식
==================================================

최종 보고는 다음 순서로 작성한다.

A. Repository audit 결과

B. 기존 구조에서 재사용한 코드

C. 새 Workspace Package 구조

D. 추가/변경한 파일

E. 현재 Real2Sim revision → workspace_001 publish 결과

F. Generic Real2Sim Environment launch 구조

G. dynamic/static/task object 처리 방식

H. reset/randomization 처리 방식

I. Source Demo 연결 방식

J. record/replay 검증 결과

K. 기존 pick_place.py와 기존 pipeline이 변경되지 않았는지

L. 테스트 결과

M. 아직 구현하지 않은 것
   - Annotation
   - Datagen
   - Dataset
   - Train
   - Eval/Rollout
   - Blender

N. 다음 Phase에서 해야 할 작업


==================================================
21. 가장 중요한 원칙
==================================================

이번 작업의 성공 기준은
"새 scene이 예쁘게 보인다"가 아니다.

성공 기준은:

현재 Real2Sim revision
        ↓
workspace_001로 publish
        ↓
Generic Real2Sim Environment에서 동일 scene/task 재현
        ↓
기존 SO-101 teleop 사용
        ↓
Source Demo 기록
        ↓
동일 workspace에서 replay

가 기존 pick_place environment를 파괴하지 않고
실제로 연결되는 것이다.