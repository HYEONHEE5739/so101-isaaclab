현재 프로젝트에는 Isaac Lab 기반으로 만든 SO-101 simulation 환경과 UI가 있다.

이 프로젝트를 분석한 뒤, 기존 UI와 전체 workflow는 최대한 유지하면서 simulation backend를 Isaac Lab에서 MuJoCo로 교체하거나 MuJoCo backend를 추가하고 싶다.

목표는 단순히 별도의 MuJoCo demo를 만드는 것이 아니라, 현재 UI에서 수행하던 아래 workflow를 MuJoCo에서도 동일하게 수행할 수 있게 만드는 것이다.

1. Environment 실행
2. SO-101 teleoperation
3. Source demonstration 수집
4. Demonstration 저장
5. Subtask annotation
6. Mimic/MimicGen 기반 dataset generation
7. 생성 dataset 확인
8. 학습된 policy rollout
9. RGB camera observation + robot state를 이용한 VLA inference

최종 목적은 Isaac Lab의 높은 GPU VRAM 사용량을 줄여서, RTX 4060 Laptop 8GB 환경에서도 MuJoCo simulation과 Qwen3-VL-2B 4-bit 기반 VLA policy inference를 동시에 실행하는 것이다.

---

## 1. 우선 현재 프로젝트 구조를 분석해줘

먼저 코드를 수정하지 말고 현재 repository를 분석해서 다음을 구분해줘.

- UI에 종속된 코드
- Isaac Lab에 직접 종속된 코드
- simulation engine과 무관하게 재사용 가능한 코드
- teleoperation 관련 코드
- source demo recording 관련 코드
- annotation 관련 코드
- Mimic / dataset generation 관련 코드
- dataset conversion 관련 코드
- policy rollout 관련 코드

그리고 어떤 부분은 그대로 재사용할 수 있고, 어떤 부분은 MuJoCo용으로 다시 구현해야 하는지 설명해줘.

특히 “기존 UI 전체를 다시 만드는 방식”은 피하고 싶다.

가능하면 다음처럼 backend abstraction을 두고 싶다.

```python
class SimBackend:
    def reset(self):
        ...

    def step(self, action):
        ...

    def get_state(self):
        ...

    def get_rgb(self):
        ...

    def get_eef_pose(self):
        ...

    def get_object_pose(self, object_name):
        ...

    def set_object_pose(self, object_name, pose):
        ...

    def is_success(self):
        ...
```

그리고:

```python
class IsaacBackend(SimBackend):
    ...

class MujocoBackend(SimBackend):
    ...
```

형태로 기존 UI가 simulation engine을 몰라도 동작할 수 있는 구조가 가능한지 검토해줘.

가능하다면 이 구조를 우선적으로 사용해줘.

---

## 2. MuJoCo 환경은 기존 Isaac 환경을 최대한 재사용하고 싶다

현재 Isaac Lab 환경과 동일한 task를 MuJoCo에서 다시 만들고 싶다.

환경의 핵심 구성은 대략 다음과 같다.

- SO-101 robot
- table
- cube/object
- target bin 또는 placement area
- RGB camera
- robot joint state
- end-effector pose
- object pose
- task success condition

기존 SO-101 URDF, mesh, STL/OBJ 등의 asset이 있다면 최대한 재사용해줘.

환경을 완전히 처음부터 새로 모델링하기 전에 다음 가능성을 우선 검토해줘.

1. 기존 SO-101 URDF를 MuJoCo에서 활용
2. 기존 robot mesh 재사용
3. 기존 cube/bin/table mesh 재사용
4. 기존 camera pose/scene layout을 최대한 동일하게 재현
5. 기존 joint limit / home pose / actuator 설정을 MuJoCo MJCF로 옮김

MuJoCo에서는 최종적으로 MJCF 기반 scene 구성을 사용하는 방향을 원한다.

예:

```text
assets/
    so101/
    objects/
    textures/

mujoco/
    so101.xml
    scene.xml
    task.xml
```

robot model과 scene을 분리해서 관리할 수 있으면 그렇게 해줘.

---

## 3. Blender를 환경 제작 workflow에 포함하고 싶다

MuJoCo scene의 visual environment를 구성할 때 Blender를 활용하고 싶다.

Blender의 역할은 physics simulation이 아니라 다음과 같이 생각하고 있다.

- table/bin/background 등 visual mesh 제작
- 기존 mesh 수정
- mesh coordinate/origin 정리
- scale 확인
- material / texture 제작
- OBJ/STL 등 MuJoCo에서 사용할 asset으로 export

그리고 MuJoCo MJCF에서 다음을 정의하고 싶다.

- body transform
- joints
- actuators
- collision geometry
- visual geometry
- camera
- light
- friction
- contact
- keyframe/home pose

Blender에서 만든 visual mesh와 MuJoCo collision geometry는 반드시 동일할 필요는 없다.

가능하면:

```text
Visual:
Blender mesh

Collision:
box / sphere / capsule 등 단순 MuJoCo geom
```

처럼 physics가 안정적이고 가벼운 구조를 사용해줘.

먼저 현재 project의 기존 mesh/URDF/USD asset을 확인해서 Blender를 거쳐야 하는 asset과 그대로 사용할 수 있는 asset을 구분해줘.

필요하다면 Blender → OBJ/STL → MJCF workflow를 제안해줘.

---

## 4. Source Demo / Teleoperation

현재 Isaac Lab에서 사용하던 source demo collection workflow를 MuJoCo에서도 동일하게 만들고 싶다.

필요한 데이터는 최소한 다음과 같다.

```text
observation:
    joint position
    optionally joint velocity
    RGB image
    end-effector pose
    object poses

action:
    SO-101 joint target / control command

metadata:
    episode index
    timestep
    success
```

기존 teleoperation 입력 방식을 최대한 재사용해줘.

현재 UI의 “Source Demo 생성” 버튼을 눌렀을 때 backend가 MuJoCo라면 MuJoCo environment가 실행되고 demonstration이 저장되는 형태가 좋다.

---

## 5. Annotation

기존 annotation UI가 simulator independent하게 재사용 가능한지 확인해줘.

가능하면 annotation 과정은 simulation backend와 분리해서:

```text
recorded episode
→ replay
→ user selects subtask boundaries
→ annotation metadata 저장
```

형태로 만들어줘.

예:

```text
approach
grasp
transport
place
```

와 같은 subtask boundary를 기록할 수 있어야 한다.

annotation 과정에서 MuJoCo physics가 반드시 필요하지 않다면, dataset replay만으로 처리하도록 해줘.

---

## 6. Mimic / MimicGen

현재 Isaac Lab Mimic 방식으로 source demo를 annotation하고 generated demonstrations를 만들고 있다.

MuJoCo에서는 가능하면 NVIDIA MimicGen 또는 동일한 아이디어를 활용하고 싶다.

다음 정보가 필요할 것으로 생각한다.

```text
eef_pose
object_poses
subtask_term_signals
target_pose
gripper_action
```

기존 Isaac Lab Mimic 코드가 어떤 정보를 사용하고 있는지 분석한 뒤, MuJoCo/MimicGen으로 대응시켜줘.

가능하면 MimicGen의 environment interface를 구현하는 방식으로 다음 workflow를 만들고 싶다.

```text
Source Demo
    ↓
Subtask Annotation
    ↓
DatagenInfo
    ↓
Subtask selection
    ↓
Object-relative trajectory transformation
    ↓
MuJoCo execution
    ↓
Success check
    ↓
Generated Dataset
```

알고리즘 자체를 불필요하게 처음부터 다시 만들지 말고, MimicGen에서 재사용할 수 있는 기능이 있다면 적극적으로 사용해줘.

---

## 7. Dataset format

최종적으로 생성한 dataset은 LeRobot 또는 현재 사용하던 training pipeline으로 변환하기 쉽게 유지하고 싶다.

가능하면 simulator-specific raw dataset과 policy training dataset을 분리해줘.

예:

```text
MuJoCo raw episode
        ↓
common episode format
        ↓
LeRobot Dataset conversion
```

이렇게 하면 향후 Isaac / MuJoCo 어느 simulator에서 수집한 데이터든 같은 converter를 사용할 수 있게 하고 싶다.

---

## 8. Policy Rollout

최종적으로 가장 중요한 목적은 다음 구조를 실행하는 것이다.

```text
MuJoCo RGB Camera
        +
Robot State
        ↓

Qwen3-VL-2B
4-bit quantized
        ↓
VL hidden features

        +

State Encoder
Action Encoder
        ↓

Mini Flow-DiT
        ↓
Action Decoder
        ↓
SO-101 Action Chunk
        ↓
MuJoCo
```

현재 RTX 4060 Laptop 8GB 환경에서 Isaac Lab + Qwen3-VL-2B가 VRAM 부족으로 동시에 실행되지 않는다.

따라서 MuJoCo environment는 최대한 GPU VRAM 사용량이 적도록 설계해줘.

가능하다면:

- MuJoCo physics는 CPU
- RGB camera는 필요한 1개만
- 낮은 rendering overhead
- offscreen rendering 지원
- debug visualization은 옵션
- env count = 1
- policy rollout 전용 lightweight mode

를 지원하도록 해줘.

---

## 9. 중요한 구현 원칙

한 번에 전체 프로젝트를 새로 작성하지 말아줘.

먼저 repository를 분석하고 다음 단계별 migration plan을 제안해줘.

### Phase 1
현재 코드 구조 분석

### Phase 2
MuJoCo에서 SO-101 + table + object + camera가 동작하는 최소 environment 구현

### Phase 3
기존 UI에서 MujocoBackend를 통해 reset / step / camera / state가 동작하도록 연결

### Phase 4
Teleoperation + Source Demo recording

### Phase 5
Annotation 연결

### Phase 6
MimicGen / dataset generation 연결

### Phase 7
Dataset converter 연결

### Phase 8
Policy rollout 연결

각 Phase마다:

- 수정할 기존 파일
- 새로 만들 파일
- 재사용할 코드
- 삭제하거나 분리할 Isaac-specific dependency
- 테스트 방법

을 먼저 설명한 뒤 구현해줘.

기존 프로젝트를 최대한 보존하면서 작은 단위로 수정하고 싶다.

---

## 10. 우선 지금 해줬으면 하는 것

지금 당장 전체 코드를 작성하지 말고 다음부터 해줘.

1. 현재 repository 전체 구조를 분석한다.
2. 기존 UI와 Isaac Lab dependency를 분류한다.
3. 기존 SO-101 robot/environment asset 중 MuJoCo에 재사용 가능한 것을 찾는다.
4. Blender가 필요한 asset과 필요하지 않은 asset을 구분한다.
5. 기존 UI를 유지하면서 MujocoBackend를 추가할 수 있는 architecture를 제안한다.
6. Source Demo / Annotation / Mimic / Data Generation 각 기능의 migration 난이도와 구현 방식을 설명한다.
7. 가장 먼저 구현할 “최소 MuJoCo environment”의 구체적인 파일 구조와 작업 순서를 제안한다.
8. 내가 검토할 수 있도록 계획을 먼저 보여준 후 Phase 2 구현으로 진행한다.

가능하면 기존 프로젝트 파일과 함수 이름을 직접 언급하면서 구체적으로 분석해줘.