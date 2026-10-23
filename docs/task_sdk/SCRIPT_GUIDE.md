# Source / Evaluation / Annotation / Mimic Script 작성 가이드

현재 계약: 2026-10-23, MuJoCo Script 실행. 기존 함수형 Task Python과 Legacy `options.tasks` Script는 유지한다. 사용자 Python은 신뢰한 코드이며 **명시 Worker 실행에서만** 실행한다. UI 준비 확인은 literal `SCRIPT`와 알려진 Placement partial의 AST만 읽는다.

## 현재 기본 API: Scene 기반 Source와 Event/Subtask Terms

새 Source Generator는 `scene.objects` **매핑**을 생성한다. 기존 목록형 `scene.objects` / `source.id,pick_object,target_object`와 `FOLLOWUP`은 Legacy 호환 경로로 유지한다. 아래 Legacy 절의 Pick/Target 필수·Native 성공 추가 제한은 그 경로의 계약이다. Evaluation Script 계약은 그대로다.

### Scene만 녹화하기

[최소 Source](../../examples/tasks/scene_source_minimal.py)의 전체 실행 선언이다. Path는 Script 기준이며 장치 포트는 사용자가 확인한다. 자동 검증은 실제 로봇 대신 가상 Leader를 사용했다.

```python
SCRIPT = {
    'stage': 'source',
    'workspace': '../../outputs/real2sim/environments/scene2_builder/v8',
    'scene': {'objects': {'cube_red': {}}},
    'options': {'backend': 'mujoco', 'storage_name': 'scene_source_minimal',
                'num_envs': 1, 'port': '/dev/so101_leader',
                'episodes': 3, 'episode_time_s': 20, 'reset_time_s': 5},
}
```

Source ID·Pick·Target·성공 함수·Annotation·Mimic은 필수가 아니다. 선택적 `source={'instruction':'...'}` 또는 `id`는 녹화 Metadata다. 새 경로는 Task Definition을 만들지 않고 success를 미판정으로 저장한다. Builder의 로봇/동적 Body·질량/Collision·Camera 계약은 여전히 필요하다. 물리 정보 없는 순수 시각 검토 Body를 임의로 Dynamic으로 바꾸지 않는다. Task catalog가 없는 Builder라도 명시적인 물리 정보가 있으면 Scene 기록이 가능하다.

`{}`는 **Builder 초기 Pose/Orientation을 그대로** 사용한다. 기존 Source의 edge-uniform을 자동 적용하지 않는다. 기존 목록형 Source의 Reset은 변경하지 않았다. 선택하지 않은 물체는 기존 Compiler가 제거하고 Table·필수 Support·Robot은 유지한다.

| 예제 | 실제 용도 |
|---|---|
| [scene_source_minimal.py](../../examples/tasks/scene_source_minimal.py) | A: red만, Target/Task/조건 없음 |
| [scene_source_two.py](../../examples/tasks/scene_source_two.py) | B: red 기본 Pose + blue XYZ |
| [scene_source_uniform.py](../../examples/tasks/scene_source_uniform.py) | B 확장: red의 기존 Uniform, blue는 기본 Pose |
| [scene_source_custom_reset.py](../../examples/tasks/scene_source_custom_reset.py) | F: 사용자 Reset, Backend RNG와 Metadata Pose 사용 |
| [scene_red_cup_a.py](../../examples/tasks/scene_red_cup_a.py) | C: 기존 Red/Cup Annotation·Mimic recipe |
| [scene_blue_cup_b.py](../../examples/tasks/scene_blue_cup_b.py) | D: Binding만 바꾼 Blue/Cup recipe |
| [scene_custom_event.py](../../examples/tasks/scene_custom_event.py) | E: `left_start` Event·Acceptance·Failure·tool0 두 Subtask |

### Pose와 Reset

```python
'scene': {'objects': {
    'cube_red': {'reset': 'random_red'},
    'cube_blue': {'pose': {'xyz': [0.04, -0.20, 0.039]}},
}}
# SCRIPT 밖:
from soarm101_lab.task_terms.reset import uniform as random_red
```

Pose는 workspace frame / metre다. `frame`은 생략 또는 `'workspace'`만 허용한다. Orientation을 생략하면 Builder 회전을 유지하며 명시할 때는 정규화된 `quat_wxyz=[w,x,y,z]`를 쓴다. `pose`와 `reset`의 동시 지정, fixed Target 이동, Support 밖/침투·Robot/다른 Body Collision은 거부한다. 현재 검사는 Metadata bounding box/기존 support와 MuJoCo contact에 따른 보수적 초기 배치 검사이며 경로 계획·도달 가능성 인증은 아니다. Builder 원본·Geometry·Physics 파일을 바꾸지 않는다.

Reset callable은 `(ctx, object_id, rng) -> 4x4 SE(3)`이다. `metadata_pose`, `uniform`, `edge_uniform`은 기존 WorkspaceRandomizer와 Backend 소유 RNG를 사용한다. Uniform은 이미 놓인 다른 dynamic Body를 기존 separation 검사에 포함하고 그 Body를 이동하지 않는다. 샘플이 Robot과 충돌하면 조용히 다른 배치로 바꾸지 않고 거부한다. Sampling margin/separation/clearance가 없는 Workspace는 이 분포를 추측하지 않으므로 사용자 Reset을 명시한다.

Source는 기존 Episode reset에서만 이 함수를 실행한다. Annotation은 **기록된 초기 상태**를 복원하며 Reset hook을 다시 실행하지 않는다. Mimic은 기존 reset 경계에서 같은 명시 Pose/Reset을 적용한다. Seed는 기존 Backend/Adapter가 소유한다. Resume는 같은 Scene/Pose/Reset 코드가 필요하다. 횟수·출력·resume Path를 바꿀 수 있고 코드 identity는 `SCRIPT` 선언을 제외한 AST로 확인한다. 전체 entry SHA는 실행 계획 변경 감지와 provenance에 계속 기록한다. import한 외부 모듈 내용까지 pin하지 않는다.

### Annotation: Event, Boundary, Acceptance

Source Script에 다음 선택적 선언을 넣는다. Source는 아래 함수·factory를 호출하지 않는다. Annotation/Mimic UI에서 **같은 Python을 명시적으로 선택**해야 한다. Dataset 안의 Python Path는 자동 실행하지 않는다.

```python
'annotation': {
    'diagnostics': {'object_id': 'cube_red', 'target_id': 'cup_a'},
    'events': {'grasp': 'grasp_event', 'place': 'place_event'},
    'signals': ['grasp'], 'final_events': ['place'],
    'manual_event': 'grasp', 'accept': 'native_acceptance',
},
'mimic': {'recipe': 'pick_place/1', 'accept': 'native_acceptance'},
# SCRIPT 밖:
from soarm101_lab.task_terms.diagnostics import grasp_completed as grasp_event
from soarm101_lab.task_terms.annotation import placed as place_event, native_acceptance
```

Event 이름은 임의 identifier다. `(ctx)->bool` 함수 또는 `{'factory':'make_event'}`를 연결한다. Factory `()->callable(ctx)`는 **환경·Episode마다 새로 호출**한다. 반환 객체의 선택적 `reset(ctx)`는 초기 상태 복원 뒤 호출한다. 일반 함수는 순수 조회 함수여야 하며 모듈 전역의 변경 가능한 상태를 공유하지 않는다. 새로운 StableGrasp tracker를 만들 필요 없이 기존 Adapter 소유 tracker를 조회한다.

Event는 기존 pre-command 관측에서 평가한다. 첫 True의 frame이 Boundary다. `signals`에 지정한 Event는 0→1 단조 신호로 저장하고 나머지는 순간 Event 값을 저장한다. `final_events`는 명시한 함수가 마지막 post-step에서 실제 True일 때만 마지막 action에 대응시킨다. 마지막 상태만으로 과거 grasp를 만들지 않는다. `manual_event`는 수동 Mark가 덮어쓸 Signal이며 기존 Mark/Skip/Pause/Step·유효 frame 범위를 사용한다. Curation timeline에서도 임의 이름을 표시한다.

`accept(ctx)->bool`은 선택한 Script의 채택 조건이다. 선택적 `failure(ctx)->None|False|category_string`은 첫 실패를 Episode에 latch하며 마지막 post-step도 검사한다. Context의 `object_pose`는 workspace 4×4/m, `joint_state`는 측정 model-rad다. pre-step 조회는 이미 확보한 관측을 빌려 추가 전체 state/render를 만들지 않는다. `event_boundary(ctx,name)`로 현재 Episode의 첫 Event 위치를 읽는다. 사용자 Term은 reset/step/render/파일 기록을 수행하지 않는다.

`diagnostics` 자체는 선택 사항이다. StableGrasp만 필요하면 dynamic `object_id`만 지정하고 Placement가 필요할 때 `target_id`와 선택적 `condition`을 추가한다. 기존 `native_acceptance`는 기존 Placement AND latched StableGrasp를 호출한다. `pick_place/1` recipe에서는 사용자 Acceptance가 이 Native gate를 우회할 수 없다. 임의 작업은 명시 `subtasks`와 자신의 채택 조건을 사용하며 Native 성공과 구분한다. report의 `playback_completed`, `task_accepted`, `native_acceptance`, 최종 `annotation_accepted`는 각각 실행 완료·사용자 조건·Native 진단·게시 완료다. 과거 `physical_replay_passed` 필드도 호환 목적으로 남아 있다.

### Mimic: 지원되는 Subtask mapping

```python
'mimic': {
    'subtasks': [
        {'name':'approach', 'object':'cube_red', 'eef':'tool0',
         'end_event':'left_start', 'offset':[0,0]},
        {'name':'finish', 'object':None, 'eef':'tool0',
         'end_event':None, 'selection':'random'},
    ],
    'accept':'accept_motion', 'failure':'failure',
}
```

기존 portable reference Generator·DataGenInfoPool·RNG·IK·scheduler·Writer를 그대로 사용한다. `recipe='pick_place/1'`은 기존 `workflow.mimic.configuration(pick,target,seed)` 그 자체다. 명시 Subtask는 `tool0`, Scene 안의 Object 또는 None, 기록된 중간 Signal을 사용한다. 마지막 Subtask는 기존 엔진 계약상 `end_event=None`으로 Episode 끝을 사용해야 한다. None Object에는 random selection만 지원한다. 현재 공개 선택은 random/nearest_neighbor_object, `nn_k`, 정수 offset, interpolation/fixed steps다. noise=0 및 기존 sampling 의미를 유지한다. 다른 EEF·SkillGen/coordination·사용자 config factory·지원하지 않는 설정은 거부한다. 실제 실행 검증은 두 Subtask다. 세 개 이상은 reference의 sequential 설정을 전달할 수 있으나 이번에 실제 실행하지 않았다. Subtask 수만 늘렸다고 새로운 로봇 작업이 성공한다고 보장하지 않는다.

기존 StableGrasp·Placement·Drop/Release 수학과 `configuration`은 단일 구현이다. 공개 Term은 `task_terms.reset`, `events.finger_contact`, `diagnostics.{grasp_completed,released,dropped,drop_failure,failure_category}`, `annotation.{placed,native_acceptance,grasp_boundary,place_boundary,event_boundary}`, `placement.placement`다. 새로운 Distance/Push/Fold Controller나 Reset engine을 추가하지 않았다. 사용자 함수는 공개 Pose/Joint API로 자기 조건을 작성한다.

### UI/CLI와 제한

기존 실행 Script 생성/선택 UI를 Source에서 사용한다. Annotation/Mimic에서는 기존 Python 선택 칸에 해당 Source/Terms Script를 지정한다. READY는 AST·실제 Dataset header/Scene을 확인하며 함수를 실행하지 않는다. 이미 Annotation된 결과의 `task_followup` identity와 다른 Python은 DataGen/Resume에서 거부한다. Source에 계약을 기록했다면 그 entry와 일치해야 한다. 계약 없이 수집한 Source에는 같은 Scene/Pose/Reset의 별도 Terms 파일을 명시적으로 선택할 수 있다.

```sh
python scripts/real2sim/workflow.py task-template --workspace <Workspace> --script-stage source --output <새 Source.py>
python scripts/real2sim/workflow.py ready --stage source --execution-script <Source.py>
python scripts/real2sim/workflow.py run --stage source --execution-script <Source.py>
python scripts/real2sim/workflow.py run --stage annotate --input <Source.hdf5> --workspace <Workspace> --storage-name <새이름> --followup-python <Source.py> --options <mujoco-options.json>
python scripts/real2sim/workflow.py run --stage datagen --input <Annotated.hdf5> --workspace <Workspace> --storage-name <새이름> --followup-python <Source.py> --options <mujoco-options.json>
```

이번 새 Source 기록은 기존 Side/Wrist 640×480 Common HDF5/MP4를 지원한다. 기존 Legacy NPY/MP4와 Native Isaac 경로는 유지하지만 새 Scene/Terms Python의 Isaac·다른 Robot/EEF·비표준 Camera 기록은 미지원이다. Source가 언어/Task를 생략했다면 LeRobot 변환에 명시 `instruction`을 전달한다(직접 converter CLI는 `--task`; Source 미판정 데이터를 포함할 때 `--include_failed`). 가짜 Task Definition/성공을 붙이지 않는다.

실제 검증은 [scene_terms_sdk/REPORT](../../outputs/workflow_validation/scene_terms_sdk/REPORT.md)를 따른다. Red/Blue의 **동일한 기록 Scene** A/B는 수치·영상이 같다. 새 Builder 기본 배치와 과거 random Reset은 다른 입력이며 성공률 동등성을 뜻하지 않는다. 새 Scene Blue의 특정 가상 데모는 7회 모두 기존 strict IK 오류로 생성 목표 미달이었다. Threshold나 배치를 바꿔 이를 숨기지 않았다. 실물/Isaac/장시간/새 Robot 조작 성공은 검증하지 않았다.

## Evaluation: 명시 Scene과 Task별 Policy 평가

Evaluation은 Scene-only Source의 매핑과 달리 `scene.objects` **ID 목록**을 사용한다. [단일 평가](../../examples/tasks/eval_pick_place.py)와 [다중 평가](../../examples/tasks/eval_multi_task.py)가 실행 가능한 전체 예제다. Workspace/Checkpoint 상대 Path는 Script 기준이며 Checkpoint는 `hf://owner/model@revision`도 지원한다.

```python
SCRIPT = {
    'stage': 'sim_eval', 'workspace': '/workspace', 'checkpoint': '/pretrained_model',
    'scene': {'objects': ['cube_red', 'cup_a']},
    'options': {'backend': 'mujoco', 'storage_name': 'red_eval',
                'num_envs': 2, 'device': 'cpu', 'max_steps': 1000},
    'evaluations': [{'id': 'my_red_task', 'instruction': 'Place red in cup A.',
                     'success': 'red_success', 'episodes': 3}],
}
from functools import partial
from soarm101_lab.task_terms.placement import placement
red_success = partial(placement, object_id='cube_red', target_id='cup_a')
```

ID와 Instruction은 사용자가 정한다. `placement`는 해당 Object/Target에 유일한 Metadata 조건이 있을 때만 condition을 생략한다. 다른 조건을 쓰려면 지원하는 관계·유한한 양의 속도·region 경계를 명시한다. 새 의미의 조건을 기존 결과와 동등하다고 표시하지 않는다. Geometry·Physics·카메라·Robot SHA를 Script에 복사하지 않는다.

각 Evaluation은 `episodes` quota를 순서대로 실행하고 Task 안에서 기존 1/2/4/8 독립 환경을 사용한다. Task 간 환경 Cache는 없다. 항목별 `scene`/`reset` override는 기존 Workspace 설정 계약만 지원한다. Policy 관측·ActionGuard·Physics·저장은 기존 `evaluate_sim` 한 경로다. Generic 사용자 성공은 `task.success/1`, 기존 Placement/Grasp/Drop은 별도 진단으로 기록한다.

임의 함수는 Script 또는 같은 폴더의 모듈에 정의할 수 있다. 이 경우 기존 Recorder/Grasp 진단에 사용할 `scene.diagnostics={'object_id': ..., 'target_id': ...}`를 성공 함수와 별도로 명시한다. 선택적 `setup`/`observe`도 함수명으로 연결한다. READY는 일반 함수 의미를 추론하거나 import하지 않는다. 알려진 placement partial의 literal 인자만 AST에서 검사하고 Worker가 실제 함수/인자와 대조한다.

```sh
python scripts/real2sim/workflow.py task-template --workspace <Workspace> --script-stage sim_eval --task-id <Metadata-ID> --checkpoint <Model> --output <새 Eval.py>
python scripts/real2sim/workflow.py ready --stage sim_eval --execution-script <Eval.py>
python scripts/real2sim/workflow.py run --stage sim_eval --execution-script <Eval.py> --root <새 출력 root>
```

## 호환성: 기존 Source·FOLLOWUP·Task Python

아래 입력은 기존 Dataset/Script를 위해 유지한다. 새 Source 기본 예제로 사용하지 않는다.

| 기존 입력 | 유지되는 의미 |
|---|---|
| 목록형 `scene.objects` + `source` | Source ID/Instruction/Pick/Target을 native 참조로 저장. 수집 성공 Gate는 없으며 기존 edge-uniform Reset 유지 |
| `options.tasks` + `active_object_ids` + `bindings` | 명시 Task/Scene 목록. 새 `scene`과 섞으면 오류. 내부 Source/Evaluation Worker는 동일 |
| 별도 literal `FOLLOWUP` | 기록된 native Pick/Target/조건을 대조하고 기존 Annotation/Mimic을 실행. 선택 `annotation` 함수는 Native 성공 뒤 채택을 추가 제한 |
| 함수형 `TASK_ID/INSTRUCTION/setup/observe/success` | 단일 MuJoCo Evaluation 및 `check_env.py --task-python` 검사. Scene 선택은 별도 명시 |
| Script 없는 Native Isaac | 기존 AppLauncher/Adapter/배치·수동 기능 유지. Scene/Terms Python SDK 지원을 의미하지 않음 |

Legacy 예제는 [Source Red](../../examples/tasks/source_pick_place.py), [Source Blue](../../examples/tasks/source_blue_cup_b.py), [FOLLOWUP Red](../../examples/tasks/followup_red_cup_a.py), [FOLLOWUP Blue](../../examples/tasks/followup_blue_cup_b.py)다. 과거 Source/Annotation의 조건·계보·Python identity를 변경하거나 자동 migration하지 않는다. 기존 조건과 다른 FOLLOWUP은 새 Source가 필요하다. Dataset에 적힌 Python Path를 실행하지 않는다.

## 지원 범위와 문서 소유권

- 현재 Script/Terms는 기존 MuJoCo SO-101/tool0 계약이다. Source는 단일 환경·Side/Wrist640×480 Common 기록, Evaluation은 기존 병렬 환경이다. 실제 새 실행 검증은 1/2환경이다.
- Builder ID가 다른 [fruit Source](../../examples/tasks/fruit_source.py)/[Evaluation](../../examples/tasks/fruit_sim_eval.py)는 Metadata/compile 예제다. front camera가 기존 ACT side/wrist Policy와 호환된다는 뜻이 아니다.
- 다른 Robot/EEF·Isaac 사용자 Script·RGB TaskContext API·SkillGen·임의 Controller·sandbox는 미지원이다. Term이 Python Backend에 접근할 수 있으므로 순수 조회 규칙은 신뢰한 코드의 계약이다.
- 여기서 공개 Script 문법과 사용법을 관리한다. 내부 소유권/Reset 순서는 [SDK 설계](../architecture/TASK_SDK_DESIGN.md), 전체 호출은 [SYSTEM_FLOW](../architecture/SYSTEM_FLOW.md), Native 학습·Hub는 [NATIVE_LEROBOT](../real2sim/NATIVE_LEROBOT.md)가 소유한다.
- 단계별 검증 이력은 각 outputs 보고서에 남긴다. 현재 정리의 A/B·제외 항목은 [architecture_cleanup/REPORT](../../outputs/workflow_validation/architecture_cleanup/REPORT.md)다.

## 내가 직접 확인할 것

1. Generator로 새 Source를 만들고 선택 물체/Pose/Reset을 확인한다. 실제 Leader 연결은 사용자가 포트·보정을 확인한 뒤 실행한다.
2. 같은 명시 Terms 파일로 Annotation/Mimic을 실행하고 Curation에서 Event/Signal을 확인한다. Blue의 기존 strict IK 실패는 성공으로 해석하지 않는다.
3. Evaluation Script와 저장 report의 Task별 quota·성공 출처·Dataset Path를 확인한다. 학습/Hub는 별도 Native 문서를 따른다.
