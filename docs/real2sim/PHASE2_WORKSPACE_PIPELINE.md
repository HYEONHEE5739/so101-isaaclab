# Phase 2 Workspace Pipeline

이 문서는 `PHASE2_WORKSPACE_PIPELINE_IMPLEMENTATION.md` 요청의 구현/실행 안내다.
기존 Phase 1 workspace와 source-demo → replay를 보존한다. 기존 `workspace_001/v1`이나
accepted profile을 수정하지 않았다. 기존 CLI는 계속 사용 가능하다.

## 실행

GUI 전용 환경에서 기존 UI를 실행한다.

```bash
.venv-real2sim-ui/bin/python -m pip install -r scripts/real2sim/requirements-ui.txt
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

UI가 Source Demo/HDF5 artifact를 직접 검사하므로 GUI 환경에도 `h5py`가 필요합니다.
`NOT_READY — No module named 'h5py'`가 표시되면 위 설치 명령 실행 후 UI를 다시 시작하세요.

새 **4 · Workspace Pipeline** 탭에서:

1. Workspace ID/version 폴더를 선택한다.
2. 실행 Python은 Isaac/LeRobot이 있는 interpreter를 선택한다. GUI venv와 별개다.
3. 기본 `Sim 작업 파이프라인`에서 Source Demo부터 단계를 선택한다. `환경 관리`에는 Revision 승인/Publish, `실제 로봇 평가`에는 Real Evaluation이 분리되어 있다.
4. 기본 제공 설정을 조정하고 Run을 누른다.
5. 프로세스 종료 후 실제 output 검증을 통과해야 SUCCEEDED artifact가 등록된다.
6. 로그에서 완료를 확인한 뒤 다음 단계와 입력 artifact를 직접 선택한다. 자동으로 다음 단계가 실행되거나 화면이 전환되지 않는다.

Capture/Fit/Inspect는 기존 탭을 그대로 사용한다. 승인 역시 명시적인 사용자 작업이며,
수치 fitting을 대신 수행하거나 자동 승인하는 기능은 아니다.
Dataset/Policy 업로드, Real 평가는 각각 별도 Run이다. 자동 업로드/자동 모터 구동은 없다.
Real 평가는 `enable_motion: true`와 follower ID/port 및 두 camera 장치를 명시해야 한다.
현재 workflow 중지 버튼은 해당 실행 process group에 SIGINT를 보낸다.

기본 상세 설정은 JSON editor로 제공한다. `output`을 생략하면 고유 run 폴더를 사용한다.
파일을 덮어쓰지 않으므로 재실행은 새 output을 지정하거나 기본 경로를 사용한다.
Artifact의 파일·폴더는 Open Folder로 확인할 수 있다. revision picker는 실제 revisions
폴더에서 시작하며 다른 picker는 현재/최근 compatible artifact의 디렉터리를 우선한다.

## 구조와 저장 위치

```text
기존 Real2Sim UI + Workspace Pipeline panel
                ↓
workflow.operations.Operations  ←  scripts/real2sim/workflow.py
                ↓
기존 CLI / 소규모 stage adapter
                ↓
내용 검증 → artifact registry → 다음 단계 입력
```

- `outputs/workflow/artifacts/<id>.json`: 타입, 경로, hash, 부모 artifact, workspace,
  representation 및 실행 provenance.
- `outputs/workflow/runs/<id>/state.json`: READY/RUNNING/SUCCEEDED/FAILED, code commit/dirty 여부.
- 같은 run의 `process.log`, 필요 시 `request.json`, 기본 output.
- `outputs/workflow/ui.json`: workspace/Python 선택. 인증 정보는 저장하지 않는다.
- 직접 선택한 source/dataset/checkpoint도 동일 Registry/Operations를 사용한다.
- 파일명으로 task나 artifact 종류를 추측하지 않는다. 수동 import 시 종류를 선택하고 내용을 검증한다.
- 이미 등록된 파일/hash/좌표 metadata가 바뀌면 실행을 거부한다. 명시적으로 다시 import해야 한다.

CLI는 같은 operation 계층을 사용한다:

```bash
python scripts/real2sim/workflow.py register --type source --path datasets/workspace_001_source.hdf5
python scripts/real2sim/workflow.py list
python scripts/real2sim/workflow.py ready --stage annotate --artifact <등록된_ID> --workspace workspace_001
python scripts/real2sim/workflow.py run --stage annotate --artifact <등록된_ID> --workspace workspace_001 --options /path/to/options.json
```

## Representation contract

Canonical identity는 `MODEL_JOINT_RAD_V1`이다. arm/gripper 모두 model/URDF rad이고,
joint order는 shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper이다.
기존 `.so101.json` schema와 데이터 숫자는 변경하지 않고 별도 명시적 descriptor를 추가했다.
구형 metadata도 의미·mapper·robot identity가 동일하면 사용한다. unknown provenance를 현재 값으로
다시 붙이지 않고 workflow에서 거부한다.

| 경계 | 표현 / 처리 |
|---|---|
| LeRobot 0.4.1 follower | use_degrees=True: arm calibration-relative degree, gripper RANGE_0_100 |
| Real observation → policy | degree/% → 기존 mapper → model rad, 한 번 |
| Policy → Real command | 저장된 postprocessor로 역정규화 → mapper inverse → degree/%, 한 번 |
| Sim action / joint_targets / IK output | 이미 model rad; mapper 적용 금지 |
| Source `actions` / Mimic | base-frame TCP delta: m×3 + rotation-vector rad×3 + gripper rad |
| Source `joint_targets` | absolute model rad×6 |
| LeRobot 변환 | joint_targets를 action으로 사용; state도 model rad |
| ACT/SmolVLA | dataset 통계로 normalize, checkpoint pre/post processor로 복원 |

LeRobot MotorsBus의 degree zero는 calibration min/max midpoint이며 URDF physical zero와
동일하다고 새로 증명한 것은 아니다. 기존 identity arm mapper와 legacy gripper mapping은 유지한다.
일반 source/IK/replay 값을 다시 degree→rad 변환하는 경로는 추가하지 않았다.

Workspace/source/dataset/policy는 mapper와 robot asset identity를 연결한다. Workspace가 다른
policy는 기본적으로 거부한다. cross-workspace 전이 정책은 별도의 검증/지원 대상이다.
Normalization processor JSON와 safetensors는 checkpoint에 포함되고 HF에도 함께 올라간다.

## Physics compiler

`real2sim/workspaces/physics.py`가 profile + task의 semantic classification을 compile한다.
새 publish에는 `physics.json`과 hash가 포함된다. 기존 v1 패키지는 변경하지 않고 기존 task
physics를 읽는 fallback을 유지한다.

| Role | 실행 표현 |
|---|---|
| ROBOT | 기존 SO-101 articulation/USD physics 유지 |
| STATIC_SUPPORT | table collision, 움직이지 않는 support |
| STATIC_GEOMETRY | task.roles로 명시한 고정 collision |
| DYNAMIC_MANIPULAND | cube RigidObject, mass/material, reset 대상 |
| FIXED_TARGET | cup kinematic object, 열린 벽/바닥 collision |
| VISUAL_ONLY | 기본 marker/background/camera body; collision 없음 |

컵은 32×4 hidden wall segments와 bottom collider를 사용한다. 상단을 막는 solid cylinder가 아니다.
Visual mesh와 collider는 하나의 logical target 안에 존재한다. dynamic cube는 static USD에서
제외하므로 중복 spawn하지 않는다.

새 task template 물성은 `default_template`이며 실제로 측정한 값이 아니다.
기존 provisional 값은 provisional로 유지한다. inertia는 PhysX가 collision/mass에서 계산한다.
compiler는 provenance와 원문 note를 기록하며 숫자를 이미지에서 추측하지 않는다.

새 task template은 four-anchor sampling과 XY jitter 0.015 m, yaw 0을 사용한다.
이는 학습 배치용 기본 설정이며 물리 측정값이 아니다. 실제 크기/여유는 profile의 support 크기와
cube 크기에서 계산하고 경계 이탈/겹침을 reject한다. 기존 task에 mode가 없으면 기존 uniform
방식을 유지한다. seed, mode, jitter, sample_index를 기록하며 새 Python workspace 파일은 필요 없다.

## Annotation / Datagen

기존 `SO101PickPlaceMimicEnv`와 IK/selection/generation 알고리즘을 재사용한다.
새 adapter는 전체 scene/reset, target object 참조, cup success 조건만 workspace에서 구성한다.

Workspace Annotation은 source의 `joint_targets`를 직접 재생한다. 이미 기록된 trajectory를
다시 IK로 재구성해서 오차를 만들지 않는다. 기록되는 Mimic action/target EEF 의미는 기존
Cartesian contract를 유지한다. Datagen은 원래 IK 경로를 계속 사용한다.

```bash
python scripts/mimic/annotate_demos.py --workspace workspace_001 \
  --input_file datasets/workspace_001_source.hdf5 \
  --output_file datasets/annotated/workspace_001_new.hdf5 --auto --headless --device cuda:0

python scripts/mimic/generate_dataset.py --workspace workspace_001 \
  --input_file datasets/annotated/workspace_001_new.hdf5 \
  --output_file datasets/generated/workspace_001_new.hdf5 \
  --generation_num_trials 10 --num_envs 1 --headless --device cuda:0
```

동일 이름의 기존 output은 직접 CLI에서도 새 경로를 사용한다. Workflow에서는 overwrite를 차단한다.
자동 annotation은 기존 grasp heuristic 및 최종 cup containment/settled 조건을 모두 만족해야 한다.
현재 검증 source demo는 마지막 cup_a 성공 조건을 충족하지 않아 **export 0**이었다.
저장 버튼의 `success=True` metadata만으로 조작 성공을 확정하지 않는다.
따라서 그 파일을 기반으로 한 성공적인 annotation→datagen 전체 실행은 아직 검증하지 못했다.
성공한 red cube→cup_a demonstration으로 재확인이 필요하다. smoke 파일은 training input에서 거부한다.

Source와 Datagen의 서로 다른 HDF5 observation/state layout은 replay adapter에서 처리한다.
기록된 object trajectory가 없는 source의 object error는 null이며 검사했다고 주장하지 않는다.

## Conversion / Train / Evaluation

기존 변환 CLI에 feature joint names와 독립 import용 `workflow_provenance.json`을 추가했다.
Guided conversion과 수동 dataset 선택 모두 같은 workspace/contract를 읽는다.

Train은 기존 `lerobot.scripts.lerobot_train`을 호출한다. 기본 SmolVLA, 선택 ACT를 지원한다.
`architecture`, `steps`, `batch_size`, `save_freq`, `base_model`, `policy_options`를 설정할 수 있다.
`push_to_hub=False`를 강제하여 학습과 업로드를 분리한다. 기본 model 다운로드가 필요할 수 있지만
이 작업에서는 새 dependency 설치나 SmolVLA 모델 다운로드/학습을 수행하지 않았다.

Sim/Real 평가는 같은 `workflow.policy` loader와 pre/postprocessor를 사용한다.
Real 쪽만 audited follower/camera adapter와 degree/% 경계 변환을 사용한다.
Real 평가는 관측/명령을 기록하지만 실제 task 성공 여부는 미측정으로 남긴다.
UI/CLI의 Run이 실제 모터 동작을 명시적으로 요청할 때만 real_eval을 실행한다.

## Hugging Face

`hf_dataset`, `hf_policy` 단계에서 명시적으로 Run한다. 정상적으로 변환/학습된 artifact를
선택하고 아래 설정을 지정한다:

```json
{"repo_id":"account/repository", "private":true, "create_repo":true, "revision":"main"}
```

기존 repo는 `create_repo:false`; 지정 visibility가 실제 repo와 다르면 거부한다.
새 branch는 `create_branch:true`를 지정한다. 기본 인증은 huggingface_hub의 로컬 로그인/HF_TOKEN이다.
토큰을 UI 설정 JSON에 넣지 않는다. 설치나 로그인은 이 구현에서 자동 수행하지 않는다.

표준 dataset/model 폴더를 HfApi.upload_folder로 업로드하고 provenance JSON를 후속 commit으로
추가한다. 입력 local artifact를 수정하지 않는다. 최종 commit/URL/revision은 remote artifact로
등록하며 실패한 upload를 성공으로 등록하지 않는다. HFPolicy는 평가 input으로 선택 가능하고
정확한 commit을 snapshot_download한다. 실제 Hub upload/download는 이번 검증에서 수행하지 않았다.

## 검증 결과와 한계

- 기존 + 신규 non-hardware/Qt/계약/physics/workflow test: **88 passed (7.48s)**. 로그: `outputs/workflow_validation/unit_tests.log`.
- 기존 actual source 420-step replay: 최대 joint error 약 0.000191 rad,
  평균 RGB 차이 약 0.83/255. object trajectory 미기록으로 position error는 null.
- 새 compiler로 독립 package 생성 후 120-step cube drop: 열린 컵 안 진입 및 안착 true.
- 실제 source 1 episode / 420 frames → LeRobot v3 변환 및 독립 import 검증.
- CPU 작은 ACT 1-step 학습/checkpoint/provenance 등록 검증. 이 모델은 성능 평가용이 아니다.
- 해당 checkpoint를 실제 Isaac GPU에서 3-step Sim 평가: 실행 성공, task success=false.
- HF 메타데이터/업로드 호출은 fake API unit test. 실제 upload는 미실행.
- 실제 follower 제어, 새 physical leader 기록, SmolVLA 실학습, 성공 annotation 기반 datagen은 미검증.
- Phase 2의 generic env는 num_envs=1 및 현재 box/cup 역할을 지원한다.
- UI 상세 입력은 아직 JSON 기반이며 production 사용자용 세부 form polish는 남아 있다.

검증용 output은 `outputs/workflow_validation/`에 격리했다. accepted profile과 기존
`workspace_001/v1`은 유지했다. 기존 pick_place, Mimic 알고리즘, mapper 숫자는 수정하지 않았다.


## 변경 파일

새 모듈 (`source/soarm101_lab/soarm101_lab/` 기준):

- `representation.py`
- `workflow/{artifacts,operations,policy,evaluation,hub,ui,__init__}.py`
- `real2sim/workspaces/{physics,mimic}.py`

추가 CLI/검증: `scripts/real2sim/{workflow,workflow_stage,check_workspace_physics}.py`,
`tests/real2sim/test_workflow.py`.

수정 기존 코드:

- `scripts/envs/teleoperation/record_mimic_dataset.py`: additive representation metadata.
- `scripts/mimic/{annotate_demos,generate_dataset}.py`: opt-in workspace adapter.
- `scripts/real2sim/workspace.py`: 독립 report 경로와 source/datagen replay layout 호환.
- `scripts/tools/convert_isaac2lerobot.py`: joint names, 독립 dataset provenance.
- `real2sim/ui.py`: pipeline 탭과 revision picker 시작 위치.
- `real2sim/workspaces/{package,scene_assets,environment,reset,demo}.py`: compiler, reset mode, contract/replay helper.
- `tests/real2sim/test_references.py`: 새 tab을 반영한 UI 회귀 assertion.
- `docs/real2sim/WORKSPACE_ENVIRONMENT.md`: Phase 2 안내 연결.

## Operator / Developer 모드

통합 UI의 **4 · Workspace Pipeline → 실행 모드**에서 선택합니다.

- **Operator (기본값)**: Isaac을 `--headless`로 실행합니다. Isaac viewport 조작 없이 같은 탭의 SIM SIDE / SIM WRIST를 보며 진행합니다.
- **Developer**: 정상 Isaac 창을 함께 엽니다. 같은 workspace, environment, action/observation, 카메라 및 기록 경로를 사용합니다. 차이는 창 표시 여부뿐입니다.

1. Workspace와 입력 artifact, 단계를 선택합니다.
2. Sim 작업에서는 SIM 영상만 표시하고, Annotation/Datagen/Sim Evaluation에는 큰 Overview도 표시합니다. 변환/학습에는 영상 패널이 없습니다. 실제 영상은 별도 `실제 로봇 평가` 화면에서 평가 루프가 이미 연 카메라를 사용합니다.
3. Run을 누르면 환경 생성 후 두 SIM 영상이 표시됩니다. 왼쪽 설정 / 오른쪽 실행 영역과 영상 / 로그 사이의 구분선을 드래그하여 크기를 조절할 수 있습니다. 이미지 비율은 보존하며 검은 UI 배경은 사용하지 않습니다.
4. Source Demo: 왼쪽의 Leader port와 Episodes를 설정하고 기존 leader 입력으로 조작합니다. **저장 / 다음 (→)**, **리셋 / 폐기 (←)**, **기록 종료**를 사용합니다. 오른쪽 영상·로그 영역에 포커스가 있으면 방향키도 동작합니다. 왼쪽 설정을 편집할 때는 방향키가 recording 명령을 보내지 않습니다. 기존 timeout 자동 저장 규칙은 그대로입니다. 기록 종료는 미완성 episode를 버리고 완료된 기록을 닫습니다.
5. Replay / Datagen preview / Sim Evaluation: **재생 / 일시정지 / 한 스텝**을 사용합니다. Real Evaluation은 재생/일시정지를 지원합니다.
6. Manual Annotation: 왼쪽 Annotation 방식을 `수동 Annotation`으로 선택하고 **재생 / 일시정지 / Subtask 표시 / Episode 건너뛰기**를 사용합니다. action index와 표시한 index가 영상 아래에 나옵니다. 성공 조건과 annotation 검증은 기존 구현을 그대로 적용합니다.
7. **현재 workflow 중지**는 해당 subprocess에 SIGINT를 전달합니다. 종료하지 않으면 5초 후 SIGTERM, 10초 후 SIGKILL로 해당 process group을 정리합니다. 중단된 부분 산출물은 자동 등록하지 않습니다.

영상은 최대 10 Hz의 진단용 JPEG 사본입니다. 원래 observation/recording/policy 입력은 변환하지 않습니다. REAL 프레임은 host read 기준 0.5초 이내 데이터만 전송합니다. 새 REAL 프레임이 2초 이상 없으면 표시를 지웁니다. SIM 영상은 저장/처리 중 또는 종료 후에도 마지막 프레임을 유지하고 상태에 새 프레임 대기/종료임을 표시합니다. REAL/SIM 동시 표시 자체가 동기화된 paired capture를 뜻하지는 않습니다.

각 실행의 `outputs/workflow/runs/<id>/presentation/`에 config, status, 최신 영상, commands/replies가 저장됩니다. 실행마다 경로가 분리됩니다. 동일 공통 `workflow/presentation.py`가 양 모드에서 기존 env의 step/reset 결과를 관찰하고 조작 명령을 전달합니다. 별도 simulator나 reconstruction/physics 구현은 없습니다. 원래 CLI를 직접 실행하여 session 환경변수가 없으면 기존 동작을 유지합니다.

CLI workflow options에서도 `"mode": "operator"` 또는 `"mode": "developer"`를 지정할 수 있습니다. 명시적 mode가 기존 headless 옵션보다 우선합니다. 수동 Annotation/Source Demo의 headless UI 조작은 통합 UI에서 실행하는 것을 권장합니다.

검증: 기존 420-step source demo의 실제 GPU headless replay 및 SIM Side/Wrist 파일 출력을 확인했습니다. 실제 leader/USB 카메라, Developer 창 수동 조작, 성공한 Datagen 및 실제 로봇 평가의 종단간 검증은 이 작업에서 실행하지 않았습니다.

## 공통 Task Selector / Language Instruction

**4 · Workspace Pipeline**의 공통 Task Selector에서 task를 설정합니다. 현재 workspace는 세 cube × 두 cup의 여섯 preset을 지원합니다:

| Pick object | Target presets |
| --- | --- |
| `cube_red` | `cup_a`, `cup_b` |
| `cube_blue` | `cup_a`, `cup_b` |
| `cube_green` | `cup_a`, `cup_b` |

Source Demo 실행 전에 preset 또는 Pick object / Target cup을 선택하세요. 객체 조합을 바꾸면 기본 영어 instruction이 채워집니다. 문장은 자유롭게 수정할 수 있지만 **문장으로 객체 ID를 추론하지 않습니다.** 예를 들어 blue→B를 선택한 뒤 문장에 red를 적어도 성공 판정 대상은 blue→B입니다.

Task Definition은 다음 정보를 함께 보관합니다:

```json
{
  "schema": "so101.task-definition/1",
  "task_id": "cube_red_to_cup_b",
  "pick_object_id": "cube_red",
  "target_object_id": "cup_b",
  "language_instruction": "Pick up the red cube and place it in cup B.",
  "success_condition": {"relation": "contained_and_settled", "max_speed_m_s": 0.03},
  "randomization_spec": {"path": "task.json#/reset", "sha256": "<workspace reset 설정의 hash>"}
}
```

성공 조건과 randomization reference는 선택한 immutable workspace의 task 설정에서 가져와 검증합니다. 기존 workspace/profile/asset을 덮어쓰지 않으며 선택은 실행별 `tasks.json`에 저장합니다.

- **Source Demo:** 한 번의 recording run은 하나의 명시적 task를 사용합니다. HDF5 env_args에 `task_definitions`, 각 `demo_N` attribute에 `task_id`, `language_instruction`, 전체 `task_definition`을 저장합니다. Source artifact에도 task 목록이 등록됩니다.
- **Annotation / Datagen:** 입력 artifact의 task를 상속하며 selector는 읽기 전용입니다. 선택한 cube의 grasp 및 cup의 place 성공 판정에 적용합니다. 여러 task가 섞인 annotation/datagen 입력은 거부하므로 task별로 실행하세요. 생성 episode에도 동일 task 정의가 저장됩니다.
- **LeRobot 변환:** 각 episode의 instruction을 LeRobot `task`에 전달합니다. `workflow_provenance.json`의 `episodes`에는 LeRobot episode index와 원본 episode, Task Definition을 연결합니다. task ID는 문장에서 재생성하지 않습니다. 명시적 episode task가 있으면 예전 CLI `--task` 문구보다 우선합니다.
- **SmolVLA / ACT Training:** dataset의 task 정의를 학습 완료 checkpoint의 `workflow_provenance.json`에 전달합니다. 학습 단계에서 입력 데이터의 task를 다른 이름으로 덮어쓰지 않습니다.
- **Sim / Real Evaluation:** 평가 범위에서 단일 task, 체크한 여러 task, workspace 전체 task를 선택합니다. `episodes`는 **task당** 횟수입니다. task별 instruction을 정책에 전달하고 결과 JSON의 `metrics_per_task`에 episode 수·step 수·성공률을 분리합니다. Sim 성공은 해당 cube가 해당 cup 안에 정지했는지 검사합니다. Real 성공은 현재 자동 측정하지 않으므로 성공률은 `null`, 상태는 `unmeasured`이며 실행 step 수만 측정합니다. 실제 물체 배치는 자동 초기화하지 않습니다.

기존 파일에 명시적 `grasp_object` / `place_bin`이 있으면 이 구조적 정보를 통해 호환 task를 결정할 수 있습니다. task 정보가 전혀 없으면 Annotation/Datagen/변환/학습의 의미를 임의로 추측하지 않고 NOT_READY 처리합니다. 기존 task 없는 source의 물리적 replay 경로는 유지합니다.

검증 기록: `outputs/workflow_validation/task_selection/`.
6개 task × 3-step GPU Sim Evaluation smoke와 기존 420-frame Source Demo의 실제 LeRobot 변환을 수행했습니다. 이는 연결/metadata 검증이며 정책의 작업 성공 성능이나 실제 로봇 검증을 뜻하지 않습니다. SmolVLA checkpoint task metadata 전달은 외부 학습 실행을 mock한 회귀 테스트로 확인했습니다.


## 진행 로그와 Annotation 문제 진단

오른쪽 로그는 실행 중 `process.log`를 증분으로 읽습니다. 기본 화면에는 기존 print의 episode 시작·저장·리셋, Replay 진행, Annotation task/subtask 실패·export 결과 및 오류를 표시합니다. 긴 Isaac 초기화 정보는 `전체 로그` 또는 `원본 로그 열기`로 확인합니다. 로그 버퍼는 최대 4000줄이고 원본 파일은 보존됩니다. 거대한 artifact JSON은 기본 로그에 출력하지 않으며 `결과 상세`의 별도 크기 조절 가능한 창에서 봅니다.

Revision 승인은 fitting 결과를 채택하는 단계이고 Publish Workspace는 accepted revision을 실행 가능한 환경 패키지로 고정하는 단계입니다. 이미 publish된 workspace로 demo를 기록할 때 반복할 필요가 없습니다.

사용자 Annotation 로그에서 확인한 원인:

- 이전 demo: `The final task was not completed` → task 성공 조건 미달로 export 0개. Replay 재현 성공과 task 성공은 다릅니다.
- 최근 demo: 저장 시 `torch.stack`에서 CUDA OOM. Workspace Annotation은 기존 Datagen의 CPU observation recorder를 재사용하고 입력 episode도 CPU로 로드하도록 수정했습니다. 초기 상태와 실행 target만 simulation device로 전달합니다.
- 예외 뒤 Isaac 프로세스가 남아 GPU를 점유한 문제: Annotation entrypoint에서 예외가 나도 app.close를 실행하도록 변경했습니다.

성공 조건이나 subtask 판정을 느슨하게 바꾸지 않았습니다. export 0개는 task/subtask 사유를 로그에 남기고 명시적인 실패로 처리합니다.

수정 후 실제 GPU 재검증: 최근 source의 5개 episode 중 4개 export 성공. 원본 `demo_2`는 최종 task 조건 미달로 제외되었으며 성공 기준은 유지했습니다. 결과는 `outputs/workflow_validation/annotation_ui/annotated_clean.hdf5`이고 task metadata 및 grasp signal 검사를 통과했습니다. 비하드웨어 회귀 테스트는 103개 통과했습니다. UI를 다시 실행한 뒤 Artifact 새로고침으로 검증된 annotated 파일을 선택할 수 있습니다.

### 단계별 작업 화면 및 시간 설정

Source Demo의 **Episode Time**(기본 20초), **Reset Time**(기본 5초)을 UI에서 조절합니다.
Reset Time은 episode 사이 대기 시간이며 Episode Time은 기록 제한 시간입니다. 시간 초과 시에는 Reset Time 동안 폐기할 수 있고, 폐기하지 않으면 기존 규칙대로 자동 저장합니다.
작업이 먼저 끝나면 **저장 / 다음 (→)** 으로 바로 저장하고 다음 episode로 넘어갑니다.
실패한 시도는 **리셋 / 폐기 (←)** 로 버립니다. 기존 recorder의 저장 조건은 유지됩니다.

| 단계 | 오른쪽 화면 | 조작 |
|---|---|---|
| Source Demo | Sim Side / Wrist, 기록 로그 | 저장/다음, 리셋/폐기, 기록 종료 |
| Replay | Sim Side / Wrist, 재현 진행 | 재생, 일시정지, 한 스텝 |
| Annotation | 큰 Overview + Side / Wrist, 판정 상태 | 재생/일시정지, 수동 모드에서만 Subtask 표시·건너뛰기 |
| Datagen | 큰 Overview + Side / Wrist, 생성 시도/성공/실패 | 재생, 일시정지, 한 스텝 |
| LeRobot 변환 / 학습 | 실시간 로그, 결과 상세 | 카메라·재생 버튼 숨김 |
| Sim Evaluation | 큰 Overview + Side / Wrist, task·episode 결과 | 재생, 일시정지, 한 스텝 |
| Real Evaluation | Real Side / Wrist | 별도 실제 평가 화면 |

Overview는 모델과 workspace 범위로 결정한 검사용 카메라입니다. Operator/Developer 모두
동일 환경에 생성하며 기존 profile, 정책 Side/Wrist 입력, 데이터셋 관측 계약을 변경하지 않습니다.
단계를 바꾸면 이전 단계의 프레임을 새 단계의 영상처럼 표시하지 않습니다.

상태 패널의 접근은 TCP–선택 물체 거리(mm)입니다. 별도 성공 임계값은 만들지 않았습니다.
Grasp는 기존 Mimic의 휴리스틱 신호이며 실제 접촉 검증을 의미하지 않습니다.
컵 안 정지는 기존 task의 contained-and-settled 조건입니다.
Annotation의 저장/제외, Datagen의 최근 완료 시도, Sim Evaluation의 성공/시간 초과는
각 실행 코드의 판정 결과에서 가져옵니다. 아직 제공되지 않은 신호는 `판정 없음`으로 표시합니다.

### 실행 설정 입력 안내

`실행 설정 · JSON 예시 / 항목 설명`을 펼치면 현재 단계에서 사용하는 항목의
자료형, 기본값/예시, 필수 여부와 설명이 표시됩니다. 아래 JSON에는 기본값이 이미
채워져 있으므로 필요한 값만 수정하세요. 예를 들어 Datagen은 `"trials": 10`,
학습은 `"steps": 10000`, `"batch_size": 8`을 조절합니다.
`output`을 빈 문자열로 두면 고유 run 폴더를 자동 사용합니다.
필수 경로와 `repo_id`의 빈 문자열은 실제 값으로 채워야 합니다.
JSON에는 주석을 넣지 않으며, `현재 단계 JSON을 기본 예시로 되돌리기`로 복원할 수 있습니다.
Source Demo의 port/episode 수/시간, Annotation 방식, Task, Python/실행 모드는
전용 UI에서 설정하므로 JSON 예시에 중복 제공하지 않습니다.
