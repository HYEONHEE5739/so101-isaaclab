> 과거 설계·실험 기록입니다. 현재 구조와 실행 경로는 [ARCHITECTURE](../../real2sim/ARCHITECTURE.md)와 [DATA_FLOW](../../real2sim/DATA_FLOW.md)를 따릅니다. 본문의 단계·제한·명령은 작성 당시 기준입니다.

# Isaac Lab / MuJoCo 두 트랙 전환 계획

검토일: 2026-10-06. 요구사항: `MUJOCOVERSION.md`.
상태: Phase 1 분석 완료. Phase 2 독립 MuJoCo backend/검증 worker 구현.
Phase 2.5 VRAM Feasibility Gate 통과 전 Phase 3 이후로 진행하지 않는다.
기존 Isaac 실행 경로와 UI는 변경하지 않았다.

## 1. 결론과 범위

Isaac Sim / Isaac Lab 트랙을 보존하고 MuJoCo 트랙을 추가한다. 기존 PyQt UI와
Source → Replay → Annotation → Datagen → Convert → Train → Sim Eval 순서를 유지한다.
UI 프로세스는 엔진을 import하지 않고 선택한 backend의 worker를 실행한다.
MuJoCo 기능은 각 단계의 검증이 끝난 뒤 활성화한다. 미지원 기능을 Isaac으로 자동 실행하지 않는다.

현재 사용자 수정이 있는 `workflow/ui.py`, `workflow/operations.py`, `real2sim/launcher.py`,
`docs/real2sim/README.md`와 신규 curation/VLM 작업을 보존한다.

## 2. 실제 코드 분류

아래 경로의 Python 패키지 기준 위치는 `source/soarm101_lab/soarm101_lab/`이다.

| 영역 | 확인한 파일 / 함수 | 재사용 또는 분리 |
|---|---|---|
| 데스크톱 진입 | `scripts/real2sim/real2sim_ui.py`, `real2sim/ui.py` | 기존 PyQt 진입과 Real2Sim 화면 유지 |
| Workflow UI | `workflow/ui.py`: `WorkflowPanel.execute`, `stage_options`, `poll_preview`, `configure_stage` | backend 선택, Python 경로, capability 표시 추가 |
| 실행 계획 | `workflow/operations.py`: `Operations.plan`, `run`, `readiness` | 검증·로그·취소 재사용, stage별 argv를 backend에 따라 분기 |
| Real2Sim 연결 | `real2sim/launcher.py`: `RuntimeLauncher.connect`, `poll` | 현재 `real2sim_sim.py` 고정. backend/session identity 확인 후 worker 선택 |
| Real2Sim runtime | `real2sim/runtime.py`, `scene.py` | Isaac scene/env 직접 접근 부분은 Isaac 전용으로 유지. MuJoCo adapter 추가 |
| 환경 구성 | `real2sim/workspaces/environment.py`: `make_config`, `task_success_details`; `assets/robots/so101.py`; `tasks/manager_based/…` | Isaac config/managers/assets/sensors는 그대로 실행 불가 |
| Workspace | `real2sim/workspaces/package.py`: `load`, `publish`, `metadata`; `profile.py`, `workflow/tasks.py` | profile/task/checksum 재사용. package 생성·검증에는 USD 필수 조건이 남아 있음 |
| Scene/physics | `workspaces/scene_assets.py`: `build_assets`; `workspaces/physics.py`; `scene.py` | 측정 geometry/물성 의미 재사용, USD authoring과 엔진별 물성 변환 분리 |
| Teleoperation | `devices/lerobot/so101_leader.py`: `SO101Leader.advance`; `so101_joint_mapping.py`; `real2sim/hardware.py` | serial/calibration/mapper 재사용. Leader의 `assets` import가 Isaac을 끌어오므로 joint 상수 분리 필요 |
| Source 기록 | `scripts/teleoperation/record_mimic_dataset.py`, `scripts/real2sim/workspace.py` | AppLauncher/Isaac recorder/env loop 교체; 타이밍·저장·중지 동작 유지 |
| Annotation | `scripts/mimic/annotate_demos.py`: `replay_episode`, `annotate_episode_in_manual_mode`, `PreStepDatagenInfoRecorder` | 현재 물리 재생과 Isaac recorder 의존. UI 명령과 경계 기록은 분리 가능 |
| Mimic | `workspaces/mimic.py`: `make_mimic_config`, `validate_input`; `scripts/mimic/generate_dataset.py` | `isaaclab_mimic` async generation을 그대로 MuJoCo에 연결할 수 없음 |
| FK/IK | `utils/so101_kinematics.py`: `SO101Kinematics.forward_kinematics`, `inverse_kinematics` | Pinocchio/NumPy 기반 재사용 후보. 5축 arm의 orientation 제약 검증 필요 |
| Dataset 계약 | `so101_dataset_contract.py`, `representation.py`, `workspaces/demo.py` | 이름·단위·action 의미 검증 재사용. `restore_initial_state`는 Isaac state adapter 필요 |
| Conversion | `scripts/tools/convert_isaac2lerobot.py`: `load_actions`, `image_to_uint8` | Isaac 런타임 import 없이 HDF5를 읽음. 공통 schema adapter로 재사용 가능 |
| 데이터 검수 | `workflow/curation.py`: `analyze`, `export_subset`; `curation_ui.py` | HDF5 필드 계약을 맞추면 대부분 재사용 |
| 프리뷰/제어 | `workflow/presentation.py`: `Presentation.publish`, `observe`, `attach`, `RecorderControls` | 파일 기반 명령·프레임 재사용. env.scene/sensors와 observation 파싱 분리 |
| 학습/추론 | `workflow/policy.py`: `load_policy`, `observation`, `train`; `workflow/evaluation.py`: `evaluate` | LeRobot ACT/SmolVLA 로더 재사용, 평가 env loop 분리 |
| Qwen 진단 | `scripts/vlm_diagnostics/test_qwen3vl.py` | 이미지 기반 VLM 진단. Qwen+Mini Flow-DiT action policy 구현으로 취급하지 않음 |

`workflow/policy.py`는 ACT와 SmolVLA만 허용하고, observation은 side/wrist 두 영상을
구성한다. 따라서 Qwen3-VL + Mini Flow-DiT와 단일 카메라는 별도 policy adapter와
checkpoint input 계약이 필요하다. 기존 두 카메라 checkpoint를 단일 카메라에 자동 연결하지 않는다.

## 3. Asset 확인 결과와 Blender

실제 확인한 원본:

- `assets/SO101/urdf/so101_isaaclab.urdf`: 6개 revolute joint, `tool0` fixed frame,
  link inertia, mesh origin, joint limit 포함.
- `assets/SO101/meshes/`: URDF가 참조하는 STL 13개 모두 존재.
  binary STL vertex bounds는 대략 0.02–0.14 크기의 부품으로, m 단위와 일치하는 범위다.
  STL에는 단위 metadata가 없으므로 FK/실측 검증 없이 임의 0.001 scaling하지 않는다.
- STL은 파일별 1,254–53,994 triangles. 최초에는 원본 visual을 재사용하고
  렌더 성능 측정 후 복사본에 decimation을 적용한다.
- `assets/SO101/usd/so101_isaaclab.usd`와 `usd/configuration/`: Isaac 트랙에 보존한다.
  MuJoCo 로봇은 URDF/STL을 기반으로 MJCF를 만든다.
- 현재 workspace table/cube/cup은 `profile.json`과 `task.json`에서
  `scene_assets.build_assets`가 USD로 생성한다. 동일 profile에서 MJCF geometry를 생성한다.

| Asset | Blender 필요 여부 | MuJoCo 처리 |
|---|---|---|
| SO-101 STL/URDF | 최초에는 불필요 | link transform/inertia 유지, visual STL 재사용 |
| Table/cube/placement | 불필요 | profile 치수와 pose를 box 등으로 구성 |
| Cup/bin | 최초에는 불필요 | 열린 바닥+분할 벽 collision. 단일 꽉 찬 convex mesh를 컨테이너 collision으로 쓰지 않음 |
| 프린트 텍스처 | 초기에는 불필요 | 기존 texture 파일과 provenance 재사용; UV 대응은 별도 확인 |
| 배경/외형 개선 | 선택 사항 | Blender에서 m 단위, origin/transform 정리 → OBJ/STL export → MJCF visual |
| Mesh 경량화/손상 수정 | 측정 후 선택 | 원본 보존, 파생 asset/hash 기록, collision은 primitive 중심 |

URDF effort limit은 10이고 `assets/robots/so101.py`의 Isaac actuator effort limit은
30이다. 둘 중 하나를 물리적 정답으로 간주하지 않는다. home pose는 workspace task에서
읽고, actuator kp/damping/force limit은 출처를 남긴 MuJoCo 설정으로 검증한다.
`tool0`의 zero inertia fixed link는 site로 옮기는 방향을 검토한다.

## 4. UI와 backend 구조

화면 상단에 `Simulation backend: Isaac Sim / Isaac Lab | MuJoCo` 선택을 둔다.
stage 목록·입력 artifact·Task Definition·실행/정지·로그·프리뷰는 유지한다.
engine별 Python executable을 저장하고 physics device와 policy device를 분리한다.
실행 중에는 backend 변경을 막는다. artifact의 수집 backend와 현재 실행 backend를 표시한다.

```text
WorkflowPanel / Real2Sim UI
  → Operations.plan / RuntimeLauncher
  → backend capability + artifact compatibility 검증
  → backend worker (별도 프로세스)
  → SimBackend
      ├─ IsaacBackend → 기존 config/runtime
      └─ MujocoBackend → MJCF / MjModel / MjData
  → 기존 Presentation 프레임·명령 / Registry / 공통 episode
```

`SimBackend`는 `reset(seed, initial_state)`, `step(joint_target_rad)`, `get_state`,
`get_rgb(camera_name)`, `get_eef_pose`, `get_object_pose`, `set_object_pose`, `is_success`,
`close`를 제공한다. replay를 위해 `get_initial_state`와 `restore_state`도 필요하다.
state는 명시적인 joint order의 measured q/qvel과 commanded target을 구분한다.
pose는 workspace frame의 4×4 matrix, 길이는 m, joint는 rad, RGB는 H×W×3 uint8로 정의한다.
physics timestep, control timestep, frame timestamp도 metadata로 남긴다.

Mimic용 `action_to_target_pose`, `target_pose_to_action`, `get_subtask_signals`는
별도 datagen interface에 둔다. MuJoCo 사용자에게 Isaac의 managers API를 요구하지 않는다.
factory는 선택한 엔진만 lazy import한다. MuJoCo worker에서 isaaclab/omni/pxr import가
발생하지 않는지 검증한다. 학습·변환 등 엔진이 필요 없는 stage는 공통으로 유지한다.

### Workspace와 데이터 계약

기존 `so101.workspace/1`은 `scene/static.usda`와 USD robot asset을 필수로 검사한다.
이 검사를 단순 삭제하면 기존 계약을 깨뜨린다. 초기에는 읽기 전용 v1 package의
profile/task/URDF/STL로 run directory에 MJCF를 생성하고 package hash와 생성 설정을 기록한다.
장기적으로 엔진별 asset entry를 가진 새 workspace schema를 추가하되 v1 loader를 유지한다.
published package 내부에는 파생 MJCF를 쓰지 않는다.

공통 episode의 후보 필드는 measured joint_pos/joint_vel, joint_targets, RGB,
eef_pose, object_poses, initial_state, timestamp, success, task definition이다.
backend/model hash, mapper/coordinate contract, camera 설정, seed를 기록한다.
MuJoCo raw qpos/qvel/act 등과 Isaac raw state는 별도 namespace로 보존한다.
새 schema는 version을 부여하고 기존 HDF5 consumer에는 adapter를 제공한다.

현재 Operations는 source/annotation/datagen에 MIMIC_SPACE 호환성을 요구하고,
Registry는 annotated data의 grasp signal을 검사한다. 단순 joint-target 녹화 파일을
기존 Mimic episode라고 등록하지 않는다. FK 기반 target pose와 action mapping 검증 후
호환 adapter를 적용하거나 새 계약을 명시적으로 도입한다.
서로 다른 엔진의 raw physics state를 직접 restore하거나 resume merge하지 않는다.

## 5. 기능별 migration 난이도

| 기능 | 난이도 | 방식 / 완료 조건 |
|---|---|---|
| Env/reset/state/RGB | 중간 | MJCF compiler + backend. home/FK/camera/settling 검증 |
| Teleop/source/save | 중간 | 기존 Leader/mapper, CPU command loop, HDF5 writer. 저장/버리기/중지/재개 검증 |
| Replay/manual annotation | 중간 | 저장 영상·state 기반 경계 선택. 오래된 데이터의 누락 pose/RGB는 원래 엔진으로 보강 |
| Auto annotation | 높음 | 기존 grasp 접촉·안정성 및 contained-and-settled 조건을 MuJoCo contact/state에 대응 |
| MimicGen generation | 높음 | 환경 interface와 pose↔action adapter. 기존 Isaac datagen 유지 |
| LeRobot conversion | 낮음–중간 | 공통 필드 adapter + 기존 converter. joint target/관측 정렬, camera feature 검증 |
| ACT/SmolVLA rollout | 중간 | inference loader 보존, env loop adapter. checkpoint camera 계약 준수 |
| Qwen+Flow-DiT rollout | 높음 | 별도 action policy/processor/checkpoint 필요. 현재 VLM 진단만으로 완료할 수 없음 |

현재 Isaac annotation은 `eef_pose`, `object_poses`, `target_eef_pose`, gripper 관련 값과
subtask signals를 수집한다. `make_mimic_config`는 `tool0`의 pick/place object_ref를 연결하고
grasp 신호와 최종 성공을 사용한다. approach/grasp/transport/place 4개 UI 경계는 도입할 수
있지만 기존 두 subtask recipe를 이름만 바꿔 호환 처리하지 않는다. 별도 task recipe가 필요하다.

MimicGen의 `DatagenInfo`에는 eef_pose/object_poses/subtask_term_signals/target_pose/
gripper_action을 대응시킨다. object-relative transformation과 source selection은
MimicGen 재사용을 우선한다. env interface 외에 generator가 요구하는 reset/state/success와
dataset metadata protocol도 spike로 확인한다. SO-101은 5축 arm이라 임의 6D target pose가
도달 가능하다고 가정하지 않고 IK residual/limit 실패를 기록한다.

## 6. 단계별 파일과 검증 계획

| Phase | 수정할 기존 파일 | 신규 파일 후보 | 재사용 / 분리 | 검증 |
|---|---|---|---|---|
| 1 분석 | 실행 코드 수정 없음 | 이 문서 | 기존 호출 경로·asset·계약 확인 | URDF 참조/limit/STL 확인 |
| 2 최소 환경 | 기존 Isaac/UI 수정 없음 | `simulation/base.py`, `simulation/mujoco_backend.py`, `simulation/mjcf.py`, run-local `scene/so101.xml`, `scripts/mujoco/check_env.py`, `scripts/mujoco/requirements.txt`, `tests/real2sim/test_mujoco_backend.py` | profile/task/URDF/STL 사용, Isaac import 분리 | compile, 6 joints, home/FK, finite step, RGB, reset/settling |
| 2.5 VLA VRAM Feasibility Gate | 기존 Isaac/UI 수정 없음 | 독립 feasibility worker와 결과 JSON (후속 단계) | Phase 2 RGB와 Qwen3-VL-2B 4-bit hidden features, 작은 action head | 8GB에서 동시 실행, 실제 RGB forward, feature shape, peak/steady VRAM, latency/FPS |
| 3 UI 연결 | `workflow/ui.py`, `operations.py`, `real2sim/launcher.py`, `workflow/presentation.py` | `simulation/factory.py`, `simulation/isaac_backend.py`, `simulation/capabilities.py`, `scripts/real2sim/workflow.py worker` | UI와 presentation protocol 유지; subprocess 실행 분기 | argv/READY/capability/session identity/offscreen UI 확인 |
| 4 Teleop/record | `devices/lerobot/so101_leader.py`, `workflow/artifacts.py`, `workspaces/demo.py`, `workflow/data_resume.py` | `simulation/robot_spec.py`, `simulation/episodes.py`, `simulation/recording.py` | mapper/calibration 재사용, Isaac 상수와 recorder 분리 | 가짜 Leader 입력, 저장/버리기/재개, initial state replay |
| 5 Annotation | `workflow/ui.py`, `operations.py`, `artifacts.py`, `presentation.py` | `simulation/annotation.py`, `scripts/mujoco/annotate.py` | 영상 replay/manual markers 공통화; auto 접촉 판정 adapter | 경계 순서·길이·필수 datagen fields·invalid annotation 거절 |
| 6 MimicGen | `operations.py`, `artifacts.py` | `simulation/mimicgen_interface.py`, `scripts/mujoco/generate.py`, task recipe | 기존 Isaac Mimic 보존, MimicGen 알고리즘 재사용 | pose/action 왕복, IK 실패 제외, seed·성공 목표 개수 검증 |
| 7 Conversion | `scripts/tools/convert_isaac2lerobot.py`, `so101_dataset_contract.py`, `representation.py` | 공통 converter entry/episode adapter | 기존 LeRobot writer/normalization 계약 유지 | 두 엔진 fixture 변환, 6D rad action, timestamp/camera keys 검증 |
| 8 Rollout | `workflow/policy.py`, `workflow/evaluation.py`, `workflow/ui.py` | MuJoCo eval adapter, 별도 Qwen Flow-DiT policy adapter | ACT/SmolVLA 추론 재사용; Isaac AppLauncher 분리 | input 계약/action chunk/reset, 동일 task eval, 동시 실행 VRAM 측정 |

각 단계 시작 전에 실제 변경 범위와 검사 항목을 다시 제시한다. 기존 Isaac 경로를
삭제하지 않는다. `tests/real2sim/`의 workflow/launcher/presentation/contract 관련 회귀 검사를
수정 영역에 맞춰 실행한다. 실제 leader 연결은 별도의 수동 teleop 검증으로 구분한다.

## 7. Phase 2 최소 환경의 구체적인 순서

1. NumPy 기반 `SimBackend`와 state/pose/action 계약을 먼저 정의한다.
2. URDF link origins/joint axes/limits/inertia와 13 STL을 기반으로 robot MJCF를 생성한다.
   zero-inertia tool frame을 site로 유지하고 FK를 Pinocchio/기존 URDF와 비교한다.
3. `so101.xml` robot 정의를 포함하는 run-local `scene.xml`과 task metadata를 만든다.
   선택 workspace profile에서 robot base/table/object/target/camera를 읽는다.
4. table/cube는 box, target은 열린 container collision으로 만든다. visual과 collision을
   분리하고 collision 크기를 바꿔 화면 오차를 보정하지 않는다.
5. workspace home pose, position actuators, CPU physics/control decimation을 설정한다.
6. camera 1개와 lazy renderer를 추가한다. viewer는 옵션이고 physics-only 모드에서도 실행한다.
7. `check_env.py`로 reset → target step → state/eef/object 조회 → RGB 저장 → close를 검증한다.
8. 완료 증거는 scene snapshot, FK 오차, finite simulation, object settling, RGB shape,
   엔진 import 분리, CPU/RAM/VRAM 측정이다. 이 단계는 teleop나 전체 UI 연결 완료가 아니다.

### Lightweight 설정 제안

env count=1, CPU physics, camera 기본 1개, 320×240 초기 성능 검증값,
overview/wrist 추가 렌더는 선택, physics rate와 camera rate 분리, renderer 지연 생성.
해상도/intrinsics는 profile 기반으로 일관되게 조정하고 provisional 값을 표시한다.
Linux OSMesa software rendering과 EGL accelerated rendering을 명시적으로 선택한다.
EGL은 GPU memory를 사용하므로 MuJoCo 전체가 VRAM 0이라고 표시하지 않는다.
기존 Operations/Launcher의 NVIDIA offload 환경변수는 Isaac 또는 명시적 EGL 선택에만 적용한다.
Qwen 4-bit와 동시 실행 가능 여부는 model/context/image/action chunk를 고정하여 실제 측정한다.

## 8. 근거와 미검증 범위

외부 API 검토:

- [MuJoCo rendering context 문서](https://mujoco.readthedocs.io/en/3.3.7/programming.html): Linux GLX/OSMesa/EGL 선택.
- [MimicGen environment interfaces](https://mimicgen.github.io/docs/modules/env_interfaces.html): DatagenInfo와 action/target pose 변환 경계.
- [MimicGen custom environment tutorial](https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/datagen_custom.md): custom interface 구현 항목.

Phase 1은 정적 코드 분석과 URDF/STL 검사였다. 이후 Phase 2에서 MuJoCo model compile,
camera rendering, reset/step, EEF FK, object settling와 성공 판정을 검증했다.
실제 teleop, MimicGen end-to-end, Qwen action policy 및 8GB 동시 실행은 아직 검증하지 않았다.
가장 먼저 사람이 검토할 부분은 joint/frame mapping, actuator 설정 출처,
열린 container/gripper collision, 카메라 convention, dataset action 의미다.

## 9. Phase 2.5 — VLA VRAM Feasibility Gate (필수)

Phase 2 완료 직후, UI migration/teleoperation/annotation/MimicGen에 앞서 수행한다.
8GB 환경의 실측 결과가 성공한 경우에만 Phase 3로 진행한다. 실행 실패/미측정/VRAM
장치 접근 불가는 미통과이며, MuJoCo만 성공했다고 VLA 동시 실행을 보증하지 않는다.

1. RTX 4060 Laptop의 실제 device memory total, 기존 점유량, 시스템 RAM, 드라이버 확인.
2. env count=1/CPU physics만 실행한 steady RAM/VRAM 측정.
3. 실제 side RGB 1개 활성화 후 renderer RAM/VRAM/FPS 측정.
4. 같은 시스템에 Qwen3-VL-2B를 4-bit로 로드. 모델 revision, quantization backend,
   compute dtype, processor image pixel/token 제한과 prompt/context 길이를 고정하여 기록.
5. MuJoCo 실제 RGB frame을 processor로 입력하고 `generate()` 없이 model forward 수행.
6. decoder/VL hidden state가 `[B, N, 2048]`인지 실제 checkpoint config 및 출력으로 확인.
   N은 image/text tokenization에 따라 달라지므로 기록한다. 임의 reshape로 shape를 맞추지 않는다.
7. 작은 dummy/Mini Action Head forward와 출력 finite/shape 검사. 학습된 VLA 성능 검증과 구분.
8. 모델 로드 peak, 첫 forward peak, 반복 forward peak/steady VRAM,
   camera FPS, CUDA synchronize를 포함한 Qwen forward latency, RAM을 기록.
   PyTorch allocated/reserved peak와 device/process 전체 VRAM을 구분한다.
9. EGL GPU와 OSMesa 또는 검증된 software EGL을 별도 프로세스로 비교.
   GL vendor/renderer를 확인하며 llvmpipe EGL을 GPU EGL 성공으로 표시하지 않는다.

통과 기준: 실제 8GB RTX 4060에서 camera+Qwen 4-bit+head가 동시에 유지되고,
warmup 후 반복 forward에 OOM/nonfinite가 없으며 필요한 hidden feature가 확인되어야 한다.
잔여 VRAM, peak, 반복 횟수, head 크기, camera FPS/forward latency를 결과에 남긴다.
최소 반복은 warmup 3회 후 측정 20회로 제안한다. 기능 통과와 별개로 FPS/latency가
실제 action chunk 주기를 충족하는지 검토한 뒤 UI 연결을 시작한다.

Phase 2 결과/실행 명령은 `MUJOCO_PHASE2.md`에 기록한다.
실제 robot MJCF는 workspace base transform과 source hashes를 보존하기 위해
`assets/`의 고정 파일 대신 run-local `scene/so101.xml`로 생성한다.
