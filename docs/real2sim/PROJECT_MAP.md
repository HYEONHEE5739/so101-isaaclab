# SO-101 Project Map

"뭘 수정하려면 어디를 봐야 하지?"에 답하기 위한 실전용 지도다.

## Real2Sim / scene

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| Real2Sim UI | `source/soarm101_lab/soarm101_lab/real2sim/ui.py` |
| 공통 Real/Sim runtime / render / pose 적용 | `real2sim/runtime.py` |
| 실제 카메라 reader | `real2sim/cameras.py` |
| 실제 follower/하드웨어 연결 | `real2sim/hardware.py` |
| camera/geometry fitting · 비교 metric | 독립 `real2sim-builder/app/calibration/`, `app/reconstruction/` |
| profile schema / provenance | `real2sim/profile.py` |
| scene/profile을 Isaac에 적용 | `real2sim/scene.py` |
| snapshot 세션/실측 저장 | `real2sim/capture_dataset.py` |
| atomic 저장/IPC | `real2sim/storage.py` |

## Robot / joint

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| Leader device 동작 | `devices/lerobot/so101_leader.py` |
| motor normalize/calibration | `devices/lerobot/motors/` |
| hardware↔model joint mapping | `so101_joint_mapping.py` 및 호출 경계 |
| SO-101 Isaac asset config | `assets/robots/so101.py` |
| teleop action config | `tasks/manager_based/soarm101_lab/base_pick_place_teleop_env_cfg.py` |
| IK action | `tasks/manager_based/soarm101_lab/mdp/so101_ik_actions.py` |

## Dataset / replay

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| dataset representation contract | `so101_dataset_contract.py` |
| Isaac HDF5 recording | `scripts/teleoperation/record_mimic_dataset.py` |
| demo annotation | `scripts/mimic/annotate_demos.py` |
| Mimic datagen | `scripts/mimic/generate_dataset.py` |
| joint-target replay | `scripts/tools/replay_joint_targets.py` |
| Isaac→LeRobot conversion | `scripts/tools/convert_isaac2lerobot.py` |

## Workspace Environment

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| workspace package/version | `real2sim/workspaces/package.py` |
| Builder export import | `real2sim/workspaces/builder.py` |
| workspace env config | `real2sim/workspaces/environment.py` |
| mass/friction/collision | `real2sim/workspaces/physics.py` |
| reset/randomization | `real2sim/workspaces/reset.py` |
| source-demo/replay contract | `real2sim/workspaces/demo.py` |
| Mimic workspace adapter | `real2sim/workspaces/mimic.py` |

### Randomization을 수정할 때

먼저 `real2sim/workspaces/reset.py`와 task/env cfg의 event/reset 설정을 확인한다.
Real2Sim calibration/capture 단계에서는 random reset이 관측 비교를 깨뜨릴 수 있으므로, **보정용 deterministic 상태와 학습용 randomization을 분리**한다.

## Unified workflow UI

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| 전체 workflow UI | `source/soarm101_lab/soarm101_lab/workflow/ui.py` |
| 화면/view 구성 | `workflow/view.py`, `presentation.py` |
| 단계 정의 | `workflow/stages.py` |
| 실제 실행 operation | `workflow/operations.py` |
| artifact 추적 | `workflow/artifacts.py` |
| task/language instruction | `workflow/tasks.py`, `task_ui.py` |
| train | `workflow/policy.py` |
| evaluation | `workflow/evaluation.py` |
| Hugging Face upload | `workflow/hub.py` |
| 옵션 설명 | `workflow/option_help.py` |

## CLI entry points

- Real2Sim UI: `scripts/real2sim/real2sim_ui.py`
- Real/Sim runtime (Isaac/MuJoCo 선택): `scripts/real2sim/real2sim_sim.py`
- Builder workspace import: `scripts/real2sim/workflow.py import-builder`
- workspace tool: `scripts/real2sim/workspace.py`
- workflow: `scripts/real2sim/workflow.py` (`list/register/ready/run/worker/publish/import-hdf5`)
- physics check: `scripts/real2sim/check_workspace_physics.py`

## 어디부터 읽어야 하나

새로 코드를 이해할 때 권장 순서:

1. `docs/real2sim/ARCHITECTURE.md`
2. `docs/real2sim/DATA_FLOW.md`
3. 이 문서에서 관심 기능의 "먼저 볼 곳"
4. 해당 파일의 public class/function
5. 관련 `tests/real2sim/` 테스트

전체 repository를 처음부터 끝까지 읽는 방식은 권장하지 않는다.

## MuJoCo / MiniVLA

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| 공통 backend 선택·stage UI | `workflow/ui.py` |
| 실행 계획·취소·등록 | `workflow/operations.py` |
| MuJoCo stage request | `workflow/operations.py::mujoco_request/mujoco_plan` |
| MuJoCo worker source/replay/train/eval | `workflow/mujoco.py` |
| canonical recording/annotation/datagen/convert | `workflow/episodes.py` |
| MJCF/CPU physics/RGB/FK/IK·명시적 MJCF adapter | `simulation/mjcf.py`, `simulation/mujoco_backend.py` |
| policy gripper 좌표·joint 단위 변환 | `so101_joint_mapping.py::RobotCoordinates` |
| 공통 카메라 화면 widget | `workflow/view.py::CameraView` |
| Qwen + action head | `source/soarm101_lab/soarm101_lab/learning/minivla/` |
| 공통 model-radian inference/train | `workflow/policy.py` |

별도 MuJoCo UI mixin과 MiniVLA subprocess service는 공통 UI/Operations로 통합했다. [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md)에 실제 구현/검증 범위를 기록한다.
