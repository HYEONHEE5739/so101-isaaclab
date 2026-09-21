# SO-101 Project Map

"뭘 수정하려면 어디를 봐야 하지?"에 답하기 위한 실전용 지도다.

## Real2Sim / scene

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| Real2Sim UI | `source/soarm101_lab/soarm101_lab/real2sim/ui.py` |
| Isaac runtime / render / pose 적용 | `real2sim/runtime.py` |
| 실제 카메라 reader | `real2sim/cameras.py` |
| 실제 follower/하드웨어 연결 | `real2sim/hardware.py` |
| camera intrinsics/extrinsics 계산 | `real2sim/calibration.py` |
| image/landmark metric | `real2sim/calibration.py` |
| profile schema / provenance | `real2sim/profile.py` |
| scene/profile을 Isaac에 적용 | `real2sim/scene.py` |
| capture/revision/IPC 저장 | `real2sim/storage.py` |
| bootstrap/reference 입력 | `real2sim/bootstrap.py`, `references.py` |

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
| HDF5 recording | `datasets/lerobot_recorder.py` 및 teleop recorder script |
| demo annotation | `scripts/mimic/annotate_demos.py` |
| Mimic datagen | `scripts/mimic/generate_dataset.py` |
| joint-target replay | `scripts/tools/replay_joint_targets.py` |
| Isaac→LeRobot conversion | `scripts/tools/convert_isaac2lerobot.py` |

## Workspace Environment

| 바꾸고 싶은 것 | 먼저 볼 곳 |
|---|---|
| workspace package/version | `real2sim/workspaces/package.py` |
| scene asset compile | `real2sim/workspaces/scene_assets.py` |
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
- Isaac runtime: `scripts/real2sim/real2sim_sim.py`
- workspace publish: `scripts/real2sim/publish_workspace.py`
- workspace tool: `scripts/real2sim/workspace.py`
- workflow: `scripts/real2sim/workflow.py`, `workflow_stage.py`
- physics check: `scripts/real2sim/check_workspace_physics.py`

## 어디부터 읽어야 하나

새로 코드를 이해할 때 권장 순서:

1. `docs/real2sim/ARCHITECTURE.md`
2. `docs/real2sim/DATA_FLOW.md`
3. 이 문서에서 관심 기능의 "먼저 볼 곳"
4. 해당 파일의 public class/function
5. 관련 `tests/real2sim/` 테스트

전체 repository를 처음부터 끝까지 읽는 방식은 권장하지 않는다.
