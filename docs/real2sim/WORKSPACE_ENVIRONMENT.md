# Real2Sim Workspace Environment — Phase 1

이 문서는 Phase 1 기록입니다. 후속 구현과 통합 UI 안내는 [Phase 2 Workspace Pipeline](PHASE2_WORKSPACE_PIPELINE.md)을 참고하세요.

## 현재 상태

기존 pick_place scene을 덮어쓰지 않고 독립적인 `workspace_001/v1`을 publish했다.
사용자 UI에서 선택된 ACCEPTED revision은
`revision_20260919_124108_a217180c9838425d835477bf1ae66ec1`이다.
이전 대화의 120159 revision으로 되돌리지 않았다.

패키지: `outputs/real2sim/environments/workspace_001/v1`
원본 profile SHA256: `b0a3dc668317f4cc5ba58dbb15c0df614a234bffae1ce7e6321e7cfa2b5fae08`
원본 accepted profile과 패키지 profile의 바이트가 동일하다.

첫 task는 기존 recorder의 pick-red/place-first-receptacle recipe를 명시적인
`cube_red → cup_a` 설정으로 옮긴 기본안이다. 사진에서 task 의미를 추론한 결과나
사용자가 target cup을 확정한 결과는 아니다. 다른 task가 필요하면 task JSON을 수정하여
새 version으로 publish한다. 질량·마찰·collision approximation은 측정된 물성이 아닌
provisional simulation 설정이다.

## A. Repository audit

| 영역 | 기존 코드 | 판단 |
|---|---|---|
| 기존 scene | `assets/scenes/pick_place.py` | 고정 table/bin/cube/camera 좌표는 scene-specific, 재사용하지 않음 |
| Robot | `assets/robots/so101.py` | SO101_FOLLOWER_CFG와 기존 USD/URDF 재사용 |
| Action / observation | `tasks/manager_based/soarm101_lab/base_pick_place_teleop_env_cfg.py` | ActionsCfg / ObservationsCfg 재사용; 기존 scene/event는 사용하지 않음 |
| Reset | `utils/episode_randomizer.py`, `tasks/manager_based/soarm101_lab/mdp/resets.py` | 기존 world 좌표를 새 workspace에 복사하지 않음 |
| Teleop | `scripts/envs/teleoperation/teleop_so101.py` | 기존 leader/control 유지, opt-in --workspace 추가 |
| Source demo | `scripts/envs/teleoperation/record_mimic_dataset.py` | 기존 기록 함수, 키보드 workflow, HDF5 구조 재사용 |
| 기존 replay | `scripts/tools/replay_joint_targets.py` | legacy replay 유지; workspace 전용 dispatcher 추가 |
| Annotation | `scripts/mimic/annotate_demos.py`, `so101_mimic_env.py` | initial_state/actions/object semantics 필요; Phase 2 통합 대상 |
| Real2Sim | `real2sim/profile.py`, `scene.py`, `runtime.py` | 기존 profile 검증 및 deterministic USD builder 재사용 |
| USD export | `real2sim/runtime.py` | 기존 export는 scene/profile snapshot; manipulation package 자체가 아님 |

위 source-relative 경로는 `source/soarm101_lab/soarm101_lab/` 아래에 있다.

## B–D. 재사용 및 추가 구조

공통 구현: `source/soarm101_lab/soarm101_lab/real2sim/workspaces/`

- `package.py`: task/package 검증, immutable publish, hash, ID/path resolution.
- `scene_assets.py`: static USD, cup visual/collision, wrist mount의 deterministic 생성.
- `environment.py`: 독립 ManagerBasedEnvCfg/InteractiveSceneCfg 구성, task success observation.
- `reset.py`: workspace-local reset/randomization.
- `demo.py`: HDF5 workspace provenance 확인 및 initial-state 복원.
- `__init__.py`: package API.

추가 실행 파일: `scripts/real2sim/publish_workspace.py`, `scripts/real2sim/workspace.py`.
수정한 기존 실행 파일: `teleop_so101.py`, `record_mimic_dataset.py`의 opt-in 분기.
후자는 기존 기록 helper를 smoke test에서 재사용하도록 AppLauncher 시작을 main guard로 감쌌다.
새 테스트: `tests/real2sim/test_workspaces.py`.

```text
workspace_001/v1/
  manifest.json       READY, ID/version, revision/hash, mapper, 파일 목록
  profile.json        원본 accepted profile 그대로
  task.json           dynamic/target, pick/place, reset, physics, success
  provenance.json     source revision, asset resolution, provisional 구분
  assets/SO101/       복사한 robot assets
  assets/textures/    내용 hash로 식별한 texture
  scene/static.usda
  scene/cup_a.usda
  scene/cup_b.usda
  scene/wrist_mount.usda
```

Generated USD는 profile/task/asset으로 재생성 가능한 representation이다.
텍스처는 패키지 내부 상대 경로로 연결한다. 원본 profile의 경로는 hash 보존을 위해
유지하고 validation 시 provenance asset map을 사용한다. 동일 ID/version은 덮어쓰지 않는다.
ID에 여러 version이 있으면 short ID 자동 선택을 거부하므로 package 경로를 지정한다.
Isaac 기본 material인 OmniPBR.mdl은 Isaac runtime dependency이다.

## E–F. Publish와 launch

Isaac Lab이 설치된 기존 `lerobot-arena` 환경에서 repository root 기준으로 실행한다.
새 dependency 설치는 필요 없다.

현재 package는 이미 publish되어 있으므로 바로 preview 가능하다:

```bash
python scripts/real2sim/workspace.py preview workspace_001 --device cuda:0 --steps 1800
```

새 task template 생성 및 검토 후 publish:

```bash
python scripts/real2sim/publish_workspace.py --revision /path/to/accepted/profile.json --workspace-id workspace_002 --write-task-template /tmp/workspace_002_task.json
python scripts/real2sim/publish_workspace.py --revision /path/to/accepted/profile.json --workspace-id workspace_002 --task /tmp/workspace_002_task.json
```

`--revision` 생략 시 현재 ui_settings의 선택 profile을 읽는다. ACCEPTED가 아니면 거부한다.
USD binding이 아직 로드되지 않은 Isaac Python에서는 publish가 headless AppLauncher를 시작한다.
실패한 publish staging은 INVALID 상태이며 READY package로 선택되지 않는다.

## G–H. Object / reset

- 세 cube는 각각 하나의 RigidObject이며 static USD에서 제거된다.
- cup_a/cup_b는 고정 kinematic RigidObject이다. 기존 cup visual에 숨겨진 바닥과
  32×4 벽 collider를 추가하며 상단은 열려 있다. 실제 cup 물성을 검증한 것은 아니다.
- table/background/workspace marker는 static scene, wrist camera body는 gripper 자식 visual이다.
- marker의 workspace-local 크기/중심, cube 크기, margin으로 XY sampling 범위를 계산한다.
- marker 표면 높이 + cube 반높이 + clearance로 Z를 계산하고 cube 간 겹침을 피한다.
- task JSON의 seed로 재현 가능하며 legacy world coordinate를 사용하지 않는다.
- Phase 1은 단일 env, box dynamic objects, cup targets, 고정 yaw만 지원한다.
- success는 cube의 8개 꼭짓점이 target cup 내부이고 선속도가 설정값 미만인지 확인한다.
  기존 recorder의 수동 저장 semantics를 자동 성공 판정으로 바꾸지는 않았다.

## I. Source Demo 연결

기존과 같은 leader port 및 calibration을 사용한다. 아래 port는 실제 장비 경로로 지정한다.

```bash
python scripts/real2sim/workspace.py teleop workspace_001 --port /dev/so101_leader
python scripts/real2sim/workspace.py source-demo workspace_001 --port /dev/so101_leader --dataset_file datasets/workspace_001_source.hdf5 --num_episodes 1
python scripts/real2sim/workspace.py replay workspace_001 --dataset datasets/workspace_001_source.hdf5 --headless --device cuda:0
```

실행 흐름은 기존 recorder와 같으며 environment source만 바뀐다.
HDF5에는 initial_state, obs/policy, actions(TCP delta + gripper rad), joint_targets(URDF rad),
states가 기록된다. workspace ID/version, revision/profile/asset/mapper/task hash를 env_args에
추가하고 coordinate sidecar에도 패키지 robot asset provenance를 기록한다.
Replay는 joint_targets를 직접 사용하므로 mapper를 다시 적용하지 않는다.
다른 workspace/version/hash로 replay하면 거부한다.

`SO101-Workspace-Source-v1`은 Phase 1 dataset 식별자이다. 현재 기존 annotation/datagen에
등록된 Mimic task라는 뜻은 아니다. 해당 연결은 후속 Phase에서 구현해야 한다.

## J–L. 검증

실제 NVIDIA GPU Isaac Sim에서 hardware 없이 다음을 수행했다:

1. 독립 workspace config로 robot, 두 camera, 세 dynamic cube, 두 cup spawn.
2. 기존 recorder helper로 60-step 작은 wrist motion 기록.
3. 같은 프로세스 reset/replay: 최대 joint error 4.7684e-7 rad, object position error 0 m,
   평균 camera RGB 절대차 1.3239/255.
4. 새 Isaac 프로세스에서 같은 파일 replay: joint/object error 동일,
   평균 camera RGB 절대차 0.7516/255.
5. config 생성 전후 기존 SO101TeleopEnvCfg scene dictionary 동일 확인.

Artifact: `outputs/real2sim/workspace_runs/workspace_001/smoke_v2.hdf5` 및 `.so101.json`.
결과: 같은 디렉터리 `replay_report.json`, `side_cam.png`, `wrist_cam.png`.
Smoke는 validation_only=true, success=false이다. 성공한 pick/place demo나 실물 teleop 검증이 아니다.
실제 leader 텔레옵과 사람이 수행한 pick/place 기록은 아직 실행하지 않았다.
일반 source demo replay는 저장된 object trajectory가 없으면 object error를 null로 보고한다.

기존 70개 + 새 6개 non-hardware test: **76 passed**.
테스트는 deterministic/idempotent publish, immutable profile hash, asset 누락/변경 검출,
ID/version/task 검증, dynamic 중복 방지, 열린 cup collision, reset 범위/분리/seed,
source metadata mismatch 거부를 포함한다. Generic loader 구성은 실제 GPU spawn으로 검증했다.

이번 Phase에서 기존 pick_place.py, robot implementation, Mimic/datagen 알고리즘,
dataset/policy/training 코드는 수정하지 않았다. 기존 workspace 미지정 실행 분기는 유지한다.
기존 robot USD의 tool0 visual reference warning은 남아 있으며 이 작업에서 원본 asset을 바꾸지 않았다.

## M–N. 후속 작업

아직 구현하지 않음: Annotation/Datagen/Dataset/Train/Eval-Rollout UI 통합, Blender,
다중 env, arbitrary task/object 타입, 통합 launcher UI.

다음 순서: 사용자가 실제 leader로 짧은 source demo 기록 → 같은 workspace replay 확인 →
workspace-aware Mimic environment와 annotation의 task/object contract 연결 → datagen 확장.
현재 accepted revision과 기존 환경을 보존한 채 진행한다.
