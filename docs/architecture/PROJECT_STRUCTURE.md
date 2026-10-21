# 프로젝트 구조와 코드 소유권

이 문서는 2026-10-19 Reader(6a)·Common Writer/Buffer(6b-1/2)·Source Writer(6b-3)·Playback(6c-1) 분리 후의 실제 구조를 설명한다. Registry/학습 준비/분할 정책의 전환은 아직 진행 전이다. 먼저 전체를 읽고, 원하는 기능은
[WHERE_TO_EDIT](WHERE_TO_EDIT.md), 실행 순서는 [SYSTEM_FLOW](SYSTEM_FLOW.md)를 따라간다.
감사 판단은 [REFACTOR_AUDIT](REFACTOR_AUDIT.md), 실제 이동은 [전후 비교](REFACTOR_BEFORE_AFTER.md)에 있다.

## 1. 전체 시스템

이 프로젝트는 **장치와 시뮬레이터를 연결하는 프로그램 + 데이터 작업 Workflow + 정책 학습 코드**다.
장면 제작 프로그램은 sibling `../real2sim-builder`이며 여기서는 export를 검증해 소비한다.
화면은 작업을 요청하고 worker가 장치·물리·학습을 수행한다. Registry는 결과의 파일 위치와 계보를 기억한다.
`Workspace`는 장면 package, `Artifact`는 등록 결과 record, `Run`은 한 번의 실행 이력이라 서로 다르다.

```text
soarm101_lab/
├── README.md, AGENTS.md, CODEX_HANDOFF.md    시작 안내·작업 규칙·최근 인계
├── pyproject.toml                          검사 도구 설정 (패키지 설치 파일과 다름)
├── source/soarm101_lab/
│   ├── setup.py, pyproject.toml             setuptools package/빌드 설정
│   ├── config/extension.toml                Isaac extension metadata
│   └── soarm101_lab/                        실제 import package
│       ├── so101_joint_mapping.py          하드웨어·정책 관절 표현
│       ├── so101_dataset_contract.py       저장 좌표/명령 의미 계약
│       ├── so101_robot_spec.py             URDF·limit·actuator 및 모델 증거
│       ├── representation.py              정책/데이터 표현의 호환 검사
│       ├── real2sim/                       장치·Snapshot·보정·실시간 비교
│       │   └── workspaces/                 Builder 소비·immutable Scene·엔진 cfg
│       ├── simulation/                     MuJoCo·공통 IK 구현
│       ├── data/                            공통 episode Reader/Writer·명시적 frame Buffer
│       ├── workflow/                       작업 계획·등록·생성·변환·학습 연결·UI
│       │   └── mimic_reference/            공식 Mimic 1.0.15 이식 본문/라이선스
│       ├── learning/minivla/               Qwen3-VL + Flow-DiT·학습·추론
│       │   └── configs/                    모델 기본 JSON·명시 Import template
│       ├── devices/lerobot/                기존 Leader/모터 adapter
│       ├── tasks/manager_based/soarm101_lab/
│       │   └── mdp/                        Isaac task action/관측/reset/recording
│       ├── assets/{robots,scenes}/         Python Isaac asset/scene 설정
│       ├── datasets/                       기존 LeRobot recorder adapter
│       └── utils/                          video·kinematics·입력/음성·공유 file identity
├── scripts/
│   ├── real2sim/                           Desktop/Runtime/Workflow/Workspace CLI
│   ├── mimic/                              Workspace/고정 씬 Annotation·DataGen CLI
│   ├── teleoperation/                      Isaac Source/직접 Teleop CLI
│   ├── tools/                              단일 변환기와 명시적인 검사 도구
│   ├── mujoco/                             MuJoCo 독립 점검·USD contact 진단
│   ├── inference/                          기존 ACT simulation 진입점
│   └── vlm_diagnostics/                    모델 환경 진단; production 학습 launcher 아님
├── assets/SO101/{urdf,usd,...}/             실제 robot mesh/URDF/USD 리소스
├── tests/{real2sim,minivla}/                계약·UI·CPU·엔진별 회귀
├── docs/{architecture,real2sim,minivla,archive}/  현재 구조·세부 계약·역사 문서
└── outputs/                                사용자 자산/실행/검증 결과; Python source 아님
    ├── real2sim/                            Snapshot·보정·immutable environments·IPC
    ├── workflow/                            아래의 단계별 결과와 Registry
    └── workflow_validation/                 검증 보고서·로그·화면; production 입력과 구분
```

`__pycache__`, `.venv-*`, `.pytest_cache`, `.git` 등은 코드 책임 트리에서 제외했다. 기존 `demo/`, `policy/`,
`annotation_frames/`, `screenshots/` 등 실험·사용자 자산 경로는 보관된 리소스이며 새 소스 package로 옮기지 않는다.
기존 최상위 `datasets/`는 현재 필요하지 않다. **Python package의 `source/.../datasets/`와 혼동해 삭제하면 안 된다.**
새 사용자 데이터의 실제 경로 계약은 [DATA_FLOW](../real2sim/DATA_FLOW.md#단계별-저장-경로입력-선택--2026-10-17)에 있다.
현재 신규 Source는 `sources/<사용자 이름>/origin`, 선별본과 명시 분할은 같은 Source 묶음의 사용자 이름 아래에 저장한다.
그 밖의 신규 Annotation/DataGen/Dataset/모델/평가는 각 Stage 저장 영역의 `<사용자 이름>/`에 직접 저장한다.
기존 중첩 결과와 Registry 부모/학습 config도 현재 계속 읽는다. Path-only 전환과 과거 역할 강제 제거는
[진행 중인 설계/현황](FUNCTIONAL_REFACTOR_PLAN.md)의 후속 단위이며 아직 완료된 기능으로 해석하지 않는다.

## 2. Real2Sim: 한 번의 연결 세션

아래 경로는 `source/soarm101_lab/soarm101_lab/` 기준이다. 각 행의 함수가 해당 기능을 찾을 시작점이다.

| 파일 | 기능·이 위치인 이유 | 호출자 → 주요 의존 | 수정할 때 |
|---|---|---|---|
| `real2sim/ui.py` | Desktop `Window`, Capture/RealSim/Workflow/보정 화면. 물리 제어 자체를 하지 않음 | UI script → launcher/Bus/WorkflowPanel/precision UI | 화면·버튼·설정 연결. 기존 위젯 순서·signal 보존 |
| `real2sim/launcher.py` | `RuntimeLauncher.connect/disconnect/poll`, worker env 구성 | Window → subprocess/Bus | 엔진 Python/env/비동기 시작·중지 |
| `real2sim/runtime.py` | `Runtime`의 connect/start/tick/command/run. 장치·엔진·save future 소유 | Runtime script → hardware/cameras/precision/scene/capture/storage | 제어 수명·Snapshot freeze. 높은 위험 |
| `real2sim/hardware.py` | `Leader`, `Follower`, `CalibrationDevice`; SDK/encoder 검증 | Runtime/Source/Calibration → LeRobot SDK·Precision | 통신·실측·정수 goal; 자동 Connect 금지 |
| `real2sim/cameras.py` | `CameraReader` read thread·RGB packet/host timestamp | Runtime → OpenCVCamera·hardware audit | 새로운 영상 입력. 노출 시각으로 오표기 금지 |
| `real2sim/core.py` | 단위/시각·`Stability/assess_pair` 비교 품질 | Runtime/Capture → joint mapper·NumPy | 안정성/timing 품질 규칙 |
| `real2sim/precision.py` | `PrecisionMapper`, `ProfileStore`, `Calibration`, `simulation_configuration` | hardware/Runtime/worker → robot spec/storage/Workspace | 보정 수학·승인·recorded model 호환 |
| `real2sim/precision_ui.py` | 보정 reference 수집/검토·승인 화면 | Window → file command/Calibration 상태 | 보정 화면; raw 장치 쓰기는 worker |
| `real2sim/profile.py` | Scene profile schema·`load/validate/render_k/transform` | Runtime/Workspace/scene → 수치·JSON | K/extrinsic·provenance 입력 |
| `real2sim/scene.py` | `build_usd/configure_environment/apply_profile` | Runtime/Workspace → profile/visuals·lazy Isaac/USD | profile의 Isaac 적용 |
| `real2sim/visuals.py` | `tapered_cup`의 profile 기반 USD 컵 mesh | scene → profile/pxr | 컵 visual. 측정/provisional 치수 보존 |
| `real2sim/capture_dataset.py` | `prepare_snapshot`, `SnapshotDataset.save/resume/finish` | Runtime writer → timing/모델 복사/storage | Snapshot 저장·측정 정합 |
| `real2sim/storage.py` | `atomic/sync_directory/Bus` | Runtime/UI/Workflow → OS·JSON·lock | 파일 IPC/원자성. 알고리즘 규칙을 넣지 않음 |
| `real2sim/api_manifest.json` | 하드웨어 API 감사 근거 데이터 | hardware audit | SDK 계약 확인; 실행 코드 아님 |

## 3. Workspace: Builder export를 실행 가능한 package로 소비

`real2sim/workspaces/`에 있는 이유는 실제 작업공간·모델·camera·물성 계약을 두 엔진이 함께 소비하기 때문이다.

| 파일 | 핵심 함수와 책임 | 호출자 → 의존 / 수정 지점 |
|---|---|---|
| `tasks.py` | `catalog/validate/selected/active_objects/configured/workspace_key`, Task 선택과 immutable Scene 구성 | 두 엔진/compiler/UI/worker → package/physics. Workflow 내부 import 없음 |
| `package.py` | `load/metadata/sha`, immutable 파일/hash·version·task 검증, ROOT | 모든 소비자 → profile/mapping. package schema |
| `builder.py` | `inspect_export/import_scene`, frame/mounted/world_pose, portable asset 검증 | import CLI/UI → package/profile/XML/YAML. Builder import 호환 |
| `environment.py` | `make_config/task_success_details`, Isaac scene/contact·robot cfg | Runtime/unified/evaluation → Isaac tasks/reset/physics. Isaac 연결 |
| `physics.py` | `compile_physics`, object 역할·mass/friction/collision | tasks/두 compiler → profile/task. 임의로 verified 물성으로 바꾸지 않음 |
| `reset.py` | `WorkspaceRandomizer/reset_workspace/source_workspace` | 두 backend/task/Source → task/계보. seed 소비/기록된 reset |
| `demo.py` | `coordinate_contract/verify_dataset`, recorded scene 연결 | Operations/unified → package/task/데이터 계약. Replay 호환 |
| `grasp.py` | `StableGrasp`의 접촉·상대 pose 기반 상태 | 두 adapter/평가 → NumPy. 집기/놓침 정의 |
| `diagnostics.py` | Workspace 물리 진단/증거 | check CLI → physics/package. production 성공 판단과 구분 |
| `mimic.py` | native Mimic developer cfg/물성 비교 | 명시 개발 도구 → environment/tasks. production 생성은 workflow/datagen |

## 4. Simulation: engine별 실제 계산

| 파일 | 기능·위치의 이유 | 호출자 → 의존 / 수정 지점 |
|---|---|---|
| `simulation/playback.py` | `restore_episode/replay_steps/step_targets`, 기록 복원·오차·command/post-step 경계 | Replay/Annotation → 순수 transforms/기존 Adapter protocol. env/UI/파일/성공 판정 소유 없음; Worker가 환경 수명/벡터 clock 소유 |
| `simulation/control.py` | `PhysicsControl/GenerationStopped`, pause·single-step·취소 | Annotation/Adapter/Mimic scheduler → threading. 엔진·Mimic import 없음 |
| `simulation/base.py` | `SimBackend`가 engine이 제공할 동작을 정의 | MuJoCo backend → ABC. 실제 engine seam |
| `simulation/mjcf.py` | `build_scene/robot_xml/load_inputs`, URDF/Workspace→메모리 MJCF/XML assets. 파일 저장 없음 | backend/check tools → package/profile/workspace_scene/task. physics 옵션/모델 생성 |
| `simulation/workspace_scene.py` | `extend_scene/add_camera/apply_builder_visuals`, object/cup/mount MJCF | mjcf → physics/Builder transform/presentation framing. scene geometry |
| `simulation/mujoco_backend.py` | `MujocoBackend` physics/FK/IK/reset/render; `PolicyEnvironment` 모델rad↔policy 환경; `MujocoSO101` 기존 adapter | Runtime/workflow/MiniVLA → MuJoCo/MJCF/IK/reset/coordinates |
| `simulation/ik.py` | `solve_pose/solve_selected`, solver/policy 검사·candidate/hold/best effort | 두 adapter/backend → SciPy/kinematics. 수식·기준은 고정 |

Isaac의 engine 연결은 `workspaces/environment.py`와 기존 `tasks/`의 cfg/MDP에 있다. 두 엔진의 폴더 모양을
억지로 대칭으로 만들지 않았다. MuJoCo process에 Isaac/Omni/USD를 import하면 fail-closed로 거절한다.

## 5. Workflow: 입력 Artifact 한 개에서 결과 한 개로

| 파일 | 핵심 책임/함수 | 호출자 → 의존 / 수정 지점 |
|---|---|---|
| `workflow/ui.py` | `WorkflowPanel` 선택·설정·readiness·execute; `TrainingLossPlot` | Window → Operations/Registry/view/settings/task. 확정 UI 보존 |
| `workflow/operations.py` | `Operations.plan/run/cancel/readiness`, `execute_request`, engine 계획 | UI/CLI → Registry/task/request/각 worker. 실행·취소·검증·등록 |
| `workflow/worker_request.py` | `MUJOCO_STAGES/mujoco_request`, 기존 protocol 검증의 단일 소유자 | planner와 MuJoCo worker → Path/JSON/Workspace/config. Qt/engine import 없음 |
| `workflow/artifacts.py` | `Registry`, `inspect_artifact/output_location/collection_at` | UI/Operations/Curation → 계약/저장/공유 identity. IDs·hash·path·lineage |
| `workflow/unified.py` | `execute/_native_isaac`; 공통 입력·환경 준비와 명시 Stage dispatch | worker → dataset/task/adapter/replay/annotation/datagen. Replay 본문/cache 삭제, Source 자동 replay 추가 금지 |
| `workflow/replay.py` | `run`, 명시 Replay 진행·기존 task/평가 Playback 결과·JSON | unified/명시 호출 → Simulation Playback/I/O/Presentation. Annotation import 없음, 환경은 빌림 |
| `workflow/annotation.py` | `annotate_episode/run`; Grasp/Place·수동 mark/skip·pause/step·동적 슬롯·순서 보존 저장·계측 | unified → Simulation Playback/기존 Adapters/DatasetWriter/PhysicsControl. Replay import·공통 물리 본문 없음 |
| `workflow/mimic.py` | `MimicGenerator/source_pool/GenerationScheduler`; 제어는 simulation.control | datagen → 공식 Mimic/Torch/lock. 시도 seed·성공 예약·중단 |
| `workflow/mimic_reference/` | 공식 Generator/Pool/selection/waypoint/math/config | mimic → portable adapter. NOTICE/UPSTREAM hash 기준 보존 |
| `workflow/datagen.py` | `run_mujoco/run_isaac/_isaac_loop`, pool·writer·scheduler 수명 | unified → adapter/mimic/dataset. 병렬 실행·Resume·실패 집계 |
| `workflow/adapters.py` | `SimulationAdapter/MujocoAdapter/IsaacAdapter/record_frame` | generator/datagen/replay → 각 engine/IK/StableGrasp/video. 관측·명령·실패 증거 |
| `workflow/dataset.py` | `EpisodeDataset`, migrate·validate_dataset·native/legacy recording 변환 | Source/unified/Curation → data.episodes/legacy FK/계보. Common Writer/배열 검증은 data 소유 |
| `workflow/episodes.py` | `source_catalog/source_split/split_episodes`, prepare/convert | Source/Curation/학습 → 정규화/Source 증거. 기존 정책 유지; Writer 삭제 |
| `workflow/source.py` | `metadata/resume_legacy/resume_common/save_common` | 기존 mujoco.source → I/O/기록된 Scene 검사. Source 준비·Resume 허용 조건·Common 저장 시 Simulation geometry 연결 |
| `workflow/data_resume.py` | 기존 HDF5 검증/병합/무손실 압축 | Operations/Source → HDF5/계약. 공통 생성 scheduler Resume와 구분 |
| `workflow/mujoco.py` | Source/legacy train·LeRobot→MiniVLA 준비·rollout·explicit import worker | execute_request/datagen → request/backend/episodes/learning. 이름과 달리 MiniVLA 연결도 현재 여기에 있음 |
| `data/episodes.py` | `EpisodeReader`, Common `DatasetWriter`, `LegacyEpisodeWriter`, `recording_args/validate_episode` | 전 Stage → NumPy/h5py·기존 atomic/영상/순수 계약. lazy 읽기·형식별 commit/manifest·Resume 파일 복사. env/FK/Stage/Registry 의존 없음 |
| `data/recording.py` | `EpisodeBuffer`, `SourceEpisode`, `source_episode_values` | 명시적 pre-step snapshots/initial/events → rows·MP4 staging/순수 Common 배열 조립. Source는 numeric leaf와 초기 state를 보유 복사, RGB는 기존 bounded queue만 소유 |
| `utils/transforms.py` | `pose_matrix/as_pose7/cartesian_actions` | Reader/Annotation/Adapter/legacy 형식 → NumPy. 원래 수치 본문 그대로 |
| `workflow/curation.py` | `analyze/select_distribution/export_subset` | Curation UI → data.episodes/기존 dataset/partition. 분석·선별 정책; 파일 Reader는 data 소유 |
| `workflow/curation_ui.py` | `CurationDialog`, 데이터 열기/동기 재생/분할 미리보기·저장 | WorkflowPanel → curation/Registry/episodes. Train/Val 잠금 보존 |
| `workflow/policy.py` | `train/load_policy/observation/resume_checkpoint/selected_minivla_checkpoint/ActionGuard` | worker/evaluation/UI → LeRobot/MiniVLA adapters. architecture 연결 |
| `workflow/evaluation.py` | `evaluate/EvaluationRecorder/episode_result/evaluation_seed` | common worker → policy/engine/SDK/task. Sim/Real 평가 |
| `workflow/hub.py` | `publication_metadata/publish` | common worker → HF API/기존 provenance. 업로드는 명시 실행 |
| `workflow/tasks.py` | `recorded_scene/from_artifact/bind_scene_options`, stamping/평가 metrics. 순수 구성은 workspaces/tasks | UI/compiler/worker → Workspace/physics. task/active object 의미 |
| `workflow/task_ui.py` | `TaskSelector`가 선택 task/object 편집 | WorkflowPanel → tasks/package. Source에서만 표시 |
| `workflow/presentation.py` | `Presentation/current/attach`, camera/status/제어 file IPC·문구 | engine worker/UI → storage/공통 상태. 진단 영상은 학습 입력 아님 |
| `workflow/view.py` | `CameraView/LogTail` | Workflow/Real2Sim UI → Qt/file incremental read. 표시 규칙 |
| `workflow/stages.py` | `controls/feeds`, stage별 제어·카메라 표 | UI/presentation → 순수 stage 이름. STAGES input/output 표와 의미가 다름 |
| `workflow/option_help.py` | 옵션 key/default/help·모델별 허용 값 | Settings/UI → 모델 config. UI default와 config default의 우선순위 |
| `workflow/settings_form.py` | `SettingsForm`의 옵션 JSON↔실제 Qt editor | WorkflowPanel → option_help/Qt. 체크된 override 직렬화 |
| `workflow/theme.py` | 확정된 stylesheet | WorkflowPanel → Qt. 이번 작업 무변경 |
| `workflow/robot_preview.py` | URDF FK/geometry·object telemetry preview | Curation UI/legacy dataset FK → XML/NumPy. 순수 pose 변환은 utils.transforms |

## 6. 학습·로봇 공통·지원 구현

| 파일 | 책임 / 호출자 / 의존 / 수정할 기능 |
|---|---|
| `learning/minivla/config.py`, `configs/default.json` | MiniVLAConfig/training_config. UI/worker/trainer가 읽음. 모델 schema·학습 override/구형 Resume 호환 |
| `learning/minivla/data.py` | load_manifest/Episodes/collate/fit_statistics/dataset_identity. Trainer/준비 worker → Parquet/MP4·NumPy/Torch·계보. sample·정규화·mask |
| `learning/minivla/backbone.py` | QwenBackbone processor/letterbox·층 feature/QLoRA. trainer/policy → Transformers/PEFT. 입력/토큰 계약 |
| `learning/minivla/head.py` | ActionHead/Flow-DiT·Flow Matching loss. train/policy → Torch. 모델 수식 |
| `learning/minivla/checkpoint.py` | architecture 검사·adapter state 읽기/쓰기. inference/기존 Trainer → config·필요 시 PEFT. Dataset/Trainer import 없음 |
| `learning/minivla/train.py` | main/compute_loss/validate_resume/save_checkpoint. subprocess → data/backbone/head. Batch·optimizer·validation·checkpoint |
| `learning/minivla/policy.py` | MiniVLA/RobotPreprocessor 추론·저장된 통계 정규화. eval/rollout → checkpoint/backbone/head. Trainer/Dataset 준비 import 없음 |
| `learning/minivla/rollout.py`, `evaluate.py` | 기존 rollout·평가 도구. policy/backend/case evidence. production engine 평가는 workflow 경로 확인 |
| `learning/minivla/act_baseline.py` | 공식 ACT 비교 adapter; Flow-DiT와 다른 loss. CPU test/명시 비교 도구에서 사용 |
| `learning/minivla/probe.py`, `profile.py` | 모델 환경·시간/메모리 진단. 사용자 명시 실험; production 자동 실행 아님 |
| `so101_joint_mapping.py` | SO101SimJointMapper/RobotCoordinates/observed_positions. 전체 시스템 공통 rad·degree·fraction 경계 |
| `so101_dataset_contract.py` | current_contract/read_contract/write_contract/require_resume_contract. 기록·변환·학습·평가가 공유 |
| `so101_robot_spec.py` | joint_limits/urdf_bytes/profile용 model evidence/limit_usd/actuator. 여러 engine 공통 수치 근거 |
| `representation.py` | describe/require_compatible/real_action/real_observation. Policy·Operations/실물 연결에 동일 의미 적용 |
| `utils/artifact_identity.py` | fingerprint(path). Registry·Dataset·MiniVLA가 공유하는 bytes identity, Registry/Qt 없이 호출 |
| `utils/async_video_writer.py` | VideoEpisode/영상 probe/relative path/PTS·checksum. Recorder/Writer/Converter가 공유 |
| `utils/so101_kinematics.py` | SO101Kinematics의 기존 Pinocchio FK/DLS. 선택 IK 및 검사용; 수학 보존 |
| `utils/episode_randomizer.py`, `keyboard.py`, `voice.py`, `constant.py` | 기존 task randomizer/입력/음성/상수. 사용자 동작·구형 task 호환. 새 공통 workflow를 중복 구현하지 않음 |
| `devices/device_base.py`, `devices/lerobot/so101_leader.py` | 기존 Isaac device·Leader advance. task/teleop CLI → motor/precision. model-rad command 생성 |
| `devices/lerobot/motors/`, `utils/` 하위 | motor bus/Feetech register·encoding·오류/데코레이터. Leader 내부 구현; 단위/SDK 정수 계약 |
| `tasks/.../base_pick_place_teleop_env_cfg.py`, `so101_mimic_env{,_cfg}.py` | Isaac cfg/환경 등록. Workspace make_config와 명시 고정 씬 CLI가 소비 |
| `tasks/.../mdp/so101_ik_actions{,_cfg}.py` | Isaac IK action/cfg pair. cfg class_type의 양방향 참조는 engine 관례 |
| `tasks/.../mdp/{observations,resets,so101_mimic_mdp,so101_mimic_recorders}.py` | Isaac 관측/reset/성공·기록, Source native 기록→공통 Writer. Workspace/Source worker가 호출 |
| `assets/robots/so101.py`, `assets/scenes/pick_place.py` | Isaac Python asset/씬 cfg. 실제 파일은 최상위 assets/SO101. 경로 이동 시 parents 계산 주의 |
| `datasets/lerobot_recorder.py` | 기존 LeRobotRecorder. Isaac 직접 record 도구 adapter; 새 공통 HDF5 Writer와 용도가 다름 |

`__init__.py`도 모두 같은 역할은 아니다. MiniVLA는 무거운 import를 lazy하게 제공하고, `datasets`는 recorder를
즉시 import하며, `tasks/assets/devices`는 Isaac 등록과 연결된다. 모듈을 옮길 때 부모 package 초기화까지 확인한다.

## 7. 실행 script와 테스트를 찾는 법

- `scripts/real2sim/real2sim_ui.py` → Window, `real2sim_sim.py` → Runtime. UI/engine process 분리를 유지한다.
- `scripts/real2sim/workflow.py` → Operations/execute_request/import/register. `workspace.py`는 package 도구,
  `check_workspace_physics.py`는 명시 진단이다.
- `scripts/teleoperation/record_mimic_dataset.py`, `teleop_so101.py` → Isaac Source/직접 Teleop.
- `scripts/mimic/{annotate_demos,generate_dataset}.py` → Workspace는 공통 unified 경로, 명시 `--task`는 고정 씬 호환.
- `scripts/tools/convert_isaac2lerobot.py` → 공통/Legacy 입력의 단일 LeRobot mapping. inspect/replay/kinematics script는 별도 진단.
- `tests/real2sim/test_{joint_mapping,precision_calibration,snapshots,runtime_liveness,...}.py`는 장치 없는 계약/fake tests,
  `test_{workflow,common_mimic,artifact_storage,dataset_collections,source_partitions,curation,...}.py`는 pipeline/storage 계약이다.
- `test_{mujoco_backend,ik_selection,active_objects,...}.py`는 실제 CPU physics/IK도 포함한다. USD tests와 process를 분리한다.
- `test_{workflow_view,mujoco_workflow_ui,settings_form,precision_ui,...}.py`는 Qt 동작을 고정한다.
- `tests/minivla/`는 CPU tensor·임시 checkpoint·LeRobot fixture 및 선택적인 로컬 processor 검사다. 실제 pretrained GPU 학습과 다르다.
- `test_architecture_contracts.py`는 요청/identity interface와 import 방향을 고정한다. 정확한 실행은 [검증](REFACTOR_VALIDATION.md).

전체 추적 Python inventory/import edge는 검증 JSON에 남겼지만, 이 문서의 소유권 설명은 실제 호출 경로를 확인해 작성했다.

## 6c-2a 진행 상태 — 2026-10-20

기록 Scene의 Isaac 생성만 `simulation/environment.py:create_isaac`로 이동하고 unified._native_isaac를 삭제했다. Worker가 identity 검사에 사용한 Workspace를 재사용한다. 105개 계약 회귀 통과, 설정/reset/close 본문은 입력 치환 외 AST 동일. 실제 Isaac baseline부터 GPU OOM으로 실패하여 native A/B 및 성능은 미검증이다. MuJoCo 생성, Source/평가의 별도 Isaac 경로, make_config의 기존 Presentation 간접 참조는 남아 있다. **6c-2 전체 완료 아님.** 근거: `outputs/workflow_validation/functional_refactor/environment/REPORT.md`.

## 6c-2b 실제 소유권 — 2026-10-21

`real2sim.workspaces.environment.make_config(..., overview_camera=False, grasp_observations=False)`는 전달된 요구만 사용한다. Workspace/Simulation의 Presentation.current/INSPECTION_STAGES 참조는 삭제했다. `workflow.unified.execute`는 연결된 DataGen의 Overview, `workflow.evaluation.evaluate_sim`은 연결된 평가의 Overview/Grasp 관측을 결정한다. 독립 Script/Source/Runtime은 false/false다. 정책 카메라 목록은 기존 Task가 소유한다. framing은 `simulation.workspace_scene.framing` 한 곳에 있고 이전 `workflow.presentation.framing`은 없다. MuJoCo는 같은 순수 구도 함수를 쓴다.

삭제 edge: workspaces.environment→workflow.presentation/stages, simulation.workspace_scene→workflow.presentation. 새 edge: workspaces.environment→simulation.workspace_scene. 다른 Stage/I/O 역참조나 전체 SCC 해소를 뜻하지 않는다. 실제8종 설정/32회 UI독립 구성·Isaac2환경 Annotation/평가·MuJoCo 모델/물리/RGB 비교 근거는 `outputs/workflow_validation/functional_refactor/presentation_config/REPORT.md`다. 이전6c-2a의 간접 UI 결합 문구는 당시 상태다. 전체6c-2는 미완료다.
