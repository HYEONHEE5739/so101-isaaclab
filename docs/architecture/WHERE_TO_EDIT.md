# 기능별 수정 시작점

## 2026-10-21 — Curation 자유 선별·분할 완료

`workflow.curation.split_dataset/export_splits/export_subset`가 Episode 단위 선택·선택적 Train/Val 배정·재선별을 소유한다. 기존 `episodes.source_split/export_source_splits` 강제 Group 연결 성분 분할은 삭제했다. 과거 Partition/Lineage는 변경 없이 검증·보존하며 새 배정은 `curation_split.json`/`curation.role`로 분리한다. UI에 전체 제외·출력 폴더 선택·저장 취소를 연결했다. 지원 편집 형식은 기존 HDF5와 Legacy NPY/MP4이며 LeRobot 직접 편집·평가 묶음 전체 편집은 지원하지 않는다. 기존 Training/LeRobot Merge 정책은 후속이다. [검증/지원 범위](../../outputs/workflow_validation/functional_refactor/curation_independent/REPORT.md). 아래 이전 완료 기록은 당시 상태다.


## 2026-10-21 — Registry 제거 / 전체 Stage Path 실행

현재 모든 Workflow Stage의 공개 입력은 Path다. `workflow/artifacts.py`와 Registry 등록·조회·promote·ID CLI는 삭제했다. 기존 `artifacts/*.json`은 사용자 과거 자료로 그대로 남지만 실행에서 읽지 않는다. 아래 이전 단계 기록은 당시 상태이며 현재 경계는 이 절을 따른다.

- `workflow/paths.py`: native marker 탐색·저장 이름·출력 경계. 목록에서 payload/모델을 읽지 않는다.
- `worker_request.stage_input/episode_context`: Path에서 Dataset/Checkpoint/Workspace 계약을 읽고 기존 Stage 실행에 전달한다. `dataset.inspect_dataset`와 `policy.inspect_checkpoint`가 형식별 검사를 소유한다.
- `Operations`: Path 요청·출력 예약/Lock·process/Cancel·runs 로그·완료 결과 Path. UI의 선택/상세/READY는 기존 executor에서 처리한다.
- Curation은 원본 Path를 직접 열어 새 결과를 저장한다. 과거 Group/Partition 정책과 MiniVLA 학습/Val 정책은 이번에 바꾸지 않았다. 공식 ACT/SmolVLA Trainer/Resume도 유지한다.
- 필수 provenance/좌표가 없는 Checkpoint는 발견되더라도 평가 불가 오류를 낸다. Registry·다른 Checkpoint로부터 정보를 복원하거나 원본을 고치지 않는다.

상세 결과와 검증 한계: [Registry 제거 보고서](../../outputs/workflow_validation/functional_refactor/registry_removal/REPORT.md).


## 7b-1 Replay/Annotation Path 실행

입력/READY: `workflow/ui.py:_refresh_episode_inputs/_poll_episode_inputs/_start_ready_check/_poll_ready_check`. 계획·출력/예약·process: `operations.py:episode_plan/Operations.run`. Native header/Scene/Coordinate: `worker_request.py:episode_context`. Worker full 검사/기존 엔진 수명: `unified.execute`. Playback·주석 알고리즘·Writer의 소유권은 바꾸지 않았다.

CLI run/ready는 이 두 Stage만 --input/--workspace/--root/--storage-name을 사용한다. UI의 Path 선택은 Curation 창으로 직접 전달하지만 창 자체의 기존 등록/분할 정책은 후속이다. 다음7b-2의 DataGen caller는 [CALLERS](../../outputs/workflow_validation/functional_refactor/path_execution/CALLERS.md)를 따른다. [실제 실행/회귀](../../outputs/workflow_validation/functional_refactor/path_execution/REPORT.md).

## 7a Path 발견/선택을 바꿀 때 (2026-10-21)

형식 marker·nested/symlink/lock 처리: `workflow/paths.py:input_format/discover_inputs`. Resume 목록/선택의 비동기 연결: `workflow/ui.py:_request_model_discovery/_poll_model_discovery/browse_checkpoint/_poll_checkpoint_metadata`. 선택 설정: `workflow/policy.py:checkpoint_metadata`, Dataset header: `data/episodes.py:dataset_metadata`. Curation Picker는 marker만 검사하고 기존 QThread 분석으로 넘긴다. CLI: `scripts/real2sim/workflow.py discover`.

발견은 실행 허가가 아니다. ID 기반 Stage 입력·실행과 등록을 바꾸려면 [7b 정확한 함수 목록과 metadata 누락 처리 조건](../../outputs/workflow_validation/functional_refactor/path_discovery/REGISTRY_RESPONSIBILITIES.md)을 먼저 읽는다. 지금은 Curation 정책·Training/Validation 변경을 섞지 않는다.

**현재 구현 경계:** 공통 제어·guard·MiniVLA 추론 core·순수 Workspace Task와 Curation Reader(6a)·Common Writer/Buffer(6b-1/2)와 Source Writer(6b-3)·공통 Playback/Replay Stage(6c-1)를 분리했다. Registry와 MiniVLA 학습·Group 정책은 후속 교체 대상이며 아직 실제로 사용된다.

먼저 [전체 구조](PROJECT_STRUCTURE.md)로 소유자를 찾고, [실행 흐름](SYSTEM_FLOW.md)에서 해당 함수의 앞뒤 데이터를
확인한다. 아래의 경로는 `source/soarm101_lab/soarm101_lab/` 기준이다. 코드 이동이 필요해도 사용자 UI/자산/단위 계약은 유지한다.

## 일상적인 수정

| 수정하려는 기능 | 먼저 볼 파일·진입 함수 | 관련 코드·영향 범위 | 변경하면 안 되는 계약 / 우선 검사 |
|---|---|---|---|
| 실제 카메라 입력 | `real2sim/cameras.py:CameraReader.start/_run/nearest/close` | `runtime.open_cameras`, `hardware.audit_camera_sources`, Window 장치 설정·capture | RGB/ID·해상도·host read interval·copy/close. 새 camera 역할은 아래 표 참고 |
| 로봇 Joint Mapping | `so101_joint_mapping.py:SO101SimJointMapper/RobotCoordinates/observed_positions` | `precision.PrecisionMapper`, `hardware.Follower.encode_goal`, `Runtime.command_target` | degree/%↔model rad↔fraction, measured와command 구분. joint/precision tests |
| Calibration 검증 | `real2sim/precision.py:fit_joint/validate_joint/review_joint/PrecisionMapper` | ProfileStore/Calibration/precision_ui/robot_spec | raw branch/identity/승인/URDF SHA·과거 profile. 수학/실장치 분리 검사 |
| Profile/카메라 K | `real2sim/profile.py:validate/load/render_k` | capture_dataset/scene/Builder 소비 | K의 원본·렌더 해상도, optical frame, provisional provenance |
| Snapshot | `capture_dataset.py:prepare_snapshot/SnapshotDataset.save` | Runtime.command/save_snapshot_requests/collect_snapshot_finalizers, storage.atomic | request 시 state/RGB freeze, freshness·실측·저장 원자성/finish 순서 |
| 장치 연결/Start/Stop | `runtime.py:connect/start/stop/disconnect_hardware/run` | launcher/Bus/Leader/Follower/CameraReader | Connect≠Start, 승인 없이 torque/goal 금지, heartbeat·예외 close. 높은 위험 |
| Workspace import | `real2sim/workspaces/builder.py:inspect_export/import_scene` | package.load/metadata, 별도 Builder의 export schema | immutable/version/hash·robot frame/좌표/출처. Builder 원본 변경 금지 |
| Workspace 메타데이터 | `workspaces/package.py:load/metadata` | builder.import_scene, demo.coordinate_contract, tasks.workspace_key, Registry inspection | 필드가 identity인지 정보인지 먼저 결정. 기존 hash/저장 파일 재작성 금지 |
| Isaac Scene | `workspaces/environment.py:make_config` | simulation.environment.create_isaac, tasks cfg/MDP, scene.configure_environment | modelrad/action order·전체 reset·관측·camera. USD와 실제 PhysX 검사 구분 |
| MuJoCo Scene/물성 | `simulation/mjcf.py:build_scene/robot_xml`, `mujoco_backend.py:MujocoBackend` | workspace_scene.extend_scene/physics.compile_physics, worker_request.prepare_mujoco → simulation.environment.create_mujoco | 기존 profile/reset·recorded model·MuJoCo 독립 process·physics defaults |
| MuJoCo 손가락 collision | `simulation/mjcf.py:robot_xml` | 저장소 루트 `assets/SO101/mujoco/finger_convex_parts.json`, 재생성 `scripts/mujoco/export_finger_collisions.py` | collision만 변경·원본 visual/URDF 불변·STL SHA·모터/물성 보존·접촉/실제 물리 검사 |
| IK 계산/실패 정책 | `simulation/ik.py:solve_selected/solve_pose` | adapters.target_eef_pose_to_action, utils.so101_kinematics | SciPy/DLS 수식·seed·strict/hold/best_effort·limit·gripper. IK parity tests |
| Task/active object/Scene 구성 | `real2sim/workspaces/tasks.py:catalog/validate/selected/configured` | 두 compiler/Workspace 환경/UI/worker; 기록/stamping은 아직 workflow.tasks | 원본 불변·기록 reset/physics·Task/Scene hash·명령문으로 identity 추정 금지 |
| 공통 pause/step/취소 | `simulation/control.py:PhysicsControl/GenerationStopped` | Annotation/DataGen/Adapter, 기존 Mimic Scheduler | pause 토큰·환경별 single-step·중단 waiter 해제, 엔진 의존 없음 |
| 정책 명령 guard | `workflow/policy.py:ActionGuard` | Sim 평가/MiniVLAJointPolicy/옛 CLI | finite·실측 범위·rate/target clamp 수학·누적 상태 보존 |
| Source 수집·Discard·Resume | `workflow/mujoco.py:source`, `workflow/source.py:resume_legacy/resume_common` | Simulation `PolicyEnvironment.recording_context/sample/geometry`, Isaac native recorder는 기존 경로 | pre-observation/target·native 초기 state·기존 Resume 거부 조건/완성분 보존 |
| Source 기록·Legacy manifest | `data/recording.py:SourceEpisode`, `data/episodes.py:LegacyEpisodeWriter.save/copy_episodes` | 기존 VideoEpisode, Source Stage가 이미 생성된 관측/명령/metadata 전달 | numeric/RGB 소유권·NPY dtype/MP4 PTS·episode 게시 후 provenance→manifest·rollback |
| Source Common 조립·FK | `workflow/source.py:save_common`, Simulation `PolicyEnvironment.recording_geometry` | `data/recording.py:source_episode_values` → 기존 DatasetWriter | FK는 저장 때만; 기존 Common 첫 관측 초기 state와 별도 native legacy_initial_state 의미 유지 |
| Replay 진행·결과 | `workflow/replay.py:run` | unified.execute의 명시 dispatch, Simulation Playback/기존 Adapter·Presentation | 기록된 joint_targets·strict task gate와 평가 기록 Playback 구분·기존 Preview/status 빈도 |
| 공통 물리 재생 | `simulation/playback.py:restore_episode/replay_steps/step_targets` | Replay/Annotation → 기존 Adapter/순수 pose 변환 | restore1회·기존 DT/단위·command→post-step barrier. 벡터 전체 관측 후에만 다음 episode 복원 |
| Annotation 판정·수동·배정 | `workflow/annotation.py:annotate_episode/run` | Playback의 before/observation/after hooks, 기존 PhysicsControl/Writer | Grasp/Place·mark/skip/pause/step·1/2/4/8 동적 배정·입력 순서 저장. Stage 간 import 금지 |
| 저장 이름·결과 경로 | `workflow/ui.py:stage_options/execute`, `workflow/artifacts.py:suggested_name/output_location` | Operations.plan/run·Registry | 모든 UI 단계 필수 이름·단계별 값 분리·Annotation flat 경로·기존 데이터/계보 보존·중복 이름 거부 |
| DataGen 알고리즘 연결 | `workflow/mimic.py:MimicGenerator/source_pool` | datagen.run_mujoco/run_isaac, adapters, mimic_reference | 공식 본문 보존·시도별 RNG·Pool1회. 신규 알고리즘 fallback 금지 |
| DataGen 성공 수/Resume | `mimic.GenerationScheduler.finish/progress`, `datagen._resume` | writer commit, report/attempts/presentation | 예약→저장→성공 확정·rollback, stop/cancel·seed/IK config 일치 |
| DataGen 실패 진단 | `adapters.py:SimulationAdapter.failure_category/failure_diagnostics/update_grasp` | datagen._attempt_details→scheduler/presentation.datagen_status_text | 실제 grasp/drop/place 증거·IK 진단 구분. 분모/기존 category/성공 조건 |
| HDF5/영상 검증 | `data/episodes.py:validate_episode/DatasetWriter.commit`, `workflow/dataset.py:validate_dataset` | utils.async_video_writer, EpisodeBuffer/기록 변환 | schema/clock/PTS/SHA·원자 commit·상대경로/소유권 |
| 파일/데이터 bytes identity | `utils/artifact_identity.py:fingerprint` | Registry.verify, learning.load_manifest/Resume, Source/catalog | 파일명·정렬·.cache 제외·HDF5 외부 영상. 기존 hash 절대 임의 변경 금지 |
| Curation 분석·선별 | `workflow/curation.py:analyze/export_subset` | data.episodes.EpisodeReader, curation_ui, episodes Source split | 원본/TrainVal 잠금·동기 seek·불명 lineage·선별 저장 경계 |
| Common 저장/완료·영상 인수 | `data/episodes.py:DatasetWriter.commit/close/validate_episode` | Source/Isaac recorder/Annotation/DataGen/Evaluation/Resume, 기존 VideoEpisode | 배열·PTS·SHA·private group·소유 staging만 rename·외부 영상 byte copy·중복 경로 거부·취소 시 완성분 정책 유지 |
| Simulation 녹화 경계 | `workflow/adapters.py:SimulationAdapter.record_frame` → `data/recording.py:EpisodeBuffer` | 두 엔진의 pre-step, Annotation의 before state, Evaluation의 이미 읽은 state/images | 관측/이벤트는 Simulation 소유, I/O는 callback 없이 전달값만 기록. q≠target·시간/초기 state·모든 종료 경로 abort |
| Telemetry/정확한 frame 읽기 | `data/episodes.py:EpisodeReader.read/frame/close` | curation.analyze/load_episode, CurationDialog.release_reader, utils.transforms/영상 utils | 측정 q·target·시각/PTS·pose 정렬 보존. Reader 순차 사용, Qt 작업 완료 후 close. 전체 RGB 읽기 금지 |
| Source Train/Val 분할 | `workflow/episodes.py:source_catalog/source_split/export_source_splits/split_episodes` | Curation preview/save, Registry.source_collection/ensure_source_origin | frame 기준 분리 금지, Source/group 중복 누수 거절, 동일 origin 여러 split |
| 결과 경로/Registry | `workflow/artifacts.py:output_location/Registry.register/verify` | Operations.run, input_directory/collection_at/discover_collection | 평평한 이름 경로·IDs/부모/SHA·예약/overwrite·원본 자산. 변경 높은 위험 |
| Workflow 데이터 선택 | `workflow/ui.py:browse_data/select_data` | Registry.input_directory/discover_collection, Window.browse_scene(Live Import만) | 단계별 입력 폴더/선택 분리·Source 버전·Train/Val 계약·선택 중 worker/장치 시작 금지 |
| LeRobot 변환 | `scripts/tools/convert_isaac2lerobot.py` | episodes.convert_dataset→동일 converter, dataset contract | measured state와joint_targets action·단위/FPS·MP4·partition |
| MiniVLA Dataset/정규화 | `learning/minivla/data.py:load_manifest/Episodes/collate/fit_statistics` | workflow.mujoco.train_from_lerobot/_lerobot_index, episodes.validate_training_inputs | Train-only 통계·Source 누수·sample시점·H40 padding·gripper 관측범위 |
| MiniVLA 추론/정규화 | `learning/minivla/policy.py:MiniVLA/RobotPreprocessor` | checkpoint.py, backbone/head/config | 저장 통계·구형 batch/accumulation 해석 유지. Trainer/Dataset 준비 import 금지 |
| MiniVLA 모델 | `head.py:ActionHead`, `backbone.py:QwenBackbone` | config/processor/compute_loss/policy | 고정 architecture schema·tokens/masks·loss. CPU tests≠실제 GPU 품질 |
| MiniVLA Checkpoint 형식 | `learning/minivla/checkpoint.py:require_architecture/restore_adapter/adapter_state` | policy, 현재 Trainer 저장 | architecture schema·adapter 상태 해석·기존 파일 불변 |
| MiniVLA 학습/Checkpoint | `train.py:main/compute_loss/save_checkpoint/validate_resume`, `config.py:training_config` | workflow.mujoco.train, settings/option_help, data.identity | Batch/accumulation·optimizer·stats/dataset identity/옛 Resume 유지 |
| ACT/SmolVLA 학습 | `workflow/policy.py:train/resume_checkpoint` | 공식 lerobot trainer/config, Operations 옵션 | LeRobot processor·dataset/modelrad·checkpoint output·명시 publication |
| 평가/Checkpoint 선택 | `workflow/evaluation.py:evaluate_sim/EvaluationProgress/EvaluationRecorder`, `workflow/policy.py:selected_minivla_checkpoint/load_policy/batch_actions` | mujoco.execute, evaluation.evaluate, curation.analyze, replay.run | 전체 평가 quota·정책 queue 환경 분리·성공/실패 label·Sim/Real 구분 |
| 실물 정책 실행 | `workflow/evaluation.py:evaluate/real_start_pose`의 real_eval 경로 | `workflow/ui.py:stage_options, real_evaluation_defaults`, representation/precision/hardware/policy | UI 시작 버튼/CLI opt-in·활성화 전 float64 자세 검사·hardware_preflight.json·strict inverse/보정 identity·finite/freshness/모터 range |
| 업로드 | `workflow/hub.py:publish/publication_metadata` | HF API, Registry·기존 provenance | 별도 명시 실행·기존 인증/정확한 commit, 자동 공개/업로드 금지 |
| 전체 UI | `workflow/ui.py:WorkflowPanel.configure_stage/refresh/stage_options/execute` | real2sim.ui.Window, view/settings_form/task_ui/theme | 확정 widget tree/visibility/defaults/연결/Stage별 폴더·선택/잠금 |
| UI 응답 지연·반복 읽기 | `workflow/artifacts.py:Registry.get/all(metadata_only=True)`, `workflow/ui.py:_start_ready_check/_poll_ready_check` | refresh/입력 목록·Window Scene 목록·CameraView.show_image | 표시용 projection은 실행에 사용하지 않음. worker의 full record·자산/계보 검증 유지. 외부 변경·Val 세대 토큰·영상 새 프레임 검사 |
| 학습 로그/그래프 | `presentation.py:training_status_text`, `view.py:LogTail`, `ui.py:TrainingLossPlot` | MiniVLA metrics.jsonl/기존 LeRobot stdout | 실제 측정 Val 점·현재 Run만 표시·Train에서만 노출 |
| worker 요청 검증 | `workflow/worker_request.py:mujoco_request` | Operations.mujoco_plan/MuJoCo.execute/CLI choices | 검사 순서·오류·options 객체·engine/Qt 미import |

## 새 기능의 변경 전파를 예상하는 법

다음은 감사용 가상 변경 분석이며 **이번에 새 기능을 구현한 목록이 아니다.**
파일 수만 줄이려고 서로 다른 계약을 합치지 않는다.

| 가상 변경 | 정상적인 시작점과 전파 | 피해야 할 결합 |
|---|---|---|
| 새 camera backend, 기존 side/wrist 역할 유지 | CameraReader 획득 구현→Runtime packet interface; camera metadata audit·tests | Training/Registry를 수정하지 않아도 packet 계약으로 공급 가능해야 함 |
| 세 번째 camera **역할** 추가 | 위 경로+profile schema/Window feeds/capture 검증/task observation_cameras/recording/변환/모델 config | 현재 두 카메라 검증과 고정 Side-only MiniVLA를 몰래 확대하지 않음. “입력 종류”보다 큰 변경 |
| Calibration 규칙 추가 | precision validate/review→mapper/store 승인·command 연결 tests | UI가 동일 규칙을 따로 재구현하지 않음; hardware safety 재검증 |
| Workspace metadata 추가 | package의 검증/metadata identity 여부→Builder 소비/기록/Registry/eval | optional 정보와 content identity 변경을 구분; 과거 파일 일괄 수정 금지 |
| MuJoCo physics 옵션 | mjcf/backend→_environment→허용 Workflow 옵션·recorded execution evidence | Isaac 전체 옵션으로 위장하거나 Resume의 기존 물성 누락 금지 |
| 새 Workflow Stage | Operations.STAGES/plan/execute_request→worker, stages controls/feeds→UI/option_help→Registry type/출력 | 이들은 각각 다른 결정. 범용 plugin framework로 감싸지 않음; 취소/등록까지 계약 검사 |
| 새 Policy Architecture | policy.train/load_policy→전용 learning 구현/Checkpoint→옵션/표시와 eval adapter | 모델 코드가 UI/Registry 내부 구조를 알아야 하는 설계 방지 |
| Dataset 검증 추가 | validate_episode/validate_dataset 또는 학습 load_manifest(검사 목적에 맞게 선택)→저장/읽기/변환 consumers | 물리 성공·표현 schema·파일 identity 검사를 한 판정으로 합치지 않음 |
| DataGen 실패 항목 | adapter evidence/category→attempt details→scheduler aggregate→공통 presentation | 두 engine의 UI 문자열/실패 집계를 독립 복제하지 않음 |

## 수정 전 확인 순서

1. 입력이 raw/native/modelrad/policy fraction/tensor 중 무엇인지 먼저 찾는다.
2. 상태 소유자와 출력 파일을 확인하고 같은 역할의 기존 구현이 있는지 찾는다.
3. [INVARIANTS](../real2sim/INVARIANTS.md)와 해당 영역 tests를 읽는다.
4. 먼저 기존 호출자 interface의 기대값/오류를 고정한 다음 작은 범위를 바꾼다.
5. GUI/Isaac/USD/MuJoCo/학습 native 의존성을 검사 환경에 한꺼번에 섞지 않는다.
6. 사용자 자산 대신 tmp fixture로 저장·실패·복구를 검사하고 실제 장치/GPU 성공을 별도로 구분한다.

## 6c-2a 진행 상태 — 2026-10-20

기록 Scene의 Isaac 생성만 `simulation/environment.py:create_isaac`로 이동하고 unified._native_isaac를 삭제했다. Worker가 identity 검사에 사용한 Workspace를 재사용한다. 105개 계약 회귀 통과, 설정/reset/close 본문은 입력 치환 외 AST 동일. 실제 Isaac baseline부터 GPU OOM으로 실패하여 native A/B 및 성능은 미검증이다. MuJoCo 생성, Source/평가의 별도 Isaac 경로, make_config의 기존 Presentation 간접 참조는 남아 있다. **6c-2 전체 완료 아님.** 근거: `outputs/workflow_validation/functional_refactor/environment/REPORT.md`.

## 6c-2b 실제 소유권 — 2026-10-21

`real2sim.workspaces.environment.make_config(..., overview_camera=False, grasp_observations=False)`는 전달된 요구만 사용한다. Workspace/Simulation의 Presentation.current/INSPECTION_STAGES 참조는 삭제했다. `workflow.unified.execute`는 연결된 DataGen의 Overview, `workflow.evaluation.evaluate_sim`은 연결된 평가의 Overview/Grasp 관측을 결정한다. 독립 Script/Source/Runtime은 false/false다. 정책 카메라 목록은 기존 Task가 소유한다. framing은 `simulation.workspace_scene.framing` 한 곳에 있고 이전 `workflow.presentation.framing`은 없다. MuJoCo는 같은 순수 구도 함수를 쓴다.

삭제 edge: workspaces.environment→workflow.presentation/stages, simulation.workspace_scene→workflow.presentation. 새 edge: workspaces.environment→simulation.workspace_scene. 다른 Stage/I/O 역참조나 전체 SCC 해소를 뜻하지 않는다. 실제8종 설정/32회 UI독립 구성·Isaac2환경 Annotation/평가·MuJoCo 모델/물리/RGB 비교 근거는 `outputs/workflow_validation/functional_refactor/presentation_config/REPORT.md`다. 이전6c-2a의 간접 UI 결합 문구는 당시 상태다. 전체6c-2는 미완료다.

## 6c-2c 실제 소유권 — 2026-10-21

- Request/파일/Source Resume/보정/Physics Profile 선택: `workflow.worker_request.prepare_mujoco(request, coordinates=..., hz=..., workspace=...) → kwargs`. 이미 검증한 원본 Workspace를 재사용하며 compiler에 구성 copy 대신 원본을 전달한다. Task/Active Objects/Simulation.reset은 명시 kwargs다.
- 실제 생성: `simulation.environment.create_mujoco(**kwargs) → 기존 PolicyEnvironment`. Stage/Dataset/Registry/Resume를 해석하지 않는다. Worker가 정상 close, PolicyEnvironment가 Backend 획득 후 초기화 실패 close를 소유한다.
- 기존 `workflow.mujoco._environment`와 output 인수·export는 없다. Unified/DataGen/Evaluation의 생성용 mujoco import를 삭제했다. Source/옛 rollout/환경검사 caller도 새 경계를 직접 사용한다.
- 계속 쓰는 Legacy/Common provenance 읽기는 `workflow.dataset.recording_provenance`로 이동했다. 새 형식·새 Writer·Wrapper는 없다. DataGen의 제공 Parent는 소유권을 이전받아 닫고 Parent가 없으면 직접 생성한다. 공유 MjModel/독립 MjData·ThreadPool·RNG·수치 알고리즘은 그대로다.
- `workspaces.reset.source_workspace → workflow.tasks/artifacts` 및 Legacy reader 위치는 남은 결합이다. Registry/Path/Training/Isaac 생성 통합을 완료했다는 뜻이 아니다.

실제 전후 결과·load/compile/close·DataGen 첫 환경 배정 차이와 검증용 배정 고정의 한계는 `outputs/workflow_validation/functional_refactor/mujoco_environment/REPORT.md`를 따른다. 숫자/MP4 일치에서 실측 timing·새 Source UUID·private 경로는 구분했다. 기존 6c-2a/b의 MuJoCo factory 미완료 문구는 당시 상태다.


## 2026-10-21 — 6c-2d 완료 / 6c-2 환경 생성 경계 완료

Isaac 설정/생성 호출35곳과 Gym 경로를 점검했다. Recorded Scene의 create_isaac, Source/Teleop, Sim 평가, Live Runtime, 독립 검사/고정 Scene CLI는 센서·Reset·수명이 달라 별도 진입을 유지한다. 공통 cfg는 기존 make_config이며 새 Factory/Manager/Wrapper를 추가하지 않았다. Source/평가/Runtime/물리 CLI는 이미 준비한 Workspace dict를 재사용한다. Runtime.build_environment(p, *, workspace=None)는 connect의 검증 snapshot을 빌리며 standalone 호출은 필요 시 현재 파일을1회 읽는다. Runtime profile 적용에는 파생 Task copy가 아닌 원본 Workspace를 사용한다. Source는 장치 연결 뒤 실제 mapping Hash 검사를 유지했다.

실제 Isaac A/B: 평가2환경×6step의 설정·3trace/72numeric·24frame/PTS·Preview3파일, Runtime/Source 생성 경계6step과 물리CLI120step의 설정·초기/명령/관절·물체속도 trace가 같다. Source는 가상 Leader이며 Recorder 전 경계만 검증했다. load는 평가생성1→0(상위1회 별도), Runtime connect3→1/standalone build2→1, Source2→1, 물리CLI3→1이다. 평가 baseline 반복과 전후 RGB 차이는 따로 기록했으며 pixel/통계적 동등성·보편적 속도향상은 주장하지 않는다. GUI/Runtime137PASS·인접57PASS·입력6PASS, core 완료 로그는 보고서 참조(중복합산 금지). 설정32회 UI독립 비교도 통과했다.

환경 생성·실제 reset event·Camera/framing에는 Workflow/UI/Registry import가 없다. 기존 workspaces.reset.source_workspace/demo.verify_dataset의 상위 입력 정책 역참조와 Task 생략 시 요청 파일 fallback은 남으며 전체 Workspace/Registry 정리 완료가 아니다. 사용자자료/설정·UI·의존성·Commit/Push 변경 없음. 실제 Source 전체 수집·실물·장시간·새 Annotation/DataGen actual·4/8환경 actual은 이번에 재검증하지 않았다. **6c-2a/b/c/d 경계는 완료, 전체 기능 단위 리팩터링은 미완료**다. 상세 호출자/검증/한계는 [6c-2d 보고서](../../outputs/workflow_validation/functional_refactor/isaac_boundaries/REPORT.md)와 해당 CALLERS.md를 따른다.
