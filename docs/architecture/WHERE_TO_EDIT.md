# 기능별 수정 시작점

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
| Isaac Scene | `workspaces/environment.py:make_config` | tasks cfg/MDP, scene.configure_environment, unified._native_isaac | modelrad/action order·전체 reset·관측·camera. USD와 실제 PhysX 검사 구분 |
| MuJoCo Scene/물성 | `simulation/mjcf.py:build_scene/robot_xml`, `mujoco_backend.py:MujocoBackend` | workspace_scene.extend_scene/physics.compile_physics, workflow.mujoco._environment | 기존 profile/reset·recorded model·MuJoCo 독립 process·physics defaults |
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
