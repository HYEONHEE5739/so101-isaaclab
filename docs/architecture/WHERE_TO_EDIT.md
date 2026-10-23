# 기능별 수정 위치

기준: 2026-10-23. [전체 지도](PROJECT_STRUCTURE.md), [실제 호출](SYSTEM_FLOW.md), [SDK 소유권](TASK_SDK_DESIGN.md). 사용법은 [SCRIPT_GUIDE](../task_sdk/SCRIPT_GUIDE.md)와 [NATIVE_LEROBOT](../real2sim/NATIVE_LEROBOT.md)를 수정한다.

| 변경 목적 | 실제 시작점 |
|---|---|
| Path 탐색/출력 이름·잠금 | workflow.paths, operations |
| Qt 선택·Preview·상태/생성 | workflow.ui:stage_options/create_task_template/_start_ready_check; task_ui |
| Worker 실행·Cancel·native 입력 | operations.execute_request; worker_request.stage_input/episode_context |
| Script AST/identity·callable binding | task_scripts.read_script/read_followup/resolve_script/load_execution_script/load_followup |
| Metadata Task/Scene·템플릿 | workspaces.tasks:configured/scene_selection/task_binding/generate_template |
| Source 수명·Recording·Resume | workflow.mujoco.source, workflow.source; data.recording.SourceEpisode |
| Object Reset·Runtime 조회 | simulation.task_context:reset_objects/TaskContext; task_terms.reset; WorkspaceRandomizer 원본 |
| Term binding·환경/Episode별 상태 | SimulationAdapter.bind_followup/after_reset/record_frame/term_boolean |
| StableGrasp / Drop·Release | workspaces.grasp.StableGrasp; SimulationAdapter.update_grasp/_update_failure_evidence |
| Annotation 채택·Manual/Boundary | annotation.annotate_episode; 공개 조회 task_terms.annotation/diagnostics/events |
| Mimic cfg/생성·Resume | mimic.configuration/subtask_configuration; datagen.run_*/_resume; 공식 reference 수학 복제 금지 |
| 평가 Policy·사용자 Hook·저장 | evaluation.evaluate_sim/EvaluationRecorder; policy.load_policy/batch_actions/ActionGuard |
| HDF5/Legacy·영상/Signal | data.episodes.DatasetWriter/LegacyEpisodeWriter; data.recording.EpisodeBuffer; workflow.dataset |
| Curation / Conversion·Merge | workflow.curation/curation_ui; episodes.convert_dataset; lerobot_merge |
| Native Dataset/Model·Hub 검사 | learning.datasets:inspect_dataset/inspect_policy/resolve_model/validate_sim_features; workflow.hub |
| 공식 학습·Fine-tune/Resume | learning.training:training_command/native_arguments/resume_checkpoint/train |
| 저장 모델 독립 Validation | learning.validation; 모델/Loss/optimizer loop는 공식 LeRobot |
| 환경·Physics·명시 카메라 | simulation.environment; workspaces.environment; Backend/IK 원본 |
| 기존 함수형 Task Python | worker_request.load_task_python; TaskContext.validate_bindings; check_env.main |
| MiniVLA 기존 모델/추론 | learning.minivla.{head,backbone,checkpoint,policy}; 신규 학습/Resume 비활성 유지 |

Runtime Terms는 Script 파서에 두지 않는다. Static Subtask/Scene 문법 검사는 엔진 없는 계획 경계에 남기며 실제 Metadata/Runtime binding 검사를 대체하지 않는다. 새 Task마다 공통 Worker의 ID 조건문을 추가하지 않는다. TaskContext는 Engine·관측 Cache·Dataset을 소유하지 않는다.
