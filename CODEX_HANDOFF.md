## 2026-10-23 Scene/Terms 아키텍처 정리

최신 계약은 docs/task_sdk/SCRIPT_GUIDE.md(사용법), docs/architecture/TASK_SDK_DESIGN.md(상태/Reset), SYSTEM_FLOW.md(호출 그래프)가 소유한다. 아래 항목은 날짜별 과거 기록이다. task_scripts는 AST/identity/Worker entry 실행, SimulationAdapter.bind_followup/term_boolean/after_reset은 Runtime binding/Event 상태를 소유한다. 기존 파서의 bind_followup/accept_followup/term_*는 Adapter 메서드로 직접 교체했다. Native/Term Event 배열·Mimic cfg·Uniform/Edge sampler 연결 중복 통합. 검증/수치/남은 부채: outputs/workflow_validation/architecture_cleanup/REPORT.md. 사용자 원본·기존 dirty·의존성·Commit/Push 보존.

## 2026-10-23 Scene 기반 Source / Reusable Terms

Scene 매핑 Source는 Task/Pick/Target 없이 Builder Pose/명시 Pose·Reset으로 Common 기록. 기존 목록형 Script/FOLLOWUP 유지. simulation.task_context.reset_objects와 기존 sampler를 사용하고, task_scripts의 기존 후속 경계에 Event/factory·Acceptance/Failure·tool0 sequential Subtask 연결. Tracker/Physics/IK/reference Generator는 원래 소유자. Event factory는 환경/Episode별 생성. UI 조회는 AST/header만, 명시 Worker만 Python 실행. Source Resume는 Scene+SCRIPT 외 코드 identity(실행 옵션 수정 허용), 실행 계획은 전체 SHA. 실제 가상 Leader/Qt Worker, Legacy Red/Blue A/B·일부 새 Builder 배치의 strict IK 실패를 구분한다. 범위/명령/근거 docs/task_sdk/SCRIPT_GUIDE.md 및 outputs/workflow_validation/scene_terms_sdk/REPORT.md. Training/Hub/MiniVLA/원본/의존성/Commit/Push 보존.

## 2026-10-23 Annotation/Mimic Task SDK

Source reference를 기존 Annotation/Mimic에 연결. task_scripts.read_followup/followup_context/load_followup/bind_followup/accept_followup과 기존 Worker·UI Python 선택을 재사용한다. Native Adapter/check_success·StableGrasp·Placement·Mimic/RNG/Recording/Source는 그대로이며 순수 사용자 Term은 Native 성공 뒤 결과 채택만 추가 제한한다. Source Python Path는 Metadata에서 자동 실행하지 않는다. Red/Blue/0.02조건/subset 실제 MuJoCo 가상 Leader→Annotation→Mimic, 기존37/60배열·4MP4 A/B 동일, Qt 실제 자식 Worker 확인. 명시 Python은 MuJoCo Pick/Place만; 임의 Boundary/새 Subtask·EEF/Isaac SDK/실물 미지원. 107PASS·UI10PASS, 실패/원인/범위는 outputs/workflow_validation/annotation_mimic_sdk/REPORT.md. API docs/task_sdk/SCRIPT_GUIDE.md. 기존 dirty/사용자 원본/Training/의존성/Commit/Push 보존.

## 2026-10-23 Native LeRobot 통합

기존 Script SDK 보존. learning.datasets가 설치 factory 목록·명시 hf://Dataset/Model·commit snapshot/native metadata/feature 검사를, learning.training이 공식 CLI argv/config/모드/native_args·기존 저장/Resume 연결을 소유한다. Training UI는 Local/Hub·Scratch/Fine-tuning/Resume·명령/Metadata·공식 로그 Loss 표시를 제공한다. 실제 ACT Hub1update→Resume2step·Local Fine-tune/converted 학습·Qt Start/Stop, SmolVLA2batch 언어 입력, 기존 Sim180배열/10영상 일치. 상세 outputs/workflow_validation/native_integration/REPORT.md 및 docs/real2sim/NATIVE_LEROBOT.md. 다른 factory Policy/Hub 업로드·GPU·실물 미검증. 사용자 원본/의존성/Commit/Push 변경 없음.

## 2026-10-23 간결한 Metadata Script API

scene.objects + source/evaluations를 기본 Generator 형식으로 제공. Legacy tasks/bindings도 유지, 혼합 거부. Metadata pair의 유일한 조건만 생략 가능, 사용자 Task ID와 물리 Scene을 분리. subset target visual alias/초기 복원·기존 Source Resume 보호. 실제 검증/제한은 outputs/workflow_validation/concise_scripts/REPORT.md, 전체 예제/계약은 docs/task_sdk/SCRIPT_GUIDE.md. 사용자 원본·Training·Commit/Push 변경 없음.

## 2026-10-23 Script-Driven SDK (부분 구현·실제 검증)

Source/Evaluation literal SCRIPT + Worker-only terms, metadata generator 확장, UI 실행 Script 모드, 기존 MuJoCo multi-task/multi-env 연결. Source 가상 Leader/실제 엔진 및 UI child Worker 확인. 실제 ACT 2task3+2episode/2env, Source·Evaluation A/B numeric/video 정확 일치. 상세 outputs/workflow_validation/script_sdk/REPORT.md, 사용 docs/task_sdk/SCRIPT_GUIDE.md. 임의 Scene/Reset·Isaac Script·사용자 Annotation/Mimic hooks는 미구현. 기존 작업·사용자 자료 보존, Commit/Push 없음.

## Task SDK Phase 4 — 단일 MuJoCo Evaluation 연결

options.task_python / CLI --task-python → 기존 Operations/Worker → evaluate_sim 연결. 명시 Scene·num_envs1·MuJoCo만 지원. Worker가 source bytes SHA256/선언/Scene 확인, Runtime binding/setup 후 사용자 Hook을 평가한다. SO101 예제는 기존 Placement/StableGrasp/Annotation/Mimic 구현 참조를 선언하고 원본 계산을 재사용. Generic success와 물리 diagnostics/annotation 의미 분리. UI/Isaac SDK/Annotation·Mimic 연결은 미구현. 실제 ACT CPU Worker와 Hold+실제 MuJoCo 비교, 29PASS 근거 outputs/workflow_validation/task_sdk_phase4/REPORT.md. 사용자 원본·Training/Hub/MiniVLA·Commit/Push 변경 없음.

## Task SDK Phase 3.5 — 사용자 Task 독립 실행

check_env.py 사용자 Task 모드는 고정 cup probe 없이 setup/observe/success를 실행한다. --scene-task-id/--full-workspace는 명시 Scene 선택이며 TASK_DEFINITION은 검증용이다. TaskContext.validate_bindings가 setup 전 선언 ID/실제 body/camera·선택적 asset/hash를 검사한다. docs/architecture/task_x_threshold.py는 catalog 없는 X좌표 조건 예제. 단일 MuJoCo만 지원, 기존 Backend 판정 위임 예제는 보존. 근거 outputs/workflow_validation/task_sdk_phase35/REPORT.md. 사용자 자료/Training/Physics/Commit/Push 변경 없음.

## 2026-10-22 Task SDK 최소 경로

Phase 1–3만 구현: simulation/task_context.py의 단일 MuJoCo borrowed Context, workspaces.tasks.generate_template, 기존 workflow.py task-template 및 check_env.py --task-python. 실제 Builder v8·두 Builder fixture/실제 MuJoCo trace 확인. SDK4PASS, 기존 Task4PASS·Qt selector1PASS. 인접 Training fixture coordinate 재쓰기 실패는 이번 범위 밖으로 남김. UI/Mimic/Isaac/Evaluation 통합 미구현. 상세 outputs/workflow_validation/task_sdk/REPORT.md, API docs/architecture/TASK_SDK_DESIGN.md. 이전 Native Learning dirty 변경 보존, Commit/Push 없음.

## 2026-10-22 Native LeRobot 호환 작업 완료

외부 Dataset native 학습 입력·HF Import UI/CLI·native Policy 재사용과 Partition 역할 강제 제거를 반영했다. 실제 PushT 다운로드/Reader/ACT CPU1update, 기존 ACT 공식 CLI fine-tuning1update 및 저장 processor Validation 성공. 상세 근거/제한: outputs/workflow_validation/native_learning/REPORT.md. CPU 학습은 ACCELERATE_USE_CPU=true를 자식에게 전달한다. Sim/Real의 기존 좌표 안전 검사는 유지한다. 같은 Train/Val Path는 경고하며 독립 검증으로 표시하지 않는다. 과거 per-episode Partition gate/union은 삭제, 역사 Hash/중복 Episode 검사는 유지한다. 현재 문서는 docs/architecture의 네 문서; 이전 기록은 Historical로 구분했다. Commit/Push 없음, 이전 dirty 변경 보존.

## 진행 상태 영역 높이 제한 — 2026-10-22

DataGen/Sim Evaluation의 task_state QLabel이 텍스트 줄 수만큼 중앙 화면을 늘리던 문제를 수정했다. 기존 QLabel/상태 생성은 유지하고 QScrollArea 내부에 넣어 viewport 높이72–160px로 제한했다. 긴 내용은 내부 세로 스크롤로 전부 확인한다. Preview/카메라·실행 본문 변경 없음. Qt offscreen Presentation17PASS(긴4환경 상태의 전체 텍스트·160px 제한·스크롤 확인). 실제 desktop 화면은 재시작 후 확인 필요. Commit/Push 없음.

## 독립 Replay 삭제 — 2026-10-22

사용자 요청으로 독립 Replay Stage/UI/CLI·workflow/replay.py·scripts/tools/replay_joint_targets.py를 삭제했다. Workspace CLI의 replay와 smoke 뒤 자동 재생·전용 helper도 제거했다. Annotation 공통 물리 재생, Curation 영상 Preview, Legacy 초기 상태/Annotation의 기존 기록 필드는 실제 기능/형식 계약으로 유지한다. 이전 Replay 수정/구현 기록은 과거 상태다. 검증·삭제 범위: outputs/workflow_validation/replay_removal/REPORT.md. 사용자 데이터/과거 결과·Commit/Push 변경 없음.

## Replay 판정 수정 — 2026-10-22

명시 Replay는 모든 형식에서 실행 완료와 Task 성공을 분리한다. `workflow.replay.run`의 evaluation 전용 예외/인수를 제거하고 `replay_task_success`를 별도 기록한다. Grasp/Place 실패는 결과에 남되 실행 실패로 올리지 않는다. 복원/물리 실행 오류는 계속 실패하며 Annotation 성공 기준은 변경하지 않았다. 실제 사용자 `테스트/origin/dataset.hdf5`를 별도 검증 출력으로 재생: 323step, joint오차0·object오차1.727281241445394e-08 그대로, exit0·입력Hash 불변. 관련39PASS/Qt1skip. 근거 `outputs/workflow_validation/replay_task_outcome/source_jfvi1sp6/`.

## 최신 완료 — Training / Offline Validation

공식 ACT/SmolVLA Trainer 설정·native Resume는 `learning/training.py`, LeRobot 입력/누수 검사는 `learning/datasets.py`, 독립 평가와 저장 processor/공식 loss는 `learning/validation.py`가 소유한다. Train 전체·선택 Val Path를 사용하며 자동 Split/과거 Partition 역할 강제는 없다. UI/Worker의 독립 `validate`와 학습 후 별도 평가 process를 연결했다. MiniVLA 전용 train/data·prepared Dataset·학습 UI는 삭제했고 모델/Checkpoint 추론 및 실제 legacy Reader는 보존했다.

실제 ACT/SmolVLA CPU1batch 평가·ACT Worker와 관련 Resume/UI/core 검사를 통과했다. ACT VAE는 posterior/KL을 보존한 train/no_grad objective이며 SmolVLA Flow loss와 직접 비교하지 않는다. 실제 SmolVLA 입력의 공유 계보 경고를 기록했다. 장시간 학습·실제 optimizer Resume·GPU/전체 통합은 미검증이다. 사용자 원본·Commit/Push 변경 없음. 아래 과거 기록은 당시 상태다.

상세: [Training/Validation 보고서](outputs/workflow_validation/functional_refactor/training_validation/REPORT.md).

## 2026-10-21 — Curation 자유 선별·분할 완료

`workflow.curation.split_dataset/export_splits/export_subset`가 Episode 단위 선택·선택적 Train/Val 배정·재선별을 소유한다. 기존 `episodes.source_split/export_source_splits` 강제 Group 연결 성분 분할은 삭제했다. 과거 Partition/Lineage는 변경 없이 검증·보존하며 새 배정은 `curation_split.json`/`curation.role`로 분리한다. UI에 전체 제외·출력 폴더 선택·저장 취소를 연결했다. 지원 편집 형식은 기존 HDF5와 Legacy NPY/MP4이며 LeRobot 직접 편집·평가 묶음 전체 편집은 지원하지 않는다. 기존 Training/LeRobot Merge 정책은 후속이다. [검증/지원 범위](outputs/workflow_validation/functional_refactor/curation_independent/REPORT.md). 아래 이전 완료 기록은 당시 상태다.

## 2026-10-21 — Registry 제거 / 전체 Stage Path 실행

현재 모든 Workflow Stage의 공개 입력은 Path다. `workflow/artifacts.py`와 Registry 등록·조회·promote·ID CLI는 삭제했다. 기존 `artifacts/*.json`은 사용자 과거 자료로 그대로 남지만 실행에서 읽지 않는다. 아래 이전 단계 기록은 당시 상태이며 현재 경계는 이 절을 따른다.

- `workflow/paths.py`: native marker 탐색·저장 이름·출력 경계. 목록에서 payload/모델을 읽지 않는다.
- `worker_request.stage_input/episode_context`: Path에서 Dataset/Checkpoint/Workspace 계약을 읽고 기존 Stage 실행에 전달한다. `dataset.inspect_dataset`와 `policy.inspect_checkpoint`가 형식별 검사를 소유한다.
- `Operations`: Path 요청·출력 예약/Lock·process/Cancel·runs 로그·완료 결과 Path. UI의 선택/상세/READY는 기존 executor에서 처리한다.
- Curation은 원본 Path를 직접 열어 새 결과를 저장한다. 과거 Group/Partition 정책과 MiniVLA 학습/Val 정책은 이번에 바꾸지 않았다. 공식 ACT/SmolVLA Trainer/Resume도 유지한다.
- 필수 provenance/좌표가 없는 Checkpoint는 발견되더라도 평가 불가 오류를 낸다. Registry·다른 Checkpoint로부터 정보를 복원하거나 원본을 고치지 않는다.

상세 결과와 검증 한계: [Registry 제거 보고서](outputs/workflow_validation/functional_refactor/registry_removal/REPORT.md).

## 2026-10-21 — 7b-1 Replay/Annotation Path 실행 완료

Replay/Annotation의 UI·CLI·Operations·Worker 입력은 Dataset Path다. Registry 없이 marker 탐색→선택 header 비동기 확인→Worker native/full 검증→기존 물리 실행/Writer→Path 결과로 완료한다. Operations(root=…)는 기존 process/예약/Cancel/로그를 공유하며 두 Stage에서 get/verify/register/promote·입력 ID·자동 접두사 이름을 제거했다. Run token은 실행 이력으로 유지한다. 다른 Stage는 아직 기존 Artifact 인터페이스다.

실제 MuJoCo Common/Legacy Replay·Annotation, Isaac2환경 Annotation A/B에서 초기/명령/물리·관측/판정·Numeric이 같다. MuJoCo MP4 bytes, 양 엔진 Frame/PTS 일치; Isaac RGB는 기준 반복부터 달라 별도 비교했다. 실제 Cancel은 완료2episode만 보존·STOPPED, 실제 CLI와 Registry 사용을 금지한 child Legacy 실행도 완료했다. 기존7a·I/O·수치·Curation 정책·환경 실행/정리 try AST를 보호했다. Report/호출자/한계는 `outputs/workflow_validation/functional_refactor/path_execution/REPORT.md` 및 CALLERS.md를 따른다.

목록의 실제32JSON/190,761,190byte sync 읽기를 제거했지만 전체 Workflow 처리량 개선을 주장하지 않는다. 선택 header와 full 검사를 구분하며, UI Stage 전환/늦은 결과/종료를 검증했다. asset scope 설명 문구만 다를 때 파일/digest·Mapping은 정확 비교한다. 사용자 자료·설정·의존성·Commit/Push 변경 없음. **7b-1 완료, 전체 Registry 제거는 미완료**다. 다음은 DataGen Path/Hash/Resume 경계(7b-2)다. Source/Conversion/Merge/Training/Evaluation/Hub·Curation 정책·MiniVLA/Validation은 남는다.

## 2026-10-21 — 7a Path 탐색 완료 / ID 실행 전환은 후속

`workflow.paths`의 순수 scandir/stat 탐색을 Resume 모델 목록·Curation 파일 선택기·기존 CLI discover에 연결했다. Resume 후보는 Path identity/상대 라벨이며 선택 checkpoint config는 기존 executor에서 읽는다. Registry.input_directory와 standalone artifacts.browse_directory 본문/옛 호출은 삭제했다. 기존 Reader/Writer·Policy 기존 함수와 6c-2 환경/물리 본문은 유지했다.

실제 Dataset 후보29개·Checkpoint16개(9 native LeRobot/7 MiniVLA)를 발견했다. 탐색은 등록·파일 복사·모델/HDF5/video/Hash 읽기를 하지 않는다. 실행 허가가 아닌 형식 후보이며 full 검사와 기존 native Resume 조건은 worker에 남는다. 일반 Stage ID 선택/실행·Curation.open_latest/loaded/save·Registry 결과 등록은 아직 남아 있다. 중간 native checkpoint7개의 좌표/provenance 누락, 오래된 payload 없는 Artifact1개, Sim 평가 Policy/payload action space 차이는 원본 보완 없이 7b 조건으로 기록했다.

신규17PASS, 기존UI202PASS, Curation/Qt49PASS, 엔진/저장 경계203PASS/1제외(겹침 합산 금지). 교체한 Resume 목록의 실제 자료 첫 조회 read190,761,190→0byte·heartbeat 최대약1.100초→0.0104초. 전체 Workflow refresh 측정은 아니다. 사용자 자료/설정·Commit/Push 변경 없음. 전체 Registry 제거·무Registry 실행 완료가 아니다. 보고서: `outputs/workflow_validation/functional_refactor/path_discovery/REPORT.md`, 세부 책임/7b 대상: 같은 폴더 `REGISTRY_RESPONSIBILITIES.md`.

## 2026-10-21 — 6c-2d 완료 / 6c-2 환경 생성 경계 완료

Isaac 설정/생성 호출35곳과 Gym 경로를 점검했다. Recorded Scene의 create_isaac, Source/Teleop, Sim 평가, Live Runtime, 독립 검사/고정 Scene CLI는 센서·Reset·수명이 달라 별도 진입을 유지한다. 공통 cfg는 기존 make_config이며 새 Factory/Manager/Wrapper를 추가하지 않았다. Source/평가/Runtime/물리 CLI는 이미 준비한 Workspace dict를 재사용한다. Runtime.build_environment(p, *, workspace=None)는 connect의 검증 snapshot을 빌리며 standalone 호출은 필요 시 현재 파일을1회 읽는다. Runtime profile 적용에는 파생 Task copy가 아닌 원본 Workspace를 사용한다. Source는 장치 연결 뒤 실제 mapping Hash 검사를 유지했다.

실제 Isaac A/B: 평가2환경×6step의 설정·3trace/72numeric·24frame/PTS·Preview3파일, Runtime/Source 생성 경계6step과 물리CLI120step의 설정·초기/명령/관절·물체속도 trace가 같다. Source는 가상 Leader이며 Recorder 전 경계만 검증했다. load는 평가생성1→0(상위1회 별도), Runtime connect3→1/standalone build2→1, Source2→1, 물리CLI3→1이다. 평가 baseline 반복과 전후 RGB 차이는 따로 기록했으며 pixel/통계적 동등성·보편적 속도향상은 주장하지 않는다. GUI/Runtime137PASS·인접57PASS·입력6PASS, core 완료 로그는 보고서 참조(중복합산 금지). 설정32회 UI독립 비교도 통과했다.

환경 생성·실제 reset event·Camera/framing에는 Workflow/UI/Registry import가 없다. 기존 workspaces.reset.source_workspace/demo.verify_dataset의 상위 입력 정책 역참조와 Task 생략 시 요청 파일 fallback은 남으며 전체 Workspace/Registry 정리 완료가 아니다. 사용자자료/설정·UI·의존성·Commit/Push 변경 없음. 실제 Source 전체 수집·실물·장시간·새 Annotation/DataGen actual·4/8환경 actual은 이번에 재검증하지 않았다. **6c-2a/b/c/d 경계는 완료, 전체 기능 단위 리팩터링은 미완료**다. 상세 호출자/검증/한계는 `outputs/workflow_validation/functional_refactor/isaac_boundaries/REPORT.md`와 해당 CALLERS.md를 따른다.



## 2026-10-21 — 6c-2c MuJoCo 입력/생성 분리 완료

기준 HEAD dd935a6. _environment/unused output/export 삭제. worker_request.prepare_mujoco가 기존 Dataset/Resume/Calibration/Physics 선택을 소유하고, simulation.environment.create_mujoco는 검증한 원본 Workspace+명시 kwargs로 기존 PolicyEnvironment를 생성한다. Dataset Legacy provenance 본문은 workflow.dataset.recording_provenance로 이동하고 실제 모든 caller 교체. Unified/평가는 이미 읽은 Workspace 재사용. DataGen Parent 제공/직접 생성 소유권 및 초기화/Writer 종료 실패 close 보강. PolicyEnvironment viewer/limits 생성 실패는 Backend close. Model/IK/Physics/Playback/Recording/UI/학습 본문은 보호했다.

실제 MuJoCo/EGL: Source/Discard/Resume37·74numeric/MP4일치, Replay432step(접촉정보/접촉력 digest포함), Annotation2env4episode/148numeric·8MP4일치, 평가2env3회×6step/108numeric·6MP4일치. DataGen/직접생성Resume는 동일 초기claim(env1→env0)을 검증 runner에서만 맞춰 전후Reset·명령·IK·물리·영상 일치. 자유 병렬 첫 실행은 env배정이 달라1성공/1실패 vs2실패; Production Scheduler/RNG를 고쳐 숨기지 않았고 무제어 병렬 bitwise 재현성 주장금지. 동일배정 A/B는1성공/1IK실패, Resume는누적1성공/3IK실패이며 목표2미달 정상domain실패다.

load는 Source/Resume4→3, Unified Annotation/DataGen/평가3→1, 직접DataGenResume2→1. compile1·공유Model1·독립primaryMjData/Renderer/실제close 동일. none준비3회 중앙1.258→1.146s(기준/현재각3회), pagecache/thermal미통제라전체속도향상 주장금지. core126PASS/Qt1제외→Qt1PASS, 계약95PASS/Qt1skip→UI110PASS에서통과, 초기정책34PASS/준비12PASS/생성43PASS. 겹침합산금지. Renderer-before-import/Qt누락/spy/timing비교의초기실패로그 보존. Source새UUID/private영상경로/실측timing만명시제외.

근거 outputs/workflow_validation/functional_refactor/mujoco_environment/REPORT.md·resource_comparison/audit/각payload와execution_comparison. Code/원래untracked로그/사용자자료/설정·HEAD보존, Commit/Push없음. Source Resume helper의 workflow 역참조·Legacy reader위치·Registry/Path/Training/MiniVLA/Isaac Source·평가 통합은남음. 실물/학습정책품질/장시간/4·8환경actual/새LegacyEGL/OSMesa·softwareEGL/developer창실조작은미검증.


## 2026-10-21 — 6c-2b 완료: 명시 카메라·관측 요구

make_config에 keyword-only overview_camera/grasp_observations(false기본)를 추가하고 Presentation.current/INSPECTION_STAGES import·분기를 삭제했다. unified는 연결된 DataGen Overview, evaluation은 연결된 Sim 평가 Overview/Grasp 요구를 결정한다. Source/독립 Script/Runtime/Mimic 검사 caller는 false/false를 명시한다. 정책 카메라는 기존 Workspace task.observation_cameras로 유지하며 별도 활성 옵션을 추가하지 않았다. framing 본문은 simulation.workspace_scene로 이동하고 옛 presentation 정의/재노출은 삭제했다. MuJoCo의 기존 Overview 생성은 그대로다.

실제 Isaac2환경(seed1047) Annotation29trace/148numeric·주석/순서, 짧은2환경 평가3trace/72numeric·관측shape/Preview, MuJoCo 모델hash/compiled20배열/30step/RGB가 전후 같다. Stage8종 설정 같고32회 UI독립 설정검사도 통과했다. RGB PTS/count 일치, pixel/MP4는 기준 반복부터 달라 비열등성은 주장하지 않는다. core122/UI145/Runtime42/평가UI16PASS(겹침 합산금지), 최종호출/import72PASS. evidence: outputs/workflow_validation/functional_refactor/presentation_config/REPORT.md. 무거운 단계실행/물리/저장 본문·UI디자인·사용자자료/설정·의존성·Commit/Push 변경 없음. 실제 Source수집/수동창 조작/장시간DataGen/학습정책 품질/실물은 미검증이다. **6c-2b 완료, 전체6c-2는 MuJoCo 환경생성 분리 등이 남음.** 이전 간접 Presentation 의존이 남았다는 문구는 과거 이력이다.


## 2026-10-21 — 6c-2a 실제 엔진 검증 완료

동일 Isaac2환경/seed1047/Dataset4episode로 기준2회·현재1회 실행 완료. Workspace load2→1, 생성/reset/close1/5/1, 설정·초기/물체속도 포함29trace·148numeric·Annotation 순서 정확 일치. RGB3048frame PTS/count 동일, pixel 평균차 최대 기준0.83587/255·전후0.84249/255로 bitwise/통계적 비열등성 주장은 하지 않는다. GPU peak 전체약4.96GiB, OOM 없음. 예외 close는 별도13Fixture통과. Production/테스트 파일hash 불변, 사용자자료/설정·Commit/Push 변경 없음. 근거 `outputs/workflow_validation/functional_refactor/environment/verification_20261021/REPORT.md` 및 summary.json. **6c-2a 구현+실제 검증 완료; 6c-2 전체 미완료.** 이전 GPU OOM/검증 보류 문구는 역사 기록이다. 다음 구조 변경은 수행하지 않음.


## 6c-2a 진행 상태 — 2026-10-20

기록 Scene의 Isaac 생성만 `simulation/environment.py:create_isaac`로 이동하고 unified._native_isaac를 삭제했다. Worker가 identity 검사에 사용한 Workspace를 재사용한다. 105개 계약 회귀 통과, 설정/reset/close 본문은 입력 치환 외 AST 동일. 실제 Isaac baseline부터 GPU OOM으로 실패하여 native A/B 및 성능은 미검증이다. MuJoCo 생성, Source/평가의 별도 Isaac 경로, make_config의 기존 Presentation 간접 참조는 남아 있다. **6c-2 전체 완료 아님.** 근거: `outputs/workflow_validation/functional_refactor/environment/REPORT.md`.

# 2026-10-19 단위6c-1 — 공통 Simulation Playback / Replay·Annotation 분리 완료

- 사용자 후속6c-1만 구현. 최신6b-3 dirty tree·보고서 확인 후 outputs/workflow_validation/functional_refactor/playback/baseline_files.json/before/에 보존. 사용자 Dataset/MP4/모델/Workspace/설정·의존성·UI·Commit/Push 변경 없음. Registry/Path·Curation/Training·OfflineValidation 섞지 않음.
- simulation/playback.py: restore_episode(adapter,values,attrs), replay_steps(adapter,values,before_step/on_observation/after_step), step_targets(adapter,command,actions). Stage/UI/Mimic/Reader/Writer/env생성 import없음. recordedtarget/DT·오차 수학·state 재사용·이전command copy·command/None barrier 의미 그대로. Step V/R예외는 iterator.throw로 같은실패row, Interrupt/close는 Worker전파.
- workflow/replay.py:run 신규 실제 Replay진행/결과 본문. Annotation import없음. 기존strict Source task gate와 명시evaluation_playback의실행완료/task성공구분 유지. per-runIsaacholdtensor1회/idle명령고정, 기존status/preview빈도와 선택realtime 유지. env/Adapter는 Worker소유, per-episodefinally iterator.close/buffer.abort.
- workflow/annotation.py:replay_episode삭제→annotate_episode(판정/manual/prepared)와 기존run. Manual mark/missing/skip/pause/step·Grasp/Place gate/실패분류·prepared/HDF/MP4 동일. scheduler본문은 annotate_episode/step_targets 실제호출자만 바꾼외AST같음. vector모든slot의poststep관측→None barrier→빈slot즉시refill/입력순서저장 유지.
- unified.replay 본문/export삭제, execute가Replay.run/Annotation.run별도dispatch. replay_cache_identity/cached_replay는production/CLIcaller0이라전용cache검사9개와삭제(승인계획425). DataGen자동Replay/Annotation금지검사유지. scripts/mimic/annotate_demos.py의동명함수는fixedscene nativeEpisodeData CLI계약이므로보존. 구Wrapper/reexport없음.
- Nativebaseline93PASS/Qt1skip→기존동등회귀84PASS/skip1(차이cache9삭제). 최종Native121PASS/Qt1skip·Qt1제외, GUI107PASS와별도진단3PASS에서Qt모두통과. 새Playback21개:command/barrier/stateidentity·V/R/Interrupt·Manual4조건·1/2/4/8실제iterator refills/Cancel·환경소유권/abort. 기존32조합scheduler pause/step/resume/idlehold도통과. overlap합산금지. core_initial경로/cwd, interfacesUSD누락/core_finalQt실패로그보존.
- 실제native검증: 기존annotation_throughput/verified_mujoco1_timed/dataset.hdf5의4episode readonly, 동일Workspace/seed1047. MuJoCoAnnotation1/2env각1524episodecontrolsteps·16trace/148numeric/70attrs/8MP4byte 일치,Replay432step4trace/결과일치. IsaacAnnotation2env4episode/16trace/148numeric/주석/상태호출수 정확일치. protected엔진/Reader/Writer/Source/UI/학습140filesbytes·schema/원본fingerprint보존. 실제4/8env는fixture만이며 nativeIsaacReplay1env/실물/장시간은미검증.
- Isaac8MP4byte는옛코드반복부터다름. baseline/전후각3048frame streamingdecode해PTS/count동일. 영상별최대평균절대RGB차0.832/255 vs0.815/255, 최대채널차86 vs78. Physics/render/seed를바꿔맞추지않았고bitwise영상동등주장금지. 첫byte일치실패로그와별도isaac_rgb_comparison.json보존. Kit종료전native.close hook에trace게시하도록검증runner만보강,원본첫report도보존. 기존SDKvisual/seed경고유지.
- 성능: 합성iterator5000frame/6회중앙5.350→5.528µs/frame(+0.178µs). actualAnnotationprocessing Mu1 12.265→12.391s, Mu2순차10.724→11.293s,Isaac2 71.147→71.726s. 초기겹친Mu2측정도별도보존. 추가state/render/tick없으나speedup/장시간thermal/IO통제성능동등보장금지. cProfile/steps_benchmark/nativecomparison에원본수치있음.
- 최종독립Stageimport2개추가후 공개계약test_playback/architecture45PASS(final_interfaces.log). 좌표sidecarJSON도native3A/B bytes같음. Isaactrace는jointq/qdot·EE/objectpose비교,전체rigidbodyvelocity trace는미비교.
- audit:annotation scheduler(호출자외)/prepared수동signal·복원/오차수학/nativeIsaacfactory AST동일,보호package140파일·baseline/무관untracked보존,212Pythonparse/package. AST133→135modules/445→449edges/SCC49유지. unified→annotation은정상dispatch로남아모듈edge감소주장금지;Replay→Annotation함수호출을삭제한것. evidence REPORT.md/audit/dependency_comparison/각comparison/core_final_verified/gui.log,구조/흐름/수정시작점/TESTING/설계안10.4갱신.
- 남은전체범위: Reader/Adapter/환경factory/SourceResume역의존·Registry/Path·Curation과과거Group정책·MiniVLA신규학습비활성화·공통Training/OfflineValidation. 이번6c-1검증경계에서중단하며다음단위자동진행하지않음. actualdevice/사용자자료변경필요검증은별도요청.

# 2026-10-19 단위6b-3 — Source EpisodeWriter 책임 분리 완료

- 최신 dirty-tree 첫5개/6a/6b-1/2를 baseline_files.json과 before/에 보존한 뒤6b-3만 구현. Commit/Push·사용자 Dataset/모델/Workspace/설정·UI·의존성 변경 없음. Registry/Path·Curation/Training/OfflineValidation 정책 변경 없음. 아래 과거 이력의 미완료 중 Source Writer만 이번에 완료했다.
- workflow.episodes.EpisodeWriter 전체 삭제, workflow.dataset.recorded_mujoco_episode 삭제. recording_args는 data.episodes로 본문 이동, Evaluation/LegacyReader/Common Source/test 실제 callers 직접 연결, 구 export/Wrapper 없음.
- Simulation PolicyEnvironment.recording_context/sample/geometry가 Scene/좌표 사실·기존 관측 기반 수집 sample·save-time FK/static pose 생성. 기존 observe/step/backend 전체 함수 보존. sample은 borrowed arrays, Recording.append 이전에 step/mutation 금지.
- data.recording.SourceEpisode: initial은 begin 때 deepcopy1회, numeric leaf/active object matrices는 필요한 frame copy, pick은 소유 matrix alias. RGB는 기존 VideoEpisode/AsyncVideoWriter bounded queue1회 copy, RAM history없음. finish는 staging만, Source finally가 abort. 기존 EpisodeBuffer는 AST 동일.
- data.episodes.LegacyEpisodeWriter: new root/metadata·NPY/MP4/replay/outcome→episode directory→provenance→manifest 원자 게시. 기존 path 충돌/외부 staging 거부, npy/rename/manifest 실패 자기 episode rollback. copy_episodes는 Source가 검사한 이전 자료를 private resume staging으로 복사 후 게시/정리. 열린 handle없어 close를 만들지 않음. 기존 Common Writer/Reader/validator AST 동일.
- 새 workflow.source(metadata/resume_legacy/resume_common/save_common)는 Source 준비/Resume 허용 검사/Common geometry 연결. 기존 mujoco.source가 수집/Discard/quit/episode/자원 수명을 계속 제어. native initial 별도 attrs와 Common 첫관측 q/zero velocity/pose의 역사적 의미, Legacy/ Common 좌표 포화집계 시점 차이도 유지. Common hidden HDF5 준비 시점만 최초save→setup,0episode close는 제거.
- Native 최종58PASS/Qt1deselected, 저장/Resume 중간76PASS, 경계82PASS/Qt1skip, 별도GUI76PASS와130PASS(Qt skip/deselected 모두 통과). 실제 MuJoCo+fakeLeader 양형식 Discard/save/interrupt·cleanup 실패 close. overlap 합산금지. ui_native_environment_attempt/source_final/after_source/source_core 초기 env/fixture/caller/harness 실패 보존, 최종 *_verified 사용.
- 실제 MuJoCo/EGL before/after 같은 Workspace/seed1047,1048·2episodes×8frames/format.40trace/RGB배열·Legacy34files byte·Common74arrays/24attrs(무작위video path제외)/4MP4byte·PTS 일치. Resume2+1episode seed1049(합성RGB)는50Legacyfiles/111Commonarrays·MP4일치, 원본 fingerprint동일. 기존 reader/validator 양형식 읽기.
- profiling128frame/64×48/7회 sample+append+boundedH264: frame median0.07668→0.07667ms,p95median0.2013→0.2049ms. 속도개선주장금지. 실제EGL 단일paired cold startup/encoding 저장시각 차이도 speedup아님. Source frame추가FK/render/hash 없음;Common save FK n개/static object1회만 spy검증.
- 근거 outputs/workflow_validation/functional_refactor/source_writer/REPORT.md·comparison/resume_comparison/append_comparison/audit/dependency_comparison와 TESTING. audit:6a/6b1/6b2 AST·Source 외Simulation/worker/Partition함수 동일,보호package134파일bytes/baseline보존,209Python parse/package/CLI. AST132→133modules,439→445edges,SCC48→49,새SourceStage가 기존정적순환 포함;전체순환개선주장금지. common I/O에 Stage/engine/env callback import없음. 전체 graph/실제남은edge 설명은 DEPENDENCIES.
- 남은범위: Source Resume가 workflow.dataset/episodes/tasks(기존형식/Catalog/recorded_scene)를 사용. Isaac native normalization/LegacyReader RobotPreview FK·Registry/Path·나머지 Stage/학습 정책/Minivla학습비활성화/OfflineValidation 미완료. 실제Isaac PhysX/장치/장시간/디스크강제종료 미검증. 이번요청은6b-3까지만이므로 이후단위 자동진행하지않음.

# 2026-10-19 단위6b-1/2 — Writer·관측/저장 책임 교체 완료, Source Legacy는 후속

- 사용자 요청 범위는6b Writer/저장만. 첫5개·6a 보고서/최신 dirty tree 확인 후 baseline_files와 이전 본문 보존. Registry/Path·Curation 정책·Training/Validation·기존UI/사용자설정 변경 없음. Commit/Push/의존성·실장치·사용자 Dataset/모델/Workspace 변경 없음. 기존 미커밋/두 untracked 로그 보존.
- 6b-1: workflow.dataset의 DatasetWriter/validate_episode/file_sha256/_write/SCHEMA/SEMANTICS→data.episodes 실제 본문. Source EpisodeWriter/Isaac native recorder/Annotation/DataGen/평가/Resume/테스트 caller 직접 연결, old export/Wrapper 없음. artifacts 수정은SCHEMA import1곳뿐. 모든 배열/시각/PTS/SHA·staging·rollback·독점 게시/취소 정책 동일.
- 6b-2: workflow.adapters의 EpisodeBuffer 본문 삭제→data/recording.py. output/hz/base/camera + begin(initial_state), append(joints,target,state,images,grasp/place/ik), finish(attrs,success)→tuple, abort. env/adapter/성공 callback 없음. 기존 SimulationAdapter.record_frame가 pre-command 관측/이미지/이벤트와 측정시간 소유. 이미 읽은 state/images는 재조회하지 않음. 각종 finally의 abort·중앙 Writer 직렬화 유지. as_pose7/cartesian_actions→utils.transforms 원본 AST 동일.
- Common Writer.commit의 datagen_info 전체 deepcopy를 제거하고 바꿀 pose map만 변환한다. 나머지 배열은 synchronous commit 동안 readonly로 빌리며 동시 mutation 금지. 숫자/원본 dtype/영상 의미 유지, encoding 구현 변경 없음.6a Reader 클래스 AST 동일.
- 단계별6b-1/2 각113PASS/Qt1skip. 최종 storage111PASS/Qt1skip(Reader8개 제외·새녹화6개 추가), GUI Reader/Curation/평가84PASS·WorkflowUI158PASS. Source/Resume6PASS(가상Leader/실제Mu), Native 경계53PASS+PyQt 없는1실패→GUI해당1PASS. 기존 legacy provenance경고 유지. 중복 suite합산 금지. 초기 import편집 SyntaxError는 고쳤고 core_owner.log에 보존, 최종core_owner_verified.log 사용.
- 전후실행: 기존 agent 소유 임시Builder Workspace readonly·실제MuJoCo12step·명시 검은RGB fixture. qpos/modelhash·numeric28개·attrs8개(무작위private경로제외)·MP4bytes 동일, 영상 inode이동 유지. 초기renderer none get_rgb실패→검은RGB fixture명시한 runtime_verified.log가 완료근거. 실제Isaac PhysX/실물/학습정책/장시간/전체성능 아님.
- profiling: 실제commit 준비3문장만 AST추출해20kframe/pose10개/7회. float64 중앙15.19→5.93ms·tracemallocpeak33.30→12.80MB, float32 7.43ms/12.82MB→0.017ms/1.48KB. wholeWriter처리량 향상으로 표현금지. 입력dtype/출력배열 동일. ASTgraph131→132모듈/428→439edges/SCC47→48: 공통I/O→기존root계약의 지연Precision/Simulation정적순환이 연결됨. 직접Stage/UI/engine import는없고 freshimport로확인. root보정계약은scope밖, 순환개선성과로주장금지.
- 근거 outputs/workflow_validation/functional_refactor/dataset_writer/REPORT.md, audit/runtime_comparison/preparation_comparison/unit_dependencies 및 로그. audit: 보호코드56개/6aReader/SourceEpisodeWriter(import외) 보존,207Python parse/package/CLI help. TESTING/구조/흐름/의존/전후/설계안10.2 갱신.
- 남은6b-3: Source workflow.episodes.EpisodeWriter의 env/FK/초기state·manifest/Legacy저장/Resume 결합, workflow.dataset recording_args/recorded_mujoco_episode/isaac_recording_episode와 LegacyReader/Dataset단위정책은 그대로다. 이것을 data로 옮긴척하지 말 것. 사용자 후속에 따라 다음Writer작업도 작은단위검증해야하며 Registry/Curation/Training을 이단계에 섞지 않는다. 전체리팩터링/MiniVLA학습비활성화/Path/OfflineValidation 미완료는 아래이력과같음.

# 2026-10-19 기능 리팩터링6a — Curation 공통 Reader, 전체 미완료

- 사용자 추가6원칙(공개 API/타입/예외/수명, 기존 기술, lazy/background I/O, Atomic/Resume/Cancel/Lock, 공식 Trainer, 실제 품질/측정)을 설계안 상단에 반영했다. 다음 작은 경계6a를 실제 교체했다. Commit/Push/사용자 데이터/설정·장치·의존성 변경 없음.
- workflow.curation의 EpisodeReader 본문 삭제→data/episodes.py. Curation 분석/events/scan은 Reader1개/context, plot도 context. UI의 기존 Qt Job/generation 그대로, release_reader가 선택·창 종료 시 in-flight 작업 완료 뒤 파일/영상/cache를 닫는다. 생성payload0, lazy HDF5 file/group1개, Canonical은 선택된 mmap 배열만 분리, 기존 RGB8frame/16MiB·영상Reader2개 유지. 순차 호출 계약, 닫힌 read/frame은 RuntimeError, 파일/키/시각 오류는 기존 예외 전파. 모든 테스트 caller도 직접 새 owner/context 사용. 호환 Wrapper/export 없음.
- pose_matrix 본문은 utils/transforms.py로 이동, 이전 robot_preview 정의 삭제. 실제 callers4개 모두 교체, AST 동일. Adapter/Annotation/Curation→robot_preview3개 edge 제거. dataset의 legacy FK→RobotPreview는 아직 실제 사용 중. data Reader의 Stage/UI/Registry/Workspace/엔진 import 없음.
- 기존34개 Curation baseline, 변경후34 및 추가 lifecycle 회귀. 계약144PASS/Qt1skip→GUI별도1PASS, UI136PASS. 최종 group 재사용 수정 후 Reader/RGB12PASS·Curation/평가UI36PASS. 예외 close·시각 mismatch·부분 RGB·cache·영상 해제·작업 중 선택전환/이전결과거부·heartbeat. 기존 legacy provenance 경고4개 유지, suite 중복 합산 금지. native와 GUI/USD process 분리.
- 소유 synthetic HDF5/동일80frame/순차 process/각7회: open1520→1·중앙0.383→0.071초, frame p50/p950.83/1.18ms(전4.49/5.79ms), 반환byte1517120→1510720. payload880배열·원본Hash 정확 일치. cProfile도 보존. OS page cache 미통제·disk-cold/전체UI/Stage/실물성능 아님. 실제 사용자 데이터/GPU/새 MuJoCo·Isaac rollout/학습 실행 없음.
- 증거 outputs/workflow_validation/functional_refactor/dataset_reader/REPORT.md·comparison.json·audit.json·unit_dependencies.json·before/after_profile와 로그. 기존 first5 파일hash 기준으로 관련6개 source만 수정됨, 새 data 패키지/transform/test 추가. 모든 baseline파일 보존. 공식 Trainer/물리/IK/수치/Hardware/Writer/Resume/설정 파일의 기존 작업 보존. 패키지탐색/AST/CLI help 확인.
- 전체 미완료: 다음6b는 workflow.dataset.EpisodeDataset의 Common/Canonical 형식 읽기·MiniVLA legacy I/O 호출자 경계 확인. Writer/Playback·MiniVLA 전용 준비/Trainer 삭제 및 신규학습 비활성화·Path/Registry·Curation 과거역할강제 제거·공식 Training 인터페이스/독립 Offline Validation은 남아 있다. 현재 MiniVLA 신규학습과 옛 Group 정책이 아직 살아 있으므로 비활성화했다고 주장하지 말 것. 앞선 first5 완료/물리 trace는 아래 기록을 따른다.

# 2026-10-19 기능 단위 리팩터링 — 첫5개 이동 완료, 전체 미완료

- 사용자가 전체 구현을 승인하되 수정4개를 요구했다. 공식 ACT/SmolVLA Trainer는 재작성하지 않고, Val은 실제 저장 Checkpoint를 사용하는 독립 Offline Validation 우선. 새 지표는 Horizon/Mask/정렬/단위 검증 전 도입 금지. Curation의 과거 Partition/Lineage는 현재 역할을 강제하지 않게 후속 교체한다. 작은 단위별 교체/삭제/회귀. Commit/Push 금지.
- 실제 완료1: mimic의 GenerationStopped/PhysicsControl 본문→simulation/control.py, Annotation/DataGen/Adapter/Scheduler 실제 호출자 교체.2: ActionGuard→workflow/policy.py, 모든 평가/옛 CLI/테스트 호출자 교체.3: MiniVLA architecture/adapter 함수3개→learning/minivla/checkpoint.py.4: RobotPreprocessor→minivla/policy.py. 추론은 train/data/Workflow를 import하지 않는다.5: catalog/validate/selected/object_ids/active_objects/workspace_key/configured와 TASK_ENV→real2sim/workspaces/tasks.py, 호출자24개 파일 교체. 원래 본문 삭제, 새 호환 Wrapper/Re-export 없음.
- Task의 recorded_scene/from_artifact/stamping/평가 규칙은 아직 workflow/tasks.py에 남아 있다. Workspace.reset/demo의 역참조도 남음. Registry/Path 전환, Dataset I/O/Replay 이동, Curation 자유 분할 정책, MiniVLA 신규 학습 비활성화, 독립 Offline Validation은 **아직 구현하지 않았다**. 현재 MiniVLA 전용 Trainer/준비/Group 정책은 작동 중인 기존 경로다. 다음은6단위 Dataset Reader의 실제 형식/호출자 경계 확인부터 진행한다. 전체 완료로 보고하지 말 것.
- 단위 회귀 control78/guard54/checkpoint28/preprocessing60/Task native17·USD6, UI196 통과. 최종 native 통합188통과와 PyQt 경로 없는6개 UI 실패→기존 GUI/Pxr 경로의 별도 process에서 해당6개 모두 통과. 겹치는 검사를 합산하지 않는다. Phase3 최초5실패는 checkpoint dict와 모듈명 충돌을 checkpoint_format으로 고친 뒤28개 재통과. 초기 PxR 미지정 collection 오류와 모든 로그 보존.
- 임시 Builder Workspace에서 실제 MuJoCo 모델 hash·compiled23배열·30step qpos trace 정확 일치. 렌더 없음, 사용자 Workspace/데이터를 실행/수정한 검증 아님. 실제 Isaac/GPU 학습·기존 MiniVLA GPU 추론·실물/새 UI 성능 측정 없음. fake 장치/CPU config/checkpoint fixtures와 실제 장치 성공을 구분한다.
- behavior_audit: 이동 본문 AST 동일(Task 상대 import만 제외), 공식 workflow.policy.train/resume 함수 AST 동일, UI widget/signal AST는 import 외 동일. backbone/head/config·물리/IK/수치 계약·Hardware/Calibration 파일 byte 보존.201개Python parse, 새3개 모듈은 기존 package 내에 있어 설치 경로 유지. CLI help 확인.
- 의존 graph d60기준 package125→128, edges409→423, 최대SCC48→47. edge 수가 아닌 실제 제거된 MiniVLA 추론→train/data, Annotation→Mimic 제어, compiler/Workspace→workflow Task edge가 근거. 기존 SourceGroup 강제 등은 아직 남음. 새 노드 Framework/Manager 없음.
- docs/architecture/FUNCTIONAL_REFACTOR_PLAN.md 수정4개·작은16단위·실제1~5현황/그래프 반영. PROJECT_STRUCTURE/SYSTEM_FLOW/WHERE_TO_EDIT/DEPENDENCIES/REFACTOR_BEFORE_AFTER와 TESTING 갱신. 보고서 outputs/workflow_validation/functional_refactor/REPORT.md, model_comparison/behavior_audit/dependency_comparison 및 단계별 로그. git baseline d60a7c2, 이전 untracked2개 로그·설계 문서 보존. 사용자 설정/데이터/모델/자산·dependency 수정/Commit/Push 없음. 새 GUI는 완전히 재실행해 반영한다.

# 2026-10-19 Workflow UI 지연 원인 계측·목록 읽기 교체

- 실제 Registry31개/136MiB. 기존 단계 전환2.35–4.86초, refresh2.07–2.21초. cProfile29.53초 중
  Registry.all25회/대용량 JSON decode22.63초. 실제 사용자 Registry/UI를 임시 복사한 동일 runner로 측정.
- Registry.get/all의 metadata_only=True가 display용 필요한 scene/task/partition만 compact JSON으로 캐시한다.
  inode/size/mtime_ns/ctime_ns가 달라지면 갱신하고 삭제/추가도 감지. 반환은 독립 dict라 UI 수정이 cache에 남지 않는다.
  기본 get/all/verify와 실제 저장 데이터는 그대로. 중복 register는 가벼운 목록을 찾고 반환 시 full record를 다시 읽는다.
- Workflow UI/Window Scene 목록의 full 읽기를 대체했다. readiness와 execute는 기존 background worker 안에서
  full get 후 기존 서비스로 전달. MiniVLA 후보의 전체 계약 검사를 기존 readiness executor로 이동하고 generation 검사.
  거절한 Val은 None으로 되돌리며 다른 Val로 조용히 대체하지 않는다. 중복 refresh_training_choices 호출 삭제.
  화면용 Workspace 재사용만 하고 실행 전 load/hash 검증은 유지. CameraView/Workflow preview는 같은 frame decode를 반복하지 않는다.
- 최종 같은 runner 단계 전환0.062–0.136초/refresh0.068–0.075초, 최초 초기화2.08→1.00초. cold 목록 읽기는 남는다.
  일반 readiness는 이 시간 측정에서 fixture. 별도 actual Train100/Val30 native preflight는 READY, heartbeat49회/
  max gap0.160초. worker/장치0, 원 Registry/UI fingerprint 보존, 실제 학습/장시간/실물 검증 아님.
- UI206PASS, 공통/Source Partition/Resume/Runtime90PASS. 이후 중복 등록·Val 자동 대체 거부 보강5PASS(중복 합산 금지).
  초기 Workspace load 횟수/mock get signature/기존 merge·enable_motion schema fixture 기대와 이미지 timestamp tick을
  고친 로그는 최종과 구분. 의존성 설치/변경 없음. 실제 실행 중인 사용자 GUI/프로세스는 건드리지 않았으므로 재실행 필요.
- 새 production 파일·Manager·Wrapper 없음. 실제 수정은 workflow/{artifacts,ui,view}.py, real2sim/ui.py의 목록 caller,
  기존 테스트3개와 문서. 이전 미커밋·모델·데이터·Workspace 보존, Commit/Push 없음.
  보고서 outputs/workflow_validation/ui_responsiveness/REPORT.md, before/after JSON·profile, interactions.json, 최종 로그.

# 2026-10-19 Workflow 데이터 선택 단일화·모든 새 결과 이름별 평면 저장

- 사용자 요청대로 Workflow의 입력 폴더 열기/묶음 버튼과 browse/browse_collection/select_collection/
  browse_workspace 본문을 삭제했다. data_select 버튼 → browse_data/select_data 한 경로로 교체.
  Registry.input_directory는 매번 Source=environments, Replay·Annotation=sources, DataGen=annotations,
  Convert=datagen, Train·Dataset upload=datasets, Eval·Policy upload=models에서 시작한다.
- Source는 manifest/READY와 실제 v<version> 폴더를 확인한 후 그 버전만 선택. Annotation Source family를
  고르면 origin/선별/분할 이름/train,val 표시, origin 기본. origin의 실제 파일명이 다른 경우도 폴더명으로 표시.
  폴더 안 호환 입력1개는 자동 선택, 여러 개는 dropdown. 단계별 input_folders/input_selections는 독립·저장/복원.
  MiniVLA는 같은 버튼으로 별도 Val 폴더를 고르되 실제 계약 검사 후 Train/범위를 유지하고 Val만 연결한다.
- output_location에서 앞 입력 이름·모델 종류·stage·merged 그룹 생성 분기와 policy_architecture/
  input_storage_name helper를 삭제. 신규 source=name/origin 유지, 이후는 store/<필수 저장 이름>/로 직접 저장.
  Source Curation의 origin/이름별 선별/Train·Val과 실제 partition·계보/예약/덮어쓰기 거부는 유지.
  모든 UI output editor/schema/help/옛 값 전달 제거. CLI/API 명시 output 및 ACT/Smol Resume 기존 위치 계약은 유지.
  기존 자료 이동/metadata rewrite 없음. 모델 내부 checkpoints/policy, 영상·보고서 형식은 유지.
- Live Real/Sim의 Scene 선택/Builder Import는 실제 사용되므로 Window.browse_scene로 본문을 이동해 보존.
  browse_directory는 이 Live Import와 Curation/Profile에서 사용되어 유지. 새 Wrapper/Manager/launcher 없음.
  Convert와 Real 평가의 입력을 상단 Sim backend 때문에 숨기던 filter를 정리했고 기존 실행 좌표/Task gate는 유지.
- 회귀 최종 UI/저장/Source version/Live183PASS, 계약115PASS(영상 reader1개는 native process 별도1PASS),
  선택 부분30PASS·경로/이름 factory6PASS(겹침 합산 금지). 초기183 이전 origin/source_demo 표시 assertion을
  폴더명으로 고쳤고, actual capture에서 드러난 Convert engine filter도 교정. 처음 native codec검사는 Qt GLib
  symbol 충돌이라 의존성 변경 없이 별도 native process로 통과했다. 초기 로그 모두 보존.
- 실제 Green Annotation17을 읽어 MuJoCo/Isaac 각2환경 목표1개 생성(414/362frame) 후 LeRobot 변환/등록.
  datagen/내가_정한_생성본, datasets/내가_정한_변환본에 저장. 실제 Source origin/14·6 목록, Train100/Val30
  별도 선택의 metadata 계약, 모델 선택 등9화면 캡처. 임시 Registry/검증 출력만 쓰고 원본/사용자 Registry 보존.
  capture의 일반 readiness는 fixture이며 선택 시 Train/Val 실제 검사와 native worker 실행은 별도 확인했다.
- 보고서 outputs/workflow_validation/workflow_data_selection/REPORT.md, final_audit.json/display_audit.json/
  native_*/result.json. 코드198개AST·11stage×2backend 경로 검사. 기존 GUI/장치 미조작·새 학습/Hub/실물 없음.
  Commit/Push 없음, 기존 미커밋 변경 보존. GUI를 완전히 다시 열어 반영한다.

# 2026-10-19 미사용 출력 인수 삭제·공통 필수 이름·Annotation flat 저장

- 사용자 후속 요청3개 완료. MujocoBackend.output_dir/PolicyEnvironment.output은 본문에서 읽지 않고,
  production 호출7곳도 이미 전달하지 않았다. 두 인수와 호환 설명을 삭제하고 테스트30개 호출의 옛
  위치 인수/child process의 불필요한 출력 경로를 제거했다. **구형 인수를 받아 버리는 wrapper/kwargs 없음.**
  두 클래스 자체는 실제 physics·좌표 변환에 사용하므로 유지. 학습 policy.output_dir/Writer 출력은 실제 사용하므로 유지.
- UI의 source_name을 storage_name으로 교체, 모든10개 Workflow 화면 같은 첫 위치에 `저장 이름 (필수)`.
  빈 이름은 경고 후 worker0, stage_options 마지막에 넣어 Train filter에서도 유지. 단계별 storage_names로
  분리하고 실행 중 잠금. 새 이름은 기존 output_location/예약/Registry 계약으로 전달. CLI의 자동 이름 API 유지.
- Annotation 새 기본은 annotations/annotation_<선택 Source 이름>/dataset.hdf5, videos/report도 같은 폴더.
  기존 Source/origin/split/train/val 중첩 생성 분기 삭제. train/val처럼 일반 이름은 Source·분할 이름 포함,
  길이 제한 시 안정 hash suffix. Source 입력 변경은 자동 이름만 갱신하고 직접 입력한 이름은 보존.
  기존 데이터/Registry/계보·분할/중복 이름 거부/ACT·Smol Resume 출력 계약은 유지. 기존 폴더 이동 없음.
- 실제 Operations→worker→저장/Registry: 양 엔진1환경 Green Source432frame·HDF5/MP42개를 평평한 기본
  경로에 저장/읽기 확인. JPG0/Scene파일0. Mu hash·compiled23배열·이전30step trace 정확 일치.
  원본 Source/사용자 Registry/UI 설정 보존. 실제 기존63개 등의 artifact count는 과거 이력이며 현재 inventory는 audit 참고.
- 최종 core147PASS, native IK27PASS, native 구성32PASS(legacy provenance warning1), USD/UI182PASS,
  마지막 이름/Preview/학습 UI 부분84PASS. 겹치는 suite는 합산하지 않는다. 198개Python AST/실제 호출7+
  테스트30개 검사. 10개 화면의 필수 이름 row 동일 y/편집/Options 전달·worker0 capture 확인.
  초기 combined suite의Pinocchio import segfault, USD와 MuJoCo 혼합 실행 guard 실패12개는 보존했고
  의존성 변경 없이 native/GUI를 분리해 통과. capture의 임시 artifacts 폴더 누락도 고친 최종 로그와 구분.
- 보고서 outputs/workflow_validation/storage_names_cleanup/REPORT.md, final_audit.json/native_*/result.json.
  Annotation 진행 화면/성능 보고서는 annotation_job_view/REPORT.md이고 최신 필수 이름 UI로 재캡처했다.
  새 GPU 학습/실물/실제 Hub 업로드/사람의 주석·Developer 창·반복 성능은 미검증. 사용자 GUI 미조작,
  Commit/Push 없음. 기존 미커밋 작업 보존. GUI 재실행 후 새 코드가 반영된다.

# 2026-10-19 Annotation 진행 상태 화면·미리보기 경로 제거

- 사용자 요청: Annotation만 realtime Sim 영상 UI를 없애고 양 엔진 같은 진행률/완료·저장·실패/환경별 frame/
  경과 시간으로 교체. 기존 view.py의 AnnotationProgress와 presentation.annotation_status_text를 사용한다.
  기존 UI/worker 진입·판정/수동기능/결과 저장은 유지. 새 production 파일·Manager·launcher·옵션 flag 없음.
- annotation.run의 preview 큐/preview 콜백/last_images 접근/JPEG publish를 삭제. snapshot 상태 provider를
  기존 Presentation에 연결하고 Writer.commit 후 저장 성공을 증가한다. 먼저 끝난 episode가 순서 저장을
  기다리면 완료/저장 대기를 분리한다. 전체 시간은 환경 준비/일시정지 포함·종료 시 고정. 오류/실패 유형도 표시.
- stages에서 Preview와 SIM 실행 단계를 구분해 Annotation feeds=(). UI는 표시되는 feed만 읽고 Annotation의
  카메라/task signal/영상 상태 문구를 갱신하지 않는다. Isaac Annotation은 Overview sensor/env binding/실카메라
  시작 호출 제거. manual Developer의 native 창 pause 렌더만 유지. CameraView/Presentation.publish/attach 본문 동일.
- 실제 Green4개1,524frame/seed1047/2환경 before-after: Mu9.41→9.07초, Isaac77.15→61.15초(엔진 준비 제외).
  전체 wall은13.42→13.22/95.73→79.51초. 각1회 순차 표본이며 부하 변동/일반 속도 우위를 주장하지 않는다.
  양 엔진37배열×4개/주석/계보/성공 정확 일치. Mu RGB24표본 정확 일치, Isaac raster session 차이는 허용하되
  영상 frame/FPS/PTS/SHA 계약 검사. 두 Isaac 실행770tick·6160substep·dt1/240·decimation8 동일.
- Before JPG Mu2/Isaac3→After0개, status.frames={}·Isaac Overview 없음/SideWrist 유지. 실제 수동1환경도 양 엔진
  각432frame에서 IPC pause→mark→resume과 mark frame 저장·MP4를 확인. 사람의 키보드/Developer 창 검증은 아님.
- 최종 core107PASS, USD/UI70PASS. 합성1/2/4/8·실패·저장 대기·controls·elapsed 고정과 다른 Stage Preview를
  확인했다. 실제 상태를 임시 Registry에 넣은5개 UI capture·원본 Source/사용자 Registry/UI 설정 보존 확인.
  보고서 outputs/workflow_validation/annotation_job_view/REPORT.md, audit.json/BENCHMARK/display_audit.
  실제8환경/여러 seed·반복 성능·하드웨어·새 학습 미검증. 사용자 GUI/장치 미조작, Commit/Push 없음.
  기존 미커밋 작업 보존. 이전 preview 구현 snapshot은 검증 폴더 before_*.txt에만 있고 production 경로 없음.

# 2026-10-19 Annotation 실행 교체·동적 슬롯·속도 비교

- 사용자 원칙: 기존 구현을 교체하고 Wrapper/Manager/호환 엔진을 누적하지 않는다. Annotation 직렬 본문을 삭제하고
  annotation.py의 replay_episode 한 상태 머신을 Replay/Annotation이 공유한다. run은 Source 동적 할당/순서 저장/계측만
  소유한다. 새 Manager/Adapter/launcher 없음. lazy 순환 import 없음. request_realtime 단순 wrapper 삭제.
- 자동1/2/4/8, 기본1. MuJoCo 공유 model/독립 MjData worker, Isaac vector tick. 빈 슬롯은 즉시 다음 Source로 재배정.
  post-step 관측을 모든 슬롯에서 끝낸 후 restore해 다른 슬롯의 접촉 관측 경계를 보존. 대기 목표 고정. 완료 numeric/영상은
  Source 순서로 Writer.commit. caller 소유 primary는 닫지 않고 worker GL은 자신의 thread에서 정리한다.
- 기존 UI/worker 진입 유지, 실행 설정의 num_envs 카드 추가. 수동 mark/skip은 기존 Env0의1환경만 명시 허용.
  Annotation은 realtime 값을 무시하고 sleep pacing 없음, pause/step 유지. Replay realtime/평가 playback 유지.
  PhysicsControl/Adapters/Buffer/Writer 재사용. DataGen/Evaluation/Training 수치·알고리즘·설정은 수정하지 않았다.
- 실제 Green Source4개1,524frame/seed1047 양 엔진 before1/after1·2·4 측정. Mu 처리12.11→12.47/9.04/9.51초,
  Isaac91.94→90.81/62.94/49.96초. 단일 표본·엔진 준비 제외. 전체 wall/처리량/실패 로그는 REPORT/BENCHMARK.
  Mu 각37배열×4episode와 주석 정확 일치·RGB72표본 일치. Isaac1 numeric148개/주석 정확 일치.
  **Isaac2/4는 명령/clock/성공4·grasp 경계가 같지만 측정 q 최대0.01151rad, place 경계−1~+3frame 차이**.
  멀티의 모든 Annotation 결과가 완전 동일하다는 요구까지 충족했다고 주장하지 않는다. 첫 EE 유도 Mimic actions의
  작은 차이와 RTX RGB session 차이도 audit에 기록. 차이를 숨기는 clipping/주석 보정/판정 완화 없음.
- 상세 Isaac4 probe: physics_step_s11.998초(3456substeps=432×8), render19.279초, native control-step minus render
  20.897초(관측 manager 포함). encoding/background 합계는 wall과 겹침. 입력 MP4 RGB decode/reencode0회,
  실제 관측 출력만 카메라별1회 인코딩. Common Source 기준이며 Legacy 내부 RGB Import는 별도 기존 경로다.
- regression_final132PASS, 최종 Annotation/공통68PASS, GUI26/Presentation16PASS(중복 합산 금지).
  8환경/controls는CPU fixture, native는1/2/4. 원본 Source hash 보존·Registry/모델 쓰기 없음. 실물/새 학습/native8 미검증.
  이전 구형 실행은 outputs/workflow_validation/annotation_throughput/before_unified.txt 검증 snapshot에만 있음.
  보고서 같은 폴더 REPORT.md, audit.json. 새 GUI 설정은 GUI 재실행 후 반영. 사용자 GUI/하드웨어는 건드리지 않음.
  Commit/Push 없음. 기존 미커밋 작업 보존. 초기 PyQt/USD 누락·synthetic Buffer events·무작위 이름 fixture 실패는
  최종 로그와 구분했다. 계측용 sim.step/render는 실행 후 원래 메서드로 복원한다.

# 2026-10-19 Workflow MuJoCo Scene 파일 저장 코드 삭제

- 사용자 요청: Source뿐 아니라 모든 단계 결과에 생성되는 scene/XML/별도 metadata 파일을 없앤다.
  mjcf.write_xml/출력 mkdir·파일 쓰기 분기를 삭제. build_scene은 XML 문자열/include bytes와 메모리 metadata 반환.
  MujocoBackend는 MjModel.from_xml_string(xml,assets)로 로드. 임시 XML 디렉터리도 생성하지 않는다.
- 기존 backend/PolicyEnvironment의 output 인수는 외부 호출 호환용으로만 수용하며 파일을 쓰지 않는다.
  Workflow _environment/Datagen/Evaluation과 check_env의 scene 경로 전달을 제거했다. Runtime도 MuJoCo용
  임시 scene 폴더를 만들지 않는다. Isaac Runtime의 세션 USD 자원·immutable Workspace 원본은 그대로다.
- 같은 ET UTF-8 직렬화를 사용해 기존 model_sha256을 보존. 실제 Workspace의 변경 전 hash·compiled23배열·
  seed1047의30step 물리 trace와 정확 일치. joint/actuator/collision/자산·수치·성공 조건 변경 없음.
- core103PASS/1skip, Runtime14PASS. 실제 Source fakeLeader1×3frame, 기존 Source의 Replay/Annotation 각1개,
  실제2환경 DataGen 목표1개, hold fixture의2환경 Sim평가2×3step 저장을 확인. 모든 결과 Scene 파일0개.
  원본 Source/Annotation/ACT hash·Registry byte 보존, 임시 검증 payload 제거. 기존 사용자 scene 폴더는 삭제하지 않음.
- 보고서 outputs/workflow_validation/mujoco_memory_scene/REPORT.md. 새 학습/실물/실제 정책 품질 검증·Commit/Push 없음.
  기존 미커밋 작업 보존. 초기 runner의 Annotation 입력 종류·직접 평가 request.renderer 누락을 고친 최종
  runtime_verified.log/runtime_result.json을 완료 근거로 사용한다. 앞선 Source 보고서의 scene 유지 설명은 과거 상태다.

# 2026-10-19 Source 필수 이름·이름별 Curation 저장·PNG 코드 제거

- GUI Source 이름은 필수. 빈 이름으로 시작하면 QMessageBox 오류·worker0. 준비 검사는 이름을 입력하기
  전에도 다른 설정을 확인하되 시작할 때 필수 검사한다. Source의 output editor/schema key/복원 값/전달 제거.
  입력 이름→기존 output_location→sources/<이름>/origin. CLI/직접 API output 계약·다른 Stage 옵션은 유지.
- Curation 저장 이름은 선별·분할 모두 노출/직접 입력. Source 선별은 sources/<묶음>/<이름>/dataset.hdf5,
  분할은 같은 <이름>/{train,val}. 과거 canonical 선별은 <이름>/manifest.json+episode들. 자동 next_split 삭제.
  collection_at/source_collection이 이름 variant를 인식해 Annotation/묶음 선택을 유지한다. 기존 split_N도 읽는다.
  관리 밖 Source는 첫 저장 때 기존 ensure_source_origin으로 복사, 관리된 파생 Source는 현재 입력을 사용한다.
  origin/빈 이름/경로 탈출/기존 결과 덮어쓰기를 거부한다. Annotation/DataGen 일반 선별 경로는 유지했다.
- MuJoCo execute.publish의 첫 프레임 PNG4종 저장 코드 삭제. 실제 MP4·UI presentation JPG·Snapshot은 유지.
  scene/ XML은 MjModel.from_xml_path 실행 입력이며 metadata는 씬/hash 증거라 유지. 학습 reader는 XML을 안 읽는다.
- GUI/USD154PASS. 실제 Common/Canonical Source20 각각17선별/14·6분할, Common629배열/34영상과 legacy
  episode 파일 보존. 별도 가상 Leader/실제 MuJoCo Source1×3frame, PNG0/MP42/Scene 컴파일/preview 확인.
  실제 장치/사람 시연/Isaac 수집/학습 없음. 원본/Registry 보존·임시 payload 제거. 기존 자료 이동/삭제 없음.
- 보고서 outputs/workflow_validation/source_named_storage/REPORT.md. GUI 재시작 필요. Commit/Push 없음.
  검증 runner의 dict 배열 비교·timestamp key 가정 실패는 고쳐 최종 runtime_final.log/result.json으로 확인했다.

# 2026-10-19 미분할 색상별 LeRobot 병합 수정

- 사용자 화면의 Source Partition evidence/hash 오류는 새 색상별50개 변환본의 partition=None을 무조건
  validate_partition으로 넘긴 버그다. 미분할끼리는 원래 Source/Mimic segment 계보를 검사해 그대로 병합한다.
  Train/Val 배정을 만들지 않으며 분할 상태 혼합·깨진 증거·중복/Source identity 충돌은 거부한다.
- simulation_model의 Task/scene 선택 차이도 로봇 물리 모델 변경과 분리했다. 원래 hash/Scene의 비Task 설정·
  관절/그리퍼/충돌/카메라 계약은 검사한다. 다Task 공통 모델의 scene_configuration=None, hash 재계산;
  episode Scene과 merge_inputs.coordinates에 원래 Task/좌표 계약을 보존한다. MiniVLA 분할 정책은 그대로다.
- 실제 사용자3×50개를 임시 Registry/worker로 병합:150개/60,942frame의 state/action/time/frame 정확 일치,
  두 카메라 RGB18표본 정확 일치·3Task·계보·MiniVLA index 확인. 원본 hash/Registry byte 보존 후 임시 출력 제거.
  실제 UI에서도 같은3개 선택→미분할150개 요약/버튼 활성화 확인. 병합24+Source계약6, GUI8개 통과.
- 초기 native suite의 GUI6개는 PyQt 경로 누락으로 실패해 기존 GUI 경로를 명령에만 넣어 별도6개 통과했다.
  보고서 outputs/workflow_validation/lerobot_merge_unpartitioned/REPORT.md. GUI 재시작 필요. Commit/Push 없음.

# 2026-10-19 Green DataGen 실패 / Source·DataGen Resume 입력 격리

- 사용자 실패2개: runs/20261019_020923...와021014...의 resume_data가 파란 Source의 origin/dataset.videos였다.
  영상 보관 폴더를 dataset directory로 읽어 manifest.json이 없어서 실패. 초록 Annotation17은 정상 입력이었다.
  Source/DataGen이 공유하던 resume_data 위젯 값이 단계 이동 때 넘어간 UI 버그를 수정했다.
- WorkflowPanel.resume_inputs가 Source/Datagen 값을 각각 보관/복원한다. DataGen은 기존 DataGen 결과(선택)
  라벨과 새 생성이면 비우는 안내, clear 버튼을 사용한다. 기존 기본 Annotation 입력과 이어 생성 입력은 별개다.
- 기존 data_resume.py에 inspect_resume_input을 추가해 Source/Annotation/영상 폴더의 잘못된 DataGen Resume를
  차단한다. unified_plan은 엔진/Workspace/Task/Annotation SHA/총 목표를 검사. unified.execute도 출력/엔진
  생성 전 경로·종류를 검사한다. 기존 정식 _resume의 seed/IK/정책/모델 검사·공식 Mimic·물리는 무변경.
- 동일 사용자 Green Annotation17/SciPy Best Effort/4환경으로 임시 목표1개를 실제 생성·검증했다. 성공1 저장,
  네 환경373/384/373/400step. 유효한 그 결과의 이어 생성 계획도 READY. 원본 SHA·기존 Registry byte 보존,
  검증 payload/임시 Registry는 제거했다. 원 사용자 목표50개 실행/하드웨어/Commit/Push 없음.
- core/UI/Resume111PASS, 안내 문구 최종 UI/Resume 부분 suite는 별도 로그. 원래 실패 로그/출력은 보존.
  보고서 outputs/workflow_validation/datagen_resume_isolation/REPORT.md, diagnosis/runtime_result.json.
  현재 GUI에서 급히 새 생성하려면 이어 생성 입력을 비우면 된다. 단계별 격리 코드는 GUI 재시작 후 적용된다.

# 2026-10-19 Source 이어 수집 선택 / 직접 저장 이름

- MuJoCo의 새 common Source는 HDF5인데 browse_resume_data만 폴더 전용이었다. 기존 Curation DatasetPicker를
  재사용해 양 엔진 모두 HDF5/과거 canonical 폴더를 고르게 했다. Curation의 평가 입력은 기본 지원을 유지하고
  Resume picker만 allow_evaluation=false. 취소/잘못된 입력은 기존 선택을 보존하며 클릭 예외는 guard로 표시.
- Source 화면 상단 source_name은 선택 사항. UI→storage_name→기존 output_location/worker→Registry 표시 이름.
  sources/<입력 이름>/origin 아래 기존 dataset.hdf5(Mu)/source.hdf5(Isaac) 형식 유지. 다른 단계에는 전달하지 않는다.
  생략 시 선택 Task가 자동 이름에 반영된다. 기존 Source 자동 이름은 Task+수집 시각으로 읽기 전용 표시하며
  실제 폴더/기존 Registry JSON을 rename/rewrite하지 않았다.
- 총 목표 Episodes는 기존 개수 포함. 선택 시 목표<=기존이면 기존+1. MuJoCo plan에도 기존 resume.validate를
  연결해 총 목표/Task/Scene/partition 오류를 장치 연결 전에 차단. MuJoCo 총 목표·Isaac 추가 개수 전달 유지.
  legacy canonical Source의 partition 잠금 확인도 top-level/provenance 모두 읽는다.
- 실제 Source20의 기존 EpisodeWriter.resume로 임시 복사:740배열·영상40 SHA 일치, 원본/Registry 보존 후 임시 삭제.
  Qt UI·양 엔진 준비 계획·기존 Source fake Leader/Resume 검사와 캡처는 별도 프로세스/임시 Registry.
  새 시연/실제 장치 연결·학습·Commit/Push 없음. 기존 사용자 GUI/worker는 종료하지 않았다.
  보고서 outputs/workflow_validation/source_resume_naming/REPORT.md. GUI를 다시 열면 새 입력/선택창이 반영된다.

# 2026-10-19 변환된 LeRobot 데이터셋 병합

- 기존 LeRobot 변환 화면에 병합 버튼/체크 선택 창. 같은 역할의 검증된 v3 변환본2개 이상을 새 이름으로 저장.
  UI 실행/취소/로그는 기존 Operations.run과 workflow.py worker, Registry parents는 선택한 모든 입력이다.
  새 launcher·의존성·자동 Hub 업로드 없음. 기본 datasets/merged/<이름>, 이력은 기존 runs.
- workflow/lerobot_merge.py는 설치된 LeRobot aggregate_datasets(roots=local)를 사용한다. 좌표/모델/영상/FPS/
  Workspace/엔진·중복·Source 배정/group 누수를 검사하고 원본을 보존한다. 실패 시 private staging만 제거.
- 다른 Source 집합의 top-level partition은 inventory를 합치고 episode_source_partitions/input 증거는 원래대로
  보존한다. episodes.partition_evidence를 validation 준비 검사와 MiniVLA index가 공유하며 기존 변환본은 fallback.
  Train과 Val은 별도 병합. 예전 unknown/미분할 변환본은 임의 계보를 붙이지 않고 거부한다.
- 실제 Train100 중 두 episode의 임시 subset 병합에서738frame state/action/시각 및12RGB 표본이 정확 일치.
  원래 Val30과 Train/Val preflight·MiniVLA CPU reader·원본 dataset hash·Registry/UI byte 보존을 확인했다.
  검증 payload는 임시 폴더에서 제거. 새 GPU학습·정책실행·실물·Hub·Commit/Push 없음.
- 상세 보고서 outputs/workflow_validation/lerobot_merge/REPORT.md, runtime_result.json과 관련 로그.
  실행 중 사용자 GUI/Source worker는 건드리지 않았다. 새 버튼은 GUI를 다시 열면 반영된다.

# 2026-10-18 사용자 지정 학습 계보만 보존 / 나머지 결과 삭제

- 사용자 명시 요청: Source20→14/6→Annotation→DataGen100/30→LeRobot100/30, ACT와 MiniVLA, ACT100회 Sim 평가만 보존.
  추가 범위 질문에 “Workflow와 검증용 테스트 결과 모두 정리”로 답해 outputs/workflow_validation 전체도 삭제했다.
  **아래 과거 이력에 적힌 workflow_validation 보고서/스크립트는 이제 존재하지 않는다.** 새 정리 기록은
  outputs/cleanup/20261018_training_lineage/REPORT.md, plan.json, runtime_validation.json, final_state.json이다.
- 이전 평가/Replay/실패 실행/.incomplete/캐시와 테스트 결과 약9.63GiB를 영구 삭제했다. 삭제 자료 사본 없음.
  Registry71→16(보존 record byte/ID/부모/SHA 동일), runs26→10. 보존한 실행 이력·모델/데이터가 사용하는 호환 링크는 유지.
  ACT100회는 성공61/실패39이며 success/failure HDF5·MP4·report 모두 보존했다. ACT 전체 Checkpoint와 Mini best18k/last20k 보존.
- 보존1714파일과 소스/자산/실사용 Workspace/보정/캡처 등 보호1563파일의 SHA/링크가 전후 동일했다.
  Registry16개 verify, Source20/14/6, 생성·변환100/30, Curation 평가100개 읽기, Mini Train100/Val30 CPU 로딩과 RGB,
  best/last Resume 계약을 삭제 경로 분리 후 다시 확인했다. 실행 중 worker 없음(사용자 GUI만 유지).
- 기능 코드·UI 설정·기존 데이터 내용 변경 없음. 새 학습/정책 실행/모터 연결/Commit/Push 없음.
  GUI를 다시 열면 정리된 Registry 목록이 반영된다. 남아 있는 runs는 보존된 계보의 실행 이력/필수 경로다.

# 2026-10-18 Curation 관절 목표 그래프 hover

- 기존 curation_ui.Plot의 actions 모드에만 mouse tracking과 QToolTip을 추가했다. 곡선 근처에서 축 번호/이름,
  degree와 원래 rad, 기록 시각/1-based frame을 커서 옆에 표시한다. 겹친 곡선은 함께 표시, 빈 곳/leave 시 숨김.
- paint에서 이미 생성한 곡선 좌표로 hit test하고 가까운 실제 표시 표본을 사용한다. sample_indices/전체 timestamps를
  따라 정확한 기록 시간을 표시하며 각도를 보간해 새 값으로 만들지 않는다. 기존 그래프 모양/타임라인/저장/물리 변경 없음.
- 실제 완료 ACT 평가 HDF5의6축 Qt mouse event·tooltip 표시/화면을 확인. sparse timestamp/샘플 index·단일 표본도 확인.
  Curation 회귀34PASS. 원본 HDF5와 Registry/UI byte 보존은 outputs/workflow_validation/curation_joint_hover/result.json.
  보고서 같은 폴더 REPORT.md. 기존 미커밋 작업 보존, Commit/Push 없음. 실행 중 GUI는 완전히 다시 열어야 새 코드가 반영된다.

# 2026-10-18 MuJoCo 손가락 collision 교정 / Real 평가 시작 자세 검사

- 사용자 지시에 따라 보이는 STL·물체 형상·관절/모터값은 유지하고 gripper/jaw collision만 오프라인 convex 조각으로 교체했다.
  `simulation.mjcf.robot_xml`이 정적 `assets/SO101/mujoco/finger_convex_parts.json`을 읽고 원본 STL SHA를 검사한다.
  Source/DataGen/Sim 평가의 기본 공통 compiler에 적용. explicit legacy_urdf는 이전 hull을 유지한다. Isaac collision은 변경하지 않았다.
- CoACD는 검증 폴더에 내려받아 분리 실행했고 conda에 설치하지 않았다. worker 의존성 추가 없음.
  원본 3개 collision mesh당96조각(총288) 상한에 도달했으므로 전체 표면의0.75mm 오차 보장은 아니다.
  기록1496frame의 새 접촉44표본 최대 간격0.000339mm, 새 ACT4 평가1660frame의91표본 최대0.000125mm.
  compiled 관절/구동기/질량11필드·visual46개·cube collision 동일. 실제 물리5초 진행 후 양쪽 접촉/15mm 이상 상승 확인.
- 기존 ACT4 고정seed1047 평가4회는 성공1/집기 실패3. 성공·실패 HDF5/영상과 격리 Registry 등록을 확인했다.
  collision 교정과 정책 성공률은 구분하며 모든 집기/튕김 해결이나 성공률 개선을 주장하지 않는다.
- 사용자 Real 실패 로그는 활성화 전 elbow raw3115가 보정상한3053 밖이어서 차단된 경우다. 범위를 완화하지 않았다.
  evaluation.real_start_pose에서 measured hold의 encoder 왕복을 float64로 검사해 float32 경계 오판을 피하고
  hardware_preflight.json에 측정값/결과를 남긴다. approved profile 오프라인 raw3053 통과/3115 거부 재현.
  실제 Follower를 범위 안에 두어야 실행 가능하다. 장치 연결/모터 활성화/실물 정책 실행은 하지 않았다.
- 회귀88PASS/1skip(별도 fake worker5PASS 포함), USD 보정2PASS. 기존 Registry/UI byte 보존은 final_audit.json 참조.
  보고서 outputs/workflow_validation/gripper_contact_fix/REPORT.md. 기존 미커밋 작업 보존, Commit/Push 없음.

# 2026-10-18 이전 Isaac Annotation 삭제

- 사용자 명시 요청으로 예전 annotated_clean.hdf5(Isaac4×600frame, 904.4MiB)와 contract sidecar,
  직접 호환 symlink2개, Registry84ba9689... 및 빈 묶음 폴더를 삭제했다. 파생 Artifact/열린 file handle 없음.
- 다른 Registry/UI 설정 byte 보존. 최신 Source/Train·Val Annotation/DataGen/모델은 삭제하지 않았다.
  삭제 데이터 사본을 만들지 않았다. 기록 outputs/workflow_validation/legacy_isaac_annotation_removal/result.json.
  코드 수정·Commit/Push 없음. 아래 보존 설명은 삭제 이전 시점의 이력이다.

# 2026-10-18 단계별 직접 입력 저장 구조 / 모델 종류 분리

- 사용자 요청으로 output_location의 최초 Source 계보 재사용을 제거. Source origin/split_N/train·val 및
  Annotation의 직접 Source 묶음은 유지. DataGen/<Annotation 실행>, datasets/<DataGen 실행>,
  models/{act,smolvla,minivla}/<학습 실행>, evaluations/<종류>/<학습 실행>/<평가 실행>을 사용한다.
  source_partition은 데이터의 의미적 분할이며 모델 저장 폴더의 분할이 아니다. 계보는 Registry/config에 유지.
- 실제10경로·666파일을 rename, Registry9개 path/collection만 갱신(ID/부모/등록 SHA와 파일 byte 보존).
  옛 config/Checkpoint 절대 경로는 호환 symlink로 유지. 이전 Source 이름 폴더에 남은 것은 링크이며 실제 모델은 종류별 폴더다.
  UI 입력 폴더는 새 단계/모델 종류에 연결. 기존 artifacts.py만 확장, UI 화면/학습/물리/장치는 변경하지 않음.
- 904.4MiB annotated_clean.hdf5는 이전 Isaac Annotation4×600frame의 내장 Side/Wrist RGB다.
  49b6...는 옛 Source Run ID이며 최신 MuJoCo Source20과 다르다. 삭제/변환하지 않았다.
- 회귀113PASS, 실제 Train100/Val30 CPU 로딩과 MiniVLA best/last Resume 통과. 실제 ACT UI READY 및 종류별 선택 확인.
  기존 Sim report1의 Workspace metadata 불일치와 ACT50k의 scheduler_state.json 누락은 이전 record/경로에서도 재현.
  파일/계약을 임의로 고치거나 완전한 ACT Resume 성공으로 보고하지 않는다. 이동 전부터 없던52개 Registry 경로도 유지.
- 보고서 outputs/workflow_validation/artifact_input_layout/REPORT.md. 검증용 이동 runner는 기존 byte/rollback 코드를 재사용.
  장치/새 학습/정책 실행/Commit/Push 없음. GUI 재시작 필요. 앞선 미커밋 작업 보존.

# 2026-10-18 Real 평가 설정 간소화

- 사용자 요청으로 모터 사용 체크/옵션 schema를 제거. UI Real 평가 시작 요청은 stage_options에서
  enable_motion=true를 내부 지정한다. readiness는 읽기 전용이며 실제 연결/활성화는 명시 시작 worker에서만 한다.
  CLI/직접 worker opt-in과 기존 장치 충돌·좌표/보정·finite/freshness/모터 범위 검사를 유지했다.
- Real 평가 설정은 현재 Window 연결 입력 또는 ui_settings.json의 follower port/ID/보정 폴더와
  Side/Wrist를 기본으로 채운다. 다른 값은 직접 입력 가능하며 단계 이동 중 사용자의 수정값을 유지한다.
- camera_devices는 추가 관측 이름→USB 장치 연결. 일반 Side/Wrist는 기존 장치 입력 사용.
  전용 한글 label/표 헤더/설명으로 수정, 임의 카메라 이름의 기존 매핑 지원 보존.
- UI50+별도 Window1, fake worker4/보정2 PASS. 기존 ACT metadata UI READY, 장치/worker 호출0,
  원 Registry/연결 설정 보존. 상세 outputs/workflow_validation/real_eval_controls/REPORT.md.
  실물 연결·정책 구동·Commit/Push 없음. 기존 미커밋 병렬 Sim 평가/구조 정리 보존.

# 2026-10-18 병렬 Sim 평가 / 실패 집계 / 성공·실패 데이터 검토

- UI Sim 평가에 num_envs1/2/4/8. episodes는 Task당 전체 횟수이며 성공 목표 재시도가 아니다.
  양 엔진 Workspace worker가 evaluation.evaluate_sim을 공유한다. MuJoCo 공유 model/독립 data thread step,
  Isaac vector action. 공식 ACT/SmolVLA batch queue, MiniVLA는 model 공유/환경별 chunk·guard.
  한 묶음 완료 후 함께 reset하고 먼저 완료된 환경/마지막 미사용 슬롯은 고정 목표를 유지한다.
- EvaluationProgress를 기존 영상 위 상태 영역에 연결. 전체 완료·성공/실패·유형·환경별 수·실행 시간을 표시.
  기존 grasp/drop/place 진단과 timeout/기하 조건을 보존한다. collision/물성/모델/학습·실물은 변경하지 않았다.
- evaluations의 기존 묶음/분할/실행 경로 아래 success/failure/dataset.hdf5+MP4 및 evaluation_datasets.json.
  Operations가 sim_eval과 하위 evaluation_dataset을 등록하며 Replay 입력 이름은 성공_/실패_로 구분한다.
  Curation에서 실행 폴더를 골라 두 분포·영상·EE/물체 경로를 비교한다. 과거 에피소드별 평가 폴더도 읽기만 한다.
  평가 검토는 읽기 전용. 실패 Playback은 명시 Replay에서만 실행 완료와 task 성공을 구분하며 Source gate는 엄격 유지.
- 실제 ACT/MuJoCo4 5회 성공3/배치 실패2. 실제 양 엔진4 각5회(6/8-step smoke)·마지막부분묶음·pause/step/resume·등록 확인.
  실패5개 native 초기 state Replay의 joint 오차0/object 최대1.47e-8m. 원래 ACT Registry hash 확인.
  core121/1skip, 후속 Replay67/1skip, USD/UI139, 마지막UI9 PASS(중복 합산 안 함).
- 보고서 outputs/workflow_validation/parallel_sim_eval/REPORT.md. 초기 실패·최종 증거/화면 보존.
  앞선 미커밋 구조 정리는 보존했다. Commit/Push 없음. 기존 열려 있는 GUI는 재시작해야 새 옵션이 보인다.
  새 SmolVLA/MiniVLA GPU rollout·8환경·장시간·실물·정책 품질 우위는 미검증. 새 production launcher/package 없음.

# 2026-10-18 MuJoCo 그리퍼 접촉 진단 — production 물리 수정 없음

- 사용자 Source/DataGen/ACT Sim 평가의 scene.xml/so101.xml은 각각 byte 동일. 평가만 collision이 다른 경우가 아니다.
- 실제 평가 완료42개/18,081 frame의 measured q와 world object pose를 독립 process에서 복원.
  그리퍼 STL의 convex hull 접촉점이 원래 메시 표면에서 최대14.3846mm 떨어짐을 확인했다.
  시각/충돌 mesh 자체는 같고 변환 차이1e-12 미만, margin/gap은0. 실패 Episode6/16에도 같은 현상.
- 원인은 오목한 손가락의 빈 공간을 덮는 공통 convex hull 근사다. 모든 튕김의 인과나 실제 접촉력은 미측정.
  형상/마찰/힘/정책/UI/사용자 worker는 변경하지 않았다. 교정 완료로 해석하지 않는다.
- 상세: outputs/workflow_validation/mujoco_contact_gap/REPORT.md, audit.json, 일반/청록 hull 비교 PNG.

# 2026-10-18 전체 코드 구조 감사 / 동작 보존 리팩터링

- 기준 HEAD `7d02d9cf029c27aac3b2c03767e85961bfb650cb`, commit/push 없음. 사용자 미추적 로그2개 보존.
- 전체 감사/구조/실제 흐름/수정 위치/전후/검증은 `docs/architecture/`의7문서. README와 기존 Architecture/Map/Testing에서 연결.
- 기존 `operations.mujoco_request/MUJOCO_STAGES` 구현은 `workflow/worker_request.py`로, `artifacts.fingerprint`는 `utils/artifact_identity.py`로 이동.
  옛 import 경로는 같은 객체 re-export. worker 상수는 Workspace package/MiniVLA config에서 직접 import. 모든 production hash-only caller는 새 소유자 사용.
- production 수정10파일은 import/추출 외 AST 동일. UI/Runtime/Calibration/physics/Mimic/model/config/scripts/assets는 무변경.
  원래 폴더는 유지하고 실제 공유 규칙2개만 기존 package 안에 소유권을 부여. 새 실행 계층/launcher/dependency 없음.
- 전후 Native407/IK27/Workspace32/Learning50PASS(2skip). USD166PASS·기존FAIL1·skip1, 한 카메라1PASS 별도.
  기존FAIL은 `test_modes_same_source_environment`의 서로 다른 자동 UUID 출력 경로 비교. 고치거나 assertion 약화하지 않음.
  신규 interface18PASS, 사용자 Registry를 tmp로 격리한 Snapshot/자식 UI2PASS. 다른 UI fixture의 validator mock 위치만 갱신.
- 고정 Registry 복사본·1500x1000·offscreen16화면 픽셀/widget tree 동일. readiness debounce는 캡처에서 동결, 실제 readiness/선택/잠금은 기존 Qt tests.
  CLI help3개 동일. 전체 증거/초기 실패 로그는 `outputs/workflow_validation/codebase_architecture/`.
- 첫 자산 baseline은 차이0. 최종 비교 때 기존3256개 동일, UI설정2개 변경/새 ACT Run7파일은 별도 사용자 UI177361→worker178370→trainer178401에서 확인.
  Run `20261018_010654_train_cube_red_to_cup_a_f53ec143`을 중단/수정하지 않았고 이번 자동 검증으로 학습하지 않음.
- 고위험 Runtime 수명/좌표·보정/Workspace task·Source partition/DataGen 동시성 이동은 추가 검증 필요로 보류.
  실제 모터/카메라/새 GPU학습/IsaacKit/Hub 검증 안 함. 전체 cycles 제거/성능 향상 주장 없음.

# SO-101 / Real2Sim 개발 인수인계

## 최신 — 최상위 datasets 경로 제거 후 검증 (2026-10-18)

사용자가 최상위 datasets 삭제를 요청해 남아 있던 Source split 호환 링크를 제거했다. 실제 Source20/14/6과
모델은 outputs/workflow의 관리 경로 그대로이며 숫자/영상/manifest/Registry/UI를 변경하지 않았다.
앞서 이 링크가 반드시 필요하다고 안내했지만 조사한 옛 datasets 참조는 과거 출처 기록이었고 현재 실행 의존성이 없었다.
옛 경로 open/listdir/scandir/mkdir를 차단한 상태로 현재 Registry15개 verify, Source 원본과14/6 열기,
양 엔진 신규 Source/Annotation 계획, Source20 복사본의 Curation14/6·10/10 저장을 확인했다.
Train100/Val30(40057/12242표본) 실제 CPU 로딩과 best/last Resume 계약도 통과했다. 옛 경로 IO 시도0.
신규 Source 기본 출력과 Curation 분할 저장은 이미 sources/<묶음>/origin, split_N/train·val을 사용하므로
새 production 수정은 하지 않았다. 과거 metadata 경로 문자열을 일괄 치환해 내용 SHA/Resume identity를 바꾸지 않는다.
회귀82PASS. 검증 스크립트의 dry plan output 가정 오류 로그는 보존했고 실제 output_location 검사로 정정했다.
보고서 outputs/workflow_validation/legacy_datasets_removal/REPORT.md, result/final_audit.json.
새 물리 기록/실물/학습/정책 추론/의존성 변경/Commit/Push 없음. Staged 되돌리기 보존.

## 최신 — 단계별 입력 폴더·선택값 격리 (2026-10-17)

공용 inputs combo의 이전 단계 ID를 refresh가 그대로 재사용하고, collection_directory가 입력 종류를 따라
기본 저장소를 바꾸던 경로를 고쳤다. input_folders는 기존처럼 단계별, 새 input_selections는 각 단계의
Artifact ID와 Train Val ID를 저장한다. inputs_stage가 현재 combo의 소유 단계를 구분하고 refresh가 그 단계의
저장값만 복원한다. 다른 단계의 전역 Scene으로 목록을 필터링하지 않는다. 명시적인 Source/Real-Sim Scene 선택은 유지.
기본 묶음 시작 위치는 단계 기준 sources/annotations/datagen/datasets/models이며 이전 입력 종류를 따르지 않는다.
빈 범위의 개별 입력 선택창도 해당 단계 폴더에서 시작한다. 버튼은 annotations / split_1처럼 실제 저장소를 표시한다.
Curation 저장은 현재 호환 단계만 갱신한다. Source에서 다른 단계의 Annotation 범위를 자동 변경하는 코드는 제거했다.

Native UI/저장/Curation168PASS/1deselected, 후속 browse64PASS/1deselected, 별도 Window/Task/Operations66PASS.
마지막 빈 입력 tooltip 제거·저장소 라벨/alias 경로 변경은 관련25PASS로 확인했다.
다른 단계로 이동·재시작할 때 공유 유형의 Source 선택이 섞이지 않는지, Curation 교차 갱신 방지와 9종 기본 경로를 검사했다.
실제 Registry 복사본에서 DataGen의 Annotation 범위를 Train으로 바꿔도 다른5단계의 폴더/ID와 Train Val이 유지됐고
재시작 후6단계 선택을 복원했다(worker0). UI 검사/보고서는 outputs/workflow_validation/stage_input_isolation/에 있다.
기존 데이터/Registry/사용자 ui.json은 수정하지 않았다. 엔진/학습/실물/Hub/Commit/Push 없음. GUI 재시작 필요.

## 최신 — 기존 실제 결과의 단계별 저장소 정리 (2026-10-17)

사용자 명시 요청으로 기존 runs/datasets 및 등록된 검증 폴더의 실제 결과를 outputs/workflow의
sources/annotations/datagen/datasets/models/evaluations로 이동했다. 28개 경로·2070파일의 byte SHA를
이동 전후 대조했고 Registry21개의 path/collection만 갱신했다(ID/부모/등록 SHA 유지).
Source20은 sources/20261016_094817_source_cube_red_to_cup_a_656da738/origin/dataset,
기존 14/6 분할은 같은 묶음 split_1/train·val이다. 이후 결과/모델도 같은 묶음·분할을 공유한다.
예전 절대 경로는 symlink로 연결해 HDF5/영상·LeRobot index·checkpoint 내 참조를 유지했다.
모델/manifest/숫자/영상은 재작성하지 않았다. 실패 결과는 단계별 .incomplete, 이력 JSON/로그는 runs다.
사용자 ui.json의 입력 묶음을 새 경로에 연결했다. 열려 있던 GUI는 재시작해야 새 코드/설정을 읽는다.
묶음 선택창은 Registry.collection_directory로 입력 저장소/분할에서 시작한다. 모델 변경 시 전체 입력을
refresh하여 MiniVLA의 Val을 Train 목록에서 제외한다. Source 외 Task 편집 숨김은 유지했다.

이동한 Registry21개 verify, Source catalog5개, Train100/Val30(40057/12242표본) 실제 CPU 로딩,
best18000/last20000 Resume 계약을 확인했다. 새 학습/정책 추론/엔진/실물은 실행하지 않았다.
UI·저장123PASS/1deselected, 별도 Window35PASS, 임시 데이터 이동 실패 rollback 확인. 정리 전부터 경로가 없던 Registry40개는
삭제·재생성하지 않고 plan.json에 보존했다. Workspace2개는 원래 위치 그대로다.
보고서 outputs/workflow_validation/storage_reorganization/REPORT.md, plan/result/runtime_validation.json.
before/에 Registry/UI 백업이 있으며 일회성 reorganize.py는 완료 후 재적용/백업 덮어쓰기를 거부한다.
Staged 되돌리기와 기존 미커밋 코드는 보존했다. Commit/Push 없음.

## 최신 — Source origin·여러 split·단계별 입력 묶음 (2026-10-17)

입력 폴더 열기 옆에 데이터 묶음 선택을 추가했다. 선택 범위를 input_folders로 단계별 저장하며 그 아래의
호환 Registry 입력만 표시한다. 빈 범위는 다른 입력으로 대체하지 않는다. 미등록 데이터는 명시 선택 폴더에서
기존 register 검증으로 발견한다. 정책 묶음은 기존 provenance만 읽고 checkpoint 파일을 수정하지 않는다.
새 Source 기본 출력은 sources/<Source 이름>/origin/이다. Curation 분할 저장은 같은 묶음 split_1, split_2…
아래 train/, val/과 source_split.json을 자동 저장한다. HDF5는 각 role/dataset.hdf5, Canonical은 role/ Dataset.
기존 Source는 명시 분할 저장 시 origin으로 복사한다. HDF5 복사본에는 Source 계보/그룹을 명시하여 경로 기반
Legacy 그룹 변경을 막는다. 원래 데이터와 파일 계약/좌표/숫자/MP4는 보존한다. 이미 분할된 입력의 잠금은 유지.
Annotation/DataGen/Convert는 기존 output_location이 Source/분할/role을 이어받아 단계별 폴더에 저장한다.
학습은 Source/분할 아래 모델별 새 실행, ACT/SmolVLA Resume의 기존 출력 계약은 그대로다.
Registry의 새 collection은 저장 그룹일 뿐, 실제 Source Partition/누수 검사에 사용되는 증거를 대신하지 않는다.
Curation 저장 후 Annotation 묶음이 연결되고 Source 기본 Curation 재열기는 origin으로 돌아가 새 분할이 가능하다.
기존 선택된 데이터의 Curation은 해당 입력을 연다. metadata 설명/Task provenance/이름 UI는 다시 만들지 않았다.
추가 요청: Scene objects/Active Objects·Task preset/Pick object/Target cup/Language instruction 영역은 Source에만
표시한다. 나머지 단계의 편집 영역은 숨기고 입력의 Task/Scene 상속·충돌 검사·평가 기능에 필요한 내부 상태는 유지한다.
실제 Qt 10단계 표시 범위와 split_2 Train/Val 양쪽 Task 상속을 검사했다(task_visibility.json).

Native UI/저장/Curation167PASS, 별도 USD/Window/Task/Operations67PASS. Fixture worker subprocess로 Train/Val
Annotation 두 경로의 기존 request/검증/등록→DataGen 묶음 선택 확인. 실제 기존 Canonical Source20을 검증 폴더에
origin+split1(14/6)+split2(10/10)로 복사·분할하고 실제 Qt dropdown 두 항목을 캡처했다. 원본 Source 및 Registry/UI64
JSON byte 보존, 엔진 worker0. Window/liveness 테스트 Registry를 임시 경로로 격리했다.
새 DataGen/학습/실물/Hub/의존성 변경/commit/push 없음. 초기 경로 기대/그룹 오류 로그 보존.
보고서 outputs/workflow_validation/dataset_collections/REPORT.md, runtime_result.json, annotation_split_2.png.

## 최신 — Task provenance 표시·전용 코드 제거 (2026-10-17)

TaskSelector의 summary QLabel/행/설명 생성과 Workflow의 Builder 설명 추가/summary 갱신을 삭제했다.
Scene/Task 구성 오류는 task_setup_error로만 보관하고 기존 NOT_READY/실행 전 검사에 전달한다.
옵션 편집은 기존처럼 실행 준비 검사와 분리한다. Task 선택/상속/충돌/review 판정, 파일 provenance와
Source Partition/Train-Val 누수 검증은 실제 기능에 필요하므로 유지했다. 이전 UI 제거의 전용 코드도 재검색했다.
UI/저장108PASS, 별도 Window/Task42PASS. Registry/UI설정64개+핵심 실행/계보 모듈5개 hash와 staged 되돌리기
diff 보존, 기존 Registry 복사본 화면에서 summary 미존재와 worker0 확인. 새 엔진/학습/실물/업로드/commit/push 없음.
보고서 outputs/workflow_validation/task_provenance_cleanup/REPORT.md. 초기 옵션 guard 위치/fixture 실패 로그도 보존했다.

## 최신 — Workflow 입력 UI 간소화 (2026-10-17)

큰 UI 리팩터링을 되돌리는 staged diff는 그대로 유지했다. 그 위의 작은 수정으로 Loss 버튼은 Train에만
표시한다. 연결·실행 환경 버튼, Workspace/Python editor, 검색/종류/Task 필터, 경로·계보 설명,
저장 이름/추천/예상 위치, Workspace/새로고침/별도 입력 선택 버튼과 전용 코드는 삭제했다.
사용자 확인에 따라 유일한 `입력 폴더 열기`는 OS 탐색기가 아니라 현재 단계의 입력 선택창이다.
선택된 입력 경로(없으면 해당 입력 유형의 최근 경로/기본 폴더)에서 시작하며 출력 경로를 쓰지 않는다.
Scene/Builder는 기존 import worker, Episode는 HDF5 또는 Canonical manifest 선택, Dataset/Policy는 폴더 선택이다.
기존 Registry 항목을 재사용하고 미등록 Episode는 생성/주석 metadata로 유형을 결정한 뒤 기존 검사를 거친다.
MiniVLA Val 입력 선택 시 Train은 유지한다. 완료/선택 뒤 내부 refresh와 CLI 이름/출력 기능은 보존했다.
Workspace는 plain `workspace_path`와 `workspaceChanged`, Python은 `worker_python`으로 실제 실행/Real-Sim에 전달한다.
선택 Scene/입력 identity를 확인하며 원본 data/model/worker/학습/IK/물리/Resume 코드는 수정하지 않았다.

검증: UI/저장 107PASS, 별도 USD/Window35PASS. HDF5 선택→등록→Scene 연결과 잘못된 DataGen 입력 거부,
10단계 입력 dialog 시작 위치/취소, 실제 worker 요청의 Python/Scene, 기존 backend/장치 연결 경계를 검사했다.
기존 Registry63+UI설정1개의 복사본으로 화면 확인, 원본64 JSON byte 보존, worker 실행0.
보고서 outputs/workflow_validation/ui_input_cleanup/REPORT.md 및 source.png/train.png/ui_probe.json.
새 DataGen/GPU 학습/실물/Hub 업로드는 실행하지 않았다. Commit/Push 없음.

## 최신 — MiniVLA 신규 학습 Batch8·Accumulation1 (2026-10-17)

사용자 요청대로 신규 MiniVLA는 Batch 기본8, Accumulation1이다. 이미 연결된 Batch QSpinBox·TRAIN_KEYS·
options 전달을 재사용했다. `training_config` 및 직접 trainer 신규 실행에서 accumulation1을 적용하며,
기존 generic batch 루프는 그 경우8샘플 collate→Forward/Backward1회→optimizer1회다.
Resume이면 누적값을 바꾸지 않는다. Config.load/from_dict는 옛 batch_size 생략을1로, 둘 다 생략을1/8로
읽는다. policy checkpoint reader와 validate_resume도 같은 해석을 사용한다. UI checkpoint 선택은 저장 Batch를
복원하며, 다른 Batch를 지정한 Resume은 준비 단계에서 거절한다. 기존 Batch8/Accum8 체크포인트도 그대로 유지.
기존 파일을 재저장/이동하지 않는다. default.json은 batch8/accum1만 변경했고 구조/LR/H40/Loss는 동일하다.
ACT/SmolVLA 설정은 유지하고 Accumulation UI·OOM 축소/재시도는 추가하지 않았다.

실제 UI→worker에서 Train100/Val30, Batch8/Accum1로 GPU1step+기존Val30/240표본·checkpoint 저장 성공.
state(8,6), actions(8,40,6), mask(8,40), images/instructions8, samples/effective8,
forward_backward_batches1 및 optimizer state 확인. 단 한 번 실행했으며 OOM/재시도/비교 벤치마크 없음.
CPU 실제 trainer 루프는 새 Batch8의 Forward/Backward1회·optimizer1회 및 옛 Resume1/8의8회·1회를 검사했다.
local Qwen processor Batch8의 언어 padding/attention mask/이미지 grid, 실제 collate/Flow loss와 OOM 전파 검사.
learning/processor28PASS, UI/Resume60PASS. 합성 Resume fixture의 pinned revision 누락 실패는 fixture에서 정정했다.
기존 원본 두 Dataset/옛 checkpoint fingerprint, head/backbone/data/compute_loss·이전 Workflow/IK 코드 hash 보존.
보고서 outputs/workflow_validation/minivla_batch_update/REPORT.md, 실제 result.json/settings.png/finished.png.
실물/정책 성능/장시간·추가 Batch 비교/실제 OOM 유발/commit/push 없음. 사용자 실행 중 Sim Eval은 유지했다.

## 최신 — 사용자 저장 이름·단계별 Artifact 폴더 (2026-10-17)

새 Workflow는 `artifacts.output_location`을 UI 미리보기/Operations 계획에서 공유한다.
Source→sources, Annotation→annotations, DataGen→datagen, Convert→datasets, Train→models,
평가/Replay/Hub receipt→evaluations 아래 사용자 이름으로 저장한다. `runs/<Run ID>`에는 요청/상태/로그/
presentation만 두며 Registry JSON은 artifacts다. 원래 엔진별 파일/폴더 및 trainer 하위 구조는 유지한다.
명시 output도 유지하되 이름/확장자 충돌을 거절한다. 이름·기존 경계 검사+배타적 예약으로 덮어쓰기/동시 저장을
차단한다. DataGen/Source/MiniVLA Resume은 새 이름, ACT/SmolVLA는 원래 Resume 위치를 유지한다.
canonical ACT worker는 확인된 in-place Resume일 때만 기존 출력 디렉터리 사용을 허용한다.

Registry는 별도 display_name을 기록하며 기존 이름 없는 항목은 읽을 때만 경로 기반 이름을 표시한다.
기존 ID/Path/SHA256/Parent를 갱신하거나 결과를 이동하지 않는다. 새 Annotation은 원본 Source와 별도로
저장/등록한다. 명시적인 promote_annotation API와 원자적 rollback 검사는 유지한다.
Curation 저장 이름/미리보기→filtered 또는 splits 제안, Train/Val 각각 이름/부모/partition 보존하여 등록.
기존 선택창에 이름/종류/Task 필터·실제 경로/계보를 추가했고 등록 모델에서 Resume/그래프를 선택한다.
ACT 중간 checkpoint import는 상위 request 또는 Registry의 runs request에서 출력 관계를 확인한다.
수동 JSONL 선택/기존 runs 탐색도 유지한다. CLI run/ready --storage-name, register --display-name 지원.

실제 UI→worker 9실행: MuJoCo/Isaac2환경 DataGen, Val 생성, 새 경로1→2재개, Train/Val 변환,
MiniVLA1→2step재개, 이름 선택 모델의 MuJoCo10step rollout. 별도 실제 Legacy3실행: Source1 Annotation,
이전 DataGen4→5개 재개, 옛 MiniVLA2→3step 재개. 원본 입력/체크포인트 fingerprint 유지.
선별/Train14-Val6는 synthetic Source20의 실제 HDF5/Registry 저장 검사다. 신규 Source teleop/실물/
ACT·SmolVLA GPU 재학습/Isaac 정책 rollout/Hub 업로드는 이번에 실행하지 않았다.
53개 수치 배열·MP4 SHA 보존, 이름 검색/모델 ID/Checkpoint/그래프 캡처 및 보호 코드 hash 확인.
USD와 MuJoCo를 혼합한 초기 suite 실패는 프로세스를 분리했다. 이전 MiniVLA UI 기대값을 현재 설정으로
정정하고 route-only fixture의 실제 데이터 검증을 분리했다. 최종 native/UI/Curation138PASS, USD25PASS,
후속 storage/Resume52PASS·planner34PASS(겹치므로 합산 안 함). 모든 Registry hash도 최종 재확인했다.
보고서 outputs/workflow_validation/artifact_storage/REPORT.md, final_audit.json, runtime_results.json,
legacy_results.json. 기존 장시간 사용자 학습 유지, 의존성/Physics/IK/Mimic/학습 알고리즘 변경·commit/push 없음.

## 최신 — MiniVLA 실제 Batch Size UI 연결 (2026-10-17)

기존 SettingsForm 숫자 입력을 MiniVLA에도 노출했다. MiniVLA 기본1, ACT/SmolVLA 기본8 유지.
`options.batch_size → training_config → MiniVLAConfig.batch_size → trainer`로 전달한다.
trainer는 accumulation×batch개를 샘플링하고 batch개씩 collate/forward/backward한다. 기존 loss/accumulation,
optimizer step/Validation/저장 로직은 유지했다. Accumulation UI·OOM 자동 축소/재시도는 없다.
실행 첫 tensor shape 및 stdout/metrics에 실제 batch를 남긴다. 과거 Resume의 생략된 batch는1로 해석하고
새 checkpoint 선택은 저장된 batch를 UI에 복원한다. 다른 batch로 Resume은 기존 config 검사에서 거절한다.

실제 UI 기본1→8 입력→기존 worker로 Train100/Val30, frozen Qwen, 기존 accumulation8을 사용해
단 한 번의 GPU 1step 검증 성공. state(8,6)/actions(8,40,6), effective64, Val30/240표본 및 checkpoint 저장 확인.
기존 사용자 장시간 학습은 중단하지 않았다. CPU trainer17PASS, UI/Resume/표시48PASS.
OOM은 합성 예외의 즉시 전파/무재시도 검사만 수행했으며 실제 Batch8에서는 발생하지 않았다.
원본 두 LeRobot fingerprint·IK/Adapter/DataGen/모델/Loss·Dataset 코드 hash 보존. 비교 벤치마크/실물/commit/push 없음.
보고서: outputs/workflow_validation/minivla_batch_size/REPORT.md, 실제 증거 result.json 및 batch8_settings.png.

## 최신 — LeRobot형 학습 로그·실시간 그래프 자동 연결 (2026-10-17)

학습 그래프가 service.session(presentation) 안에서 metrics를 찾아 선택창을 띄우던 경로 오류 수정.
`WorkflowPanel.training_metrics_path`가 run state.plan.output/resume_output, 선택 policy, 동일 Train 입력의
최근 run을 찾는다. custom 출력 및 새 UI 재연결 지원. 일반 그래프 버튼은 선택창 없음, 첫 기록 전 대기,
1.5초 incremental LogTail 읽기, nonmodal dialog. 선택 버튼으로만 다른 metrics.jsonl을 연다.
`presentation.training_status_text`를 trainer stdout과 UI LogTail.summary가 함께 사용한다. 기존 JSON 로그도
요약 표시. LeRobot 현재 공식main의 metric 항목을 참고(step/smpl/loss/grdn/lr/updt/data/prep/memory), 실제 Val만
추가한다. 새 trainer는 load/collate·device prepare·update 시간도 JSONL에 보존. 구조/학습 수치/일정/데이터 변경 없음.

실제 사용자124006 RUNNING 학습을 read-only로 붙여688→692step 자동 갱신,20k 목표·custom 출력·재연결·기존
JSON 로그 요약 캡처 확인. 학습 중단/새 GPU 학습 실행 없음. 현재 열려 있는 예전 UI는 아래 실제 metrics를
직접 선택하면 기존 timer로 실시간 확인 가능하다. 새 UI부터 자동/비모달 표시가 반영된다.
outputs/workflow/runs/20261017_124006_train_cube_red_to_cup_a_8652f7da/policy/policy/metrics.jsonl

UI/표시37PASS, 실제 CPU trainer loop14PASS(주기/early-stop/SIGINT·새 stdout/JSON 필드).
이전 IK/관측/모델 코드 hash 보존. 보고서 outputs/workflow_validation/training_live_view/REPORT.md.
공식 참고: github.com/huggingface/lerobot의 main/src/lerobot/{scripts/lerobot_train,utils/logging_utils}.py.
새 라이브러리 설치/모델·원본 데이터 변경/commit/push 없음. 다음 검증 주기 전까지 그래프 Val 점은 미측정이 정상이다.

## 최신 — MiniVLA Val30 측정 그리퍼 범위 오류 수정 (2026-10-17)

사용자122600 학습은 Train100/Val30 계보 검증 후 Val episode5 frame43/44의 measured gripper
−0.04528482/−0.04189385rad가 기존2% 허용(하한−0.03490659rad)을 넘어 실패했다. action은 정상이고
HDF5/Parquet가 일치한다. 실행 URDF의 하한−0.174533rad 안이다. 원본 변환 오류가 아니다.
`episodes.training_coordinates`가 recorded simulation_model.robot_model의 URDF 경로/SHA를 검증하여
측정 범위를 가져온다. `RobotCoordinates`의 선택적 measured_gripper_limits는 관측만 모델 범위로 검사하고
기존 fraction 포화 유지. action/팔/물리/IK는 불변. 새 index와 checkpoint coordinates에 범위를 보존하고
Direct loader·공통 Isaac MiniVLA wrapper·MuJoCo PolicyEnvironment에 전달한다. 옛 index/checkpoint 또는
URDF 근거 없는 Legacy의2% 규칙은 자동 변경하지 않는다. 모델 밖 관측/잘못된 명령은 계속 거절한다.

실제 동일 사용자 Train100/Val30로 기본 frozen Qwen/accum8,2step+Val30/240고정표본 계산 통과.
원본 두HDF5/두LeRobot fingerprint, Train의mean/std, IK/Adapter/DataGen/head/backbone/train.py hash 보존.
실제 Readiness READY, learning33/core46 PASS. Synthetic는 명령 거절/물리 범위 거절/URDF hash 변조/Legacy 및
공통 추론 wrapper의 음수 measured 상태를 확인한다. 보고서 outputs/workflow_validation/minivla_gripper_observation/REPORT.md.
데이터 재변환 불필요. 같은 입력으로 새 학습을 시작하면 된다.20k 전체 학습/실물/정책 품질/commit/push 없음.

## 최신 — MiniVLA Workflow 통합 (2026-10-17)

사용자 첨부 명세의 구현/검증을 완료했다. 일반 UI는 Train Dataset / 호환되는 Validation Dataset,
원본 provenance 자동사용(source_groups 숨김), 고급 Config/Legacy/Patience, MiniVLA 초기값50k/val1k/save2.5k/
Early Stopping OFF다. 체크한 일정 > JSON > 기본 JSON이며 최종 config 저장. ACT/SmolVLA 기본값 보존.
`episodes.validate_training_inputs`는 엔진 없는 Readiness/worker 공통 검증이다. Partition/Lineage/Group/원본 ID,
Robot/Workspace/좌표·6축·gripper·카메라/FPS/Task 검증. Train 전용 재분할 금지. Legacy는 기록된 독립 그룹+
고급 명시 옵션만 허용한다. Val은 원본 직접 변환 또는 Val Source만 별도 DataGen→변환이며 자동 실행하지 않는다.
기존 Dataset 폴더 선택으로 Val 등록 시 Train을 유지하고 Val 선택기에 연결한다. 사용자 Registry에는 테스트 artifact를 자동 추가하지 않았다.

Trainer는 val_every와 save_freq 독립, 번호별 policy/checkpoints/step_XXXXXX.pt, best/last 보존.
메트릭 기존 필드+training_loss/측정 validation_loss/LR/시간/VRAM, 검증 방식·표본 수 별도 JSON.
Loss UI는1.5초 갱신 및 다시 열기. MiniVLA Resume은 원본 metadata/양쪽 데이터 byte identity를 유지하고
새 출력에 저장(기존 파일 있는 출력 거절), 일정·early/patience 변경 허용. 선택 중간 파일보다 미래의 best는 복사하지 않는다.
SIGINT는 완료 step 저장을 위해 지연 처리하고 parent worker가 trainer를 기다린다. 기존 Workflow cancel 자동 등록 제한은 유지한다.
저장 policy는 명시적으로 등록 가능하다. 중지 시 준비가 끝나지 않았거나 강제 종료/OOM이면 마지막 저장 파일 기준이다.
Sim 평가에서 best/last/번호별 파일 선택→두 엔진이 정확한 파일 로드, 상위 policy에서 계약 읽기,
결과 checkpoint_path/step 및 selected_checkpoint.json 보존. Isaac JSON과 상세 폴더 충돌 수정.

실제 Train100/Val6(Source14/6)3step, 2→총5step 재개, 실제 SIGINT4step 저장, 양 엔진 선택2step 파일4step 평가,
양 엔진4환경 DataGen 목표1개 및4worker IK 호출 통과. core97 / learning+head36 / USD-UI51 PASS.
검증용 frozen NF4 Qwen 캐시 revision 사용. 품질/50k/실물/ACT·Smol 전체 GPU 재학습은 미검증.
원본 데이터/체크포인트 fingerprint·시작 시 IK/Adapter/DataGen hash 보존. head/loss/preprocess/Mimic 본문 보존.
보고서와 재현 명령: outputs/workflow_validation/minivla_workflow/{REPORT,VALIDATION}.md, 최종 audit.json.
일부 초기 실패 로그도 남아 있다. 의존성 변경/commit/push 없음. 앞선 IK/실패진단 미커밋 변경을 유지했다.

## 최신 — MiniVLA Validation 안내·체크포인트 간격 (2026-10-17)

사용자094548 학습 오류는 Train100개/Val 미선택. Registry에 변환된Val 없음. UI의 모호한 자동분리 문구를
‘별도 검증 데이터 없음 (Train 전용 입력은 선택 필요)’로 교체하고 Operations readiness에서 사전 안내.
원본 Source Partition 재분할 없음. 기존 Val Source 또는 Val 생성물을 LeRobot 변환 후 별도 선택해야 한다.
MiniVLA TRAIN_KEYS에 save_freq 추가→worker config 전달→기존 학습 루프 번호별 step_XXXXXX.pt 저장.
기본1000, 최종/early-stop 저장, 기존 validation 기반 best/last 유지. 옛 config의 save_freq 생략 호환,
Resume에서 저장 간격만 변경 가능. 기존 best.pt 동반 요구 유지(최초 validation 전 snapshot 단독 Resume 미지원).
실제 사용자 요청 NOT_READY 확인. CPU 실제 loop/optimizer/checkpoint IO5step으로2/4/5번 저장 확인.
learning21 / integration18 / UI39 PASS. 최초 혼합pytest USD/MuJoCo 경계 오류는 분리실행으로 해결.
GPU 재학습/새데이터/실물/commit/push 없음. 보고서 outputs/workflow_validation/minivla_training_controls/REPORT.md.


## 최신 — DataGen 실패 원인·생성 시간 표시 (2026-10-17)

공통 Adapter에 진단 전용 drop_failure 추가. 기존 StableGrasp 완료 이후, 그리퍼 명령을 열기 전에
양쪽 접촉이 모두 없고 잡았을 때 대비 상대 위치가6mm 초과 이탈한 상태가0.2초 지속되면 놓침 기록.
성공 조건·제어·Mimic은 불변이다. commanded grip이 잡은 시점보다1e-4rad 이상 증가하면 release로
구분해 의도적 놓기를 drop으로 분류하지 않는다. 물리적 놓침의 완전한 추론은 아니며 접촉 기반 진단이다.
Attempts에 failure_diagnostics 보존, 양 엔진 공통 UI에 전체/Env별 집기 실패·이동 중 놓침·컵 배치 실패.
Scheduler의 target_completed_seconds는 마지막 성공 commit 후 고정, UI는 생성 경과/목표N개 저장 완료를
분·초 표시. 환경 준비 제외, pause 포함, Resume은 이번 실행만임을 표시. 상세 검증 보고서는
outputs/workflow_validation/datagen_failure_timing/REPORT.md. 기존100개 실행은424.53초·place_failure5이며
과거 데이터는 수정/재분류하지 않았다. GUI 재시작 및 새 worker부터 적용. Commit/push 없음.


## 최신 — IK 실패 정책 분리 (2026-10-17)

기존 solver 선택에 `ik_failure_policy=strict|hold|best_effort`를 추가했다. 기본 Strict는 그대로다.
UI SettingsForm 두 번째 QComboBox→options→Operations→worker→Adapter→solve_selected 연결.
Hold는 수치 실패에 측정 팔 유지, Best Effort는 유효 candidate 적용 후 physics/Mimic 계속 실행.
SciPy solve_pose에는 reject_on_error 반환 옵션만 추가, 수치 본문·DLS·공식 Mimic·성공 기준은 보존했다.
Hold/Best Effort의 팔 명령 점프 기준은 이전 명령 대비 관절별 0.5rad/step이며 초과/비유한/shape/limit
후보는 abort한다. Strict의 jump는 진단만. invalid measured hold도 별도 abort, clipping/fallback 없음.
Resume은 solver+policy/config 검사, 옛 policy 생략=Strict. HDF5 프레임/attrs와 report/attempts에 정책·native
수렴·pose gate·candidate 적용/hold/invalid를 기록한다. 시도당 8개 물리 전후 측정 증거도 보존한다.

실제 같은 Source14/seed1047 Mu1: DLS Hold4/7, Best Effort4/4; SciPy Strict4/11, Best Effort4/5.
Best Effort 실패 후보 적용+physics 진행 DLS57/SciPy47회; 실제 이동 합1.6379/0.4279rad.
Mu2/4 및 Isaac1 두 solver smoke 포함9개 UI→worker 실제 실행 통과. 단일 seed이며 품질/일반 성능 우위 아님.
회귀 core82/native27/USD-UI39 PASS. 원본 hash/알고리즘/assets 보존 및9개 HDF5/영상 검사 통과.
보고서·벤치·LeRobot 변환 증거는 outputs/workflow_validation/ik_failure_policy/ 아래에 있다.
Source 자동 사전 replay 없음, Overall+Env0 Side/Wrist 유지. GUI 재시작 후 정책을 명시 선택한다.
실물/새 학습 없음. 사용자 요청대로 commit/push하지 않는다. 이전 IK 선택 변경도 미커밋 상태다.


## 최신 — 공통 DataGen IK 선택·실제 비교 (2026-10-16)

기준 GitHub/local HEAD8f2cd29. `ik_solver=scipy_ls|pinocchio_dls`를 기존 SettingsForm QComboBox,
UI options→Operations→worker→unified→두 Adapter에 연결했다. 기본 SciPy, unknown 즉시 거부,
숨겨진 fallback 없음. 기존 SO101Kinematics.inverse_kinematics와 SciPy solve_pose 본문은 변경하지 않았다.
DLS는 native 설정 weight1/.01/max200/tol.001/damping.001/step.3, measured seed, 실패 hold다.
SciPy는 previous-command seed/기존3mm·.12rad 탈락이다. MuJoCo world→base/기록 URDF/worker별
Pinocchio를 사용한다. 측정 hold가 MuJoCo 기존 command limit 밖이면 held_measured_joint_limit IKFailure.

report/attempts/데이터 generation_configuration과 episode attrs/ik arrays에 선택·수렴·오차·시간을 기록한다.
Native 수렴, candidate common pose gate, post-step 실제 tool0 추종, 최종 grasp/place를 구분한다.
Resume은 solver/config 불일치면 복사 전 거부, 이전 생략값은 SciPy다. Source 자동 물리 재생 없음.
기존3뷰 유지. 진행 상태가 두 줄 높이로 잘리던 실제 화면 문제를 최소 높이 계산으로 수정했다.
GUI는 재시작해야 새 선택/높이 코드가 반영된다. 기존 실행중 worker를 자동 재시작하지 않았다.

실제 Source14의560목표 동일 seed offline: DLS0.518ms/0.831mm/.612°, SciPy4.083ms/
.000034mm/.000170°, 모두 수렴. MuJoCo vs Pinocchio FK 최대2.12e-13m/1.55e-12rad 미만.
MuJoCo1 최종 재측정 DLS2/2성공·13.48/min, SciPy2/4·6.72/min(준비 제외).
Isaac1 DLS2/5·.80/min, SciPy2/2·2.29/min. Isaac DLS 미수렴hold가 집기 실패로 이어져
offline 속도가 그대로 end-to-end 우위가 되지 않았다. MuJoCo1/2/4, Isaac1/4 실제 두 IK 실행,
네 조합 UI QComboBox→request→worker 전환경 선택/호출 확인. 초기 일부 동시 부하/단일 seed 한계 명시.
core79PASS / native16PASS / USD-UI39PASS. Qt+Pinocchio 합친 pytest import segfault 로그 보존,
기존 프로세스 경계대로 native/GUI를 분리 검사했다. 의존성 변경 없음.

보고서: outputs/workflow_validation/ik_solver_comparison/REPORT.md 및 BENCHMARK.md.
검증 runner/원시 결과/audit.json/display_audit.json/verified_display.png도 같은 폴더다.
최종 MuJoCo1은 *_1_verified, MuJoCo4 DLS는 ui_mujoco_pinocchio_dls_4_verified를 따른다.
초기1환경 Isaac DLS는 frame failure_reason 문자열 추가 전 표본이며 최신 Isaac4에서 해당 필드 확인.
원본 Source hash 불변, 원본 solver/공식 Mimic/assets diff 없음. 실물·새 정책·장시간/여러 seed·Isaac2/8 미검증.
사용자 요청에 따라 이 기능은 commit/push하지 않았다. 기존 사용자 로그2개는 그대로다.


## 최신 — DataGen 자동 Source 사전 재생 제거 (2026-10-16)

사용자 요청: 실제 생성 전에 Source14개를 순차 물리 재생하지 말고 바로 병렬 생성할 것.
unified.execute에서 `or reader.backend!=backend` 분기를 제거했다. 명시 Replay/Annotation만 replay를
호출한다. DataGen은 계약 검사→엔진 준비→기존 run_isaac/run_mujoco로 직접 실행한다.
자동 cached_replay와 cross_replay_cache 결과 필드도 실행 경로에서 제거했다. 캐시 helper는 개발 검사 전용이다.
입력 계약/주석/영상 검사·전체 Isaac reset·Mimic/IK/새 생성의 실제 성공 판정·3뷰·통일 UI는 유지한다.
이전 Source 사전 검증 phase를 설명한 아래 항목들은 당시 실행 기록이며 현재 DataGen 흐름과 다르다.

실제 같은 사용자 Source14개로 WorkflowPanel→Operations→worker→Registry 실행 완료.
양 엔진4환경 모두 자동 Replay0회/검증 phase0회, 준비→생성만 관측했다.
Isaac22.23초 후 생성 시작→성공2/시도9/실패4, MuJoCo9.09초→성공2/시도6/실패1.
이번 실행은 양쪽 모두 Env1·2가 각각1개 성공했다. 전체 환경100step 이상·실행 중 성공 증가·최종 UI=report,
각 HDF52개/주석/MP4 계약·원본 hash 불변 통과. 시작 시간은 환경 준비를 포함한 단일 표본이다.
common/MuJoCo/partition73PASS, UI/Workflow39PASS. 신규14개 Synthetic dispatch 검사는 엔진을 stub으로 대체한다.
처음 fixture의 누락 backend 속성으로2실패한 로그는 보존하고 수정 후 전 suite 통과했다.

보고서/화면/로그: outputs/workflow_validation/datagen_immediate/REPORT.md.
실행 중인 이전 worker에는 코드가 자동 교체되지 않는다. 기존 작업을 중지하고 새 DataGen을 시작하면 적용된다.
현재 변경은 worker 분기이므로 이 변경만 적용하기 위한 GUI 전체 재시작은 필요하지 않다.
실물/새 정책 품질 검증·dependency 설치·commit/push 없음. 새 Production launcher/사용자 설정 없음.

## 최신 — Isaac/MuJoCo DataGen 진행 UI 통일·사용자14 Source 실제 UI 검증 (2026-10-16)

Presentation.start_datagen/source_progress/bind_generation으로 준비→Source 검증→생성을 구분한다.
양 엔진 모두 Scheduler.progress provider를 status 저장·pause·close 때 읽는다. UI는 동일한
datagen_status_text로 전체성공/목표, 시도/실패/성공률, Env별 상태/시도/성공/실패/step, 최근 결과 Env를 표시한다.
Scheduler는 latest_env_id/latest_attempt/latest_category와 환경별 step을 전달한다. 전체 목표 달성과
최근 개별 환경의 category는 분리했다. 각 poll에서 task_state 문구도 한 번만 갱신한다.

성공0개에서 사용자 중지는 report.passed=false를 유지하되 정상 STOPPED로 처리하며 빈 artifact를
Registry에 넣지 않는다. 생성 전 Source 검증 도중 중지에도 report.json을 남긴다. UI가 빈 결과의 path/id를
읽으려던 경로도 막았다. 영상은 기존 Overall+Env0 Side/Wrist, 공식 알고리즘/모델/계약은 그대로다.

이번에는 사용자 annotated Source14개를 private Registry에서 실제 WorkflowPanel.execute→Operations.run
→worker→검증/등록까지 실행했다. Isaac4: Source14/14물리 통과→성공2/시도9(Env1=1/Env2=1),
MuJoCo4: 성공2/시도6(Env1=1/Env3=1). 양 엔진 실행 중 성공 증가/전 환경100step 이상과 최종 status=report
카운터 일치를 확인했다. 각 HDF5 공통 계약과 사용자 입력 hash 불변 통과. core69PASS, UI/Workflow39PASS.
최종 문구 수정 뒤 실제 완료 status로 UI를 다시 렌더링했다. 원래 runtime 기록도 보존한다.
보고서/화면/결과: outputs/workflow_validation/datagen_progress_parity/REPORT.md.
열려 있던 GUI 프로세스에는 화면 코드가 자동 반영되지 않으므로 완전히 닫고 다시 열어야 한다.
실물/새 정책 성능 검증, dependency 설치, commit/push 없음.

## 최신 — Isaac 사전 재생의 전체 초기화·대기 관절 목표 수정 (2026-10-16)

사용자 Isaac4환경 실행에서 Env0만 움직이고 나머지가 퍼진 원인 확인. 해당 실행은 Source14개
교차 재생 중2/14·166step에서 끝났고 Mimic trial0개였다. 사전 재생은 원래 Env0에서 순차 수행한다.
그러나 ManagerBasedEnv 생성 후 전체 reset이 없어서 나머지3개는 Task 초기 자세가 아닌 거의0rad였다.
공통 physics는 전 환경을 step하며, 매 tick 측정 각도를 다시 목표로 삼아 중력 drift도 누적했다.

unified._native_isaac은 생성 직후 전체 reset, replay는 최초 batch hold 목표를 고정하고 선택 행만
교체하도록 수정했다. MuJoCo 경로·공식 생성/IK/계약은 유지한다. 전체 초기화로 첫 randomizer 샘플
소비 순서는 과거 코드와 달라질 수 있다. 사용자 요청에 따라 시도했던4분할 미리보기/관련 테스트/
전용 runner는 제거했다. 원래 Overall + Env0 Side/Wrist 유지; ui.py/test_presentation.py 추가 변경 없음.

실제 PhysX4환경 생성 직후 q와 configured q를 측정하고, 이전 초기화 누락/명령 방식을 재현했다.
Source1개401step에서 대기3개 최종 자세 오차2.26/2.72/2.56rad→모두0.01533rad,
최대0.01547rad 미만/대기command변화0, 활성 Source replay는 전후 모두 성공했다.
이후4환경412batch step, 전 환경 측정 관절 범위2.80–2.83rad/동시4행 명령변경315회/목표2 저장 확인.
common/MuJoCo/partition68PASS, 입력 hash·HDF5검사·영상3뷰 크기 확인. 실물·정책 품질 검증 아님.
사용자14개 전체의 최신 실행을 완료한 것으로 과장하지 않는다. Source1개 재현과 사용자 로그를 구분한다.

보고서/실패 로그/관절 수치/전후 영상: outputs/workflow_validation/isaac_idle_hold/REPORT.md.
최종 실제 결과는 initialized/result.json. 첫 probe는 사전 전체reset으로 초기화누락을 가렸으며,
초기 검증스크립트 모듈 참조 실패도 보존했다. 신규 dependency/장치/commit/push 없음.

## 최신 — 공통 Production 경로 정리 및 DataGen 병목 개선 (2026-10-16)

중단 작업 재개 완료. 기준 HEAD8da97ca, commit/push 없음. 전체 보고서는
`outputs/workflow_validation/architecture_cleanup/REPORT.md`, 파일 삭제/이동·A–G 판단은
`STRUCTURE_AUDIT.md`, 원시 성능과 부하 제한은 `BENCHMARK.md`를 따른다.

Operations.plan/execute_request가 Workspace replay/annotate/datagen을 형식·엔진과 무관하게
unified.execute로 보낸다. 유효 common 입력은 그대로 사용하며 과거 입력만 별도 Import한다.
MuJoCo legacy replay/annotation/datagen 및 episodes 중복 helper, Workspace CLI 조기 return 뒤
native 중복 경로, Canonical 변환용 임시 mapping을 제거했다. 공개 고정 씬 --task CLI와
개발용 물성/수치 replay 도구는 유지한다. 과거 reader/checkpoint/장치·보정 경계도 유지한다.

Pool은 worker당1회, 시도별 Generator/RNG/trace다. Scheduler는 상태lock 예약→별도 Writerlock
commit→성공 확정, 저장 실패는 rollback한다. private finalized MP4 staging은 검증 후 이동,
기존 입력/Resume은 복사한다. Isaac native Source는 EpisodeData에서 공통 Writer에 직접 저장해
중간 Legacy HDF5/final migration이 없다. 공식 Mimic 생성 본문/라이선스·6D모델 계약은 유지한다.

Positive 교차 물리 재생 캐시는 Source/IDs/Workspace/Scene/모델·보정/물리·엔진버전/전체코드/판정/Hz
identity를 검사한다. 실제401step/26.52초→hit0step/0.024초 확인, 실패 Source는 계속 차단·비캐시다.
코드를 수정하는 동안 cache miss는 정상이다. 기존 실패 로그와 GPU OOM 표본은 보존했다.

실제 양 엔진1/2/4환경 목표2 저장, SIGINT1개 보존→새출력2개 재개, 원래 numeric35배열/영상 보존,
설치 공식14-Source 목표/선택/경계 parity0, UI/Registry worker 계획 확인. 기존Train14와 새Mu2/Isaac2/
nativeSource1의 LeRobot 변환·state/target/정규화/계보·partition 비교 통과. 원본20/14/6/주석14와
기존checkpoint4개 hash를 보호한다. core364PASS/6skip, USD68PASS, learning49PASS/1skip,
common40PASS(중복 suite, 합산 금지). 최종 산출물은 final_audit.json/final_dataset_checks.json에 기록한다.

Python194개 분석, Cartpole/PPO10파일(8py/2yaml) 삭제, 과거문서3개 archive 이동.
새 Production 모듈/launcher 없음. 현재 구조는 ARCHITECTURE, 계약은 DATA_FLOW, 수정 위치는 PROJECT_MAP이다.
SCENE_REFINEMENT_HANDOFF와 과거환경freeze는 원래 위치에 유지한다. 기존 dirty HANDOFF/로컬 로그 보존.

성능은 각설정1회이며 초기 CPU검사·일부 사용자GPU부하가 섞였다. afterMu4는 깨끗한 GPU비교 표본이 아니다.
wall감소 전체를 코드효과로 주장하거나 4환경의 일괄가속을 주장하지 않는다. NativeSource 검사는
MuJoCo에서 파생된 scripted PhysX자료다. 독립 teleop/사람키보드주석/8환경/새정책품질/실물 미검증.
의존성 설치/버전변경/실물 연결/자동commit 없음.

## 최신 — Isaac / MuJoCo 공통 Episode·공식 Mimic·병렬 생성 (2026-10-16)

`workflow.dataset`는 `so101.common-episode/2` HDF5+Side/Wrist640×480 H264, 원자적 저장·검증·Legacy Import를 담당한다.
두 엔진의 새 Source/Annotation/DataGen은 공통 형식이고 기존 NPY/HDF5는 읽기만 한다. UI→operations.unified_plan
→기존 workflow.py worker→unified.execute→계약/교차 physical replay→공통 Mimic→backend adapter→중앙 Writer다.
새 production launcher는 없다. 기존 Isaac CLI도 Workspace 자동 Annotation/DataGen을 같은 core로 보낸다.

`mimic_reference`는 설치된 공식 Isaac Lab Mimic1.0.15의 5생성 모듈/수학 함수 이식본(라이선스/UPSTREAM.json 보존)이다.
생성 함수 본문을 변경하지 않았다. SO101 단일arm/두subtask/SkillGenfalse를 지원한다. nn_k3/interp5/fixed0/noise0,
grasp offset(0,10)/마지막(0,0), 원본 재선택false/이전command에서 보간. Pool은 공식 HDF5 key 순서
(demo_0, demo_1, demo_10…)를 유지한다. Curation 숫자 정렬과 혼동하지 말 것. trace→원본 이름도 이 순서로 매핑한다.
실제14개 Pool에서 설치된 공식 클래스와 선택/경계/pose/gripper/noise 차이0을 확인했다.

MuJoCo는 한 MjModel + 환경별 MjData/FK/IK scratch, 실제 ThreadPool이다. GL은 소유 thread에서
생성·사용·종료하고 render만 lock으로 직렬화한다. PyOpenGL GLX glGetString이 EGL worker에서 segfault하던
문제는 EGL proc lookup으로 수정했다. 실제4환경 모두 RTX4060/softwarefalse 확인. Isaac은 vector scene와
환경별 coroutine/action/reset mailbox, 독립 RNG/reset/episode를 사용한다. Kit loop 안에서 physics를 돌리지 않는다.
IK는 기존3mm/0.12rad 유지, 5축 제약/seed/limits/위치·각도 잔차를 기록하며 task 실패와 분리한다.

Scheduler가 전체 목표 성공/max_attempts를 관리하며 Writer callback을 lock 안에서 호출한다. 목표 초과 등록 없음.
시도 이력은 HDF5 root generation/attempts_json에도 넣어 bundle 이동 후 Resume한다. 원본 Pool byte/좌표/seed/엔진을
검사한다. Isaac default SIGINT가 저장 전에 프로세스를 종료하므로 AppLauncher 뒤 기존 deferred_interrupt를 설치하고
physics/pause 경계에서 종료한다. 양 엔진 실제 중지1→재개2 및 원래 numeric 보존 확인. incomplete는 등록하지 않는다.

Source ID·Train14/Val6·594개 원본 numeric 배열·MP4 byte/hash 보존. 실제 Source20 Isaac Annotation은19통과/1배치실패.
DataGen은 cross replay 실패 Source를 조용히 통과시키지 않는다. Annotation은 명시적으로 통과분만 별도 저장한다.
생성은 mimic_segments의 subtask Source ID·경계를 저장한다. A+B / A+C 같은 공유 Source 조합은 split 전에 연결 성분으로
묶어 누수를 막는다. UNKNOWN은 독립 split 불가. 초기 상태는 엔진 고유 blob을 다른 엔진에 적용하지 않고 portable pose/q/velocity로 복원한다.

Curation 기존 bounded reader/export, 단일 LeRobot Converter, ACT/MiniVLA 정규화·H40/6D·모델 구조 유지.
새 평가 summary는 evaluation-episode/1 + 공통 HDF5/두 MP4, 완료된 실패 episode도 보존. Legacy 단일카메라/Real Eval 유지.
실제 ACT2step, MiniVLA2step(원본Train14/Val6 및 양 엔진 생성train1/Val6), 반대 엔진8step평가를 실행했다.
짧은 연결 검사이며 정책 task 성공/실물 검증이 아니다. 원본20/14/6/주석14와 immutable Workspace는 수정하지 않았다.

문서: docs/real2sim/DATA_FLOW.md 맨 위. 검증/벤치마크/실패 로그: outputs/workflow_validation/unified_mimic/REPORT.md.
최종 회귀 수와 실행 한계는 보고서를 확인한다. 기존 dirty Curation 작업·로컬 로그 보존, dependency/하드웨어/commit 변경 없음.

최종 확인: `unified_mimic/final_audit.json` PASS, 공통 데이터24개 전체 계약 검사 및 원본4개 hash 불변.
최신 기존 Isaac CLI Annotation401step + DataGen2환경/목표1 성공(시도6, IK탈락4, 목표도달중단1)을 확인했다.
core359PASS/6skip, 별도USD69PASS, runtime47PASS/1skip, 최신 계약·MuJoCo·분할46PASS(서로 중복, 합산 금지).
보고서와 순차1/2/4 성능표 작성 완료. 4환경이 2환경보다 빠르다고 주장하지 않는다. 독립 Isaac teleop가 아닌
MuJoCo에서 파생된 scripted PhysX Source를 역방향 검사에 사용했다는 범위도 보고서에 명시했다.


## 최신 — Curation 저장 버튼 통합 (2026-10-16)

고급 설정의 export_split 버튼을 제거하고 하단 export_button 하나를 사용한다.
미리보기 split_plan이 있으면 Train / Val 저장 → save_source_split, 없으면 데이터 저장 → 기존 subset 저장이다.
save가 기존 분할 저장 함수로 분기하며 export/Registry 알고리즘은 유지한다. 기록된 분할은 재분할하지 않는다.
미리보기 취소·설정 변경 시 버튼도 데이터 저장으로 돌아온다. 설명 팝업·사용 문서도 같은 흐름으로 수정했다.
합성 자료의 실제 버튼 클릭→파일/Registry 저장 3경로 및 회귀46 PASS,
실제 Source20/저장 Train14의 읽기 전용 UI·원본 fingerprint 확인 완료.
검증: outputs/workflow_validation/curation_single_save/REPORT.md.

## 최신 — Curation 설정 설명을 팝업으로 이동 (2026-10-16)

데이터셋 구성의 긴 설명 QLabel을 제거하고 설명 버튼→show_curation_help의 정보 팝업으로 이동.
입력 이름은 사용할 데이터 수, 선별 버튼은 적용이다. auto_select 동작은 그대로다.
기본 split_status는 숨기고, 미리보기/기록 분할이 있을 때 실제 Train/Val 개수만 간결히 표시한다.
검증/화면: outputs/workflow_validation/curation_settings_compact/REPORT.md.

## 최신 — Curation 확인 사유 / Train·Val 미리보기 설명 (2026-10-16)

flag_outliers는 기존 비교 규칙 그대로 flag_details(value/threshold/median/group_size)를 분석 row에 추가한다.
Curation 목록·영상 위·상세 분석에 이유를 표시한다. 원본 annotation/outcome이나 자동 제외는 바꾸지 않는다.
분포 split fill은 Train초록/Val파랑, 확인은 주황 테두리, 선택은 흰 테두리, 제외는 회색이다.
자동 선별(남길 개수)과 분할(학습비율/Seed)을 UI 1/2단계로 설명하고, 제외 취소·모두 포함으로 명칭 변경.
미리보기를 만들면 분석 그림을 펼치고 저장 전 실제 개수를 표시한다. 같은 원본 그룹 수동 이동 유지.
비율/Seed/포함 변경은 임시 split_plan을 지우고 generation으로 예전 비동기 결과를 폐기한다.
기록 source_partition은 유지하며, 포함을 바꾼 뒤 오래된 plan으로 저장하던 누락/불일치 경로를 막는다.
검증·실제 데이터 읽기 전용 미리보기: outputs/workflow_validation/curation_split_clarity/REPORT.md.

## 최신 — Source 가장자리 배치 / 반듯한 분포와 로봇 위치 (2026-10-16)

source_workspace의 새 수집 runtime reset만 margin_m=0으로 변경했다. uniform/yaw0/xy_jitter0,
기존 크기 기반 limit·separation sampler 유지. Scene2 지지면150/물체24 mm → 중심 범위126 mm.
configured는 original·과거 uniform(원래 margin)·새 uniform(margin0)만 허용한다. 원본 Task hash 유지.
과거 Source Resume/Replay/Annotation/DataGen은 기록 reset을 상속, 새 Source 파생은 margin0을 상속한다.
Curation 분포는 90도 단위 방향으로 테두리를 수평/수직 유지하고, base_to_workspace 원점 위치를
보라색 네모로 표시한다. robot_xy를 fit에 포함, 같은 배율이며 네모 크기는 표시용이다. 화살표 제거.
UI·MuJoCo 회귀76/별도 USD Workflow·Task31 PASS, 실제 Scene2 MuJoCo32reset·3물체 모서리접촉·10step,
1000배치·기존9episode UI/hash 확인. Isaac 런타임/실물/작업 성공 검증은 미실행.
outputs/workflow_validation/edge_touch_placement/REPORT.md 참조. 원본 Workspace/사용자 데이터 보존.

## 최신 — Curation 최신 단계 자동 연결 / Annotation 시점 (2026-10-16)

CurationDialog.open_latest는 현재 Workspace의 source/annotated/datagen을 한 번에 Registry.latest로
비교한다. Source 우선 fallback 제거, 가장 최근 SUCCEEDED·존재하는 경로를 자동 로드한다.
EventTimeline은 기존 QSlider를 확장해 기록 events만 표시한다. initStyleOption의 handle/groove와
sliderPositionFromValue로 클릭 좌표를 맞추며, 주석 버튼에 경과 초·1-based frame을 표시한다.
마커/버튼→jump_to_event→기존 seek/Job→두 RGB/로봇/관절 동기화, 재생 중에도 정확한 frame에서 정지.
시각은 원본 timestamps[event]-timestamps[0], 없으면 frame*dt. Episode 교체 시 마커·버튼을 비운다.
수집/주석/생성 원본이나 annotation 경계를 수정하지 않는다. 검증/미리보기:
outputs/workflow_validation/curation_annotation_timeline/REPORT.md.

## 최신 — Curation 검토 중심 화면 (2026-10-16)

기존 workflow.curation_ui의 기본 화면을 목록 + Side/Wrist 동시 영상으로 정리했다.
로봇 경로는 정사각형 탭, 필터/상세 분석/데이터셋 구성은 접힌 상태다. 기존 선별/분할/저장은 유지.
WorkflowPanel.open_curation이 현재 TaskSelector Workspace identity를 넘기고 QTimer로
Registry.latest의 완료 Source를 자동 로드한다. Source 없으면 Annotation/DataGen, 다른 Workspace는 제외.
데이터 열기 하나에서 canonical 폴더/HDF5를 모두 선택한다. EpisodeReader.frame은 기존 단일 camera와
복수 camera tuple을 받으며 같은 관측의 rgb_frames/rgb_timestamps를 반환한다. LRU/decoder 제한 유지.
초기 분포의 XY min/max를 제거하고 WorkspaceRandomizer의 target-center envelope(크기/margin)를 표시,
로봇→지지면 중심을 위로 맞춘다. 필터에 따른 자동 배율 변경 없음. 원본 Workspace/데이터 불변.
검증/실제 9episode Source 화면과 원본 hash 확인: outputs/workflow_validation/curation_review_layout/REPORT.md.
실물/엔진 신규 실행 없음. 기존 dirty ui.py/문서 변경을 보존했고 자동 commit/push하지 않았다.

## 최신 — GUI Curation yaml/av 의존성 보완 (2026-10-16)

GUI .venv-real2sim-ui는 include-system-site-packages=false여서 lerobot-arena에 설치한 패키지를 보지 않는다.
실제 새 Source 분석→precision.simulation_configuration의 yaml import 실패를 재현했다.
requirements-ui.txt에 PyYAML>=6,<7, av>=15,<16 추가. GUI venv에만 PyYAML6.0.2/av15.1.0을 --no-deps로 설치했다.
conda/worker 패키지는 변경하지 않았다. 실제 GUI Python에서 사용자 Source9episode 분석,
Side/Wrist first/middle/last 640x480 MP4 decode 및 offscreen CurationDialog load PASS.
재현/결과: outputs/workflow_validation/curation_ui_dependencies/result.json. 원본은 읽기만 했다.
GUI 의존성 설치는 .venv-real2sim-ui/bin/python -m pip install -r scripts/real2sim/requirements-ui.txt.

## 최신 — Workflow 설정 입력 지연 제거 (2026-10-16)

ui.WorkflowPanel.update_ready는 widget snapshot만 만들고 200ms singleShot 뒤 단일 ThreadPoolExecutor에서
기존 Operations.readiness를 실행한다. poll_preview가 완료 future를 GUI thread에서 반영한다.
검사 중 Run 비활성, generation으로 이전 입력/실행 중 결과 폐기, 대기 snapshot은 마지막 한 개만 유지.
close에서 대기 취소, worker에는 plain data만 전달한다. 실행 시 service.run/plan/Registry.verify와 장치 검사 유지.
새 Controller/서비스/설정/검사 생략이나 Workspace hash cache 없음. readiness 호출 자체는 원래 full 검증이다.
실제 Scene2v8 입력5건: 이전0.56s/event(Workspace load20회), 이후0.24~0.40ms/event.
상세/회귀76PASS/프로파일: outputs/workflow_validation/ui_responsiveness/REPORT.md.
실물/worker 자동 연결/토크 없음. 이번 수정은 자동 commit/push하지 않음.

## 최신 — Dataset RGB MP4 Streaming (2026-10-15)

새 MuJoCo Source/DataGen의 EpisodeWriter는 숫자만 RAM에 남기고 기존 AsyncVideoWriter로 H264/CRF18,
카메라별8frame queue에 RGB를 전송한다. 완료 영상 frame count 확인 후 manifest를 마지막에 publish한다.
VideoFrames는 PyAV exact PTS/index와 최대2decoder, Canonical/MiniVLA/Curation에 연결했다.
Isaac Source 및 Workspace Annotation/Mimic DataGen은 EpisodeData private writer→기존 HDF5 handler subclass,
숫자만 HDF5/외부 MP4 refs는 episode attribute. upstream Reader/Pool은 numeric/datagen_info만 사용한다.
Subset/Resume는 refs와 영상 복사, fingerprint/source payload는 외부영상 bytes 포함. 기존 입력은 변환하지 않는다.
외부영상 Annotation은 원본 Source 교체 대신 별도 artifact 등록. 과거 NPY/내부 HDF5 읽기 유지.
MuJoCo Convert는 Canonical RGB를 복제하지 않고 numeric+readonly refs adapter→기존 LeRobot framewise API.
새 Convert artifact는 LeRobot 자체, 기존 nested Canonical artifact는 계속 읽는다. 임의 MP4 복사 없음.
SO101_VIDEO_CRF로 새 기록 품질 지정. 숫자 precision/보정/모델H40/6D/partition/stats/기존checkpoint 유지.
구현/용량/RAM/실제 엔진·변환/실패와 한계는 outputs/workflow_validation/rgb_video_storage/REPORT.md.
실제 장치/torque/새 정책 학습 없음. dependency 변경/자동 commit/push 없음. 이전 dirty 작업 보존.

## 최신 — Source 연속 XY 배치 / MuJoCo 음성 (2026-10-15)

새 Source는 reset.source_workspace가 runtime 복사본에 uniform XY/yaw0을 적용한다. 두 엔진 공통
WorkspaceRandomizer의 물체 크기·margin·separation 검사를 재사용한다. immutable task.json은 그대로다.
scene_configuration.reset에 실제 정책을 기록하고 Source Resume/Replay/Annotation/DataGen은 기록값을 상속한다.
configured의 original_reset은 원본 Task 정의 hash를 유지하기 위한 실행 정보다. 원본 또는 uniform/yaw0 정책만 허용한다.
MuJoCo Source에 기존 utils.voice.log_say의 시작/저장/리셋/종료 알림을 연결했다. 음성 실행 실패는 로그로 남긴다.
최근 사용자 Source render_context는 NVIDIA RTX4060, software=false였다. CPU 물리와 EGL GPU 렌더는 별개다.
현재 실행 중 worker를 자동 재시작하지 않았다. 새 세션부터 적용, 과거 Source 이어 수집은 당시 정책 유지.
검증/코드/사용법: outputs/workflow_validation/source_uniform_voice/REPORT.md.

## 최신 — Source Partition / Mimic / 별도 Validation 입력 (2026-10-15)

기존 workflow.episodes의 source_split/export_source_splits와 Curation UI에서 80/20·seed·수동 소속을
계획/새 subset 저장/Registry 등록한다. 원본 ID는 bounded payload hash와 기록 Dataset/episode,
group 별칭으로 추적한다. 복사·재번호·파일 간 편입 후에도 중복 원본은 다른 split으로 나눌 수 없다.
Mimic validate_input은 실제 Source Pool을 검사, preserve_metadata는 검증된 Partition 입력 증거와
exact_source_ids=null을 기록한다. subtask 원본을 하나로 가정하지 않는다. Source 파일 교체/혼합 Resume 금지.
Annotation은 실제 export 성공 순서의 원본 이름을 전달하므로 건너뛰기/재번호 후에도 Source ID를 유지한다.
Curation/Convert가 split·계보·Task/Scene/보정/Review를 보존한다. Train의 MiniVLA Validation Dataset
선택은 기존 Direct Episodes의 storage.sources/source_index로 두 원본을 읽고 NPY/RGB를 복제하지 않는다.
명시 Partition은 자동 split보다 우선하고 원본 교집합/증거 부족은 Qwen 로드 전에 거절한다.
기존 Legacy group 자동 분리·Canonical·ACT/SmolVLA·모델/보정 계산·장치 정책 유지, dependency 변경 없음.
요약/핵심 코드/실제 검증 범위: outputs/workflow_validation/source_partition_training/REPORT.md.
사용법: docs/real2sim/MINIVLA_INTEGRATION.md, 계약: docs/minivla/DATA_CONTRACT.md, DATA_FLOW.md.
실제 Isaac Pool/selection/HDF5 기록기와 Workflow Train subprocess/Policy Registry, GPU train/validation을
synthetic 자료로 검사했다. 물리 DataGen 성공/실물 평가 아님.
기존 AGENTS.md 변경과 로컬 로그 보존. 자동 commit/push 없음.

## 최신 — MiniVLA LeRobot Direct / 공통 Sim 모델 계약 (2026-10-15)

docs/minivla/DATA_CONTRACT.md, VALIDATION.md와 docs/real2sim/PRECISION_CALIBRATION.md,
DATA_FLOW.md, JOINT_CONTROL.md 우선. Episodes가 원본 MP4/Parquet를 읽고 학습 준비는
index/statistics만 생성한다. CPU exact decode/FFmpeg thread1/decoder2/RGB8frame·16MiB,
기존 batch/H40/6D/RobotCoordinates/Train-only stats와 source-group split 유지.
Source Group이 없거나 UNKNOWN인 생성 계보는 독립 split을 만들지 않는다.
새 checkpoint dataset_identity로 입력 byte/manifest 변경 Resume 거절. 구형 Canonical은 원래
Workflow byte hash 증명이 필요하며 원래 loader로 재개한다. 기존 추론은 유지한다.
precision.simulation_configuration이 numeric URDF frame과 Sim residual geometry를 분리한다.
Scene2v8 잔차는 Builder 증거의 Follower mapping binding이고 Source Leader hash는 달라도 된다.
새 Source/Runtime은 실제 role mapping 검사, Replay/Annotation/DataGen/SimEval은 기록 계약 상속.
Isaac IK URDF와 Physics USD를 같은 모델로 선택한다. Real Eval은 승인 Follower/모델 검증 후
기존 inverse만 사용하며 Sim offset을 motor action에 더하지 않는다. 명시 identity debug 유지.
435 PASS/2 SKIP, 실제 Qwen F/B/optimizer/Resume 및 Isaac/MuJoCo FK/render 완료.
작업 성공률/실물 구동 검증 아님. 원본 Profile/Workspace/모델/데이터 보존, 보정 재계산 없음.
상세/실패/재현/메모리: outputs/workflow_validation/lerobot_direct_calibration/REPORT.md.
기존 미커밋 Curation 변경을 유지했다. 현재 작업 자동 commit/push 없음.

## 최신 — Data Curation Robot/Object 동기 재생 (2026-10-15)

기존 workflow.curation/curation_ui/robot_preview와 두 엔진 Recorder를 확장했다. 새 앱/엔진 없음.
Dataset summary는 저용량 telemetry chunk, 선택 episode는 최대1500 plot sample, RGB는 단일 frame
background read/8frame·16MiB LRU. 최신 seek만 표시하며 robot/EE/object/graph/RGB를 같이 갱신한다.
MuJoCo 기존 target matrix는 재사용하고 distractor만 float32 xyz+wxyz 추가. Isaac Source/Mimic은
obs/object_poses/<id>와 obs/timestamp를 pre-step 관측에 기록한다. post-state를 같은 frame으로 쓰지 않는다.
기록 Workspace identity/변환/해당 URDF로 base frame에 맞춘다. 근거 없는 overlay/movement/success는 UNKNOWN.
과거 initial-only는 Trajectory unavailable; 기존 post-state는 initial[0]+post[:-1]로 복원한다.
Task/active/Scene/group/status filter와 composition별 XY 비율+group 균형 선별. Export는 원본 보존,
Registry→Convert→MiniVLA adapter에 group/Task/Scene/annotation/success 근거를 전달한다.
정확한 원본 demo 계보가 없는 Isaac 생성 결과는 공통 UNKNOWN group으로 묶어 독립 split을 만들지 않는다.
자동 성공 인증/새 학습 모델/실제 하드웨어 검증이 아니다. 사용자 원본/보정/Workspace 불변.
사용법: docs/real2sim/MINIVLA_INTEGRATION.md, telemetry: DATA_FLOW.md와 docs/minivla/DATA_CONTRACT.md.
실행 근거/초기 실패/한계: outputs/workflow_validation/curation_object_playback/REPORT.md.

## 최신 — Workflow Active Object Set (2026-10-15)

`docs/real2sim/BUILDER_WORKSPACE.md` 맨 위와 `DATA_FLOW.md`, `docs/minivla/EVALUATION.md` 우선.
TaskSelector에서 dynamic 체크/Workspace별 기존 ui.json 저장. 기본 전체, Source/SimEval만 변경.
기존 tasks.configured가 runtime 복사본을 필터링하고 원본 identity+active+Task+reset+backend hash.
MuJoCo profile/roles/Builder overlay 및 Isaac Spawn/Reset에서 비선택 객체 완전 제외.
Source→Replay/Annotation/DataGen은 저장 구성을 상속하고 Resume mismatch 거절.
Canonical/HDF5/LeRobot episode별 Task·구성 보존, 혼합 구성 학습 허용. 파생 시 동종 subset 선택.
과거 명시적 dynamic/state 근거만 복원; UNKNOWN을 3개로 추측하지 않는다.
SimEval은 다른 구성 허용/분포변경 기록, 단일 객체 선택 정확도 N/A, 모든 distractor 거리 추적.
MiniVLA architecture/Calibration/원본 Workspace/Builder export 불변, 새 production 파일 없음.
실제 두 엔진/새 Builder object/변환 및 회귀 결과·초기 실패: outputs/workflow_validation/active_scene_objects/REPORT.md.
실제 하드웨어나 학습된 새 정책의 pick/place 성능 검증으로 해석하지 않는다.

## 최신 — MiniVLA Flow-DiT 교체 (2026-10-15)

`docs/minivla/DESIGN.md`, `README.md`, `VALIDATION.md` 및
`docs/real2sim/MINIVLA_INTEGRATION.md` 우선. 기존 패키지를 직접 교체했다.
Qwen NF4 block15 output/letterbox256square/side64image tokens, adapter256, state+40actions,
AdaLN-Zero FlowDiT4, Euler8/replan4. ActionHead7,319,558params. 기본config/default.json.
Deterministic/head선택/UV head/옛4presets 제거. 공용 grounding data/ACT/SmolVLA 유지.
Checkpoint schema so101.minivla.flow-dit/1, 구형 MiniVLA 가중치 거절(원본 보존).
실제 frozen/QLoRA GPU F/B와8~15층 Action gradient 통과. 짧은2step 합성 모델은 작업용 아님.
검증 결과/실패/VRAM/남은 성공률 검증은 outputs/workflow_validation/minivla_flow_dit 참조.


## 최신 — v8 충돌 / 미끄러짐 (2026-10-14)

`outputs/workflow_validation/scene2_v8_contact_physics/REPORT.md` 참조.
정정: v8 finger collision은 USD instance proxy 내부에 존재한다. 앞선 누락 진단 오류.
복원 helper는 instance proxy까지 검사해 중복 생성하지 않도록 수정했다.
후속 Isaac 실제 렌더에서 왼쪽 틈/오른쪽 겹침 재현.
Builder Isaac dt1/240, decimation/render8로 변경해30Hz 제어 유지.
동일 합성 자세의 fixed-pad 평면 최소간격5.51→0.54mm, 실제 grasp 완전 해결 아님.
후속 근거/오진 정정은 같은 폴더 FOLLOWUP.md 우선.
원본 USD/Workspace 수정 없음. Builder isaac_matched MuJoCo noslip_iterations=10.
비관통 합성 유지6/6, 열기 bilateral 해제6/6. 실제 pick/lift/Sim2Real 검증 아님.
Isaac GPU/CPU AppLauncher는 CUDA 초기화 오류로 동적 검사 미완료.
UI 동일v8, Precision ON/Scene보정제외 OFF. 관절 보정/패드 두께/마찰계수 변경 없음.


## 최신 — Scene2 새 접촉 캡처 v8 (2026-10-14)

`outputs/workflow_validation/scene2_final_contact_refinement/REPORT.md` 우선.
새9Pair(3–9 접촉; 사용자 블록 윗면 중앙/컵rim 확인), static001 영상14개와8/9높이 제약.
Builder scene2-final-contact-review-20261014-932c8f2e/final_contact_review export/import→scene2_builder/v8.
평면 tilt2.604°+물체위치 보정; 기존 v7 signed Sim residual/Precision/카메라/원본 보존.
사용자가 base/table 접촉·수평강제를 해제했다. 물리측량된 기울기라고 해석하지 않는다.
contact-height fitting RMS3.95→1.50mm, Side RGB3.2%개선/Wrist0.14%악화, OPEN/UNVALIDATED.
UI v8, Precision ON, Scene보정제외 OFF, 직접 재연결. 실제 grasp/동적응답/Sim2Real 미해결.


## 최신 진행 중 — v7 grasp 대응 진단 (2026-10-14)

`outputs/workflow_validation/scene2_v7_grasp_alignment/REPORT.md` 참조. 실행 개선본은 아직 없다.
새 v7 두 Pair의 실물 grip0.431° 대비 Sim5.859/-9.913°, 동일모델 FK 높이차+6.927/+6.382mm.
MuJoCo full-dynamic18조건과 Isaac GPU2합성 fixture를 실행했다. 초기 관통이 있어 pick 성공 아님.
팔 q 강제 고정한 이전 probe의 낙하는 실제 Runtime 실패로 인용하지 않는다.
그리퍼 offset6/10°는 깊이/접촉 형상과 결합되어 미채택; 새 export/Workspace/Profile 변경 없음.
사용자가 실제 pad는 finger flat surface 크기의 약0.5mm 스티커라고 확인했다(추정 두께).
0.5mm MuJoCo 메모리 실험6조건 후에도 깊이 불일치가 남는다. user_pad_evidence.json 참조.
다음은 flat surface 충돌형상/부착 위치 확인, 비관통 접근/lift 및 Isaac bilateral contact 검사다.

## 최신 UI — Real/Sim Scene·보정 설정 통합 (2026-10-14)

Real/Sim 오른쪽 맨 위 Scene/관절 보정 영역에서 기존 Registry의 Workspace 선택/Import,
저장 Precision 사용 및 identity-Sim 제외 옵션을 설정한다. Workflow와 Workspace 경로를 공유한다.
Capture와 보정 체크박스는 동일 위젯이며 기존 ui_settings.json에 선택을 저장한다.
Pair 저장 설정도 Real/Sim에 표시한다. 설정 변경/탭 전환에는 worker/device 시작이 없다.
현재 worker의 workspace/precision_run_configuration과 다음 Connect 선택을 구분해 표시한다.
현재 연결의 camera tiles는 다음 Scene 선택만으로 바뀌지 않는다. 직접 재연결한다.
사용법: `docs/real2sim/BUILDER_WORKSPACE.md` 맨 위, 검사: `tests/real2sim/test_live_scene_selection.py`,
실제 v7 UI payload/화면: `outputs/workflow_validation/real_sim_scene_controls/REPORT.md`.

## 최신 — Scene2 v7 접촉 보정 실행 연결 (2026-10-14)

UI 검토용은 `scene2_builder/v7`. Builder `scene2-v6-corner-contact-review-20261014-56e6a8bf`
`corner_joint_review`를 실제 재평가·Blender·export/import했고 두 엔진8pose FK/6task smoke 완료.
기존 offsets에 precision_mapping_sha256 binding을 추가해 정확한 Precision 뒤의 Sim residual만 허용한다.
역사적 unbound 중첩 차단은 유지한다. active Profile/raw q/motor target/URDF는 변경하지 않았다.
UI 재시작→v7→저장된 관절 보정 ON→URDF 원형 검사(Scene 보정 제외) OFF→직접 재연결.
model residual을 제외하는 checkbox가 켜지면 v7 보정도 보이지 않는다.
접촉 proxy RMS12.49→3.07mm; RGB MAE는 소폭 악화, 독립 검증/실물 grasp 미완료,OPEN/UNVALIDATED.
보고서/코드 계약/실패/테스트: `outputs/workflow_validation/scene2_v7_contact_application/REPORT.md`.
아래 중첩guard 때문에 export 못했다는 v6 진단은 이 후속 계약 수정 이전의 기록이다.
이 당시 Workflow Source는 Precision이면 identity-Sim을 골랐다. 현재는 위 2026-10-15 공통 모델
계약으로 교체했다. 전체 작업 성공/실물 E2E는 계속 미검증이다.

## 최신 진단 — Scene2 v6 네 꼭짓점 (2026-10-14)

새 snapshot_20261014_110211_377fb860의8자세는 전부scene2_builder/v6.
사용자가 모두 같은 고정 finger 끝 접촉이라고 확인했다.8장 전부 measured-q 분석/import.
COPIED Builder `scene2-v6-corner-contact-review-20261014-56e6a8bf`: geometry-only REJECTED,
corner_joint_review OPEN draft. 접촉 proxy RMS12.49→3.07mm의 추정 zero+marker 제안이다.
현재 Builder/Runtime가 Precision 관측에 역사적 simulation offsets 중첩을 거절하여 export/import하지 않았다.
Guard/서명/Reference를 우회하지 않았다. transient32렌더는 실제 Candidate acceptance가 아니다.
UI용은 여전히scene2_builder/v6. Calibration/active/원본/URDF를 변경하지 않았다.
계수 일부±2° prior 경계, contact point·pose 결합 및 독립 검증 부족이 남는다.
보고서/제안/재현: `outputs/workflow_validation/scene2_v6_corner_refinement/REPORT.md`.
이전 snapshot_20261014_103225_018883f6의8장은scene1_builder/v1이므로 혼동하지 않는다.

## 최신 — Scene2 베이스·상판 수평 접촉 (2026-10-14)

작업용 검토 Workspace는 `scene2_builder/v6`, Builder 같은 project의 `base_table_contact`다.
v5에서 base CAD 최저점보다 상판이15.898mm 높고0.27736° 기울어진 문제를 수정했다.
상판 normal=robot +Z, 최저 base mesh와 상면 높이 일치. 물체들은 지지면을 따라 이동했다.
로봇/base/q/카메라/보정/물성/6개 workflow는 그대로다. v5/원본은 보존한다.
이후 정합에서도 베이스 매립/상판 기울기로 영상 오차를 숨기지 않는다.
Wrist RGB는 악화했고 Real fingertip 접촉·grasp/slip 해결은 미검증이다. OPEN/UNVALIDATED 유지.
검사/비교/UI: `outputs/workflow_validation/scene2_base_table_contact/REPORT.md`.

## 현재 — Scene2 Workflow 연결 (2026-10-14)

사용자 지정대로 단일 블록→컵6조합을 Builder `single_block_workflow`에 추가했다.
Project `scene2-visual-physics-refinement-20261013-bbb16a35`, 실행 Workspace `scene2_builder/v5`.
Scene1 성공/랜덤 Reset을 재사용한다. v4 geometry_review는 보존한다.
상세 검사와 사용법: `outputs/workflow_validation/scene2_workflow_binding/REPORT.md`.
Source 사전검사/두 엔진 판정·Mimic 연결 완료, 실제 데모→학습 E2E와 그리퍼 slip 해결은 미완료다.

## 현재 — Scene2 외형/물리 후속 (2026-10-13)

최신 검토 후보는 sibling Builder `scene2-visual-physics-refinement-20261013-bbb16a35`의
`mounted_clip_detail_review`, 실행 Workspace는 `scene2_builder/v4`다. 기존 v2/v3는 보존한다.
Scene1 camera mesh의 장착/클립을 수정하고 roller pose와 cup ink pose를 보정했다.
배경색/조명/광학 카메라/q/보정/실측 물체 치수는 유지했다.
명시적 canonical mass가 있는 geometry_review는 기존 physics로 dynamic block와 열린 고정 cup을 만든다.
픽앤플레이스 task/success를 자동 생성하지 않는다. Source/Datagen은 별도 workflow metadata가 필요하다.
상세 증거·불확실성·UI 절차: `outputs/workflow_validation/scene2_visual_physics_refinement/REPORT.md`.
아래 이전 Scene1 기록은 역사적 맥락이다. 엔진 smoke와 실물 grasp 성공을 혼동하지 않는다.

## 최신 — 2026-10-13 / Teleop 모터 범위 초과 처리

사용자 요청으로 Runtime.command_target의 기존 clip에 Follower 모터 명령 가능 범위를 포함했다.
PrecisionMapper.command_limits는 현재 URDF/Sim과 모터 범위의 정수 encoder 교집합을 계산한다.
정상적인 목표 초과는 경계로 제한하고 계속 추종한다. 저장 profile/active/URDF는 그대로다.
모호한 branch/비정상 값/identity/통신 오류는 계속 거절한다. Policy inverse는 strict 유지.
이전 “motor range 부족이면 Teleop 전체 거절” 설명은 현재 정책이 아니다.
실제 로봇 자동 테스트 없음. TELEOP_LATENCY_DIAGNOSIS.md의 후속 결과 참조.


## 후속 확인 — 2026-10-13 / Teleop 지연 및 Start 오류

사용자 재현 로그에서 목표 전송 간격은 약33ms, send0.10–0.15ms였다.
모든 모터 Goal_Velocity338–347이 Connect/Start 전후 유지되었고 사용자의 모터 전원
재시작 후0으로 초기화됐다. 이전 속도 제한 잔류가 지연의 유력 원인이다.
이후 shoulder_lift가 기존 모터 보정 범위를 넘는 목표를 받아 Start에서 거절됐다.
위 Teleop 목표 clip 수정 후 사용자가 정상 동작한다고 확인했다. 외부 센서로
지연/정확도를 실측한 결과는 아니다. 재보정/보정 파일 변경은 하지 않았다.
진단은 기존 runtime 로그/status에 남기며 모터 속도/Phase를 임의로 쓰지 않는다.
관련 최종 검사171 PASS,1 SKIP(Blender 선택 검사); 실제 Profile 1001개 목표 offline 변환 통과.
상세 로컬 기록: `outputs/workflow_validation/unified_joint_control/TELEOP_LATENCY_DIAGNOSIS.md`.
outputs는 Git 제외이므로 코드 사용 계약은 `docs/real2sim/JOINT_CONTROL.md`를 참조한다.

## 최신 — 2026-10-12 / Limit-only Profile 자동 호환

`outputs/workflow_validation/joint_limit_extension/COMPATIBILITY_REPORT.md`와
`docs/real2sim/PRECISION_CALIBRATION.md`를 우선 읽는다. 이번 사용자 지시로 이전 승인형
Limit Migration UI/worker/ProfileStore.rebind_limits를 제거했다. 기존 active Profile을 재저장하거나
재승인하지 않는다. `precision.profile_for_model`은 exact hash에 맞는 Git/보존 URDF를 읽어
lower/upper 외 모델 전체가 동일함을 확인하고 실행 복사본에만 선택 URDF의 제한을 적용한다.
원본 Profile/active/Sign/Scale/Zero/References를 변경하지 않는다. model evidence 부재,
Axis/Origin/geometry/topology 변경 및 모터 identity 변경은 계속 거절한다. 장치 제어는 기존
SDK/Precision strict inverse를 사용한다. 새 컨트롤러/Guard/Migration 설정은 만들지 않는다.
Calibration의 실제 보정 Save/사용자 승인 Activate/Restore는 그대로다.

Pan/Lift/Elbow/Flex ±5°는 이미 적용되어 추가 확장하지 않았다. Roll/Gripper도 그대로다.
원본 USD/Workspace는 보존. 실행 USD는 `assets/SO101/usd/so101_joint_limits.usda`, 새 Workspace는
`scene1_builder/v10`. Real-only 기본은 root URDF, Workspace는 자신의 URDF/USD로 재현한다.
이전 검증 보고서의 Migration 지침은 현재 정책이 아니다. 사용자가 기존 worker를 종료하고
UI 재시작 → Connect하면 된다. 자동 장치 구동/설정 설치/사용자 데이터 수정은 하지 않는다.


## 최신 — 2026-10-12 / 관절 제어 단순화 (현재 정책)

사용자 요청으로 별도 Controller/MotionGuard/자동 복귀 및 전용 UI/IPC를 제거했다.
`docs/real2sim/JOINT_CONTROL.md`와
`outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md`를 우선 읽는다.
기존 Runtime.command_target이 Leader observed_positions → URDF/선택 엔진 clip → Follower
Precision inverse → 실제 SDK 정수 변환 → 같은 q를 Real/Sim에 전달한다. Motor range 부족은
명령 전 거절한다. Start 5°/30°/s/10° Guard나 RETURNING_TO_LIMIT 경로가 없다.
Connect는 읽기 전용이며 유효한 범위 밖 Follower 피드백도 허용한다. Start는 명시 사용자 요청이다.
Precision/보정 승인·복구/모터 identity·유효성/통신/heartbeat/stall/Sim limit 검사 유지.
새 Controller/launcher/config를 만들지 않았다. 관측 계산은 기존 so101_joint_mapping에 통합했다.
새 Snapshot에서 controller metadata를 생성하지 않으며 역사적 applied q reader만 유지한다.
SDK 정수 conversion은 strict inverse의 1-count 부동소수점 문제와 Real/Sim target 동일성을
위해 Hardware에 한 개 유지한다. 이전 guard 설정을 다시 복원하지 않는다.
URDF는 이미 ±5° 확장되어 추가 수정하지 않았다. v10 및 기존 사용자 Profile/Workspace/Snapshot/
HDF5/checkpoint를 보존한다. 실제 보정 변경의 activation만 사용자 승인이다. Builder/GLB 변경 없음.


## 현재 기능 — 2026-10-12 / Joint Precision Calibration

상세 계약/사용법은 `docs/real2sim/PRECISION_CALIBRATION.md`, 검사/변경 이력은
`outputs/workflow_validation/precision_calibration/REPORT.md`와 `docs/real2sim/CHANGELOG_AI.md`를 읽는다.

- UI 네 번째 탭 → 기존 RuntimeLauncher/IPC → `real2sim_sim.py --backend calibration`
  → `precision.Calibration` → `hardware.CalibrationDevice`. 새 launcher/Builder 변경 없음.
- 명시적 read-only Connect와 승인한 선택 joint torque release만 지원한다.
  Goal/Homing/Range/Drive register 쓰기, 자동 장치 연결/로봇 이동은 없다.
- 새 기록은 raw + 손으로 맞춘 URDF 기준각으로 통일한다. split=train/reference_kind=angle은
  worker가 고정하며 validation/open/closed/opening 입력과 예전 precision.validate 실행 명령은 제거했다.
  fit → review(손으로 움직이며 확인) → save → 승인 activate. 검토는 REVIEWED_MANUAL,
  independent_validation=False/physical_verified=False이며 실물 정확도 인증이 아니다.
- `validate_joint`와 legacy_endpoints는 기존 저장 Profile을 읽고 증거를 재검사하는 데 사용한다.
  이 reader를 삭제하면 이전 저장값의 로드/복구가 깨진다. 새 기록 UI/worker 경로로 노출하지 않는다.
- 0°/90°는 fit 기준점이다. raw_domain은 증거이며 사용 한계가 아니다.
  UI/Calibration/Runtime/Compare는 joint_raw_coordinate/joint_angle로 URDF 범위 전체를 계산한다.
  goal은 기존 calibrated motor range 및 유일한 encoder branch도 확인한다.
  Precision gripper는 기록 구간으로 clamp하지 않는다. legacy gripper 제한은 유지한다.
- Real-only Capture의 STOPPED/passive 관측은 calibrated_to_observation으로 범위 밖 측정도 보존한다.
  precision_observation_only/joint_limit_warnings를 기록하고 사진/측정값을 자르지 않는다.
  motion/physics는 calibrated_to_urdf/inverse의 기존 제한을 유지한다. encoder ambiguity는 거절한다.
- raw Present_Position은 firmware Homing 이후 값이다. Homing/LeRobot normalization을 이중 적용하지 않는다.
  live read는 같은 raw packet의 SDK normalization, record는 별도 read/안정성 검사를 유지한다.
- 보정 전은 현재 저장/적용 값이다. 새 Candidate가 없으면 전/후/현재가 같다.
  0.5°는 기록 재현 residual이며 두 점 fit에서는 raw 간격의 정당성을 검증하지 못한다.
- 선택 자세 삭제는 원본 journal을 유지하고 exclusions에 ID/관절/이유/시각을 기록한다.
  해당 Candidate만 초기화하고 다른 joint/기존 active를 유지한다.
- save는 해당 role의 모든 확인된 Candidate와 기존 active를 합친다. partial activate는 다른 joint를 유지한다.
  동일 저장/적용은 파일/pointer 재사용, 실제 변경은 새 immutable 버전이다.
  같은 device/model 재연결은 active reference를 새 세션 표에 복사하고 loaded_profile_id로 출처를 남긴다.
- ProfileStore는 `<UI workspace>/joint_calibration/{leader,follower}`, Fake는 `fake/` namespace다.
  사용자 최신 active는 store를 직접 읽는다. 2026-10-12 현재 Follower profile_1a377745dd6441f0bacee6ff3f0a5690,
  Leader profile_a4ea4642b1da4587943331fb8083c322, 양쪽 6관절이다.
  이전 Follower 복구 Profile: profile_0af8c777427a4c999b47ca67e8e6298a.
  source journal: references_b2834b1d41884b80b281de6a09a24411.json.
  기존 4관절 보존, wrist_flex/gripper만 사용자 원본으로 복구했고 이후 사용자가 보정을 추가했다.
- Real/Sim/Capture는 저장 active가 있으면 다음 Connect에 Precision/원본 URDF 설정을 기본 선택한다.
  Teleop은 두 장치의 같은 보정 joint set과
  URDF/모터 범위 preflight가 필요하다. nonzero Scene Sim offsets는 충돌 검사한다.
  명시적 identity-Sim session은 immutable Scene 수정 없이 MuJoCo session/Isaac 원본 USD를 사용한다.
  hardware.Leader/Follower와 standalone SO101Leader는 Connect 시 승인 active를 로드/identity 검증한다.
  Source는 동일 mapper로 model rad를 기록하며 collection_coordinates로 hardware_mapping/URDF frame을 남긴다.
  원본 LeRobot JSON/register/전역 DEFAULT_MAPPER를 변경하지 않는다.
  새 Source와 후속 Annotation/Datagen/Convert/Train/Sim Eval은 기록된 frame을 이어받고
  기존 데이터는 당시 frame을 유지한다. real_eval은 현재 Follower mapping을 경계에서 사용한다.
  이미 URDF rad인 값에 재보정/자동 데이터 migration은 하지 않는다.
  Calibration이 포트를 소유하면 UI Source/real_eval을 차단한다.
- 이전 코드가 이미 worker에 로드됐으면 사용자가 보정 작업 종료 → UI 재시작 → 명시 Connect한다.
  저장 보정은 재기록/재저장하지 않아도 된다. agent가 사용자 worker를 자동 재시작하지 않는다.
- Fake worker+Qt/수학/실제 MuJoCo FK preview 검사가 있다. 실제 serial/torque/물리각/Teleop/
  Isaac 새 session 경로는 HARDWARE_NOT_TESTED다. 사용자 Profile/Scene/Snapshot/HDF5를 보호한다.


## 최신 — 2026-10-11 / v7 새 2 Pair → v8 Wrist K 보정

`outputs/workflow_validation/v7_wrist_camera_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_222612_a9610eee` 2자세(v7) train, 이전6자세(v6) 회귀 비교.
새 Builder project `scene1-v7-wrist-camera-refinement-20261011-58823cd8`,
선택 `wrist_view_review`: Wrist K만 fx/fy[560.8899,558.3139], cx/cy[326.8775,262.6275].
원본 q/geometry/Workspace축/robot/base/joint zero/mount/Side/task/물성은 그대로다.
새 paper IoU .8290→.9459, .8207→.8938; 이전6도 모두 개선.
영상전체 MAE 및 일부 center 회귀로 공식 nonregression FAIL, OPEN/미 promote/UNVALIDATED.
그리퍼 벌어짐은 여전히 다르다. 영점 probe는 닫힌 자세 개선/열린 자세 악화로 적용하지 않았다.
pose+focal 대안은 실제 camera body 내부 가림으로 제외. 모든 대안은 REJECTED로 닫았다.
review XML/USD export와 sibling 소비 Workspace `outputs/real2sim/environments/scene1_builder/v8` 생성.
MuJoCo 16RGB/raw-q/FK/camera/reset/200step, Blender Wrist3렌더, export USD load/구조 확인.
Isaac v8 runtime/contact/실물 calibration/독립 blind validation 미검증. 장치 접근 없음.
새 실험의 disposable Blender payload/mask 약1.11GB만 제거; source/Scene/PNG/metrics 유지.
기존 v7/원본/canonical/evidence 보존, production/dependency 변경 없음.
UI 선택은 자동 변경하지 않았다. 사용자가 v8 선택 후 직접 재연결하고 Pair version/hash 확인.
상세 실패 로그·보존·실험 recipes/·다음 finger mapping 검증은 위 REPORT.md 참조.


## 최신 — 2026-10-11 / v6 새 6 Pair → v7 Wrist 작업공간 검토 후보

`outputs/workflow_validation/v6_wrist_workspace_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_145450_d2cf22b8`는 전부 v6이며 6자세/12이미지 모두 사용했다.
새 Builder `scene1-v6-wrist-workspace-refinement-20261011-b3a03339`의
`paper_pose_guarded`에서 종이/테두리만 XY[-3.380,+1.292]mm, yaw+1.448° 수정했다.
원본 q, robot/base/zero, camera K/mount, 컵·블록·table의 world pose/치수/물성 유지.
Wrist 종이-only 진단 IoU는 6자세 모두 개선했으나 Side 일부 mask와 RGB 회귀가 남아
공식 nonregression FAIL, **OPEN / 미 promote / UNVALIDATED**다.
자세5/6은 fitting 제외 비교지만 같은 session이며 독립 validation 인증이 아니다.
`paper_pose_wrist_priority`는 Side 회귀로 REJECTED. camera-only 대안은 적용하지 않았다.
review XML/USD export와 `outputs/real2sim/environments/scene1_builder/v7` 생성.
MuJoCo 12RGB/raw-q/EEF/camera/reset/200step, Blender 두 자세4렌더 완료.
v7 Isaac runtime/실물 contact/독립 session validation 미검증. actual hardware 접근 없음.
Adapter의 Workspace 축은 수정한 support yaw를 따르므로 runtime-local 좌표는 달라지지만
나머지 물체 world pose는 동일하다. Source Reset 영역은 새 종이를 따라 이동/회전한다.
원본/canonical/evidence/기존 v6/export 보존, production/dependency 변경 없음.
이번 실험의 disposable Blender payload/mask 약1.39GB만 정리했다. recipes/ 재현 가능.
UI는 자동 변경하지 않았다. 사용자가 v7 선택 후 직접 Disconnect→Connect하고
새 Pair version/hash를 확인한다. 상세 metrics/보존/다음 단계는 위 REPORT.md 참조.


## 최신 보정 — 2026-10-11 / 실제 v5 새 4 Pair → v6 검토 후보

`outputs/workflow_validation/v5_new_capture_refinement/REPORT.md`를 우선 읽는다.
새 `snapshot_20261011_140108_47734fec` 4자세/8이미지는 실제 v5이며 전부 사용했다.
raw q 보존. tracking 차이의 모델 FK 거리 5.42–9.92mm를 scene 오차와 분리했다.
새 Builder `scene1-v5-cup-approach-refinement-20261011-56afacbf`의
`approach_contact_review`에서 작은 Sim zero/base만 보정, 카메라/컵/블록 world geometry 잠금.
추정 컵 rim52.94→45.50px, 이전 추정 landmark 평균 개선이나 개별2px 증가 허용,
Wrist 전체 MAE는 악화했다. **미 promote / UNVALIDATED / NEEDS_MORE_DATA**다.
review XML/USD export와 `outputs/real2sim/environments/scene1_builder/v6` 생성.
MuJoCo 8영상/FK/camera/reset/200step 실제 검사와 Blender 두 자세 렌더 완료,
v6 Isaac runtime/실물 grasp/place/독립 validation 미검증. v5와 원본 모두 보존.
UI/장치 연결은 자동 변경하지 않았다. 다음은 블록 접근·컵 자세의 사용자 정지 Pair와
독립 session 검증이며, Teleop 중 real/Sim plant 차이를 임의 영점에 흡수하지 않는다.

## 최신 읽기 전용 진단 — 2026-10-11

새 7 Pair(`snapshot_20261011_121308_65bb7b1f`)의 저장된 Runtime은 모두 v1이며
v5가 아니다. hash/profile/14 camera transforms로 확인했다. 모든7자세를 같은 measured q로
오프라인 재렌더했지만 calibration 식별/새 독립 holdout/real object annotation 부족으로
NEEDS_MORE_DATA다. 새 scene/export/production 변경 없음. v5는 기존 검토 후보로 보존한다.
`outputs/workflow_validation/v5_root_cause_20261011/REPORT.md`와
`SCENE_REFINEMENT_HANDOFF.md` 최상단을 우선 읽는다. UI Workspace 선택은 Connect 시 적용되므로
v5 선택 후 사용자 재연결과 새 Pair identity 검사가 필요하다. 로봇을 자동 구동하지 않는다.

## 최신 재개 상태 — 2026-10-11

이전 v3 상태보다 이 문단을 우선한다. Builder V2 우선순위 구현 후 새 project에서 v2 컵 배치와
모든12자세/24이미지로 Sim 영점/base/Wrist/블록을 추정 보정했다. 최신 실행용 검토 Workspace는
`outputs/real2sim/environments/scene1_builder/v5`. original q/model/사용자 데이터 보존.
실제 두 엔진 raw-q/FK/RGB smoke 통과, 물리/독립 validation 미완료, 후보 미 promote.
`outputs/workflow_validation/builder_v2_scene_refinement/REPORT.md`와
sibling `docs/V2_IMPLEMENTATION_REPORT.md`, `SCENE_REFINEMENT_HANDOFF.md` 최상단을 먼저 읽는다.
v4는 USD source hash 충돌을 발견한 중간 실패 버전이므로 선택하지 않는다.


기준일 2026-10-10. 실제 source와 결과 JSON을 확인하여 작성. 이 문서는 코드 변경이 아니다.
상대 경로는 이 프로젝트 기준. Builder 인수인계는 `../real2sim-builder/CODEX_HANDOFF.md`.
기존 상세 설계 원칙은 `docs/AGENTS.md`; 현재 root AGENTS.md를 우선하여 작업한다.

## 목적과 실행 구조

SO-101은 실제 Leader/Follower·카메라 제어, 실측 Snapshot 수집, Scene 소비 및 학습 Workflow 프로젝트다.
Builder는 독립 offline Scene reconstruction toolkit이며 실제 장치를 제어하지 않는다.
외부 Codex가 이미지 해석·가설 선택을 담당한다. Builder run은 deterministic 도구이지 내부 LLM이 아니다.

실제 진입점 `scripts/real2sim/real2sim_ui.py` → `real2sim.ui.Window`.
Capture/Real-Sim → `real2sim.launcher.RuntimeLauncher` → `scripts/real2sim/real2sim_sim.py`.
Isaac은 `real2sim.runtime.Runtime`, MuJoCo 구현은 `simulation/`에 있다.
Pipeline → `workflow.ui.WorkflowPanel` → `workflow.operations.Operations`.
공통 CLI `scripts/real2sim/workflow.py`; MuJoCo stage는 `workflow/mujoco.py`.
Isaac source/replay는 workspace.py와 scripts/teleoperation, annotation/datagen은 scripts/mimic,
변환은 scripts/tools/convert_isaac2lerobot.py. 세부 호출은 `docs/real2sim/PROJECT_MAP.md`와 DATA_FLOW.md 참조.

## Builder → Workspace 변환

Builder canonical `real2sim.scene/1` + 선택적 `so101.scene-workflow/1`
→ export scene.yaml/XML/USD/scene_manifest.yaml/SCENE.md
→ `real2sim/workspaces/builder.py: inspect_export, profile_and_task, build_usd_assets, import_scene`
→ immutable `so101.workspace/1`: manifest.json/profile.json/task.json/physics.json/provenance.json,
builder 사본, SO101 제어 assets, USD wrappers
→ package.load checksum 검사 → Real/Sim 또는 Operations.

Geometry/Camera K·pose는 profile, 역할·pick/place·reset·success는 task, 물성은 physics로 연결된다.
두 엔진은 기존 SO101 articulation을 사용하고 Builder reference robot을 두 번째 고정 로봇으로 넣지 않는다.
Isaac은 USD visual reference, MuJoCo는 exported geometry + 기존 actuator/contact를 사용한다.
동일 recipe를 쓰지만 renderer/solver 차이로 영상·동역학 동일성을 보장하지 않는다.
상세 schema·예제·지원 범위는 `docs/real2sim/BUILDER_WORKSPACE.md`.

## 기능과 제한

Metadata camera ID 기반 UI/view·관측·변환·정책 입력, object collision/physics/reset/성공 영역,
Source/Replay/Annotation/Datagen/Convert/Train/Eval 경로가 있다.
범용성은 SO101 + 기존 pick/place recipe 범위다. 임의 로봇/임의 task 알고리즘 자동 생성은 없다.
MuJoCo Datagen bounded IK는 Isaac MimicGen과 동일 알고리즘이 아니다.
MiniVLA 구현은 `source/soarm101_lab/soarm101_lab/learning/minivla/`: 단일 Flow-DiT head(위 최신 문서 참조).
Qwen 이미지 진단은 action policy와 다르다. 전체 새 Scene의 성공 grasp→annotation→datagen→훈련→rollout은 미검증.
Camera mount world/gripper 지원; Snapshot producer는 Side/Wrist 계약이다.
새 camera 입력을 기존 checkpoint에 자동 치환하지 않는다.

## 데이터·사용법

실측: `outputs/real2sim/snapshots/snapshot_<시각>_<id>/`.
Start→Disconnect 한 세션; 공통 robot_model/calibration을 한 번 저장하고 snapshots/<shot>에
side/wrist PNG, state.json, metadata.json, COMPLETE 저장. schema `real2sim.snapshot/1`.
이전 snapshot_dataset_* 원본도 보호한다. COMPLETE와 manifest index/hash를 확인한다.
단위 m/rad, named joint order, measured/command 분리, host timestamp는 exposure 시간이 아니다.
Mapping legacy_unverified를 물리적으로 확인하지 않고 calibrated로 바꾸지 않는다.
Capture Start는 실제 장치만 연결; Teleop은 별도 Start Teleop. Real/Sim은 backend 선택 후 Connect.
오른쪽 연결 설정, 아래 로그. UI 시작과 탭 전환은 자동 연결을 하지 않아야 한다.
사용법·계약: `docs/real2sim/CAPTURE_DATASET.md`.
해당 문서의 “downstream adapter 별도/미지원” 문장은 오래된 상태이며 현재 BUILDER_WORKSPACE.md를 따른다.

현재 실제 Scene workspace: `outputs/real2sim/environments/scene1_builder/v1`.
Builder 제작 프로젝트: `../real2sim-builder/projects/so101-scene1-snapshot-reconstruction-2202744f`.
사진/설명: `../real2sim-builder/inputs/scene1/`.
UI Pipeline에서 workspace 선택; Real/Sim에서 Pipeline Workspace 사용을 켜고 backend 선택.
새 export 폴더를 UI에서 선택하거나:

```bash
python scripts/real2sim/workflow.py import-builder /absolute/export --workspace-id scene_new --version 1
```

새 package는 checksum 고정. 변경은 Builder candidate/new export/new workspace version으로 한다.
Task 선택 후 Source→Annotation→Datagen→Convert→Train/Eval을 각각 실행한다.
Snapshot은 실측 scene fitting 자료이며 학습 Source HDF5와 다르다.
Workflow registry `outputs/workflow/artifacts/*.json`은 데이터 자체가 아니라 identity/parent/path 기록이다.
runs/는 request/state/log/presentation IPC 및 stage 결과. real2sim/ipc는 UI-worker 명령·응답이다.
workflow_validation은 개발 실행의 증거이며 runtime 입력과 혼동하지 않는다.
기존 Workspace v1/v2 producer 삭제와 이력 복구 제한은 BUILDER_WORKSPACE.md 참조.

## 검증 증거와 최근 수정

`outputs/workflow_validation/builder_scene1/report.json`: 실제 scene1 두 엔진/카메라/reset·success smoke.
`outputs/workflow_validation/metadata_adapter/paths.json` 및 report.json: 별도 synthetic 과일 scene.
workspace `.../metadata_adapter/workspaces/metadata_fruit/v2`는 실제 과일 reconstruction이 아니다.
front 1개, sphere apple/compound banana/open basket; MuJoCo·Isaac 실제 환경 렌더와 reset/안정화,
Mimic binding, synthetic HDF5→LeRobot, head forward 검증. 물체를 놓아 성공 영역 검사한 것이지 grasp 성공이 아니다.
기록된 회귀 144 passed/2 skipped. 결과 report에 첫 실패·수정 경위를 보존했다.
실제 Leader/USB/카메라 수집과 새 task 훈련/성공 rollout은 해당 회귀에서 NOT_TESTED.
사용자는 Snapshot을 직접 수집했지만 자동검사에서 하드웨어 통합을 인증한 것은 아니다.

최근 사용자 Source 오류: EGL developer viewer를 먼저 켜고 lazy Renderer 생성 시 EGL_NOT_INITIALIZED.
`simulation/mujoco_backend.py: PolicyEnvironment`에서 RGB context를 viewer보다 먼저 생성,
`workflow/mujoco.py: source`에서 Leader.connect 전 observe preflight 추가.
RTX4060 EGL + developer + device import 60 RGB/physics step(320×240) 실제 통과,
backend/minivla 관련 30 tests 통과. 실제 로봇 Source 수집은 수행하지 않았다.
다른 mode/renderer 비교는 fresh process. OSMesa 라이브러리가 없을 수 있으며 software-egl 선택 가능.

## 이어갈 작업

후속 재개 최신: `scene1_builder/v3`는 표면 색상 후보를 promote/export/import한 버전이다.
holdout MAE0.13643→0.12364, 4장 모두 nonregression PASS. Geometry/contact calibration은
미검증이며 v2 컵 배치를 채택하지 않았다. 상세는 `SCENE_REFINEMENT_HANDOFF.md` 최상단과
`outputs/workflow_validation/paired_refinement_160851/surface_refinement_result.json`.
아래는 v2 인계 당시 기록이다.

2026-10-10 최신 상태는 `SCENE_REFINEMENT_HANDOFF.md`를 먼저 읽는다.
새 Pair `snapshot_20261010_160851_00c94af5` 8쌍은 모두 import/evaluate했다.
영점/카메라 및 컵 배치 후보는 holdout 검사를 통과하지 못했다. 사용자 요청으로
컵 배치/밝은 종이컵 후보를 `UNVALIDATED` 검토용 XML/USD로 export하고
`scene1_builder/v2`로 import했다. v1과 기존 canonical은 보존한다.
v2는 검증 성공이나 physics/실물 grasp 완료가 아니다. 파일/검사 결과는
`outputs/workflow_validation/paired_refinement_160851/layout_export_result.json`.
사용자는 다른 Codex 계정으로 작업을 인계하며, 현재 턴에서는 추가 fitting을 시작하지 않는다.
이후 wrist/side/contact 차이를 기존 도구로 보정하되 measured q는 고정하고,
수치 영점을 controller/URDF의 실제 calibrated 값으로 임의 적용하지 않는다.

## 실행 환경과 검사

기존 `/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python` (Python3.11), RTX4060 Laptop8GB.
GUI venv `.venv-real2sim-ui`; root pyproject는 package installer가 아니라 tooling 설정이다.
root에 pip -e . 하지 않는다. 설치/버전 변경 없이 기존 환경을 사용한다.
SO UI 진입은 `scripts/real2sim/real2sim_ui.py`; GUI dependency는 해당 기존 venv를 확인한다.
Blender `/home/hyeonhee/apps/blender/blender`. Builder 환경 명령은 sibling docs/LOCAL_RUN.md.

```bash
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -B -m pytest tests/real2sim/test_mujoco_backend.py tests/real2sim/test_minivla_integration.py tests/minivla/test_head.py -q -p no:cacheprovider
```

MuJoCo 검사는 pxr/Isaac을 import한 pytest 프로세스와 분리한다.
Qt/USD 검사는 기존 PyQt6/pxr 경로가 필요하며 docs/real2sim/TESTING.md 및 Builder LOCAL_RUN.md 참고.
이전 /tmp/builder_test_modules PyQt symlink는 임시파일이므로 새 계정에서 존재를 가정하지 않는다.
명령별 env를 사용하고 GUI site-packages·USD PYTHONPATH를 worker에 전역 전달하지 않는다.

## Git·보호 범위

작성 전 HEAD `ca923a737e1c5dcafb8f8e8da9f2bf0a2db8b58a`, status 113 entries(문서 작성 전).
README/docs/runtime/workflow 등 수정, bootstrap/reference/구 publish/teleop launcher 등 삭제,
simulation/learning/adapter/snapshot/tests 등 untracked. 이는 진행된 구현·정리이며 복구하지 않는다.
Builder sibling도 현재 Git 저장소다. 미커밋 변경 및 ignored project/inputs/outputs를 별도 인계해야 한다.
outputs는 Git으로 보존되지 않을 수 있다. status/diff/ignored 목록을 확인하고 사용자 데이터도 별도 백업한다.
이 문서는 commit하지 않는다. reset --hard, clean -fd, 전체 checkout 금지.
기존 사용자 photos/snapshots/HDF5/checkpoint/scene1/reference 프로젝트는 변경·삭제하지 않는다.

## 추가 변경: Pair 관측과 프리뷰/Source

Source는 실행별 fresh seed(명시 seed 재현 가능), MuJoCo preview는 기존 RGB 재사용,
Pipeline 전달/Qt polling은 최대30Hz. 실제 FPS는 renderer/기기에 따라 다르다.
Pair Capture는 Runtime cached workspace context + 기존 Snapshot 저장 확장이다.
Builder import-paired와 기존 Candidate의 three-way diagnostics를 사용한다.
현재 사용법·데이터 계약·미검증 범위는 `docs/real2sim/PAIRED_CAPTURE.md`를 먼저 읽는다.
