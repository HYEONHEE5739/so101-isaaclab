# SO-101 / Real2Sim 개발 인수인계

## 최신 — Workflow 작업별 화면과 Controller 분리 (2026-10-17)

사용자 명세 기준52c4e06에서 UI만 리팩터링했다. 기존 미추적 MUJOCO_LOG.TXT와 docs/minivla/test-output.txt 보존.
WorkflowPanel은217줄의 상위 통합/화면 lifecycle로 축소했고 ui_components에 Context/Controller/Navigation,
ArtifactPicker, run별 Monitor/Preview/TrainingMetrics와 Source/Replay/Annotation/DataGen/Convert/Training/
Sim/Real Eval/Publication/Curation/Explorer Screen을 분리했다. Navigation5그룹은 STAGES 실행 계약과 별개다.
모델별 TrainingScreen 인스턴스가 옵션/입력/Resume을 각각 보존한다. Camera 없는 단계에는 CameraView를 생성하지 않는다.
Replay/Annotation/DataGen은 editable TaskSelector 대신 입력에서 상속한 Task/Scene 요약을 쓴다.

Controller는 plain snapshot/deep copy→기존 Operations.readiness/run을 호출한다. debounce/token/worker validation 유지,
Val 후보 검사도 background에서 처리. 실행 중 Context/Navigation/설정 잠금, 중복 실행 차단, 시작 전 취소 지원.
현재 화면의 run만 로그/결과와 연결한다. MiniVLA는 실제 metrics 그래프를 재사용하고 이전 기록이면 구분한다.
변환/업로드에 그래프·카메라·기록 컨트롤 없음. Real Motion은 기본OFF+명시 확인, Upload도 대상/범위 확인 후 기존 worker.
Window는 Context+scene_tasks()를 사용하고 기존 getter/browse/select/service/busy 호환 유지. Backend 동기화는 잠금을 존중한다.

Explorer는 Registry 읽기 전용 필터/계보/정확한 ID 이동, MiniVLA checkpoint를 Policy 하위 파일로 표시한다.
LeRobot 중간 checkpoint import는 기존 평가 화면의 폴더 등록을 사용한다. Curation은 기존 dialog를 임베딩했고,
episode 선별/Source Group 분할/원본 보존은 변경하지 않았다. 숨긴 real_enabled/real_side/wrist/options_help 및
section/stage combo, 이전 Navigation/show-hide와 미사용 stylesheet를 제거했다. option_help schema·CameraView/
LogTail·TrainingLossPlot·TaskSelector와 기존 SettingsForm은 재사용한다.

화면/Curation149PASS, USD/Window75PASS, 공통 계약83PASS, MiniVLA CPU38PASS. 실제 Registry63개 읽기 전용11화면
캡처는 worker0, Registry/UI 설정 byte 보존 확인. 처음 USD 경로 누락·모드 비교의 임의 출력명·라우팅 fixture의
Task metadata 의존·캡처 runner QWidget.scroll 동명 오류를 수정했고 실패 로그는 남겼다. 모델/IK/물리/Mimic/
Dataset/Operations/Registry/CLI/학습 코드는 변경하지 않았다. 새 GPU 학습/DataGen/Hub 업로드/실물 검증/commit/push 없음.
사용법·옵션/Stage 계약·제거 근거: docs/real2sim/WORKFLOW_UI.md. 보고서·캡처·명령/감사:
outputs/workflow_validation/ui_architecture/. 최종 작업은 git diff로 검토할 상태다.

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
