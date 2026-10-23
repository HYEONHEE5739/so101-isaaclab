# 현재 프로젝트 구조

기준: 2026-10-23 로컬 작업 트리. 과거 설계는 [Historical 설계안](FUNCTIONAL_REFACTOR_PLAN.md), 실행은 [SYSTEM_FLOW](SYSTEM_FLOW.md), 수정 위치는 [WHERE_TO_EDIT](WHERE_TO_EDIT.md).

패키지 루트: `source/soarm101_lab/soarm101_lab/`.

| 영역 | 실제 소유 모듈 | 책임 |
|---|---|---|
| UI/실행 | workflow/ui.py, operations.py, worker_request.py, paths.py | Path 선택·비동기 준비 확인·프로세스·예약/취소/로그 |
| Source | workflow/mujoco.py, source.py, scripts/teleoperation/record_mimic_dataset.py | 수집 수명·장치/환경 연결·Resume |
| Annotation/DataGen | workflow/annotation.py, datagen.py, mimic.py, unified.py | 판정·수동 입력·동적 배정·공식 Mimic 호출 |
| Dataset | data/episodes.py, recording.py; workflow/dataset.py | Reader/Writer·snapshot 누적·형식 검사·Legacy 변환 |
| Curation/Conversion | workflow/curation.py, curation_ui.py, episodes.py, lerobot_merge.py | 자유 선별/선택적 분할·LeRobot 변환/병합 |
| 학습 | learning/datasets.py, training.py, validation.py | Native 입력 검사·공식 Trainer 실행·저장 Checkpoint 독립 평가 |
| 추론/평가 | workflow/policy.py, evaluation.py | ACT/SmolVLA/MiniVLA 추론·guard·Sim/Real 평가 |
| 엔진 | simulation/environment.py, mujoco_backend.py, playback.py, control.py; workflow/adapters.py | 명시 설정 생성·물리 재생·pause/step·엔진 Adapter |
| Workspace/장치 | real2sim/workspaces/, runtime.py, hardware.py, cameras.py, precision.py | Scene/Physics/보정·장치/앱 수명 |
| Hub | workflow/hub.py | 공식 HF 다운로드·native 모델 폴더 명시 업로드 |
| 모델 핵심 | learning/minivla/{config,backbone,head,checkpoint,policy}.py | 기존 MiniVLA 모델/Checkpoint 추론. 신규 학습·Resume 비활성화 |

Registry 등록 저장소와 독립 Replay Stage는 없다. Curation Preview는 기록 영상/배열을 읽고 Annotation은 공통 Simulation Playback으로 물리를 실행한다. 둘은 다른 기능이다.

`workflow/episodes.py`, `dataset.py`, `mujoco.py`, `unified.py`, `adapters.py`에는 실제 기능이 남아 있다. 모든 파일이 목표 디렉터리로 이동됐다고 주장하지 않는다. 공식 ACT/SmolVLA 네트워크·Loss·optimizer loop는 설치된 LeRobot이 소유한다. Builder 제작은 sibling 저장소의 책임이다.

Script/Terms의 공개 사용법은 [SCRIPT_GUIDE](../task_sdk/SCRIPT_GUIDE.md), 상태 소유권은 [TASK_SDK_DESIGN](TASK_SDK_DESIGN.md)에서 관리한다. `workflow.task_scripts`는 정적 선언/Worker byte 실행, 기존 `SimulationAdapter`는 Runtime binding/Tracker/Event를 소유한다. 새 Registry/Manager/런처는 없다. Native 학습·Hub의 독립 계약은 [NATIVE_LEROBOT](../real2sim/NATIVE_LEROBOT.md)를 따른다.
