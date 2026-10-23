# MiniVLA 현재 지원 범위

기준: 2026-10-22. 기존 모델 핵심과 `.pt` Checkpoint 추론은 유지한다. 신규 학습·학습 Resume는 비활성화됐고 전용 train.py/data.py는 삭제됐다.

실제 소유자는 learning/minivla의 config.py·backbone.py·head.py·checkpoint.py·policy.py다. Workflow 추론/guard는 workflow/policy.py, Sim/Real 평가는 workflow/evaluation.py가 연결한다. probe/profile은 모델 진단이며 학습 실행 대체 경로가 아니다.

확장 위치는 [기능별 수정 위치](../architecture/WHERE_TO_EDIT.md), 현재 학습 계약은 [실행 흐름](../architecture/SYSTEM_FLOW.md)을 본다. 이 디렉터리의 DESIGN/DATA_CONTRACT/VALIDATION/EVALUATION은 과거 설계·실험 기록으로 보존한다. 과거 학습 명령·Source Group 강제를 현재 기능으로 사용하지 않는다.
