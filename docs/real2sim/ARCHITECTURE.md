# Real2Sim 구조

현재 기준: 2026-10-22. 중복 구조 표 대신 다음 문서를 사용한다.

- [현재 모듈 구조](../architecture/PROJECT_STRUCTURE.md)
- [UI → Worker → 데이터/학습 실행](../architecture/SYSTEM_FLOW.md)
- [기능별 수정 위치](../architecture/WHERE_TO_EDIT.md)
- [남은 의존성](../architecture/DEPENDENCIES.md)

장치 연결·Calibration·Snapshot 수명은 real2sim/runtime.py가, Workflow 실행은 workflow/operations.py가 소유한다. Builder Workspace는 읽기 전용 소비자 계약을 따른다.
