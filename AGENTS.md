# Codex 작업 규칙

- 먼저 `CODEX_HANDOFF.md`와 해당 영역의 문서를 읽고 실제 호출 경로를 확인한다.
- SO-101은 로봇 제어·Snapshot·Workflow 소비자이고, Scene 제작은 sibling `../real2sim-builder`가 담당한다. Builder 원본 export를 수정하지 않는다.
- 기존 Teleoperation, 장치 안전, Snapshot, 두 엔진 경로를 보존한다. UI 시작/탭 전환만으로 장치를 연결하거나 로봇을 움직이지 않는다. 하드웨어 검증은 별도 명시적 요청이 있을 때 수행한다.
- 사용자 Snapshot/사진/HDF5/학습 모델/immutable Workspace를 삭제하거나 metadata를 임의로 바꾸지 않는다. 정리 요청은 명시된 범위만 수행한다.
- 기존 모듈을 확장하고 중복 launcher·임시 버전·새 시스템을 만들지 않는다. 엔진은 별도 worker에서 실행하며 MuJoCo worker에 isaaclab/omni/pxr를 import하지 않는다.
- measured joint와 commanded target을 구분한다. legacy_unverified mapping 및 추정 물성/카메라를 검증 완료로 표시하지 않는다.
- conda dependency 버전을 임의로 변경하거나 설치하지 않는다. GUI·USD 경로를 전체 shell 환경에 섞지 않는다.
- 미커밋 작업이 많다. reset/clean/일괄 checkout/자동 commit을 하지 않는다. 변경 전 상태를 확인한다.
- 기능 변경 후 관련 회귀 검사와 가능한 실제 실행 검사를 수행한다. synthetic/단위 검사와 실물 End-to-End 성공을 구분하여 보고한다.
