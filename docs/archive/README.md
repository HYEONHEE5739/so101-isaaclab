# 과거 설계·검증 기록

현재 아키텍처는 [ARCHITECTURE.md](../real2sim/ARCHITECTURE.md), 데이터 계약은 [DATA_FLOW.md](../real2sim/DATA_FLOW.md), 수정할 파일은 [PROJECT_MAP.md](../real2sim/PROJECT_MAP.md)를 따른다.

| 보존 문서 | 의미 |
|---|---|
| [DUAL_BACKEND_PLAN.md](real2sim/DUAL_BACKEND_PLAN.md) | 2026-10-06의 두 엔진 전환 설계; 현재 구현 지시가 아니다. |
| [MUJOCO_PHASE2.md](real2sim/MUJOCO_PHASE2.md) | 최초 독립 MuJoCo 환경의 실제 검증 이력. |
| [MUJOCOVERSION.md](real2sim/MUJOCOVERSION.md) | 당시 사용자 요구사항 원문. |
| [DATA_FLOW_BEFORE_COMMON.md](real2sim/DATA_FLOW_BEFORE_COMMON.md) | 공통 Episode 이전 NPY/Recorder/운영 설명. |

기존 보정 근거인 root `SCENE_REFINEMENT_HANDOFF.md`는 원래 위치와 내용을 유지한다.
`docs/minivla/versions_before_minivla.txt`는 과거 환경 freeze 기록이며 설치 요구사항으로 사용하지 않는다.
`CODEX_HANDOFF.md`는 로컬 작업 이력이고, 현재 코드 구조의 기준 문서는 위 세 문서다.
