# AI Change Log

AI/Codex가 큰 변경을 만들 때 사람이 전체 diff를 처음부터 읽지 않아도 변경의 의도와 검증 근거를 추적할 수 있도록 남기는 로그다.

## 작성 규칙

- 사소한 typo/UI 문구 변경은 생략 가능하다.
- 구조/데이터/robot/camera/dataset/workflow 의미가 바뀌는 작업은 기록한다.
- 새 항목은 위에 추가한다.
- "완료"가 아니라 **무엇을 증명했고 무엇은 아직 모르는지**를 적는다.

## 템플릿

```md
## YYYY-MM-DD — <change title>

### Problem
무엇을 해결하려 했는가.

### Approach
어떤 구조로 해결했고 왜 이 방법을 선택했는가.

### Changed
- path/to/file.py — 변경 이유
- ...

### Data-flow impact
입출력/단위/좌표계/schema가 바뀌었는가.

### Validation
- command/test:
- result:
- before/after metric:

### Not verified
실제 hardware, Isaac runtime, held-out data 등 확인하지 못한 부분.

### Risks / failure modes
어디에서 틀릴 수 있는가.

### Human-review hotspots
사람이 직접 볼 파일/함수와 이유.

### Future edit point
다음에 같은 기능을 수정할 때 어디부터 보면 되는가.
```

## 2026-09-21 — Documentation consolidation

### Problem
Real2Sim 문서가 Phase 작업지시, audit 보고서, 현재 사용법으로 분산되어 현재 구조와 수정 위치를 찾기 어려웠다.

### Approach
현재 코드 중심의 living documentation으로 통합했다.

### Changed
- `README.md` — 문서 진입점과 현재 실행 개요
- `ARCHITECTURE.md` — 모듈 책임과 시스템 구조
- `DATA_FLOW.md` — joint/camera/dataset/workspace 데이터 흐름
- `PROJECT_MAP.md` — 수정 목적별 코드 위치
- `INVARIANTS.md` — 깨뜨리면 안 되는 계약
- `TESTING.md` — 증거 중심 테스트 전략
- `CHANGELOG_AI.md` — 이후 AI 변경의 검증 기록 형식

과거 Phase 구현지시/audit/status MD의 핵심 현재 규칙은 위 문서들에 흡수하고 중복 문서는 제거한다.

### Validation
문서가 참조하는 현재 주요 source/workflow/workspace/test 경로를 main branch에서 확인했다.

### Not verified
이 문서 정리 자체는 runtime code를 변경하지 않으므로 Isaac/hardware test를 실행하지 않았다.
