# SO-101 Real2Sim Documentation

이 폴더는 현재 `so101-isaaclab`의 Real2Sim / Workspace / imitation-learning workflow를 설명하는 **living documentation**이다.

과거 Phase별 작업지시서와 audit 보고서는 현재 구조에 흡수했다. 세부 과거 기록이 필요하면 Git history를 본다.

## 먼저 읽을 문서

1. [ARCHITECTURE.md](ARCHITECTURE.md)  
   전체 구조와 각 모듈의 책임.

2. [DATA_FLOW.md](DATA_FLOW.md)  
   Leader/Follower → model radians → Isaac → HDF5/replay까지 값의 단위와 흐름.

3. [PROJECT_MAP.md](PROJECT_MAP.md)  
   "카메라/랜더마이제이션/HDF5/joint mapping/UI를 어디서 수정하지?"에 대한 파일 지도.

4. [INVARIANTS.md](INVARIANTS.md)  
   Codex/agent가 변경할 때도 깨뜨리면 안 되는 robot, coordinate, dataset, Real2Sim 규칙.

5. [TESTING.md](TESTING.md)  
   실행 성공이 아니라 correctness를 증명하기 위한 테스트/metric 기준.

6. [CHANGELOG_AI.md](CHANGELOG_AI.md)  
   큰 AI-generated 변경에서 문제, 접근, 변경 파일, 테스트, 미검증 영역, 위험을 기록.

## 실행 진입점

### Real2Sim UI

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

UI와 Isaac runtime은 별도 프로세스이며 같은 `--workspace`의 file IPC를 사용한다.

### Isaac runtime

Isaac Lab/Isaac Sim이 설치된 Python 환경에서:

```bash
python scripts/real2sim/real2sim_sim.py
```

### Workspace / workflow

주요 entry point:

```text
scripts/real2sim/publish_workspace.py
scripts/real2sim/workspace.py
scripts/real2sim/workflow.py
scripts/real2sim/workflow_stage.py
scripts/real2sim/check_workspace_physics.py
```

자세한 코드 위치는 [PROJECT_MAP.md](PROJECT_MAP.md)를 본다.

## 현재 Real2Sim 원칙

목표는 실제 SO-101 환경의 관측과 동일 robot configuration의 Isaac Sim 관측을 정량적으로 맞추는 것이다.

우선 보정 대상:

- camera intrinsics / extrinsics
- robot base alignment
- object pose
- table / scene alignment

LLM은 측정하거나 최적화할 수 있는 camera/object/physics 숫자를 이미지에서 임의 추측하지 않는다.
값은 가능한 한 다음 출처 중 하나를 명시한다.

- measured
- calibrated
- optimized
- configured
- provisional / assumed

기존 SO-101 environment, teleoperation, Mimic, datagen, dataset, policy 구조는 가능한 유지한다.

## Repository에서 유지하는 데이터/예제

다음은 문서가 아니라 코드/테스트가 실제 사용하는 자료이므로 유지한다.

- `measurements.example.json`
- `VALIDATION.json`
- `bootstrap/`

이 파일들은 단순 설명 문서처럼 정리/삭제하면 안 된다.

## AI/Codex 작업 규칙

non-trivial 변경 후에는 최소한 다음을 남긴다.

1. 해결하려던 문제
2. 접근 방법과 선택 이유
3. 변경 파일과 이유
4. data-flow / unit / coordinate 영향
5. 실행한 테스트와 결과
6. 검증하지 못한 부분
7. 가능한 실패 조건
8. 사람이 직접 볼 고위험 코드
9. 다음 수정 시 시작할 위치

큰 변경은 [CHANGELOG_AI.md](CHANGELOG_AI.md)에 기록한다.

프로그램이 실행되었다는 사실만으로 correctness를 주장하지 않는다.
