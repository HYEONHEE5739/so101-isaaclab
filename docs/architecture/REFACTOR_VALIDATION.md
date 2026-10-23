> Historical / Archived — 2026-10-22. 과거 감사 기록입니다. 현재 코드는 [PROJECT_STRUCTURE](PROJECT_STRUCTURE.md)를 따릅니다.

# 전체 구조 리팩터링 검증

2026-10-18 · 기준/최종 HEAD `7d02d9cf029c27aac3b2c03767e85961bfb650cb`. Commit/Push하지 않았다.

- production 변경은 worker request와 file identity 구현 이동, 그 import 연결뿐이다.
- UI·Runtime·보정·물리·공식 Mimic·모델 수식·학습 loop·저장 schema를 변경하지 않았다.
- 전후 16화면의 픽셀·위젯 트리가 동일하고, 기존 회귀의 실패 수는 증가하지 않았다.
- 변경 전부터 있던 자동 출력 이름 비교 테스트 1개는 그대로 실패한다. 아래에서 별도 설명한다.
- 실제 하드웨어/새 GPU 학습/Isaac Kit 실행/Hub 업로드는 이번 검증으로 수행하지 않았다.

구조 판단은 [감사](REFACTOR_AUDIT.md), 실제 변경 이유/코드는 [전후 비교](REFACTOR_BEFORE_AFTER.md),
개발자 안내는 [구조](PROJECT_STRUCTURE.md) → [흐름](SYSTEM_FLOW.md) → [수정 위치](WHERE_TO_EDIT.md)다.

## 변경 파일과 검증 근거

| 파일 | 변경 | 보존 근거 |
|---|---|---|
| `workflow/worker_request.py` | 신규, 기존 MUJOCO_STAGES/mujoco_request 구현 이동 | 기존 함수 본문 AST 동일·payload/오류/option ownership·fresh import |
| `workflow/operations.py`, `workflow/mujoco.py` | request import/re-export, ROOT/MODEL_ROOT 원래 소유자 연결 | 기존 CLI/help·plan/worker·Qt 선택/READY tests |
| `utils/artifact_identity.py` | 신규, fingerprint 구현 이동 | 고정 bytes·cache/name/tamper·HDF5 외부 영상 중복/경로 오류·AST 동일 |
| `workflow/artifacts.py` | 기존 fingerprint 이름을 동일 함수 객체로 제공 | Registry/hash/Source 분할/저장/Resume 회귀 |
| `workflow/{curation,datagen,dataset,episodes,unified}.py`, `mujoco.py` | hash-only import 위치만 수정 | 전체 기존 production AST의 import 외 동작 동일 |
| `learning/minivla/{data,train}.py` | manifest/옛 Resume의 hash import만 수정 | CPU 데이터·통계·checkpoint/Batch/Resume와 bytes 변조 검사 |
| `test_architecture_contracts.py`, `test_mujoco_workflow_ui.py` | interface/구조 guard 신규, 이동한 validator의 fixture 의존 위치 갱신 | 기존 assertion 유지; production 변경 전 특성화14개 확보 |
| `test_snapshots.py`, `test_provenance.py` | Window/자식 process Registry를 tmp fixture로 격리 | 같은 UI assertion으로2개 PASS, 사용자 Registry 쓰기 방지 |

위 경로의 공통 prefix는 `source/soarm101_lab/soarm101_lab/` 또는 `tests/real2sim/`다.
문서는 `docs/architecture/`7개, 기존 README/ARCHITECTURE/PROJECT_MAP/TESTING/CODEX_HANDOFF에 연결했다.
전체 실제 변경 목록은 검증 폴더의 `changed_files.json`에 있다. scripts/resource/package 설치 설정은 무변경이다.

## 기준선과 최종 결과

아래는 서로 분리한 suite다. 추가적인 부분 재실행과 합산하지 않는다.

| 구분 | 변경 전 | 변경 후 | 확인한 범위 |
|---|---|---|---|
| Native/Workflow/Qt/MuJoCo CPU | 407 PASS,1 deselect | 407 PASS,1 deselect | Registry·Curation·partition·공통 Mimic/예약·Resume·보정·UI·영상/명령 |
| 독립 IK | 27 PASS | 27 PASS | 기존 DLS/SciPy 수치·policy·frame·worker 독립성 |
| Workspace/MuJoCo | 32 PASS | 32 PASS | active objects/reset·보정 residual·Source recorded geometry |
| MiniVLA CPU/native | 50 PASS,2 skip | 50 PASS,2 skip | 실제 CPU tensor·tiny random Qwen/ACT·LeRobot fixture·dataset/Resume·OOM fixture |
| USD/Window/Workspace | 166 PASS,1 FAIL,1 skip,1 deselect | 동일 | USD asset·Builder contract·Window·Snapshot/file IPC·plan/presentation |
| USD 1-camera UI | 1 PASS,39 deselect | 동일 | 기존 한 카메라 Workspace/상위 Window |
| 신규 interface 특성화/guard | 14 PASS(구현 이동 전) | 18 PASS | 기존 import 동일 객체·request/hash 계약·fresh import/방향4개 추가 |
| 사용자 Registry 격리2개 | 첫 하나는 위 suite에서 tmp 기본 root guard로 검사, API 자식검사1개는 보류 | 2 PASS (별도) | 명시 tmp root로 변경 후 기존 assertion 전부 실행 |

Native/Workspace의 Legacy unknown provenance 경고는 기존 실패 정책을 유지한 결과다.
MiniVLA skip2개는 `QWEN_PROCESSOR_PATH` 미지정, USD skip1개는 Blender executable이 없어서다.
실제 pretrained processor/GPU 학습 성공으로 대체 해석하지 않는다.

### 기존 실패 한 개

`tests/real2sim/test_presentation.py::test_modes_same_source_environment`는 두 번의 `plan`에 생성되는
서로 다른 `source_<timestamp>_<UUID>/origin/source.hdf5`까지 같은 argv라고 비교한다.
변경 전과 후 모두 이 출력 경로 비교에서 실패했다. production의 기존 자동 고유 이름을 같게 만들거나
기존 assertion을 약화해서 통과시키지 않았다. 실패 JSON/XML/log를 보존했으며 별도 테스트 설계 검토 항목이다.

### 최초 검증 과정에서 발생한 문제

- 최초 `before_usd` 실행은 MuJoCo 검사도 같은 process에 넣어 engine isolation guard12개에 걸렸고,
  병행 부하로 Snapshot freshness1개, 위 기존 이름 비교1개가 실패했다. guard를 우회하지 않고 process를 나누었다.
  `before_workspace_native`와 `before_usd_isolated`에서 freshness/engine 혼합 실패가 해소됐다.
- request 이동 후 UI mock이 옛 `operations.load`만 바꿔1개 실패, 잘못된 fixture 확장으로 metadata 없는 새 모듈에
  mock을 적용해39 setup error가 발생했다. mock 대상만 수정한 `request_step_final.log`는53 PASS/1 deselect다.
- identity 부분 검증에서 Qt와 TorchCodec을 한 process에 섞어 GLib symbol 오류4개가 났다.
  dependency를 설치/변경하지 않고 학습/native suite를 별도 process로 분리해50 PASS를 확인했다.
- 최초 UI capture helper가 model combo의 이름을 잘못 참조했다. 실제 `architecture` widget으로 고쳐16화면을 확보했다.

초기 오류 로그를 최종 통과 증거로 사용하지 않는다. 관련 파일은 모두 검증 폴더에 보존한다.

## UI·CLI·계산 보존

`outputs/workflow_validation/codebase_architecture/`의 결과를 사용한다.

- `before_ui/`, `after_ui/`, `ui_comparison.json`: 동일 1500×1000, offscreen/PyQt 환경에서 상위4탭·Workflow10작업
  (Train3모델)을 합친16상태의 픽셀 차이0, widget tree/텍스트/크기/순서/표시·활성/선택 값 동일.
- capture는 Registry JSON/UI 설정의 고정 복사본을 사용하며 모든 출력은 검증 fixture에 저장했다.
  readiness debounce를 중지해 정적인 CHECKING 화면을 비교했다. Worker/장치 실행0.
  실제 readiness/설정 저장·복원/단계별 선택·버튼/단축키/실행 잠금은 별도의 기존 Qt tests로 검사했다.
  캡처만으로 모든 사용자 상태나 실제 device 영상 동작까지 증명했다고 주장하지 않는다.
- `code_audit.json`: 변경한 production10파일은 import 및 추출한 정의 외 AST 차이0.
  이동한 두 함수도 import 경로를 제외한 AST가 같다. 기존 나머지 tracked source/resource167파일을 유지했다.
  UI/theme/settings/runtime/precision/physics/Mimic/head/backbone/config·scripts/asset diff는 없다.
- `cli_before.json/cli_after.json`: Workflow, worker, Runtime CLI help/exit/stdout/stderr3개 동일.
  기존 setuptools `find_packages`에서 workflow/utils를 발견하며 설치/리소스 경로 계산은 바꾸지 않았다.
- 전체 wrapper/call 계층·새 dependency/config/production launcher를 추가하지 않았다.
  실제 performance benchmark는 하지 않았다. suite 시간 차이는 부하/환경 차이가 있어 성능 개선 근거가 아니다.

## 사용자 자산과 별도 실행

초기 `outputs/workflow`·`outputs/real2sim`의3258개를 inventory로 보관했다.
metadata/URDF/USD/모델 파일은 SHA256, 다른 numeric/video는 경로·크기·mtime, symlink는 대상을 기록했다.
첫 기준선 직후에는 차이0이었다.

최종 비교 시3256개 기존 파일은 같은 기록이었다. 기존 UI 설정2개는 변경됐고, ACT 학습 실행의 새 파일7개가 추가돼 있었다.
새 Run은 `20261018_010654_train_cube_red_to_cup_a_f53ec143`이며 worker178370의 부모는
사용자 UI177361(그 부모4237), 그 아래 official LeRobot trainer178401이었다. Codex 검증 process와 구분했다.
기존 Source/HDF5/모델/Calibration/Workspace/Registry JSON의 삭제·이동·변경은 비교 inventory에 없었다.
이 외부 실행/사용자 설정을 되돌리지 않았으며 실제 ACT 학습을 이번 자동 검증으로 계산하지 않는다.
시간이 흐르면 진행 중 실행의 출력은 더 늘 수 있다. `asset_comparison.json`은 그 비교 시점의 기록이다.
미추적 원래 로그2개의 이전 수정 시각도 유지했다.

## 재현 명령과 환경

작업 디렉터리는 repository root다. 설치/upgrade 없이 기존 Python3.11 환경과 별도 Qt/USD 경로를 사용했다.
같은 폴더의 `run_suite.py`/`baseline_guard.py`는 검증 전용이며 production launcher가 아니다.
각 `before_*_command.json`, `after_*_command.json`에는 실제 **전체 pytest argv·env·exit code**가 있다.

```bash
TASK_PY=/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python
TASK_CHECK=outputs/workflow_validation/codebase_architecture
"$TASK_PY" "$TASK_CHECK/run_suite.py" after native
"$TASK_PY" "$TASK_CHECK/run_suite.py" after learning
"$TASK_PY" "$TASK_CHECK/run_suite.py" after ik
"$TASK_PY" "$TASK_CHECK/run_suite.py" after workspace_native
"$TASK_PY" "$TASK_CHECK/run_suite.py" after usd_isolated
"$TASK_PY" "$TASK_CHECK/run_suite.py" after usd_extra
PYTHONPATH=source/soarm101_lab "$TASK_PY" -m pytest tests/real2sim/test_architecture_contracts.py -q
"$TASK_PY" "$TASK_CHECK/audit_code.py"
git diff --check
```

`before`는 production 변경 전에 같은 runner로 실행했다. helper 없는 checkout에서 신규 계약 검사만 실행하려면
위 PYTHONPATH pytest 명령을 사용한다. 기존 suite는 command JSON의 파일 목록과 아래 환경을 적용한다.

```bash
TASK_USD=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
QT_QPA_PLATFORM=offscreen \
PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:$TASK_USD \
LD_LIBRARY_PATH=$TASK_USD/bin "$TASK_PY" -m pytest \
  tests/real2sim/test_snapshots.py::test_tab_changes_share_streams_and_do_not_connect_hardware \
  tests/real2sim/test_provenance.py::test_ui_is_api_free_and_loads_profile -q
```

Qt와 학습 TorchCodec의 경로를 섞지 않고, USD/MuJoCo 검사는 독립 process에서 실행한다.
core/native runner는1-camera test를 제외하고 `usd_extra`가 그1개를 담당한다.
자식 UI API 검사는 처음에 사용자 Registry 보호를 위해 제외했고, fixture 수정 후 위 명령으로 실행했다.

## 남은 위험·미검증과 직접 확인

고위험 Runtime lifecycle·Calibration 수학·Workspace task/partition 소유권·DataGen 병렬 실행 재배치는 보류했다.
같은 코드와 회귀는 강한 보존 근거지만 실제 장치 timing·새 GPU/OOM·장시간 policy 성공률·실제 Isaac Kit는 검증하지 않았다.
기존 positive replay cache가 code-file hash를 사용하면 변경된 import 파일로 정상 무효화될 수 있다.
이 개발용 cache 무효화를 피하려고 옛 hash를 유지하거나 재작성하지 않았다. DataGen 자동 Source 재생도 추가하지 않았다.

**내가 직접 확인할 것**

1. 현재 학습이 끝난 뒤 GUI에서 Source/Annotation/DataGen/Train을 이동해 기존 화면, 단계별 입력 묶음/선택, Train 전용 Loss Graph를 확인한다.
2. `worker_request.mujoco_request`와 `utils.artifact_identity.fingerprint`를 읽고 두 기존 import 경로가 동일 객체인지 신규 tests를 실행한다.
3. 기존 저장 데이터/Checkpoint로 하던 작업을 명시적으로 실행할 때 Run request/state/log와 원본 경로를 확인한다.
   실행 중인 ACT Run은 그대로 두고 이번 CPU/Qt 검증의 결과와 구분한다.
