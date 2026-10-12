# Outputs 정리 결과

2026-10-08. 사용자 요청으로 오래된 결과/임시 파일을 실제 삭제했다.
실행 코드, 환경/dependency, 원본 datasets/policy/demo, 독립 Builder 프로젝트는 변경하지 않았다.
삭제 전체 목록과 개별 이유는 `outputs/cleanup_report.json`에 있다.

## 후속 코드 정리: Snapshot만 남기기

2026-10-08. 사용자 요청으로 SO-101 UI에서 없어진 기능의 구현과 호출 경로를 제거했다.
현재 UI → `capture_snapshot` → `SnapshotDataset` → Start/Disconnect 세션 저장을 유지한다.

| 삭제/정리 대상 | 근거와 변경 |
|---|---|
| `real2sim/legacy_capture.py` | 연속 recorder와 paired→continuous 변환. 현 UI 호출 없음. `capture_dataset.__getattr__` 호환 연결도 제거 |
| `real2sim/agent.py` | 없어진 내부 API-agent/자동 보정 UI의 구현. 실제 source/script caller 없음 |
| `real2sim/calibration.py` | 옛 capture dataset/landmark fitting/이미지 metric. Runtime의 옛 rerender 전용 연결을 함께 제거 |
| `real2sim/bootstrap.py`, `references.py` | 제거된 Reference import/bootstrap 기능과 전용 옛 CLI. 현 UI/Workflow 호출 없음 |
| `storage.CaptureStore`, `captures`, `revision` | 옛 paired 저장/탐색/보정 revision helper. 현재 writer와 Workflow에는 caller 없음 |
| Runtime `capture`, `render_dataset`, `export` | 현 UI/Workflow가 보내지 않는 옛 명령. 큐·status·rerender·옛 저장 결과 분기도 제거. Snapshot 외 저장 명령은 미지원으로 거절 |
| Profile fitting/style helper 7개 | 위 삭제 모듈의 전용 함수이며 다른 source/script caller 없음. validation/load/hash/scene pose/active format loader는 유지 |
| `scripts/mujoco/compare_reference.py` | 삭제된 paired 입력/Isaac rerender 결과만 읽는 옛 색상 비교 실험 |
| `scripts/envs/teleoperation/record_lerobot_dataset.py` | current Workflow caller 없음. 이미 존재하지 않는 `soarm101_lab.datasets.LeRobotRecorder`를 import하는 옛 스크립트. 현 Source→HDF5→Convert 경로 유지 |
| `utils/episode_recorder_async.py` | 클래스명/모듈명 import와 caller가 전체 source/script에서 없음 |
| `ui_extension_example.py`, `mdp/rewards.py` | 호출/등록되지 않은 extension 예제와 사용하지 않는 reward 함수 |
| `test_agent.py`, `test_references.py`, `test_capture_dataset.py` | 삭제 기능 전용 검사 제거. Snapshot multi-pose fitting/rollback 검사는 `test_snapshots.py`로 통합 |
| `REAL2SIM_ITERATION.md` | 옛 UI 내부 재구성 프로토콜 안내. 현 Capture guide와 독립 Builder toolkit 문서로 대체 |

파일 14개 삭제, Runtime/storage/profile 안의 옛 코드도 제거했다. source/scripts/tests의
`__pycache__` 30개 디렉터리(179파일, 약 2.42 MiB)도 정리했다. 새 운영 파일/launcher/config는
추가하지 않았다. 실행 안내/구조 지도/기존 검증 harness의 삭제 경로를 갱신했다.

명시적 CLI인 MiniVLA profile/probe/rollout, 기존 ACT inference, 진단/asset 추출 도구는
UI import가 없다는 이유만으로 삭제하지 않았다. 이들은 독립 실행 진입점이다.
Gym task/config는 `tasks.import_packages`와 문자열 entry point로 등록되어 직접 import
검색만으로 삭제할 수 없다. 현재 Real/Sim의 Profile 적용, Teleop, Snapshot, Workspace
Source/Replay/Annotation/Datagen/Convert/Train/Eval/Upload와 학습 데이터/모델은 유지했다.
독립 Builder source와 기존 snapshot/scene/HDF5/policy는 수정하거나 삭제하지 않았다.

### 검증과 한계

- 삭제 모듈/심볼의 source/script import·호출 잔존 검색 0건. Python 179파일 AST parse PASS.
- Snapshot/Runtime/UI/Builder 중심 suite: **143 PASS, 2 SKIP** (37.63 s).
- 범위를 전체 real2sim/minivla/Builder 검사로 넓힌 첫 실행: **266 PASS, 3 SKIP,
  6 FAIL, 16 ERROR** (61.97 s). 모두 동일 프로세스에서 software-EGL 렌더를 import한 후
  physics-only GL 모드를 선택해 guard에 걸린 충돌이었다.
- 해당 physics/통합 두 모듈을 새 프로세스에서 실행: **29 PASS** (9.56 s).
  GL mode guard는 바꾸지 않았다. 두 실행의 test ID를 합치면 **288 PASS, 3 SKIP,
  남은 FAIL/ERROR 0**이다. SKIP은 Blender 실행 경로를 비활성화한 두 검사와 로컬
  Qwen processor 경로가 지정되지 않은 한 검사다.
- 실제 Leader/Follower/USB 장치 연결과 학습 모델의 실제 장시간 학습은 NOT_TESTED다.
  UI/worker 재시작 후 적용된다. dependency/conda는 변경하지 않았다.

결과: 기존 `outputs/implementation/snapshots/startup-fix/`의
`legacy-removal-tests.xml`, `cleanup-full-tests.xml`, `cleanup-physics-tests.xml`.
기계 판독 삭제 목록은 cleanup_report.json의 마지막 `followups`에 있으며 임시 데이터는 `/tmp`에 있다.

## 후속 정리: 기존 captures 전체 삭제

2026-10-08, Snapshot 저장 전환 후 사용자 요청으로 남아 있던
`outputs/real2sim/captures`를 폴더째 삭제했다. 옛 paired capture 14샷의 파일 84개,
논리 크기 약 12.94 MiB(할당 공간 약 13.14 MiB)다.

Builder 샘플 프로젝트에는 옛 원본 경로가 provenance 문자열로 남아 있지만 실제
analysis/render는 프로젝트 내부 `input/media` 사본을 읽는다. 해당 Side/Wrist 두
사본의 SHA-256을 확인했으며 Builder 프로젝트는 수정하지 않았다. 기존 scene의
input_session도 역사적 출처로 남겼고 published profile/hash를 변경하지 않았다.
삭제된 옛 세션을 경로로 지정해 재import하는 작업은 할 수 없다.

현재 UI 캡처는 `snapshots`에 저장한다. 이전 도구 호환용 paired-capture 코드는 이번
결과물 삭제 범위에 포함하지 않았다. Snapshot/scene/원본 datasets/policy/outputs HDF5의
168개 파일 inode·size·mtime은 삭제 전후 동일했다. 결과물만 삭제해 pytest/장치 실행은
하지 않았다. 아래 본문은 **앞선 정리 당시** 기록이며, 당시 보존했던 captures 세션은
이 후속 작업으로 삭제됐다. 기계 판독 기록은 cleanup_report.json의 `followups`에 있다.

## 삭제 결과

- 234개 파일/폴더 경로, 내부 파일 합계 3,200개 삭제. 삭제 실패/건너뛴 항목 없음.
- 파일 논리 크기 약 909.94 MiB, 디스크 할당 공간 감소 약 **923.22 MiB**.
- outputs 할당 용량: 22,889,906,176 → 21,921,837,056 bytes. 약 21.32 → 20.42 GiB.
- Workflow artifact JSON: 42 → 19개. 보존한 결과의 부모 이력은 남겼고 missing parent는 없다.

| 삭제 영역 | 삭제 이유 / 대략적 논리 크기 |
|---|---|
| `real2sim/captures`의 옛 세션 15개 | 현재 scene/Builder reference가 쓰지 않는 이전 paired capture, 45.89 MiB |
| `real2sim/runs`, `renders`, `generated`, `branches`, `analysis`의 이전 결과 | 렌더/최적화 실험과 중간 scene. 최신 엔진별 생성 scene과 최종 provenance 보고서는 보존 |
| `real2sim/environments/workspace_001/.publish_*` | 중단된 publish 사본, 37.96 MiB |
| `real2sim/workspace_runs` | `validation_only=true`인 smoke HDF5/재생 시험, 37.95 MiB |
| `implementation/multipose`와 `implementation/snapshots`의 이전 staging/backup/regression/test-temp | 이미 배포된 코드 사본과 합성 시험 결과, 약 524.18 MiB |
| `mujoco` | 옛 Phase 2 smoke 결과, 0.33 MiB |
| `workflow_validation/act_smoke_v2` | steps=1이 확인된 smoke checkpoint, 158.15 MiB. 실제 학습 모델은 보존 |
| 그 외 `workflow_validation` 결과 | 옛 UI/물리/분기 검증과 episode 없는 HDF5. 등록된 데이터셋과 nonempty HDF5는 보존 |
| `workflow/runs`의 실패/빈 출력·일부 presentation | 완료 데이터가 없거나 UI preview인 파일. 실제 데이터/학습 run은 보존 |
| `workflow/artifacts`의 불필요한 기록 23개 | 원본이 이미 없거나 이번에 삭제한 시험 결과를 가리키며 보존한 결과의 조상도 아닌 기록 |

학습 데이터로 쓸 수 있는 HDF5를 이름만 보고 지우지 않았다. 실제 데이터가 없는 파일,
명시적인 smoke 데이터, implementation의 pytest 합성 fixture만 삭제했다.

## 보존 항목과 이유

- 최신 선택 scene: `real2sim/environments/workspace_001/v2` 전체, profile/task/physics/manifest/asset 포함.
- `workspace_001/v1` 전체: `datasets/workspace_001_source.hdf5` 등의 workspace_path가 v1을 요구하므로
  이전 버전이지만 삭제하지 않았다. HDF5를 재생할 때 임의로 v2로 대체하지 않는다.
- 엔진별 마지막 생성 scene: `generated/scene_mujoco_20261008_011549_…`,
  `generated/scene_20260921_135310_…`. 후자의 날짜가 오래된 이유는 workspace 연결에서 이후
  scene 폴더가 비어 있고, 완성된 USD 파일을 가진 마지막 생성 결과가 이 폴더이기 때문이다.
- `real2sim/captures/session_20260919_121858_…` 한 세션: Builder
  `projects/so101-reference-bb83b89f/input/reference_observation.yaml`의 실제 source 경로이며,
  최신 scene profile의 input_session이기도 하다. captures 전체를 삭제하지 않은 이유다.
- 새 `real2sim/snapshots`, 실제 사진 `references`, texture `assets`, profile `revisions`,
  device/UI 설정, `workspace_tasks`, `bootstrap_inputs`, runtime IPC는 보존.
- 실제 annotation 28 episodes, datagen 50 episodes, clean annotation 4 episodes와 좌표 sidecar.
- 최상위 `datasets/`의 모든 파일. 이름이 gif_test여도 실제 episode가 있으므로 보존.
- 최신 MuJoCo canonical source 1 episode의 NPY/RGB/state/manifest/provenance 전체.
- 등록된 LeRobot 데이터셋 3종 경로와 최신 학습 데이터셋.
- 실제 학습 policy와 10개 학습 checkpoint 전부, `checkpoints/last → 100000`, optimizer/RNG 등 resume 자료.
- 최신 UI/Isaac startup 진단 자료와 기존 test_runner/screenshots/regression_case 도구는
  `implementation/snapshots`에 남겼다. 무거운 복사본·test-temp는 삭제했다.

남은 outputs 용량의 대부분은 실제 annotation/datagen HDF5, MuJoCo RGB 데이터,
학습 checkpoint다. 이번에 공간을 줄이려고 이 데이터를 손실 압축하거나 재작성하지 않았다.

## 검증

- 보존 대상 176개 파일: 삭제 전후 inode/size/mtime 비교, 변경/누락 0개.
  대형 HDF5 전체 SHA를 재계산한 검사는 아니며 파일 내용을 수정하는 명령도 실행하지 않았다.
- workspace v1/v2 loader PASS: manifest에 기록된 각각 27/28개 파일의 SHA와 계약 검증.
- 보존된 HDF5 11개 read-only open/episode 수 확인 PASS. 일부 entry 배열 전체를 읽는
  end-to-end 학습/재생 테스트는 수행하지 않았다.
- 최신 MuJoCo canonical source 구조 검사 PASS, episode count=1.
- latest checkpoint의 모델/optimizer 파일과 last 링크 확인 PASS.
- Registry 19개 중 17개는 원본이 존재한다. 나머지 2개는 보존한 annotation 결과의
  부모 Source 이력이어서 남겼다. 모든 parent ID가 Registry 내부에서 해결된다.
- Builder reference 원본 Side/Wrist 경로 존재 확인 PASS.
- 정리 시작 시 real-only Capture runtime이 RUNNING이었다. IPC/실측 데이터 경로를
  삭제하거나 stop/disconnect/teleop 명령을 보내지 않았다. 마지막 관찰 상태는 CLOSED이며,
  정리 작업에서 runtime 종료 명령을 실행한 것은 아니다.
- 코드 변경이 없어 pytest/Isaac 실행으로 임시 결과를 다시 생성하지 않았다.

이전 구현 보고서에서 링크한 staging/fixture/old validation 결과 일부는 이번 사용자 요청으로
삭제됐다. 문서에 남은 과거 검증 기록과 현재 디스크 파일의 존재 여부를 구분해야 한다.

## 현재 주요 결과 구조

```text
outputs/
  cleanup_report.json
  real2sim/
    environments/workspace_001/{v1,v2}/
    captures/session_20260919_121858_…/  # 필요한 원본 한 세션
    snapshots/                          # 실제 새 snapshot 데이터
    assets/ references/ revisions/
    generated/                          # 엔진별 마지막 생성 scene
    ipc/ ui_settings.json runtime_startup.log
  workflow/
    artifacts/                          # 결과와 부모 관계 19개
    runs/                               # 실제 data/model run
  workflow_validation/
    annotation_ui/annotated_clean.hdf5 + sidecar
    lerobot_phase2/ lerobot_phase2_v2/
    unified_pipeline/conversion/conversion_input.hdf5 + sidecar
    replay_actual_source.json
  implementation/snapshots/             # 최신 소형 검증 자료/도구
```


## 책임별 통합 정리 (최신)

2026-10-08. 현재 실행 코드의 import/caller와 독립 CLI를 확인한 뒤, 같은 책임을
통합했다. 아래는 이번 후속 작업이다. 앞 절의 output 트리는 최초 정리 당시 기록이며
`outputs/real2sim/captures`는 후속 요청에서 이미 삭제했다.

| 기존 위치 | 현재 위치/변경 이유 |
|---|---|
| `learning/minivla/worker.py` | `workflow/mujoco.py`: 모델 정의가 아닌 stage 실행 책임 |
| `learning/minivla/episodes.py` | `workflow/episodes.py`: Source/Annotation/Datagen/Convert의 공통 데이터 처리 |
| `learning/minivla/service.py` | `workflow/operations.py`: 요청 검증·실행 계획·worker dispatch를 공통 operation에 통합 |
| `learning/minivla/runtime.py`, `teleop.py`, `mujoco_env.py` | 엔진/환경/명시적 MJCF adapter/leader target 제한은 `simulation/mujoco_backend.py`; `RobotCoordinates`는 `so101_joint_mapping.py` |
| `learning/minivla/import_hdf5.py` | `workflow/episodes.py::convert_hdf5`와 공통 CLI `import-hdf5` |
| `workflow/overview.py`, `progress.py` | `workflow/presentation.py`: inspection framing/progress/프리뷰 제어를 함께 관리 |
| `simulation/render_profile.py` | `simulation/mjcf.py`: MJCF material/render 상수 통합 |
| `scripts/mujoco/workflow.py`, `scripts/real2sim/workflow_stage.py`, `publish_workspace.py` | `scripts/real2sim/workflow.py`의 `worker`와 `publish`: 같은 stage 요청을 실행하는 launcher 중복 제거 |
| `real2sim/ui.py::ImageLabel` | `workflow/view.py::CameraView`: Capture/Real-Sim/Workflow가 같은 aspect ratio 위젯 사용 |
| `scripts/envs/teleoperation/*.py` | `scripts/teleoperation/*.py`: 불필요한 중간 `envs` 디렉터리와 빈 `__init__.py` 제거 |
| `test_bootstrap.py`, `test_bootstrap_visuals.py` | `test_provenance.py`: 실제 Profile/scene/visual 테스트를 한 책임으로 통합 |
| 루트 `qwen_*` 6개 | `outputs/workflow_validation/vlm_diagnostics/`: 참조 caller 없는 진단 결과를 묶어 보관 |
| 루트 `versions_before_minivla.txt` | `docs/minivla/versions_before_minivla.txt`: 설치 전 환경 기록 보존 |

통합된 옛 파일 13개와 빈 package marker 1개를 삭제했다. Python 파일은 179 → 165개다.
옛 import를 대신 받는 재-export/호환 launcher는 남기지 않았다. 새 독립 실행 파일이나
config를 추가하지 않고 기존 CLI/class/module을 확장했다. `.pytest_cache`도 제거했다.

### 현재 기능별 구조

```text
source/soarm101_lab/soarm101_lab/
  real2sim/                 UI·장치·Runtime·현재 Snapshot·Profile
    workspaces/             published package·Isaac 환경·physics/reset·Mimic adapter
  simulation/               MuJoCo 엔진·MJCF·workspace scene·환경 adapter
  workflow/                 stage 계획·worker·episodes·Registry·preview·train/eval/upload
  learning/minivla/         Qwen/action head·모델 학습·추론·모델 설정
  so101_joint_mapping.py    하드웨어/model/policy 좌표·단위 경계
  assets/ devices/ tasks/ utils/  기존 로봇/장치/등록 task/수학 구현
scripts/
  real2sim/                 UI·상주 Runtime·통합 Workflow CLI·Isaac workspace 실행
  teleoperation/            Teleop·Isaac source 기록
  mimic/                    Isaac annotation/datagen (AppLauncher 경계 유지)
  mujoco/                   독립 physics 검사·USD contact asset 추출
  tools/ inference/ vlm_diagnostics/  명시적 데이터·모델·진단 도구
outputs/
  real2sim/                 기존 scene/profile·장치 IPC·Snapshot 세션
  workflow/                 Registry·실제 stage runs/데이터/checkpoint
  workflow_validation/      독립 검사·기존 보존 학습 입력·Qwen 진단 결과
  implementation/snapshots/ 기존 통합 검증 자료
```

등록 HDF5/checkpoint, 최신 scene, profile/reference asset, 실제 snapshot 데이터는
이동/재작성하지 않았다. 경로는 학습 resume·workspace manifest·Registry의 계약이다.
Snapshot은 Start부터 Disconnect까지 한 세션 안에 공유 robot/calibration과 각 샷을
저장하는 기존 방식이다. Isaac Workspace Replay의 preview는 이제 `--report`의 부모
run에 함께 저장하므로 별도 `real2sim/workspace_runs` 폴더를 생성하지 않는다.
독립 `check_env.py`의 기본 출력도 `workflow_validation`으로 모았다.

### 실행 흐름과 바뀐 명령

UI 실행 명령은 동일하다:

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
python scripts/real2sim/workflow.py --help
python scripts/real2sim/workflow.py worker --request /path/to/request.json
python scripts/real2sim/workflow.py publish --revision /path/to/profile.json --workspace-id ID --task /path/to/task.json
python scripts/real2sim/workflow.py import-hdf5 /path/to/recipe.json /path/to/new_dataset
```

`Window → RuntimeLauncher → real2sim_sim.py`는 실물/Snapshot/Real-Sim 연결이다.
`WorkflowPanel → Operations → worker → backend/episode/policy`는 단계별 실행이다.
Isaac Source/Replay/Annotation/Datagen의 독립 AppLauncher 경계는 유지했다.
MiniVLA의 Frozen/QLoRA × deterministic/flow 네 config와 explicit CLI, 동적 Gym task
등록은 실제 사용 경로이므로 삭제하지 않았다. 독립 Builder source도 변경하지 않았다.

### 최종 검증

- Python 165파일 syntax PASS, 존재하지 않는 project import module 0건, `git diff --check` PASS.
- 첫 core 실행: 258 PASS, 3 SKIP, 1 FAIL. 통합 전 `ImageLabel.source` 내부 속성을
  참조하던 테스트를 표시 pixmap 검사로 바꾸었다. 최종 전체 재실행: **260 PASS, 3 SKIP**.
- GL 모드가 다른 physics/통합 모듈은 별도 프로세스: **29 PASS**.
  최종 고유 test ID 합계: **289 PASS, 3 SKIP, FAIL/ERROR 0**.
- SKIP: 이 suite에서 Blender를 비활성화한 검사 2개, 로컬 Qwen processor 경로 미설정 1개.
- 실제 published Workspace v2로 통합 CLI의 physics-only reset/step와 software-EGL
  Side/Wrist/Overview render 실행 PASS. 세 영상은 640×480 RGB이며 `forbidden_imports=[]`.
- 통합 HDF5 import CLI의 synthetic fixture 변환 PASS; common accept worker가 Torch/MuJoCo/
  Isaac/omni/pxr를 import하지 않는 fresh-interpreter 검사 PASS.
- 보존 Workspace v1/v2 manifest/hash loader PASS. 장치 연결/모터 이동, 전체 Isaac runtime,
  실제 Qwen GPU 로드·장시간 학습은 이번 정리에서 실행하지 않았다. dependency 변경 없음.

증거는 기존 `outputs/implementation/snapshots/startup-fix/cleanup-full-tests.xml`과
`cleanup-physics-tests.xml`을 최신 검사로 갱신했다. 임시 scene/dataset/render는 `/tmp`에 있다.
정확한 이동·통합·삭제 목록과 사유는 `outputs/cleanup_report.json`의 마지막 followup이다.
UI/worker를 재시작하면 새 import와 CLI 경로가 적용된다.


## Builder 전환 후 옛 scene 제작 자료 정리 (2026-10-09)

- `outputs/real2sim/{bootstrap_inputs,references,revisions,runs,generated}` 삭제:
  51파일, 20,036,834 bytes. 옛 scene 제작 입력·보정 결과와 재생성 가능한 runtime 출력.
- Registry `f73be58eb321447b8ae04c6837d4a26f`의 profile은 Workspace v1의
  byte-identical 사본으로 연결했다. ID, fingerprint, 부모 이력과 package hash는 유지한다.
- UI의 bootstrap 기본 profile 자동 선택, MJCF의 workspace 미지정 bootstrap fallback,
  profile schema v1 자동 변환을 삭제했다. `check_env.py --workspace`는 필수다.
- 옛 docs 입력 JSON 2개는 `tests/real2sim/fixtures/{partial_profile,scene_profile}.json`으로
  옮기고 bootstrap provenance를 제거했다. 엔진/FK/카메라/unknown-value 검사용 데이터이며
  production import/load가 없다. 대응 테스트는 명시적 profile을 사용한다.
- `generated`는 Builder 완성본이 아니라 Connect 때 임시로 컴파일한 MJCF/USD였다.
  이제 `ipc/scene_<backend>_<id>/`에 저장하고 worker의 정상 Disconnect 종료 시 지운다.
  rebuild/rollback 중 이전 scene 파일은 worker 수명까지 보존한다.
- Snapshot, 학습 HDF5, 모델, 현재 Workspace v1/v2는 보존한다. `assets`의 컵 texture는
  current Profile의 직접 렌더/패키징 경로가 사용하므로 보존한다.
- `scene.py`, `simulation/mjcf.py`, `workspaces/scene_assets.py`, Workspace publish/accept는
  현재 Real/Sim과 Pipeline의 엔진 실행/패키징 기능이므로 유지한다. 이미지에서 scene을
  추정하는 옛 bootstrap/agent/calibration/reference 알고리즘은 이미 삭제되어 있다.
- 기존 Workspace와 Snapshot의 provenance에 있는 과거 경로는 수정하지 않는다.
  이 이력 문자열을 지우면 immutable package/dataset hash를 손상시키기 때문이다.

검증: 전체 회귀 **267 PASS / 3 SKIP**, 별도 MuJoCo physics/통합 **29 PASS**;
최종 **296 PASS / 3 SKIP / 0 FAIL**. 첫 실행의 옛 기본 profile을 가정한 테스트
1개 실패는 명시적 profile 선택으로 수정했고 최종 suite가 통과했다. 실제 Workspace v2
reset/step/software-EGL RGB PASSED, forbidden imports 0. AST 165파일 / diff check PASS.
Snapshot/Workspace/texture/학습 데이터/모델 등 보호 대상 **228파일 변경 0**.
실제 로봇·USB 장치와 전체 Isaac runtime은 NOT_TESTED. Blender 실행 비활성화 2건과
로컬 Qwen processor 경로 미지정 1건은 SKIP이다. 독립 Builder 코드는 변경하지 않았다.

기존 Registry의 source 기록 2개는 원본 경로가 이미 없고, annotated 결과의 부모 이력으로 남아 있다.
이번 삭제 대상은 그 source 경로와 무관하며 새 누락 경로를 추가하지 않았다.
과거 parent ID를 유지하기 위해 이 기록은 삭제하지 않았다.
검증 XML은 기존 `outputs/implementation/snapshots/startup-fix/cleanup-{full,physics}-tests.xml`
위치에 갱신했다. 전체 삭제 파일/이유와 검증은 `outputs/cleanup_report.json`에 기록했다.
