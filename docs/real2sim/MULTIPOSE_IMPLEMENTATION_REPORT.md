# Snapshot Multi-Pose 구현 결과

검사일: 2026-10-08. 이번 요청의 구현은 Snapshot 중심으로 기존 경로를 수정했다.
의존성 설치/업그레이드, conda 변경, 실제 장치 구동, 사용자 데이터 삭제는 하지 않았다.
아래 수치의 보정 사례는 합성 fixture이며 실제 SO-101 정합 성능으로 해석하면 안 된다.

후속 정리에서 옛 paired/continuous capture, 변환기, 내부 Reference/API agent/fitting 코드를
제거했다. 아래 초기 구현의 "호환 코드 유지" 기록은 현재 상태가 아니며,
[OUTPUT_CLEANUP_REPORT.md](OUTPUT_CLEANUP_REPORT.md)의 **후속 코드 정리**가 최신 상태다.

## A. UI 변경

| 이전 | 현재 |
|---|---|
| Capture Record/Stop Recording/Save/Event | Side/Wrist REAL 두 화면 + 현재 상태 캡처 + 개수/결과/목록 |
| 연속 프레임·demo/event 수집 | 한 번 클릭하면 한 observation 자동 저장 |
| Inspect/Profile JSON editor/수동 fitting/export UI | Inspect 제거, 필수 Profile/장치 설정은 Real/Sim 연결 설정으로 이동 |
| Real/Sim paired Capture 버튼 | 제거, 기존 4-view 유지 |
| Connect/Start/Stop/Disconnect | 안전상 유지, 명칭 Start Teleop/Stop Teleop으로 명확화 |
| Capture에 Sim 필요 | Capture 자체 Start로 real backend 연결, 엔진 없이 동일 runtime/device/camera 사용 |
| Workspace Pipeline | 기존 panel/기능 유지, 탭 번호만 3으로 변경 |

Connect는 기존 보정 확인 후 leader torque를 해제하지만 follower 새 목표는 보내지 않는다.
Start는 실측 hold/readback 확인과 torque 활성화 후 leader 추종을 시작한다. Stop은 새 목표를
중지하고 follower가 마지막 목표를 유지하게 한다. Disconnect는 기존 장치를 닫는다.
따라서 Connect/Disconnect 두 버튼으로 합치지 않았다. Capture/탭 전환은 새 목표나 장치
연결을 만들지 않는다. 실제 하드웨어 동작은 NOT_TESTED이며 mock + 파일 IPC로 검사했다.

화면 검증:

- [Capture 화면](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/ui_capture.png)
- [Real/Sim 4-view](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/ui_real_sim.png)
- [유지된 Workspace Pipeline](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/ui_pipeline.png)

위 화면은 기존 이미지로 만든 offscreen layout 사례다. 실제 live 장치를 연결한 화면이
아니며 상단에 prerecorded / hardware NOT_TESTED로 표시했다. 저장/연결 설정은 접어서
영상 공간을 확보한다. 실제 UI를 재실행해야 새 화면을 사용한다.

## B. Snapshot 저장과 데이터 계약

`real2sim.snapshot/1`을 사용한다. 기본 저장 위치는 UI workspace/snapshots 아래이며:

```text
snapshot_dataset_<time>_<uuid>/
  capture_manifest.yaml
  calibration/{cameras,sensors}.json
  robot_model/robot.urdf
  robot_model/<hash>.stl
  snapshots/snapshot_000001/{side.png,wrist.png,state.json,metadata.json,COMPLETE}
  snapshots/snapshot_000002/...
  COMPLETE
  .capture.lock
```

공통 모델/calibration은 한 번만 복사한다. 모델/source Profile/hash와 장치 calibration
identity가 다른 상태를 기존 dataset에 조용히 섞지 않는다. 연결을 끊었다 재시작한 뒤
manifest가 있는 기존 dataset 폴더를 출력 위치로 선택하면 호환성/기존 hash를 확인하여
이어 저장한다. 원본 snapshot은 덮어쓰지 않는다.

Snapshot은 이미지 두 개와 전체 실측 observation이다. Native follower feedback, 모델
q_measured, nullable q_command, command timestamp, explicit joint 순서/단위/mapping,
그리퍼 좌표의 한계, stability history, 각 이미지 원본 해상도/K/distortion/mount/parent,
sequence/read interval/blur/skew/정렬된 measured q, Scene state와 파일 hash를 보존한다.
Capture 요청 처리 시 관측을 복사하며 writer 지연으로 나중 관측을 저장하지 않는다.

Host monotonic read timestamp는 exposure timestamp가 아니다. Arm은 100 ms 이내 bracket
실측으로 보간하고 gripper는 마지막 실측을 사용한다. Missing/moving/stale >200 ms/skew
>100 ms/단위·mapping·해상도 오류는 저장을 거부한다. Blur variance <10은 경고와 함께
저장하되 fitting에서 제외한다. Textureless 이미지도 이 heuristic에 걸릴 수 있다.

PNG/state/metadata를 flush한 complete 폴더와 원자적 manifest index로 publish한다.
실패는 완료 개수에 포함되지 않으며 이전 index를 유지한다. Crash 잔여 미등록 폴더는
덮어쓰거나 유효 관측으로 사용하지 않고 다음 ID를 확보한다. Import는 shared lock에서
완료 prefix를 복사하며 새 snapshot만 추가한다. 파일·측정값 변경은 freshness signature를
변경한다. 모델/calibration에는 공통 hash, snapshot에는 metadata/image/state hash가 있다.

Scene state는 사용자가 물체/base 배치를 유지했다는 선언이다. 자동 물체 이동 검출은
없다. 물체를 옮기면 복원하거나 다른 state를 사용해야 하며, Builder는 다른 state를 한
static scene으로 합치지 않는다. Train/validation 지정은 import 시 한다.

## C. Builder 연결

SO-101 센서 코드나 제어 코드를 Builder로 import하지 않는다. 독립 importer가 portable
URDF/MJCF, joint order/unit/hash와 각 이미지 시점의 실측 q를 검증해 기존 observation
경로에 연결한다. Event 파일은 필요 없다. 동일/유사 q를 train/heldout으로 나누면 거부하고
같은 split의 반복 자세는 그룹을 공유한다. 기존 split은 재할당하지 않는다.

```bash
cd /home/hyeonhee/soarm_isaaclab/real2sim-builder
python -m app.cli new 'SO101 snapshots'
PROJECT='projects/so101-snapshots-REPLACE_ID'
python -m app.cli import-snapshots "$PROJECT" /absolute/path/to/snapshot_dataset_... \
  --scene-state static_001 --validation snapshot_000009 snapshot_000012
python -m app.cli inspect "$PROJECT"
```

프로젝트/경로/ID는 실제 결과로 바꾼다. 한 snapshot도 import는 가능하지만 보정 성공
조건은 충족하지 못한다. Scene state가 하나이면 --scene-state는 생략 가능하다.
기존 snapshot을 추가 import할 때 이미 지정한 validation ID도 다시 넣는다.

`--kind reference`는 핸드폰 사진을 agent context로 등록한다. 알 수 없는 camera의 사진을
side-camera calibration/pixel 비교에 섞지 않는다. Reference 이미지는 복사·hash·inspect로
읽을 수 있지만 fitting/rectification에서 제외한다. 기존 `--kind image`는 calibrated
camera 관측용이며 동작은 유지한다.

## D. Codex agent loop와 보존한 알고리즘

```text
Reference photos / brief / measurements + snapshots
→ Codex가 이미지와 현재 scene/metrics를 읽고 가설 선택
→ candidate edit 또는 Blender author/sync
→ 한 shared parameter group fit
→ 모든 snapshot별 measured-q FK / camera pose / render
→ train + 독립 holdout compare / nonregression
→ promote 또는 reject / rollback / 다른 가설
→ fresh final validate
→ dual export 또는 NEEDS_MORE_DATA
```

```bash
export REAL2SIM_BLENDER=/home/hyeonhee/apps/blender/blender
python -m app.cli add "$PROJECT" /data/photos/overview.jpg --kind reference
python -m app.cli add "$PROJECT" /data/brief.md --kind brief
python -m app.cli add "$PROJECT" /data/measurements.yaml --kind measurements
python -m app.cli add "$PROJECT" /data/correspondences.yaml --kind measurements
python -m app.cli candidate "$PROJECT" create wrist_01
python -m app.cli candidate "$PROJECT" fit wrist_01 --target camera:wrist
python -m app.cli candidate "$PROJECT" render wrist_01 --engine mujoco
python -m app.cli candidate "$PROJECT" evaluate wrist_01 --engine mujoco --validation
python -m app.cli candidate "$PROJECT" evaluate wrist_01 --engine blender --validation
# persisted held-out nonregression 통과 후에만:
python -m app.cli candidate "$PROJECT" promote wrist_01
python -m app.cli validate "$PROJECT" --engine blender
python -m app.cli export "$PROJECT"
```

Fit은 weighted soft-L1 bounded least squares, train-only objective/Jacobian rank,
2개 이상 서로 다른 train pose/8개 이상 train correspondence/독립 heldout 조건을 유지한다.
Camera/K/base/object pose/dimension을 그룹별로 보정한다. Measured/calibrated 값은 잠그고
q_measured는 최적화하지 않는다. Wrist는 base × FK(q) × mount, 두 renderer 모두 각
observation에 해당하는 robot/camera를 사용한다. 기존 reprojection/MAE/silhouette IoU/
center-error heldout gate와 candidate freshness/rollback을 유지한다.

CLI 수치 최적화는 deterministic tool이다. 시각적 원인 분석과 다음 가설 선택은 외부
Codex가 한다. 내부 LLM/semantic detector를 새로 넣지 않았다. FK의 3D link/landmark는
계산할 수 있지만 실제 사진의 2D uv/mask는 자동 탐지하지 않는다. 실제 측정/확인된
correspondence가 필요하며 Codex 제안 픽셀·geometry는 추정값으로 기록해야 한다.

Metrics와 이미지:
`scratch/agent/<name>/renders/<engine>/metrics.json`,
`previews/<index>_<camera>/{real,sim,overlay,difference}.png`, `posed/<index>/`.
Candidate scene/fit/signature가 남으므로 같은 이름을 다시 열어 이어 진행할 수 있다.
Blender workspace를 가진 후보의 canonical 값을 바꿨다면 build/sync freshness도 필요하다.

## E. 제거와 Legacy 호환

- UI: Record/Stop Recording/Save/Event/Mark Event/timer와 전용 handler 제거.
- UI: Inspect/editor/수동 reference fitting/export widgets/slots와 paired Capture 버튼 제거.
- Runtime: dataset_record/dataset_stop/dataset_mark/dataset_save 및 매 tick 연속 append 제거.
- Builder: active physics-replay CLI/stub 제거. Snapshot 경로에는 grasp/reset/moved filtering 없음.
- 기존 Continuous CaptureSession/convert_paired는 `real2sim/legacy_capture.py`로 이동하여
  명시적 tools/tests에서만 사용한다. 옛 capture_dataset import도 lazy forwarding으로 호환한다.
- 기존 COMPLETE continuous/paired captures와 Builder projects/reference는 삭제·수정하지 않았다.
  Builder의 과거 event exclusion은 legacy_static_reasons에서 legacy 입력에만 적용한다.
- Legacy paired store는 명시적 옛 capture API를 호출할 때만 생성한다. 실측-only worker를
  열기만 했을 때 불필요한 빈 legacy capture session을 만들지 않는다.

기존 backend calibration/profile/reference/paired storage 함수는 사용처가 있어 남겼다.
학습 Source/Annotation/Datagen/Train flow의 episode/event 계약은 이번 snapshot 수집과
별개이며 제거하거나 재구현하지 않았다.

## F. 변경 파일과 책임

SO 패키지 기준은 `source/soarm101_lab/soarm101_lab/`이다.

| 파일 | 이번 변경 |
|---|---|
| `real2sim/ui.py` | Snapshot UI, Inspect 제거, 공유 streams/4-view, 간단 연결 설정 |
| `real2sim/runtime.py` | real-only branch, 관측 freeze/비동기 save, snapshot IPC, legacy 저장 lazy 생성 |
| `real2sim/launcher.py`, `scripts/real2sim/real2sim_sim.py` | 기존 진입점에 real backend 추가, 새 launcher 없음 |
| `real2sim/capture_dataset.py` | atomic portable snapshot 저장/정렬/quality/hash/resume |
| `real2sim/legacy_capture.py` (신규) | 기존 recorder/paired 변환의 명시적 호환 경로 |
| `tests/real2sim/test_snapshots.py` (신규) | producer↔importer·무결성·UI·비구동·renderer·CLI lifecycle 검사 |
| 기존 capture/desktop/references/provenance/outside_limits 테스트 | 새 UI assertion, 기존 backend/safety 검사 유지 |
| `docs/real2sim/CAPTURE_DATASET.md`, `MULTIPOSE_IMPLEMENTATION_REPORT.md`, `README.md`, `ARCHITECTURE.md` | 실제 사용 및 한계 |

독립 Builder:

| 파일 | 이번 변경 |
|---|---|
| `app/core/capture.py` | snapshot 검증/append import/split/state guards, 기존 FK 경로 재사용 |
| `app/core/evidence.py`, `app/cli.py` | import-snapshots, context-only reference, physics-replay 제거 |
| `app/calibration/preprocess.py`, `app/reconstruction/analysis.py` | context-only 사진에 calibrated camera를 가정하지 않음 |
| `app/exporters/package.py` | scene state와 snapshot pose/split/time/hash metadata |
| `tests/test_multipose.py` | SO 패키지 없이 generic MJCF snapshot import/FK 검사 |
| README, ARCHITECTURE, MULTIPOSE_CAPTURE 문서 | Snapshot workflow와 legacy/loader 경계 |

기존 FK/fitter/candidates/renderers/exporters를 통째로 새로 작성하지 않았다.
삭제된 source 파일은 없으며, 기존 recorder 책임만 legacy 모듈로 이동했다.
변경 전 SO 코드와 Builder 원본 백업/검증 해시는
`outputs/implementation/snapshots/before/`, `builder-change-manifest.json`에 있다.
기존 다른 작업의 git diff는 보존했으며 위 목록 밖 변경을 이번 작업 성과로 포함하지 않는다.

## G. 테스트와 실행 증거

- 적용된 두 프로젝트 전체 회귀: **162 PASS, 0 FAIL, 0 SKIP**, 82.79 s.
  [JUnit XML](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/deployed-tests.xml).
- 마지막 Runtime 저장 lazy 생성/CLI boot 검사 후: **23 PASS**, 나머지 140 deselected.
  [Runtime XML](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/runtime-tests.xml).
  새 CLI case 포함 총 163개의 서로 다른 검사 항목을 이 두 실행에서 검증한다.
- 실제 Blender 실행 포함: snapshot/legacy 각각 measured pose별 render 차이, MuJoCo
  software-EGL rendering, USD binding load, dual structural parity, candidate promotion/
  rollback/heldout regression/locked parameters, 품질·hash·transaction/resume 검사.
- 새 Python에서 isaaclab/omni/pxr/mujoco import를 강제로 막아 real runtime init와 기존
  worker CLI의 IDLE → Disconnect → CLOSED/exit 0을 확인한다. 실제 장치는 연결하지 않는다.
- Mock Leader/Follower: Connect에서 follower activate/send 없음, Start 후 기존 native
  목표 전달, Stop 후 추가 목표 없음, Disconnect close. Capture는 hardware 접근 금지
  객체로 검사하며 request의 관측을 저장하고 나중 자세로 바꾸지 않음을 확인한다.
- 첫 실행에서는 UI 테스트 mock의 responsive/session 계약 부족으로 Qt callback이 abort했다.
  Mock을 실제 IPC 계약에 맞게 고쳤다. Sim-limit 회귀 stub에도 새 measured history를 추가했다.
  초기 실패를 숨기지 않았으며 최종 실패는 없다.
- 의존성 설치/업그레이드 없음. EGL probe driver 경고가 있었지만 software-EGL 렌더/로드는
  성공했다. GPU EGL/OSMesa를 이 결과로 검증했다고 주장하지 않는다.

실행 사례:
[regression/report.json](/home/hyeonhee/soarm_isaaclab/soarm101_lab/outputs/implementation/snapshots/regression/report.json).
합성 Snapshot 3개(카메라 관측 6개), train 2 pose / heldout 1 pose, wrist mount 25 mm 오류:

- Jacobian rank 6/6.
- Train reprojection RMSE: 16.6399 px → 약 5.85e-12 px.
- Heldout RMSE: 16.6399 px → 약 5.81e-12 px.
- Nonregression 통과 → promote(DRAFT) → 실제 Blender validation → XML/USD export/verify.
- Structural parity PASS. 물리/model mapping은 unverified이므로 최종 package는 **UNVALIDATED**.
  이 오차 수치는 synthetic ground truth에서 얻은 도구 검증값이며 실측 성능이 아니다.

실제 기존 SO-101 reference/paired capture의 **94개 파일 hash 불변**도 확인했다.
별도 새 프로젝트로 import/render/export가 되지만 wrist blur, 독립 holdout/geometry evidence/
verified mapping 부족으로 NEEDS_MORE_DATA 및 UNVALIDATED 상태다. 원본은 수정하지 않았다.

실제 Leader/Follower/USB 카메라 통합, 실제 물리 관절 대응, Isaac runtime/접촉/그립:
**NOT_TESTED**. 이번 성공을 physics replay나 pick/place 성공으로 표현하지 않는다.

## H. 처음 실제 Snapshot 하나 수집하기

```bash
cd /home/hyeonhee/soarm_isaaclab/soarm101_lab
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

1. Workspace Pipeline에서 실행 Python을 `/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python`
   으로 선택한다. 기존 calibrated Profile/Workspace를 사용하고 provisional bootstrap을
   실측이라고 취급하지 않는다.
2. 공통 연결 설정에서 Profile, 실제 포트/ID/calibration, 두 카메라를 입력한다.
   Capture 탭의 **Start · 카메라/로봇 연결** → REAL/feedback 상태 확인. Sim은 실행하지 않는다.
3. 같은 탭에서 Start Teleop → 직접 조작 → 안정적으로 정지 → 저장 설정에서 output 선택
   → **현재 상태 캡처** → Saved/경로/경고 확인. 별도 Save 버튼은 없다.
4. 아래 환경/경로로 Builder에 한 snapshot을 import한다.

```bash
cd /home/hyeonhee/soarm_isaaclab/real2sim-builder
# 기존 환경. 아래 env는 이 프로세스에만 적용되며 conda를 수정하지 않는다.
PY=/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python
"$PY" -m app.cli new 'SO101 captured scene'
PROJECT='projects/so101-captured-scene-REPLACE_ID'
"$PY" -m app.cli import-snapshots "$PROJECT" /absolute/path/to/snapshot_dataset_...
"$PY" -m app.cli inspect "$PROJECT"
```

한 snapshot이면 train-only import이며 fit/validate는 추가 자세·heldout·correspondence를
요청한다. 이후 다양한 pose와 별도 measurements/reference/brief를 제공하고 Codex에
보정 반복을 요청한다. Renderer/dual USD 검사에 필요한 기존 라이브러리 경로는
Builder `docs/LOCAL_RUN.md`에 있으며 환경을 설치/변경할 필요 없다.

## 최종 package와 현재 loader 한계

동일 canonical scene에서 scene.yaml/XML/USD/manifest/SCENE.md/공유 visual·collision/
comparison images/validation report를 export한다. Pose/K/frames/object IDs/role/dimension/
asset·engine mapping/scene-state/snapshot provenance와 uncertainty를 포함한다.
Normal export는 fresh validation PASS가 필요하며, 검토용은 --allow-unvalidated로 명시한다.

**현재 SO-101 Real/Sim UI는 generic Builder package를 직접 로드하지 않는다.**
Profile so101.real2sim/2 또는 so101.workspace/1을 사용하며, Builder의 reference robot은
posed geometry/FK model이고 runtime controller가 아니다. 최종 scene.xml/scene.usd를
선택해 제어 가능한 SO-101 task/camera/robot에 연결하려면 downstream adapter가 필요하다.
이번 UI 중심 작업에서 복잡한 loader를 억지로 추가하지 않았다. 기존 Profile/Workspace
Isaac/MuJoCo 4-view는 유지한다. Pixel-identical 엔진이나 Isaac runtime 검증을 주장하지 않는다.

## 추가 수정: 명시적 Start와 오래된 runtime 복구

UI를 새로 열었을 때 이전 Isaac runtime이 10시간 이상 BUILDING_ENVIRONMENT 잠금을
잡고 있어 Capture 화면과 backend 선택이 막히는 문제를 수정했다. 실제로 새 UI가 engine을
자동 생성하는 코드가 있던 것이 아니라 이전 상태를 자동으로 받아 선택/화면을 잠그는
문제였다. 기존 engine 프로세스는 실제 장치 FD가 없음을 확인한 복구 경로에서 종료했다.

- 처음에는 연결/제어/heartbeat를 시작하지 않는다. 기존 runtime 상태를 자동 접수하지 않는다.
- Capture 자체 Start는 항상 real-only이며, Sim backend/Pipeline Workspace 선택을 사용하지 않는다.
- 두 탭은 공통 Profile/장치 설정과 단일 runtime을 사용한다. 추가 장치 controller는 없다.
- Real/Sim 상단의 backend 선택은 이전 프로세스 때문에 잠기거나 바뀌지 않는다.
- 모드 변경은 기존 runtime 종료 완료를 기다리고 새 worker를 시작한다. 자동 Teleop은 없다.
- BUILDING 중 Disconnect도 동작한다. script/workspace/UID/상태/열린 실제 장치 FD를 검사하고
  장치가 없는 engine 초기화만 취소한다. 실제 연결된 장치는 기존 graceful Disconnect를 사용한다.
- UI 종료 시 미처리 연결 예약을 취소하며 다음 UI 시작에서 연결을 재개하지 않는다.
- Start Teleop은 연결이 끝난 뒤에만 활성화한다.

실제 Leader/Follower/USB 장치 검증은 여전히 NOT_TESTED다. 수정한 UI는 재실행해야 적용된다.

이번 수정 검증: 관련 pytest **32 PASS, 137 deselected** (9.51 s).
초기 UI/탭 변경 시 worker/장치 명령/heartbeat 미실행, Capture의 real-only 강제,
오래된 상태와 backend 선택 분리, worker 종료 후 모드 변경, 연결 취소 후 Start 재활성화,
기존 real-only 및 MuJoCo worker 동작을 확인했다. 기존 conda/dependency는 변경하지 않았다.

- 결과: `outputs/implementation/snapshots/startup-fix/tests.xml`
- 초기 Capture 화면: `outputs/implementation/snapshots/startup-fix/ui_capture_idle.png`
- 초기 Real/Sim 화면: `outputs/implementation/snapshots/startup-fix/ui_real_sim_idle.png`
- 위 화면은 offscreen UI 검증이며 실제 장치 연결 성공을 의미하지 않는다.

## 추가 확인: Isaac 환경 생성 지연과 재연결 정지

2026-10-08 기존 `outputs/real2sim/runtime_startup.log`와 Kit 상세 로그를 대조했다.
첫 실행은 scene 생성 0.813 s, simulation 시작 1.771 s로 완료했다. 이후 실제 Side
카메라의 요청 width 640 / actual width 160 불일치로 연결에 실패했다. 약 09:29:18
재연결에서 두 번째 ManagerBasedEnv/context 생성이 시작됐으나 이후 진행 로그가 없었고,
앞선 복구에서 20:14:01에 종료됐다. 오래 남아 있던 BUILDING_ENVIRONMENT는 초기
geometry 생성이 계속 계산 중이었던 증거가 아니다. 두 번째 context/RTX 초기화 정지로
판단되지만, 내부 native thread의 정확한 deadlock 원인은 stack dump 없이 확정하지 않는다.

확인 중 MuJoCo용 `--renderer` 기본값 `egl`이 Isaac AppLauncher에도 전달되는 오류를
찾았다. 설치된 AppLauncher/SimulationApp 소스는 이 문자열을 `/rtx/rendermode`에 설정한다.
Isaac에서는 해당 옵션을 전달하지 않아 기존 Isaac 기본 RaytracedLighting을 사용한다.
이 오류가 위 정지의 직접 원인이었다고 단정하지 않는다.

변경 사항:

- `real2sim/launcher.py`: Isaac 환경이 이미 있는 worker에 다시 Connect하는 경우 graceful
  Disconnect 후 기존 worker 종료를 기다리고 새 worker를 실행한다. 카메라 연결 실패 뒤
  동일 Kit 프로세스 안에서 context/RTX 환경을 재생성하는 경로를 피한다.
- `scripts/real2sim/real2sim_sim.py`: MuJoCo GL renderer 옵션과 Isaac RTX renderer를 분리하고
  AppLauncher 시작 시간을 기록한다.
- `real2sim/runtime.py`: 기존 환경 close, config/assets, ManagerBasedEnv 초기화,
  reset/profile/camera warmup의 시작/소요 시간을 stdout에 기록한다. 초기화와 warmup
  단계는 IPC 상태 메시지에도 표시한다.
- camera 해상도/보정 및 CPU 전력 설정은 변경하지 않았다. 기존 실제 camera 오류는
  장치 지원 모드와 calibration을 확인해야 하며 이번 무장치 프리뷰 검증으로 해결을 주장하지 않는다.

관련 자동 테스트 **37 PASS, 137 deselected** (9.45 s). 기존 real-only/MuJoCo worker,
Isaac 재연결의 종료 순서, RTX로 EGL 인자가 전달되지 않음을 포함한다.

같은 기존 `workspace_001/v2`로 실제 RTX 4060 GPU에서 두 번의 headless preview를 실행했다.
두 경우 모두 hardware={} / preview_only=true / real_camera_preview=false를 사용했다.
실제 로봇/USB 카메라 연결과 물리 명령은 없으며, 물리 장치 동작은 여전히 NOT_TESTED다.

| 단계 | 첫 프리뷰 | 재연결 프리뷰 |
|---|---:|---:|
| 전체 프리뷰 준비 (재연결은 이전 worker 종료 포함) | 13.539 s | 15.867 s |
| AppLauncher / Kit / RTX | 9.541 s | 9.468 s |
| Scene config / assets | 0.066 s | 0.063 s |
| ManagerBasedEnv / context / scene / cameras | 2.949 s | 2.606 s |
| Reset / profile / camera warmup | 0.358 s | 0.338 s |

두 프리뷰 모두 PREVIEW와 SIM Side/Wrist 이미지 생성을 확인했고, 재연결은 서로 다른
worker PID/session에서 실행됐다. 종료 후 owner lock 해제도 확인했다. 위 단계별 시간은
모든 Python import/IPC/종료 시간을 포함하는 전체 시간과 합이 다르다. 동일 조건 두 번의
측정이며 모든 환경에서 동일한 시간을 보장하거나 수정 전후 속도 향상을 입증하지 않는다.
가장 큰 확인된 정상 시작 비용은 Kit/RTX 초기화다. CPU powersave 경고도 남지만,
전력 모드 변경 효과는 측정하지 않았다.

실측 로그/수치: `outputs/implementation/snapshots/startup-fix/isaac-preview-210251/`
(`report.json`, `cycle_1.log`, `cycle_2.log`). 기존 UI는 재실행해야 새 실행 경로가 적용된다.

## 추가 수정: Start → Disconnect 단위 Snapshot 저장

2026-10-08. 같은 profile인데도 Runtime과 SnapshotDataset이 서로 다른 JSON 직렬화
방식의 digest를 비교해서 매 캡처마다 새 dataset/model을 생성하던 오류를 수정했다.
기존 workspace/hash 계약은 바꾸지 않고 Runtime의 잘못된 rollover 비교를 제거했다.
SnapshotDataset의 기존 profile/model/device 검사로 실제 변경을 확인한다.

```text
outputs/real2sim/snapshots/
  snapshot_20261008_213000_a1b2c3d4/   # Start → Disconnect 한 연결
    capture_manifest.yaml            # session ID, 시작/종료 시각, 샷 목록
    calibration/cameras.json
    calibration/sensors.json
    robot_model/robot.urdf            # URDF/mesh/calibration은 세션마다 한 번
    robot_model/<mesh-hash>.stl
    snapshots/snapshot_000001/        # 각 샷의 이미지·측정 state·metadata
    snapshots/snapshot_000002/
    COMPLETE
    .capture.lock
  snapshot_20261008_220000_e5f6a7b8/   # 다음 Start의 독립 세션
```

- Start 성공 시 연결 세션 ID를 생성하고, 첫 캡처 때만 디렉터리와 공유 파일을 만든다.
- Stop Teleop/탭 전환은 같은 세션을 유지한다. Disconnect는 이미 저장 중인 샷 뒤에
  manifest 종료 기록을 예약한다. 로봇/카메라 종료 경로는 디스크 쓰기를 기다리지 않는다.
- 첫 캡처 이후 UI의 저장 위치/Scene 이름을 잠근다. Scene state는 샷마다 변경 가능하다.
  저장 위치 또는 calibration/model/device가 바뀌면 새 Start를 요구하며 자동 분할하지 않는다.
- schema는 `real2sim.snapshot/1`을 유지하며 optional `session` metadata만 추가한다.
  기존 두 실제 snapshot dataset은 변경/병합/삭제하지 않았고 Builder reader로 읽기를 확인했다.
  이전 파일에는 연결 경계가 없으므로 동일 profile만으로 세션을 추측해 합치지 않는다.
- 기존 Runtime/SnapshotDataset/UI를 확장했다. 새 launcher/controller/config는 없다.
  Builder source는 그대로이며 두 프로젝트의 Capture 사용 문서만 갱신했다.

검증: 관련 regression **62 PASS, 1 SKIP, 114 deselected** (23.07 s).
같은 연결의 3회 캡처가 단일 dataset과 robot copy 1회로 끝나는지, 공유 파일 hash/mtime이
유지되는지, Stop 이후 세션 유지, 저장 중 Disconnect의 비차단 장치 종료와 순차 finalize,
연결 재시작 시 새 ID, 저장 위치 변경 거절, Builder import/MuJoCo pose rendering을 확인했다.
SKIP은 harness가 Blender 실행 경로를 비활성화한 Blender rendering 검사다.
추가 UI 저장 설정 잠금/해제 검사 **1 PASS, 176 deselected** (1.11 s).
실제 Leader/Follower/USB 카메라 연결은 **NOT_TESTED**이며 수정 적용에는 UI/worker 재시작이 필요하다.

결과: `outputs/implementation/snapshots/startup-fix/snapshot-session-tests.xml`.
테스트 임시 dataset은 `/tmp`에 생성해 production outputs에 샷/model 결과물을 남기지 않았다.


## 후속 UI 수정: 오른쪽 연결 설정과 MuJoCo 재연결

2026-10-08. Capture의 연결 설정 버튼과 화면 아래 공통 토글이 중복되어 있었고,
하단 최대 높이 330의 설정 영역 때문에 항목이 잘렸다. `real2sim/ui.py::Window`의
기존 필드/handler를 그대로 오른쪽 공유 패널로 옮겼다. 새 UI/launcher 모듈은 없다.

- Capture/Real-Sim 모두 왼쪽 프리뷰 + 오른쪽 연결 설정, 로그는 전체 화면 아래.
- 오른쪽 패널은 독립적인 세로 스크롤과 드래그 폭 조절을 제공한다. 작은 화면에서는
  왼쪽 Capture/Real-Sim 내용도 스크롤할 수 있다. 중복 연결 설정 버튼/토글은 제거했다.
- Capture에는 실물 연결과 Snapshot 저장 설정, Real-Sim에는 연결/Simulation 설정을 표시한다.
  Workflow 탭은 자체 설정을 사용하며 오른쪽 공통 패널을 숨긴다.
- UI를 여는 것과 탭 전환은 장치/Sim을 연결하지 않는다. Start/Connect와 Teleop Start는 별개다.

MuJoCo 오류의 실제 traceback은 `configure_rendering`의
`OSMesa shared library unavailable`이었다. 사용자 환경에 OSMesa 공유 라이브러리가
없었다. dependency/conda 설치 없이 기존 software-EGL CPU 경로를 사용하도록 수정했다.

- Live 기본 renderer는 `software-egl`. GPU EGL은 그대로 선택할 수 있다.
- OSMesa가 없으면 UI에서 해당 항목을 비활성화하고 tooltip으로 이유를 표시한다.
- Live launcher/직접 runtime CLI에 OSMesa가 지정되어도 엔진 import 전에 software-EGL로
  전환한다. 직접 CLI는 전환 사실을 로그에 남긴다. 엔진의 strict renderer guard는 유지한다.
- Runtime status에 renderer를 표시한다. ERROR 상태나 renderer 변경 시 이전 MuJoCo
  worker에 Connect만 다시 보내지 않고, Disconnect 종료를 기다린 뒤 새 worker로 연결한다.
- `launcher.py`, `runtime.py`, 기존 `real2sim_sim.py`를 수정했다. 기존 Isaac/real 연결과
  장치 종료 대기/연결 중 강제 종료 방지 경계는 유지했다.

검증: Snapshot/UI/Runtime/launcher/workflow/독립 Builder regression **153 PASS, 2 SKIP**
(39.02 s). SKIP 두 개는 이 실행에서 Blender 경로를 비활성화한 검사다.
1280×720에서 두 탭의 우측 패널 위치, 하단 로그 위치, 설정 끝까지 scroll, 필드 공유,
초기 미연결을 검사했다. 1280×900의 실제 Qt theme 화면도 렌더해 확인했다.
실제 subprocess의 software-EGL과 OSMesa 요청→software-EGL 전환을 각각 실행해
MuJoCo Side/Wrist JPEG, PREVIEW, Disconnect/CLOSED, Isaac/omni/pxr import 부재를 검증했다.
실제 Leader/Follower/USB 카메라 연결과 GPU EGL/OSMesa 자체는 이번 검증에서 실행하지 않았다.

검사 결과: 기존 `outputs/implementation/snapshots/startup-fix/tests.xml` 갱신.
두 UI preview도 기존 `ui_capture_idle.png`, `ui_real_sim_idle.png`를 갱신했다.
합성 테스트 데이터와 runtime 로그는 `/tmp/so101-sidebar-tests`에 있다. 실제 scene,
Snapshot, HDF5와 checkpoint는 변경하지 않았다. UI를 재시작하면 적용된다.


### 후속: Stop 후 남은 구버전 OSMesa worker 재사용 방지

재발한 실제 status의 session은 `dc9392e5c7534ceaa92ac673c72e6cff`였고,
renderer 필드가 없는 구버전 MuJoCo worker였다. ERROR 이후 Stop으로 IDLE 상태가
되어도 구버전 process의 OSMesa args는 그대로였다. 기존 재연결 판정은 ERROR 또는
알려진 renderer mismatch만 처리해 IDLE/unknown renderer를 재사용할 수 있었다.

`RuntimeLauncher.connect`는 이제 MuJoCo renderer를 모르는 worker도 종료를 기다린
뒤 새로 실행한다. 같은 renderer를 명시한 정상 worker의 재사용은 유지한다.
IDLE/PREVIEW의 unknown renderer 회귀 사례를 기존 테스트에 추가했다.
실제 worker의 live lock과 최신 상태를 확인했고, 실제 robot/camera/Sim/저장 세션이
연결되지 않은 IDLE 상태에서 정상 Disconnect 명령으로 종료했다. CLOSED와 lock 해제를
확인했으며 강제 kill, 장치 재연결, 모터 이동 명령은 보내지 않았다.

Launcher/실제 MuJoCo preview·software-EGL·OSMesa fallback 검사: **22 PASS**
(3.59 s; 나머지 135개 deselected). `git diff --check` PASS. 마지막 UI core 검사는
위 153 PASS/2 SKIP 기록이며 이번 추가 검사는 renderer 재연결 영역을 대상으로 한다.
검사 XML은 기존 startup-fix/tests.xml에 별도 suite로 덧붙였다. 장치 연결은 사용자가
다시 Connect로 실행한다. 라이브러리 설치 없이 CPU software-EGL 경로를 사용한다.
