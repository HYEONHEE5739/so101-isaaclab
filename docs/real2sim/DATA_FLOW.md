# SO-101 Real2Sim Data Flow

## 사용자 이름·저장 경로 — 2026-10-17

새 Workflow 실행은 **저장 이름**을 실제 경로와 Registry `display_name`에 사용한다.
단계별 이름을 자동 추천하며 수정할 수 있다. 한글/영문/숫자/밑줄/하이픈을 지원한다.
이름은 100자·UTF-8 200바이트 이내이며 경로 구분자·점·공백·장치 예약 이름은 거절한다.
`artifacts.output_location`이 UI 미리보기와 Operations 계획에 같은 파일/폴더 계약을 제공한다.

```text
outputs/workflow/
  sources/<이름>/              # Isaac: source.hdf5 / MuJoCo: dataset.hdf5 등
  annotations/<이름>/         # 공통 dataset.hdf5 + 상대 MP4
  curation/filtered/<이름>.hdf5  # 과거 canonical이면 <이름>/ 폴더
  curation/splits/<이름>/     # train.hdf5, val.hdf5, source_split.json
  datagen/<이름>/             # dataset.hdf5 + 영상·생성 report
  datasets/<이름>/            # LeRobot meta/Parquet/MP4
  models/<이름>/              # 기존 trainer의 policy/ 또는 checkpoints/ 구조
  evaluations/<이름>/         # 엔진별 report/기록; Replay·Hub receipt도 이곳
  runs/<고유 Run ID>/         # request.json, state.json, process.log, presentation/
  artifacts/<고유 ID>.json    # 데이터가 아닌 Registry 인덱스
```

필요한 폴더는 새 저장 시 생성한다. Worker에는 실제 결과 경로를 전달하고 request/state.plan.output에
동일하게 기록한다. 직접 지정한 `output`은 유지한다. 저장 이름도 지정했다면 경로의 마지막 이름(파일은
stem 또는 포함 폴더)과 일치해야 한다. 파일형 Source/Evaluation은 HDF5/JSON 확장자를 검사한다.
기존 결과가 있으면 거절하고, 동시 실행은 저장 경계 옆의 배타적 예약 파일로 중복 사용을 차단한다.
중단 결과도 삭제/덮어쓰지 않는다. DataGen/Source 이어 수집·MiniVLA Resume은 기존 입력과 **새 저장 이름**을
선택한다. ACT/SmolVLA는 기존 Resume 출력 계약을 유지하며, 기본 Resume 위치는 미리보기에 표시한다.
CLI에서 ACT/SmolVLA 재개의 새 이름을 지정하려면 일치하는 새 `output`도 명시해야 한다.
강제 kill로 예약 파일이 남았다면 자동 삭제하지 않고 해당 파일에 기록된 Run의 종료 여부를 먼저 확인한다.

새 Annotation은 지정한 위치에 등록하고 Source를 부모로 연결하여 원본을 보존한다.
기존 명시적 `Registry.promote_annotation`의 검증·rollback API는 유지한다. Curation은 같은 저장 버튼으로
선별 결과와 분할 결과를 각각 filtered/splits에 제안한다. 이름 입력 후 저장창에서 최종 경로를 고를 수 있다.
Split의 Train/Val은 `<이름>_train`, `<이름>_val`로 따로 등록하며 기존 Hash/Inventory/Assignments를 사용한다.

입력 선택기는 저장 이름·종류·Task로 좁히고 날짜/episode/partition/backend/짧은 ID를 표시한다.
선택 내용의 실제 경로·부모 계보를 확인하고 **입력 폴더 열기**로 열 수 있다. 현재 단계에 호환되는 유형만
선택할 수 있다. 등록 모델은 Resume과 Loss Graph 목록에서도 선택한다. 그래프는 현재 실행/선택 정책/
Train 입력의 등록 모델에서 metrics.jsonl을 자동 탐색하고 수동 JSONL 선택도 유지한다.
Checkpoint import는 모델의 상위 폴더 또는 Registry의 Run request에서 기록된 출력 관계를 검증한다.

기존 runs 내부 결과는 이동/수정하지 않는다. 이름 없는 Registry 항목은 읽을 때 경로 기반 이름을 보여주며
ID/Path/SHA256/Parent는 그대로 사용한다. `workflow.py run|ready --storage-name 이름`,
`register --display-name 이름`도 같은 경계를 사용한다. 명시적 `worker --output` 경로는 그대로 유지한다.

## 통합 Episode 계약 — 2026-10-16

새 Workspace Source / Annotation / DataGen은 두 엔진 모두 `so101.common-episode/2`를 사용한다.
숫자·주석·메타데이터는 HDF5, 영상은 상대 경로로 연결한 Side/Wrist **640×480 H.264 MP4**다.
Backend는 `data.attrs.env_args.backend`에 기록하며, 좌표·모델 계약은 원본을 상속한다.
기존 Canonical NPY+MP4와 Isaac HDF5는 계속 읽는다. Workflow가 필요한 Legacy 입력을 새 결과 폴더에
자동 Import하므로 엔진을 바꾸기 위해 사용자가 파일을 수동 변환할 필요가 없다.

```text
기존 NPY/Isaac HDF5 ── 읽기 전용 Import ─┐
새 MuJoCo/Isaac Source ───────────────┼─ 공통 HDF5 + MP4
                                     ├─ Source Partition → Annotation
                                     └─ 공식 Mimic 공통 코드 → 물리 Adapter → 중앙 Writer
                                                          ↓
                           Curation → 기존 LeRobot Converter → ACT / MiniVLA → 양 엔진 평가
```

`workflow.dataset.EpisodeDataset` / `DatasetWriter` / `validate_dataset`가 계약을 담당한다.
Curation의 기존 `EpisodeReader`는 같은 HDF5 키와 external video resolver를 읽으며,
전체 RGB를 RAM에 적재하지 않는다. 영상·숫자 복제는 Import/선택 저장/새 결과 작성 시에만 수행한다.

### DataGen IK 선택

기존 SettingsForm의 DataGen `IK Solver`에서 `scipy_ls`(기본) / `pinocchio_dls`를 선택한다.
MuJoCo와 Isaac 모두 같은 옵션을 사용한다. UI 드롭다운 자체가 `options.ik_solver`의 편집기이며
별도의 JSON 덮어쓰기는 없다. 실행 중/완료 상태에도 선택값이 표시된다.
공통 CLI `workflow.py worker --backend mujoco|isaac --stage datagen ... --ik-solver pinocchio_dls`를
사용할 수 있다. `--ik-solver`가 `--options` JSON보다 우선하며, immutable `--request` 사용 시에는
request의 options 안에 설정해야 한다. Workspace Isaac CLI는 `--ik_solver`를 받는다.
잘못된 이름은 엔진 초기화 전에 거부한다. Source를 자동 물리 재생하는 단계는 추가하지 않는다.

`Mimic → Adapter → simulation.ik.solve_selected → 기존 solve_pose / SO101Kinematics.inverse_kinematics`
순으로 실행한다. 기존 두 계산 본문과 공식 Mimic은 변경하지 않는다. DLS는 원래 Isaac 설정
(position/orientation weight 1/.01, 200 iterations, tolerance .001, damping .001, step .3)이다.
MuJoCo world 목표를 기록된 robot base로 변환하고, 기록된 simulation model의 URDF를 사용한다.
관절 영점은 URDF/MJCF에 이미 반영되므로 별도 보정을 더하지 않는다. Pinocchio Model/Data는 worker별 소유다.

SciPy는 이전 명령, DLS는 측정 관절에서 시작한다. 별도 `IK Failure Policy`는 기본 `strict`이며
`hold` / `best_effort`를 선택할 수 있다. Strict는 기존 SciPy 3 mm / .12 rad 탈락, DLS native 미수렴 hold를
보존한다. Hold는 native 미수렴 또는 pose 오차 초과 시 측정 팔 각도를 유지하고 다음 목표로 진행한다.
Best Effort는 수렴 여부와 무관하게 유효한 candidate를 적용하고 실제 grasp/place로 최종 성공을 판단한다.
SciPy의 `reject_on_error=False`는 candidate 반환만 허용하며 least_squares 수치 알고리즘은 그대로다.

공통 candidate의 5축 순서·finite·관절 제한을 검사하고 gripper는 원래 명령을 유지한다.
Hold/Best Effort는 이전 팔 명령 대비 어느 관절이든 1 step에 0.5 rad를 넘으면 attempt를 중단한다.
이 값은 급격한 명령을 검출하는 고정 기준이며 Strict에서는 기존 동작 보존을 위해 진단만 기록한다.
잘못된 candidate나 제한 밖의 measured hold는 적용하지 않는다. clipping/solver fallback/제한 완화는 없다.
CLI는 `--ik-failure-policy best_effort`, Workspace Isaac CLI는 `--ik_failure_policy best_effort`다.
UI → options → worker → Adapter.configure_ik → solve_selected → 기존 physics 경로를 사용한다.

`generation_configuration`, report/attempts와 episode attrs에 solver/config 및 `ik_failure_policy`,
`ik_policy_configuration`을 기록한다. optional `ik/*`에는 `ik_converged`, candidate 위치(mm)/방향(deg) 오차,
`ik_candidate_applied`, `ik_held`, `ik_invalid_candidate`, 시간(ms)과 실패 이유를 저장한다.
Native 수렴과 `common_pose_passed`는 독립이다. `ik_unconverged_applied` 집계는 native 실패 **또는** pose gate
실패 후보의 적용 수다. invalid candidate와 invalid hold command도 별도 집계한다. 실패 시도는 attempts에 남는다.
`ik_motion_evidence`는 시도당 최대 8개의 실패 시점 candidate/실제 명령/물리 전후 측정을 기록한다.
물리 추종 오차와 candidate 오차는 구분하며 기존 grasp/place 성공 조건을 바꾸지 않는다.
Resume은 solver·policy·설정 불일치를 복사 전에 거부한다. 옛 생략값은 SciPy/Strict로 해석한다.
실제 비교 및 변환 검사는 `outputs/workflow_validation/ik_failure_policy/REPORT.md`에 있다.

### 키·단위·시점

각 episode는 `data/demo_N`이고 `data.attrs.env_args`와 episode attributes는 JSON metadata다.
`schema_contract`에서 JOINTS 순서, rad, robot_base, quaternion 순서, action semantics를 검증한다.

| 경로 | 의미 / shape / 단위 |
|---|---|
| `obs/policy/joint_pos` | 명령 전 측정 관절 `[T,6]`; JOINTS 순서, gripper 포함 model rad |
| `joint_targets` | 관측 t 직후 적용한 절대 관절 목표 `[T,6]`, model rad |
| `actions` | `[T,7]`: 이전 명령 EE 목표에서 현재 목표까지 base-frame translation m + 좌측 rotation delta rotvec rad + raw gripper rad. 첫 delta는 초기 측정 EE 기준. 정책 학습의 action은 `joint_targets` |
| `obs/datagen_info/eef_pose/tool0` | 측정 EE `[T,4,4]`, robot_base SE(3), m |
| `obs/datagen_info/target_eef_pose/tool0` | 제어 EE 목표 `[T,4,4]`, robot_base. 관절 전용 Legacy 입력은 기록된 목표의 FK로 복원; 다음 측정 상태를 목표로 대신 쓰지 않음 |
| `obs/datagen_info/object_pose/<id>` | 명령 전 동적/목표 물체의 base-frame SE(3) |
| `obs/datagen_info/subtask_term_signals/grasp` | `[T,1]`, 0→1의 단일 증가 경계. Source/실패 평가에는 없을 수 있고 Mimic 입력에는 실제 경계가 필수 |
| `obs/events/grasp`, `obs/events/place` | 관측한 이벤트; 최종 action 뒤 배치 성공은 마지막 action frame에 표시하고 annotation metadata에 명시 |
| `obs/timestamp`, `action_timestamp` | 같은 제어 tick의 observation/action 초. 관측을 먼저 읽고 명령을 적용. 생성/평가는 episode-relative, Legacy Source의 시작 offset은 보존 |
| `obs/camera_timestamps/{side,wrist}` | 대응 observation timestamp와 동일. MP4 PTS는 `frame/fps`, 원본 control Hz 유지 |
| `obs/object_poses/<id>` | world-frame xyz+wxyz `[T,7]`, Curation telemetry |
| `initial_state/{articulation,rigid_object}` | 초기 root pose xyz+wxyz/velocity와 robot joint position/velocity. Legacy 미기록 velocity는 0 복원임을 표시하고 물리 replay로 별도 검증 |
| `legacy/*` | Import 원본 숫자 배열을 dtype/값 그대로 보존. Mimic pose만 공식 런타임에 맞춘 float32; 원본은 여기 남음 |
| `mimic_plan/<subtask>/*` | IK 전 목표 EE pose / gripper / noise, 선택 원본·경계는 episode의 `mimic_source_segments` |
| `generation/attempts_json` (root) | 압축된 시도 이력. 새 생성 파일을 이동해도 Resume의 누적 시도/실패 통계 유지 |

`initial_state`는 각 episode의 첫 관측과 일치해야 한다. action/gripper/pose/clock/MP4의 frame 수,
실제 H.264 codec, 모든 packet PTS, checksum과 상대 경로를 검사한다. numeric/video를 함께 복사·이동한다.
`external_videos`에는 파일 상대경로·shape·FPS·frame 수·codec·SHA256이 있으며, 파일 부모 밖을 가리키는
참조는 읽기에서 거절한다. 새 저장/Import의 입력 원본 경로만 명시적으로 허용한 뒤 결과 내부에 복사한다.

Workspace/Task/active objects/scene/reset, Coordinate/Calibration/robot model, seed/engine version은
`env_args`에 보존한다. episode에는 `source_group`, `source_lineage`, success와 판정 근거/실패 이유,
annotation을 기록한다. Source/operator 저장 라벨과 물리 검증 성공은 구분한다.

### 공통 Mimic / 병렬 실행

`workflow.mimic_reference`는 설치된 Isaac Lab Mimic **1.0.15**에서 이식했다. 다섯 생성 모듈의
실행 본문은 그대로이고 import 경로만 바꿨다. 라이선스/원본 hash는 해당 폴더의 NOTICE/UPSTREAM.json에 있다.
공식 HDF5 Pool의 group 순서(`demo_0, demo_1, demo_10…`)도 유지한다. UI 표시 순서는 별도로 숫자 순서다.
SO101 설정은 object nearest-neighbor `nn_k=3`, grasp end offset `(0,10)`, 마지막 `(0,0)`,
interpolation5 / fixed0 / noise0, subtask별/arm별 원본 재선택 false, 마지막 command에서 보간이다.
SO101 단일 arm 두 subtask를 지원하며, SkillGen·multi-arm coordination은 이 adapter 범위에 포함하지 않는다.

`workflow.adapters`만 FK/IK/상태/접촉/이미지/물리 엔진에 의존한다. 공통 bounded 5축 IK는
위치3mm·회전0.12rad를 각각 검사하고 seed/관절 한계/목표 frame/두 잔차를 기록한다.
6D 목표의 모든 자세가 5축 로봇에서 도달 가능한 것은 아니다. IK·계획 탈락, 물리 실행 오류,
집기·배치 실패, 시간 초과, 성공을 분리한다. 허용 오차를 늘리거나 탈락을 성공으로 바꾸지 않는다.

Isaac은 한 batch clock에서 환경별 coroutine/action/reset mailbox를 사용한다. episode 종료와 reset은
각 환경이 독립적으로 수행한다. reset이 다른 환경의 q/rigid state를 바꾸지 않았는지 실제 검사한다.
Kit의 별도 asyncio loop 때문에 physics/reset은 generator loop 바깥에서 실행한다.
MuJoCo는 `MjModel` **한 개**와 worker별 `MjData`/FK scratch/IK scratch를 사용한다. ThreadPool worker가
독립 실행하며 GL 생성·사용·종료는 소유 thread에서, render/GL 호출만 lock으로 직렬화한다.
MP4는 카메라별 bounded8frame encoder queue로 흘린다. physics/IK를 렌더 lock으로 감싸지 않는다.

Source Pool은 worker 초기화 때 한 번 로드하고 읽기 전용으로 재사용한다. Source 선택·경계·RNG·trace는 매 시도의 Generator가 소유한다.
`GenerationScheduler`가 상태 lock에서 성공 슬롯을 예약하고 별도 Writer lock에서 완료 episode만 commit한다.
commit 실패는 예약을 취소하며 성공 수를 늘리지 않는다. 저장 중에도 다른 환경의 상태 조회·claim은 계속된다.
먼저 끝난 환경은 즉시 다음 시도를 가져간다. 목표를 넘는 완료는 추가 등록하지 않는다.
중지 시 완료 데이터를 보존하며 미완성 영상은 버린다. Resume은 Source Pool byte hash·좌표·seed·엔진을
검사하고 새 출력으로 이어 쓴다. max_attempts는 재개 전 시도까지 포함한 전체 한도다.
성공률은 성공/(성공+분류된 실패), 목표 도달로 중단한 다른 환경과 진행 중 시도는 제외한다.
새 HDF5의 시도 이력이 없거나 모순되면 재개를 거절한다. 구형 파일의 이력이 없으면 누적 성공률은 UNKNOWN이다.

새 Source는 양 엔진 모두 처음부터 공통 Writer에 저장한다. Isaac의 `EpisodeData`는 메모리상의 native API 경계이며
중간 Legacy HDF5를 만들지 않는다. finalized `VideoEpisode`의 private staging은 검증 후 UUID 영상 폴더로 이동한다.
기존 입력·Resume 영상은 복사한다. 두 경로 모두 해상도/FPS/H264/전체 PTS/SHA 검사와 원자적 publish를 유지한다.

### 다른 엔진의 입력 / 계보

DataGen은 입력 엔진과 관계없이 모델·단위·frame·초기 q/object·시간·주석 계약을 검사한 뒤
환경을 준비하고 곧바로 선택한 환경 수로 병렬 생성한다. Source 전체를 자동으로 물리 재생하지 않으며,
교차 Replay 성공이나 캐시 유무를 생성 시작 조건으로 사용하지 않는다. 계약 검사 통과는 물리 성공을 뜻하지 않는다.
새 생성 episode의 성공 여부는 해당 실행에서 측정한 집기·배치 조건으로 계속 판정한다.

명시적으로 선택한 Replay/Annotation은 **기록 목표 관절을 실제 재생**하고 집기와 최종 배치를
`cross_replay.json`에 남긴다. Annotation은 통과한 episode만 별도 결과에 저장하고 제외 이유/개수를
함께 반환한다. 원본을 교체하지 않는다. 개발용 교차 재생·positive proof 캐시 helper는 별도 검사에만 사용한다.

Isaac vector scene 생성 직후 전체 환경을 reset해 Task 초기 관절/물체 상태를 적용한다.
명시 Replay에서 대기 로봇은 최초 hold 목표를 유지한다. 매 tick 측정 관절을 새 목표로 덮어쓰지 않는다.
MuJoCo는 환경별 MjData를 따로 step하지만 Isaac은 대기 환경까지 공통 physics step을 진행하므로 이 구분이 필요하다.
Workflow 영상은 기존 Overall 및 Env 0의 Side/Wrist 세 뷰를 유지한다.

Train14/Val6는 새 split을 만들지 않고 원래 partition/Source ID를 상속한다. 생성 episode는
각 subtask의 실제 Source ID·경계를 `mimic_segments`로 기록한다. 미분할 생성 자료를 학습 분리할 때는
A+B / A+C처럼 원본을 공유한 조합을 연결 성분으로 묶어 같은 split에 놓는다. 추적이 UNKNOWN이면
독립 split을 허용하지 않는다. 원래 Partition 파일과 원본 데이터는 수정하지 않는다.

### UI / CLI / 학습 / 평가

Workflow에서 입력과 실행 Backend를 선택하고 성공 데이터 수, 환경 수1/2/4/8, 생성 seed(미지정=Auto),
전체 최대 시도 수를 설정한다. 상세 설명은 기존 설명 버튼에 있다. 실행 중 환경별 상태와 실패 종류,
짧은 IK 오차를 보여준다. 원본 로그와 `attempts.json`에는 전체 진단을 보존한다.

두 엔진의 DataGen 진행 UI는 같은 `Presentation`의 Scheduler provider와 `datagen_status_text`를 사용한다.
환경 준비/생성을 표시하며, 전체 성공/목표·시도·실패·성공률과 환경별 상태/시도/성공/실패/step,
최근 결과의 환경 번호를 표시한다. DataGen에는 Source 사전 재생 단계가 없다.
실패 원인은 전체/환경별 집기 실패·이동 중 놓침·컵 배치 실패 및 IK/계획/저장 오류로 표시한다.
놓침은 안정 집기 확인 뒤 release 명령 전에 양 접촉 소실+상대 위치6mm 이탈이0.2초 지속된 진단이다.
기존 성공 조건은 그대로이며 attempts의 failure_diagnostics로 판정 증거를 확인한다.
생성 경과는 wall_seconds, 목표 저장 완료 시간은 마지막 성공 commit 시 고정한 target_completed_seconds다.
분·초로 표시하며 환경 준비 제외/일시정지 포함, Resume이면 이번 실행 구간만 표시한다.
종료/일시정지에서도 provider의 최신 snapshot을 반영한다. 성공0개에서 사용자 중지는 STOPPED이며 빈 artifact를 등록하지 않는다.

```bash
# 필요할 때만 명시 Import. 기존 입력은 읽기 전용이며 새 경로가 필요하다.
PYTHONPATH=source/soarm101_lab python scripts/real2sim/workflow.py migrate /path/to/legacy /path/to/new.hdf5
# 기존 worker JSON 경로: backend, stage, workspace, input, output, options.
PYTHONPATH=source/soarm101_lab python scripts/real2sim/workflow.py worker --request request.json
# 기존 Isaac CLI도 Workspace일 때 같은 공통 core를 사용한다. --output_file 경로는 그대로 보존한다.
PYTHONPATH=source/soarm101_lab python scripts/mimic/generate_dataset.py --workspace /path/to/workspace \
  --input_file /path/to/annotated.hdf5 --output_file /path/to/generated.hdf5 \
  --num_envs 4 --generation_num_trials 100 --max_attempts 10000 --headless
```

기존 LeRobot Converter 하나가 양 엔진의 HDF5를 처리한다. `joint_targets` model rad/JOINTS6와
Source lineage/Partition을 보존하고 FPS를 변경하지 않는다. 현재 LeRobot은 정수 FPS만 지원하므로
분수 FPS를 임의 반올림/재샘플링하지 않고 거절한다. LeRobot 저장 codec은 기존 라이브러리 설정을 유지한다.
ACT/MiniVLA는 기존 관측/6D action/정규화/체크포인트를 사용한다. MiniVLA의 현재 고정 모델은 Side 하나를
입력으로 쓰며 저장 영상은 Side/Wrist 모두 남는다. 별도 Val 입력과 train-only statistics를 유지한다.

두 엔진의 Workspace Sim 평가에는 `so101.evaluation-episode/1` summary와 공통 HDF5+두 MP4를 추가한다.
초기 물체 배치, checkpoint, grasp/place, timeout/controller/IK, steps/duration/failure reason 및 실제
state/action trace를 보존한다. 기존 배치 성공 조건과 안정 집기 측정은 별도 필드다. 미완성 평가는
정식 episode로 저장하지 않는다. Legacy 단일 camera/review scene과 실제 로봇 평가 경로는 유지한다.
검증 보고서: `outputs/workflow_validation/unified_mimic/REPORT.md`.

## MiniVLA Train/Val과 학습 기록 — 2026-10-17

Registry의 Train/Validation LeRobot 입력→`episodes.validate_training_inputs`(메타데이터 사전검사)→
`mujoco.train_from_lerobot`(기존 Parquet/MP4를 읽는 작은 index)→train-only 통계→기존 MiniVLA trainer로 이어진다.
검증된 Source Partition/Lineage를 최우선 사용하며 Train 전용 데이터를 다시 나누지 않는다. 일반 UI에
`source_groups` 입력은 없고 기록과 충돌하는 매핑은 거절한다. Partition 없는 Legacy는 검증된 독립 Group과
고급 호환 옵션이 있어야 기존 `split_episodes`를 사용할 수 있다. Val 준비 경로와 UI 사용법은
[MiniVLA 통합](MINIVLA_INTEGRATION.md#minivla-학습-화면과-체크포인트--2026-10-17)을 참조한다.

일정 설정은 체크된 UI 값→사용자 JSON→기본 JSON 순서다. MiniVLA UI 초기값50k/검증1k/저장2.5k,
Early Stopping OFF이며 기존 ACT/SmolVLA 설정과 분리한다. 매 step train loss, 검증 시에만 val loss를 기록한다.
검증은 episode당 최대8개의 고정 시점 평균을 episode별 동일 가중 평균한다. 실제 표본 수는
`validation_protocol.json`에 남긴다. 저장 주기는 검증과 독립이며 번호별/best/last를 보존한다.
Resume은 원본 모델·통계·양쪽 dataset identity를 검사하고 새 출력에 저장한다. Sim 평가의 명시적
체크포인트 파일 선택과 실제 step을 두 엔진의 결과에 기록한다. 모델·손실·전처리·Mimic 본문은 유지한다.

## 관련 계약과 이력

- 실시간 관절: [JOINT_CONTROL.md](JOINT_CONTROL.md), 보정: [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md).
- Snapshot: [CAPTURE_DATASET.md](CAPTURE_DATASET.md), Builder 입력: [BUILDER_WORKSPACE.md](BUILDER_WORKSPACE.md).
- 학습/평가: [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md), 모듈 지도: [ARCHITECTURE.md](ARCHITECTURE.md).
- 통합 이전 NPY/Recorder/운영 설명은 [보존 기록](../archive/real2sim/DATA_FLOW_BEFORE_COMMON.md)이다. 현재 실행 경로의 기준으로 사용하지 않는다.
