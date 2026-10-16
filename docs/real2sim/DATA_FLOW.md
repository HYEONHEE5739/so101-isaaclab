# SO-101 Real2Sim Data Flow

## 통합 Episode 계약 — 2026-10-16

새 Workspace Source / Annotation / DataGen은 두 엔진 모두 `so101.common-episode/2`를 사용한다.
숫자·주석·메타데이터는 HDF5, 영상은 상대 경로로 연결한 Side/Wrist **640×480 H.264 MP4**다.
Backend는 `data.attrs.env_args.backend`에 기록하며, 좌표·모델 계약은 원본을 상속한다.
기존 Canonical NPY+MP4와 Isaac HDF5는 계속 읽는다. Workflow가 필요한 Legacy 입력을 새 run에
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

## 관련 계약과 이력

- 실시간 관절: [JOINT_CONTROL.md](JOINT_CONTROL.md), 보정: [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md).
- Snapshot: [CAPTURE_DATASET.md](CAPTURE_DATASET.md), Builder 입력: [BUILDER_WORKSPACE.md](BUILDER_WORKSPACE.md).
- 학습/평가: [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md), 모듈 지도: [ARCHITECTURE.md](ARCHITECTURE.md).
- 통합 이전 NPY/Recorder/운영 설명은 [보존 기록](../archive/real2sim/DATA_FLOW_BEFORE_COMMON.md)이다. 현재 실행 경로의 기준으로 사용하지 않는다.
