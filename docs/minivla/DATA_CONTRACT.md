> Historical / Archived — 2026-10-22. 과거 모델 설계/실험 기록이며 삭제된 학습 명령은 실행 안내가 아닙니다. 현재 지원 범위는 [README](README.md)를 따릅니다.

# 데이터 규격

기존 데이터의 축 순서와 단위를 자동 추측하지 않는다. 기존 `Episodes`가 숫자 Canonical NPY + RGB NPY/MP4와
LeRobot Parquet/MP4를 모두 읽는다. 새 LeRobot 학습은 작은 manifest/statistics만 생성하며
`canonical_source`나 전체 RGB NPY 복사본을 만들지 않는다. Canonical 입력도 계속 지원한다.

## LeRobot Direct (2026-10-15)

기존 Workflow에서 Convert한 LeRobot artifact를 MiniVLA Train에 선택한다. `train_from_lerobot`
→ 기존 `train` → `Episodes` 순서로 실행된다. MP4/Parquet는 원본 경로에서 읽으며 새로운
`dataset/manifest.json`의 `storage`가 원본 경로·전체 SHA256·좌표 계약·gripper endpoints·camera key를
참조한다. 각 episode는 원본 episode/frame 구간, Source Group, Instruction, Task/Scene/Review를 가진다.
축 순서/6차원/model-radian 계약과 명시적인 `joint_targets` Action 출처를 검사한다.

숫자 Parquet만 episode별로 읽어 기존 RobotCoordinates로 5rad+gripper fraction을 구성한다.
통계는 train split만 사용하고 RGB를 디코딩하지 않는다. 명시적인 Source Group이 없거나
검증된 Source Partition도 없는 UNKNOWN 계보이면 split 전에 거절한다. 변형 episode를 별도 독립 group으로 만들지 않는다.
MP4는 CPU torchcodec exact seek, FFmpeg thread 1, decoder 최대 2개, RGB 최대 8frame/16MiB
LRU다. 기본 학습 프로세스에 DataLoader worker를 추가하지 않는다. 모델을 띄우기 전에
camera 존재/숫자 shape·단위·시각을 확인하고, decode 실패·누락·PTS 불일치는 episode/frame과 함께
오류로 표시한다. decoder tensor는 RGB uint8 NCHW, sample은 RGB uint8 HWC다.
640×480 원본은 유지하고 기존 Backbone에서만 256×256 Letterbox로 변환한다.

Batch 형식과 H40 padding/mask는 아래와 동일하다. chunk는 episode를 넘지 않는다.
신규 학습은 UI Batch Size(기본8)와 accumulation1로 실행한다. collate는 state `[B,6]`,
actions `[B,40,6]`, mask `[B,40]`를 만들고 이미지/문장은 B개 목록으로 유지한다. 기존 Qwen processor가
언어 길이를 padding하고 attention mask·이미지 grid를 각 샘플에 맞춘다. optimizer step당 Forward/Backward는
한 번이다. Resume은 저장된 accumulation을 유지하며, batch_size가 없는 과거 설정은1로 읽는다.
새 checkpoint는 전체 dataset byte identity와 manifest identity를 저장한다. Resume는 동일한
데이터·통계·config·주파수와 optimizer를 요구한다. 기존 Canonical 학습은 같은 저장 포맷으로
재개하며 Direct로 자동 전환하지 않는다. 구형 Canonical checkpoint의 byte identity는 원래 Workflow의
`project_provenance.json.dataset_sha256`로 증명해야 한다. 근거가 없으면 Resume를 거절하되
추론 로딩은 유지한다. 기존 파일은 재저장하지 않는다.

검증 결과: [VALIDATION.md](VALIDATION.md), 상세 근거:
`outputs/workflow_validation/lerobot_direct_calibration/REPORT.md`.

## 측정 그리퍼와 명령 범위 — 2026-10-17

LeRobot의 state/action은 원본 model rad 그대로 읽는다. 명령의 닫힘/열림 endpoint와 측정 관절의
유효 범위는 구분한다. 예를 들어 닫힘 명령0rad 이후 관성이 있는 측정값−0.045rad는 명령으로는
거절하지만, 기록된 실행 URDF의 gripper 범위[−0.174533,1.74533]rad 안이라면 관측으로 허용한다.
`workflow.episodes.training_coordinates`는 recorded simulation_model의 실행 URDF 경로·SHA256을
실제 Workspace 파일과 비교한 뒤 범위를 읽는다. 새 학습 index/checkpoint의 canonical coordinates에
`measured_gripper_limits_rad`를 저장하고, 로더와 MuJoCo/공통 MiniVLA 추론에서 동일하게 사용한다.

관측은 URDF 범위를 검사한 후 기존 fraction 변환과[0,1]포화를 적용한다. 팔5축/명령/원본 rad는 바꾸지 않는다.
URDF 밖 측정값, 비유한 값, endpoint 밖 명령은 계속 거절한다. 오류는 dataset 경로·episode·column을 표시한다.
URDF 근거가 없는 Legacy나 기존 checkpoint/index는 기존2% measured tolerance를 유지한다. 과거 manifest,
통계, 데이터셋과 checkpoint를 자동 수정하지 않는다. 새 범위 metadata는 manifest/statistics fingerprint와
Resume dataset identity에 포함되므로 재개 시 정규화 계약이 임의로 바뀌지 않는다. 이는 시뮬레이션 모델
범위 검사이며 실물 관절/물성 검증을 의미하지 않는다.

## 명시 Source Partition / 별도 Validation artifact

새 Source 분할은 기존 `episodes.source_split`/Curation subset export를 쓴다. `source_partition`은
원본 Dataset identity·episode payload ID·group 별칭·전체 분할 계획·선택 Pool·split·seed를 보존한다.
복사/재번호 후에도 원본 ID를 유지하고, 같은 원본이나 group이 train/val에 겹치면 거절한다.
이미 지정된 partition split은 자동 `split_episodes`가 다시 배정하지 않는다.

Isaac Mimic은 subtask를 재조합하므로 정확한 episode별 원본을 하나로 지정하지 않는다.
`source_lineage.kind=partition`, `exact_source_ids=null`과 검증된 `source_partition_input`을 함께 남긴다.
이 증거는 실제 입력 Source 파일과 그 안의 원본 집합을 검사한 뒤 생성한다. 출처를 모르는 과거
UNKNOWN group에 이름만 바꿔 붙이는 방식은 허용하지 않는다. 원래 group 값은 `source_group`에 보존하고
학습 분리용 group은 검증된 partition identity를 쓴다. 물리적 성공 증거와 계보 증거는 별개다.

별도 LeRobot Train/Val을 선택하면 manifest의 `storage.sources`가 각 원본 MP4/Parquet를 참조하고,
episode의 `source_index`가 해당 reader를 지정한다. 기존 단일 `storage.root`와 Canonical 입력도 지원한다.
camera/주파수/gripper endpoints/Workspace/보정 모델 계약을 확인하며 Task·Scene·Review는 episode에 유지한다.
통계는 Train만 읽고 검증 시 가중치를 갱신하지 않는다. Val-only Canonical 변환의 statistics.json은
`not_fitted` 표시일 뿐이며, 최종 Train/Val 결합 학습 준비에서 Train 통계를 계산한다.
Resume는 두 입력 원본의 byte identity를 모두 확인한다. 모델 구조/H40/6D/Best/Early Stopping은 유지한다.
사용법: [MINIVLA_INTEGRATION.md](../real2sim/MINIVLA_INTEGRATION.md#source-기반-train--validation-분리).

## Scene 구성 metadata

Workspace 물체 수는 RGB/state/action tensor 모양을 바꾸지 않는다. 실행별 composition과
Task는 canonical manifest의 각 episode에 `scene_configuration`, `task_definition`으로 보존한다.
HDF5/LeRobot와 checkpoint provenance의 저장 위치·legacy 복원·혼합 데이터 계약은
[DATA_FLOW.md](../real2sim/DATA_FLOW.md#실행별-scene-구성-2026-10-15)를 참조한다.
혼합 조합의 학습에서도 source group 단위 split을 유지한다. Direct manifest도 같은 episode
metadata를 보존하며 학습 tensor 차원은 바꾸지 않는다.

Curation용 object telemetry는 선택적 review 정보다. Target 기존 matrix와 active distractor의
float32 xyz+wxyz를 관측 시각/좌표계와 함께 보존하며 모델의 state/action 차원을 늘리지 않는다.
형식과 frame 변환은 [DATA_FLOW.md](../real2sim/DATA_FLOW.md#curation-object-telemetry-2026-10-15)를 따른다.
Converter의 episode별 `episode_review`는 annotation/성공 근거/pose reference를 보존하고,
MiniVLA manifest와 checkpoint provenance에도 전달한다. 이 정보는 학습 입력 tensor가 아니다.
원본 group 계보가 불명확하고 검증된 Source Partition도 없는 생성 자료는 공통 UNKNOWN group으로 묶는다. 독립 source가 부족하면
validation split을 만들 수 없으며, frame/생성 episode 단위로 독립성을 꾸며내지 않는다.

## 하나의 sample

| 항목 | shape/dtype | 의미 |
|---|---|---|
| images | camera별 uint8 [H,W,3] 리스트 | RGB, 0..255, 파일의 camera 순서 |
| instruction | str | episode 단위 task; 공백 정리, 임의 잘라내기 금지 |
| state | float32 [6] | train 통계 z-score, clipping 없음 |
| actions | float32 [40,6] | 관측 직후부터 40개 absolute target의 z-score |
| mask | bool [40] | True=유효 target, False=episode 끝 padding |
| grounding | float32 [2,2] | 첫 camera 원본상의 target/destination 중심 [u,v], 0..1 |
| grounding_mask | bool [2] | UV 라벨의 유효성, 없으면 전부 False |

B=1이 기본. Qwen processor 결과는 input_ids int64 [B,N], attention_mask [B,N],
pixel_values [total_patches, patch_dimension], image_grid_thw [B*camera_count,3].
Qwen language block15 출력 feature BF16 [B,N,2048], head의 VL adapter는 [B,N,256].
좌표 [B,N,2], camera id int64 [B,N], text=0/image=1..4.
추론 출력은 float32/64 [40,6] canonical 물리 단위. joint limiter 적용 전 출력이다.

## 폴더

```text
data/
  manifest.json
  episode_000/
    state.npy               float32 [T,6]
    action.npy              float32 [T,6]
    timestamp.npy           float64 [T] seconds
    action_timestamp.npy    float64 [T] seconds
    side.mp4                새 수집: H264 RGB 영상, 원본 해상도/FPS
    side.npy                과거 형식: uint8 [T,H,W,3] (둘 중 하나)
    side_timestamp.npy      float64 [T]
    grounding.npy           optional float32 [T,2,2]
    grounding_mask.npy      optional bool [T,2]
```

joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
앞 5개 rad, gripper는 닫힘=0 열림=1의 **calibrated fraction**이다. LeRobot percent는
일반적으로 100으로 나누되 실제 calibration 범위/방향을 확인한다. MuJoCo gripper joint가
rad 또는 m인 경우 mapping의 mj_per_canonical/mj_offset으로 명시적으로 변환한다.
그리퍼를 팔의 rad 변환과 함께 처리하지 않는다. 현재 코드는 velocity input을 사용하지 않는다.

manifest 예:
```json
{
  "schema_version": 1,
  "joint_names": ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
  "units": ["rad","rad","rad","rad","rad","fraction"],
  "control_hz": 30,
  "action_semantics": "absolute_target_after_observation",
  "episodes": [
    {"id":"episode_000","path":"episode_000","split":"train","group_id":"source_000","instruction":"pick the red cube"},
    {"id":"episode_001","path":"episode_001","split":"val","group_id":"source_001","instruction":"pick the blue cube"},
    {"id":"episode_002","path":"episode_002","split":"test","group_id":"source_002","instruction":"pick the green cube"}
  ]
}
```

## 시점
obs[t]는 a[t] 적용 **이전**의 이미지/상태다. a[t]는 그 직후 시작하는 제어주기 명령이다.
qpos[t+1]를 command라고 재명명하지 않는다. leader 명령과 follower 관측 차이를 보존한다.
관측은 일정 control_hz로 재표본화해야 한다. 이미지 timestamp는 관측보다 미래이면 안 되고
최대 2 control periods 이내여야 한다. 원본 녹화의 timestamp가 없으면 수집기의 정렬 계약을
먼저 확인한 뒤 생성한다. 이 패키지는 가짜 timestamp를 자동으로 붙이지 않는다.

## 누출 방지와 quality
- split은 frame 기준이 아니라 episode/group 기준. 한 source demo의 Mimic 변형, 동일 장면의
  counterfactual 지시 예시는 같은 group_id. 코드가 group이 split을 넘으면 거부한다.
- train-only Welford 결합 통계, std 최소 .01. 통계 파일의 fingerprint는 train manifest metadata를
  검증하며 데이터 바이트 변조를 탐지하는 hash는 아니다. 데이터를 수정했다면 통계를 재생성한다.
- 초기 무동작·종료 idle trim은 명령/상태/모든 카메라를 같은 timestamp 기준으로 적용한다.
- 성공/실패 및 grasp 품질은 큐레이션 후 manifest에 포함한다. 파일 형식 validator가
  trajectory 성공 여부까지 판단하지는 않는다.
- validation은 고정 시점 최대 episode당 8프레임; checkpoint 선택용 proxy다. 최종 평가는 rollout.
- OOD splits: attribute_ood, composition_ood, object_ood, spatial_ood. train/val에 넣지 않는다.
- 새 object 이름을 instruction에만 넣거나, trajectory 그대로 색만 바꾸면 grounding 학습의
  증거가 되지 않는다. 동일 scene에 복수 후보와 instruction별 올바른 trajectory가 필요하다.
- paraphrase는 원래 target/destination이 보존되는 검수된 문장만 episode 복제/변형으로 제공한다.

## 이미지 처리
256×256 LANCZOS letterbox(640×480→256×192+상하32px 검정) 후 Qwen processor 정규화. crop, flip, hue jitter 없음.
brightness/contrast만 .9~1.1. 과도한 조명 변형은 색 grounding을 망칠 수 있다.
max_instruction_tokens=96, max_tokens=512는 초과 시 예외. 임의 truncate로 image tokens를 자르지 않는다.
2 cameras를 쓰면 카메라 순서와 config, 데이터, MuJoCo mapping을 함께 변경하고 VRAM을 재측정한다.

새 RGB 저장/외부 HDF5 영상 참조의 상세 계약은 [DATA_FLOW.md](../real2sim/DATA_FLOW.md#새-rgb-저장-계약-2026-10-15)를 따른다. Canonical MP4도 숫자 배열·시각·frame 수를 먼저 확인한 뒤 요청 frame만 읽는다. 기존 NPY checkpoint의 dataset byte identity 계산은 유지한다.
