# 데이터 규격

기존 데이터의 축 순서와 단위를 자동 추측하지 않는다. 실제 데모 파일이 제공되지 않았으므로
본 형식으로 변환하거나 명시적 HDF5 recipe를 사용해야 한다. LeRobot parquet/video를 직접
읽는 loader는 현재 포함하지 않는다. decoded RGB/상태/명령을 이 형식으로 내보내면 된다.

## Scene 구성 metadata

Workspace 물체 수는 RGB/state/action tensor 모양을 바꾸지 않는다. 실행별 composition과
Task는 canonical manifest의 각 episode에 `scene_configuration`, `task_definition`으로 보존한다.
HDF5/LeRobot와 checkpoint provenance의 저장 위치·legacy 복원·혼합 데이터 계약은
[DATA_FLOW.md](../real2sim/DATA_FLOW.md#실행별-scene-구성-2026-10-15)를 참조한다.
혼합 조합의 학습에서도 source group 단위 split을 유지한다. 현재 Workflow의 LeRobot→canonical
adapter는 기존 decoder를 통해 RGB/state/action을 이 규격으로 내보낸다.

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
    side.npy                uint8 [T,H,W,3]
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
