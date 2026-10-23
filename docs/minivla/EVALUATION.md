> Historical / Archived — 2026-10-22. 과거 모델 설계/실험 기록이며 삭제된 학습 명령은 실행 안내가 아닙니다. 현재 지원 범위는 [README](README.md)를 따릅니다.

# Scene 구성과 목표 일반화 평가

Workspace Pipeline의 `Scene objects`와 Task를 각각 선택한다. 사용법과 물체 등록은
[BUILDER_WORKSPACE.md](../real2sim/BUILDER_WORKSPACE.md)를 참조한다.

| 실험 | 학습 Active / Target | 평가 Active / Target | 의미 |
|---|---|---|---|
| A | Red / Red | Red / Red | 기본 동작 |
| B | Red·Blue·Green / Red | 동일 3개 / 각 색 | 같은 장면에서 새 목표 명령 |
| C | Red / Red | Red·Blue·Green / Blue 또는 Green | 구성과 목표를 함께 변경 |

B와 C를 같은 언어 일반화 실험으로 해석하지 않는다. C의 실패는 distractor 증가와 목표 변경이
결합된 결과다. 원본 Workspace/로봇/카메라/Action 계약 검사는 유지한다. Object Set만 다르면
허용하며 `training_active_object_sets`, `evaluation_active_object_ids`,
`scene_composition_distribution_shift`를 결과에 기록한다. 학습 구성 근거가 없으면 null이다.

각 episode는 active count/IDs, target/distractors, selected ID 또는 UNKNOWN,
target_selection_accuracy, wrong_object_approach, 모든 물체의 현재/최소 TCP 거리·접근 여부,
grasp/transport/place/full_task_success를 기록한다. 물체 1개일 때 선택 정확도·오접근은 null(N/A)이며
자동 성공으로 계산하지 않는다. 선택 ID는 grasp 전 40mm 내 첫 최근접 물체라는 **기하 proxy**다.
실제 의미 인식의 정답 라벨이 아니다. UNKNOWN도 정답으로 세지 않는다. Grasp는 기존 StableGrasp,
transport는 grasp 이후 40mm 이동, place는 선택한 Task의 기존 내부 포함·정지 조건을 사용한다.
Full success는 세 조건 모두 필요하다. 기존 success_rate는 place 조건이며 별도로 해석한다.
MiniVLA 실제 replan의 `inference_latency_s`와 매 select_action의 `action_call_latency_s`를 구분한다.
실제 추론 시점을 알 수 없는 정책은 inference 값 null로 남기고 action call 시간을 제공한다.

Scene 구성은 모델의 새로운 입력 차원이 아니다. Qwen3-VL-2B/256×256 letterbox/
4-block Flow-DiT/H40/6D 및 checkpoint schema는 그대로다. 성공률은 독립 rollout으로 검증해야 한다.

## Red-only → Blue Zero-shot 평가

학습은 UI에서 Red Task의 Source/Datagen만 선택한다. Blue는 같은 RGB의 distractor로
남길 수 있지만 Blue-target episode를 train/val에 넣지 않는다. generated variants는
source group 단위로 train/val 분리하고 normalization도 train에서만 계산한다.
새 architecture checkpoint를 평가한다. 구형 MiniVLA 가중치는 호환하지 않는다.

UI Sim Eval에서 Blue task를 선택하면 `workflow.tasks.configured`가 pick_object_id,
place 대상 및 success 조건을 바꾼다. 지시문만 Blue로 바꾸고 Red 성공을 측정하지 않는다.
사용자 지정 문구와 identity는 독립이다. identity를 자연어에서 추측하지 않는다.
같은 원칙이 MuJoCo Backend와 Isaac task_success_details에 연결된다.

동일 관측의 Red/Blue Action 비교:

```bash
python -m soarm101_lab.learning.minivla.probe --checkpoint /path/best.pt \
  --manifest /path/manifest.json --episode episode_0000 --frame 0 \
  --instruction-a 'Pick the red cube' --instruction-b 'Pick the blue cube' \
  --seed 42 --out /tmp/action-swap.json
```

같은 원본 이미지/state/Flow noise를 사용한다. 정규화된 action RMS 차이, 각 instruction의
latency/VRAM을 남긴다. 차이가 있다는 사실은 Blue 선택의 정확도 증거가 아니다.
삭제된 UV head 값을 semantic 측정치로 사용하지 않는다.

Qwen의 시각/언어 인식만 별도로 확인:

```bash
python -m soarm101_lab.learning.minivla.probe --manifest /path/manifest.json \
  --episode episode_0000 --frame 0 --vlm-question 'What color is the cube on the right?' \
  --expected-answer blue --out /tmp/qwen-qa.json
```

공식 Qwen ConditionalGeneration loader와 동일 letterbox로32token 이내 greedy answer를 생성한다.
expected-answer가 있으면 exact match를 기록하나 실제 장면의 별도 정답 라벨이 있어야 한다.
이 QA는 Action Expert를 실행하지 않는다. QA 정답률과 action 접근/집기 성공률은 분리한다.

물리 rollout의 paired trial은 기존 `rollout.py`와 configs/cases.template.json을 사용한다.
동일 robot_state/free_joint_poses/destination/seed를 주고 target_body와 distractor_bodies를
Red/Blue로 서로 바꾼다. `pair_id`도 동일하게 준다. 모든 물체의 reset pose를 명시한다.
초기 관측 hash/state와 noise seed가 다르면 paired comparison을 거절한다.
각 replan은 case_seed+step으로 같은 Flow noise를 사용한다.

```bash
python -m soarm101_lab.learning.minivla.rollout --checkpoint /path/best.pt \
  --mapping /path/robot_mapping.json --cases /path/paired-cases.json --out /path/new-eval
```

case split을 red_id/blue_zero_shot으로 나누면 summary도 별도로 나온다.
기록 지표는 target selection, wrong-object approach, grasp, transport, place_success,
full_task_success, action difference, latency, peak VRAM이다. full success는 grasp/transport/place
모두 충족해야 한다. grasp 이전 TCP 접근 proxy는 의미 정확도의 완전한 측정이 아니다.
처음부터 컵 안에 둔 fixture의 place=True를 full task 성공으로 세지 않는다.

추론 중 simulation을 멈추고4개 action을 실행한다.30Hz physics/control 계약이 있어도 실제
wall-clock30Hz를 보장하지 않는다. 프로파일의4/30초 replan budget과 실제 latency를 비교한다.
동일 reset의 전체 물리 rollout/새 자세가 있어야 학습 정책의 성공률을 보고할 수 있다.
현재 실제 수행된 검증 및 미수행 성공률은 VALIDATION.md 참조.
