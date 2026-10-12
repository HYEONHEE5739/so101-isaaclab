# 실험 기록 기준

주 비교군은 ACT, Frozen-Qwen deterministic, QLoRA-Qwen deterministic.
동일 manifest/statistics/camera/action units/H16/replan4/cases/seed를 사용한다.
ACT는 ImageNet vision prior만 있으며 자연어를 받지 않는다. 무언어 baseline이 다중 instruction
조건에서 갖는 한계를 명시하고, 학습 task별 ID 성능도 함께 표시한다.
head 구조까지 변경하는 Flow 실험은 adaptation 실험과 별도다.

## 결과 표 (실측값으로 채울 것)

| Policy | Split | Seed | N episodes | Task success | Target accuracy | Wrong-object rate | Grasp | Transport | Latency p95 | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| ACT | ID | | | | | | | | | |
| Frozen | ID | | | | | | | | | |
| QLoRA | ID | | | | | | | | | |
| Frozen | object_ood | | | | | | | | | |
| QLoRA | object_ood | | | | | | | | | |

ID/attribute/composition/open-vocabulary/spatial 결과를 섞어 한 성공률로 표현하지 않는다.
각 seed의 성공 수/전체 수, 평균 및 분산을 보고한다. 소수 사례에서 높은 비율이 나와도
일반화를 입증한 것으로 해석하지 않는다. test/OOD 결과로 checkpoint를 고르지 않는다.

## 구현된 지표와 한계

- grounding success: grasp 이전 target 거리 감소와 distractor 대비 preference, 또는 처음
  approach radius 안에 진입한 대상. distractor 없는 episode는 null.
- target selection accuracy: 처음 접근한 후보가 target인가. 어느 후보에도 도달하지 않으면 null.
  coverage가 낮을 수 있으므로 count도 함께 보고한다.
- grasp: 두 gripper 접촉 그룹이 target에 접촉 + target lift. 물체 모양/접촉 proxy가 부정확하면
  env.measure를 자신의 장면 기준으로 보정해야 한다.
- transport: grasp 후 xy displacement와 lift. place: transport 후 destination 범위에 release 상태로 .3초.
- target/distractor distance는 EEF site와 object body origin 거리. 손가락 surface distance가 아니다.
- command acceleration RMS는 각 관절별 기록. rad/s²와 fraction/s²를 섞은 단일 점수를 만들지 않는다.
- joint limit violation은 raw predicted command가 bounds를 벗어난 step 수; 실행 명령은 clipped.
- completion time은 sim time, wall_seconds는 inference pause 포함 실제 경과시간.
- allocated/reserved는 PyTorch 메모리. MuJoCo/드라이버 포함 전체는 nvidia-smi로 따로 측정.

failure 순서:
TARGET_SELECTION_FAIL → APPROACH_FAIL → GRASP_FAIL → TRANSPORT_FAIL → PLACE_FAIL.
이것은 root cause 확정이 아닌 stage 기반 분류다. grounding_success와 failure를 같이 읽는다.

## 데이터 설계

먼저 같은 scene에 red/blue/green을 함께 두고 instruction마다 다른 target을 조작한다.
source scene과 trajectory 변형은 같은 group으로 묶되 train/val/test에 서로 다른 source groups를
충분히 수집한다. 색/위치/목적지 간 고정 상관관계를 깨고 의미와 동작 둘 다 검수한다.
UV labels는 가능하면 simulation object center projection으로 만들되, 첫 camera 기준 및 occlusion
유효 mask를 저장한다. object identity만 바꾸고 무효해진 grasp trajectory를 성공 demo로 넣지 않는다.

counterfactual paired cases는 초기 robot/object pose와 seed가 같고 instruction/target annotation만
다르도록 한다. Flow 비교 시 noise도 동일해야 한다. probe.py는 matched noise를 사용한다.
새 skill(open drawer)을 한 번도 시연하지 않았다면 그 motor sequence 생성은 이번 성공 기준이 아니다.

## 다음 우선순위

1. 한 clean task에서 tiny real subset 과적합 및 imitation rollout.
2. 같은 장면의 instruction-swap가 올바른 target을 바꾸는지 확인.
3. Frozen/QLoRA 비교: ID 개선이 OOD grounding 손실을 동반하는지 측정.
4. trajectory mode averaging이 관측되면 Flow 비교.
5. H8/H16, head 2층/4층 중 한 번에 한 요소만 변경.

오프라인 action loss가 낮아도 closed-loop success가 높다는 보장은 없다.
