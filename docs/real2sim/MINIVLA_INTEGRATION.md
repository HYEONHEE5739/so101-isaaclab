# MiniVLA — 기존 Workspace Pipeline 통합

2026-10-15 현재 구조. 과거 Deterministic/Flow head 선택·별도 UV head 설명을 대체한다.
MiniVLA 모델 package는 기존 `soarm101_lab.learning.minivla` 하나다.
모델 상세는 [DESIGN.md](../minivla/DESIGN.md), [검증](../minivla/VALIDATION.md),
[평가](../minivla/EVALUATION.md), 공통 데이터는 [DATA_FLOW.md](DATA_FLOW.md)를 참조한다.

기존 UI를 사용한다:

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

1. Workspace Pipeline에서 Workspace/Task/backend와 기존 엔진별 Python을 선택한다.
2. Source를 사용자 조작으로 기록한다. 새 Scene의 측정/관절 좌표 계약과30Hz를 유지한다.
3. Replay로 영상/목표를 확인한다. Annotation으로 grasp/place 경계를 생성·검토한다.
4. Datagen/Curation에서 성공한 Source의 변형과 source group을 보존한다.
5. Convert에서 기존 joint-radian/LeRobot 데이터로 변환한다. RGB 원본과 calibration은 보존한다.
6. Train의 학습 모델에서 **MiniVLA · Qwen3-VL + Flow-DiT** 하나를 선택한다.
   head 선택창은 없다. 기본 configs/default.json은 Frozen Qwen/256square/4blocks/H40/flow8/replan4다.
   기존 데이터셋/steps/output/model_config/source_groups 선택과 Resume/Stop은 유지한다.
   QLoRA는 같은 JSON의 adaptation을 qlora로 지정한다. 별도 preset 모델은 없다.
7. 새 policy artifact를 선택하고 Sim Eval을 실행한다. 기존 best.pt를 실제 MiniVLA로 로드한다.
   새로운 task를 고르면 객체 identity와 성공 판정도 바뀐다. Red 학습/Blue 평가는 평가 문서 참조.

이 단계들은 기존 개별 실행/Registry를 사용한다. 이전에 제거한 자동 단계 예약이나 VRAM gate는
추가하지 않았다. Source는 Qwen을 로드하지 않는다. 실제 장치는 UI 시작/탭 전환으로 연결하지 않는다.
Source/Replay/Annotation/Datagen의 Isaac/MuJoCo 분리도 유지한다. MuJoCo worker에 pxr/Isaac import 없음.
모델 추론·학습은 기존 별도 프로세스에서 실행한다. ACT/SmolVLA 비교 기능은 유지했다.

호출 경로:

```text
WorkflowPanel → Operations → 기존 stage worker
Train → workflow.policy.train / workflow.mujoco.train_from_lerobot 또는 train
      → learning.minivla.train → QwenBackbone + ActionHead
MuJoCo Sim Eval → workflow.mujoco.rollout → MiniVLA.predict → 기존 ActionGuard
Isaac Sim Eval → workflow.evaluation.evaluate → MiniVLAJointPolicy → 같은 MiniVLA.predict
```

Canonical 데이터의 state/action 정규화와 gripper fraction을 유지하고, model rad 변환은 기존
RobotCoordinates/Runtime boundary에서만 적용한다. Precision을 데이터에 중복 적용하지 않는다.
새 기본 모델은 Side만 소비하며 기존 Dataset/환경의 Wrist 영상은 유지한다.

체크포인트 architecture_schema=`so101.minivla.flow-dit/1`. 구형 가중치는 명확히 거절하고
사용자 checkpoint를 바꾸지 않는다. 같은 구조의 Resume은 config/stats/control_hz 검사 후
optimizer/step/best를 복원한다. 실제 MiniVLA config 변경은 학습 단계의 기존 JSON 입력을 쓴다.

검증용 2-step 합성 checkpoint를 두 엔진의 기존 Sim Eval 경로에서 실행했다. 이는 경로 검사이며
새 Scene의 Source→실물 성공 데모→학습→성공 rollout 전체가 검증됐다는 의미는 아니다.
결과와 실패 기록: `outputs/workflow_validation/minivla_flow_dit/`.
