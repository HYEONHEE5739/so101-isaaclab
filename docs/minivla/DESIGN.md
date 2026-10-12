# SO-101 MiniVLA 설계 결정

2026-10-06. 첨부 1·2를 통합. 현재 직접 요청한 코드 제작을 우선한다.
본 패키지는 학습 가능한 연구 구현이며 학습된 policy/실측 성공률을 제공하지 않는다.

## 선택
수십~수백 데모에서는 Frozen Qwen + deterministic chunk Transformer를 주 baseline으로
선택한다. 생성 head가 우월하다는 증거가 아직 없고, deterministic은 한 번의 head forward로
학습/추론하며 flow time/noise 분산이 없다. 여러 유효 궤적이 섞여 평균 행동이 문제라면
동일한 토큰 인터페이스의 flow head를 비교한다. 이 deterministic head는 ACT 자체가 아니다.
ACT의 CVAE + 이미지 encoder baseline은 별도 official LeRobot adapter로 제공한다.
Frozen과 QLoRA 비교 시 head 종류·초기화 seed·split·학습 step을 반드시 동일하게 유지한다.

| 요소 | 확정 기본값 |
|---|---|
| VLM | Qwen/Qwen3-VL-2B-Instruct, revision 기록 |
| 정밀도 | NF4 + double quant; BF16 compute (지원 GPU 필수) |
| adaptation | frozen / 마지막 language 8층 q_proj,v_proj LoRA r=4, alpha=8 |
| 이미지 | 최대 256×320 안에서 종횡비 보존; Qwen 32배수 smart resize |
| 카메라 | side 1개 기본, camera list로 wrist 추가 가능 |
| VL 표현 | Qwen3VLModel.last_hidden_state [B,N,2048], cache/전체 hidden states 끔 |
| 토큰 | 모든 유효 토큰, N<=512 hard limit; global pooling 없음 |
| projection | 공유 LayerNorm + Linear 2048→256, 명시적 xy/type/camera embedding |
| state/action | arm 5축 rad + gripper calibrated fraction [0,1]; 고정 6D |
| action | 관측 직후 적용할 absolute position target |
| chunk | 16 @30Hz, 앞 4개 실행 후 replan |
| head | noncausal self-attention + cross-attention, 2층, d=256, 4 heads |
| FFN | GELU, 1024; pre-LayerNorm, dropout=.1 |
| state | MLP 6→256→256; 각 action token에 additive conditioning |
| position | learned 16 action embeddings; VL image 좌표 별도 projection |
| flow option | sinusoidal time 256 + MLP; uniform t, velocity MSE, Euler 8step |
| objective | masked SmoothL1 deterministic / masked weighted MSE flow |
| semantic auxiliary | optional target/destination UV regression, weight .1; 없는 라벨 mask |
| optimizer | AdamW head 1e-4 / LoRA 1e-5; wd .01; clip 1; cosine warmup 5% |
| train | microbatch 1, accumulation 8, episode-balanced sampling |
| execution | synchronous paused-simulation baseline; 실제 wall-time latency 별도 기록 |

Qwen은 causal이다. instruction을 이미지 **앞에도** 배치하고 이미지 뒤에 반복해,
visual tokens가 instruction을 볼 수 있고 suffix tokens가 image를 볼 수 있게 한다.
Qwen hidden feature만 연결한다고 open-vocabulary 행동이 보장되지는 않는다.
선택적 UV supervision, 같은 장면/다른 지시 데모, instruction-swap 평가로 검증한다.

## 연구 비교 (직접 복사한 소스 없음)
- GR00T: generalist action expert의 복잡성을 그대로 가져오지 않는다. embodiment routing 제거.
- π0/openpi: continuous chunk/flow 유용. joint large expert와 pretraining scale은 적용하지 않는다.
- SmolVLA: VLM/action expert 분리와 receding chunks 참고. 100M급 expert는 본 규모에서 과함.
- ACT: low-data action chunk/CVAE baseline. 언어 prior 없는 비교군을 유지한다.
- Diffusion Policy: multimodality 장점, iterative denoising 비용. 이 프로젝트는 flow 하나만 추가.
- OpenVLA/RT-2: action token autoregression은 이 하드웨어/6D 연속 target에 우선 채택하지 않음.
- Octo: multi-embodiment/generalist 기능보다 single-robot 데이터 정합성에 집중.

공유 projection은 반복되는 2048D K/V projection을 피한다. d256, 2층이면 각 층 직접
K/V는 약 2.10M weights, 공유 projection+작은 K/V는 약 .79M weights다.
N을 유지하므로 token compression 정보 손실은 없다. 80~수백 토큰에서 learned resampler를
추가해 새로 학습할 이유가 약하다. 2 camera 또는 고해상도로 N을 늘릴 때 먼저 실측한다.

## VRAM 예산 (추정, 8GB 보장 아님)
사용자 측정 load 3.6GB를 출발점으로 한다. head 2.98M/3.11M 파라미터의 FP32
weights+grad+Adam states는 대략 약 48~50MB. B1,N512,2048 BF16 마지막 feature는 2MiB,
256D projected context는 .25MiB. 실제 피크를 결정하는 것은 Qwen 내부 activation,
vision DeepStack, FP32로 승격되는 quantization 준비 파라미터, allocator와 CUDA workspace다.
Frozen은 no_grad로 graph를 만들지 않는다. QLoRA는 language checkpointing과 작은 rank를
사용하나 8GB를 넘을 수 있다. MuJoCo render context도 별도 메모리가 필요하다.
훈련은 simulation을 끄고 수행한다. rollout에서만 Qwen+MuJoCo를 함께 실행한다.
프로파일러의 peak allocated/reserved + 외부 nvidia-smi를 함께 기록해야 한다.
OOM 시 camera=1 → image bound 축소 → max_tokens 축소 → LoRA 적용층 축소 순서로 조정.
부족한 메모리를 CPU offload로 몰래 해결하지 않는다. 작은 배치의 accumulation은
한 샘플 자체의 activation peak를 줄이지 않는다.

## 수식
Flow: z~N(0,I), t~U(0,1), x=(1-t)z+t a, u=a-z.
L=sum(mask * joint_weight * (v-u)^2)/sum(mask * joint_weight).
Padding action은 학습 입력에서도 제거하고 self-attention key padding mask로 차단한다.
Euler: x_{k+1}=x_k+v(x_k,k/K)/K, K=8. 4step 품질은 평가 대상.
Deterministic: 같은 mask denominator의 SmoothL1. Gripper weight 기본 1; 필요시 비교.
Absolute target은 delta 누적 오차를 피하고 dataset의 position-control 계약을 직접 따른다.
State/action 각각 train-only mean/std, std floor .01. 정규화 clipping은 하지 않아 경계를
조용히 왜곡하지 않는다. 출력은 실행 전에 physical joint bounds와 속도 제한을 적용한다.

## 검증 계획
ID / attribute / composition / object vocabulary / spatial OOD를 별도 split한다.
source demo 및 counterfactual scene group 전체를 같은 split에 묶는다.
동일 initial scene에 다른 instruction을 주고 target preference/EEF distance를 비교한다.
올바른 대상 접근 후 grasp 실패는 grounding success + GRASP_FAIL이다.
최소 ACT/Frozen/QLoRA, 동일 seeds/scenes/splits/action mapping, 3 seeds 권장.
추가 ablation 우선순위: deterministic vs flow, instruction swap, 2 vs 4 head layers.
Train loss 감소나 random tensor의 language sensitivity는 semantic generalization 증거가 아니다.

## 공식 참조
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- https://huggingface.co/docs/peft/en/developer_guides/quantization
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py
- https://github.com/tonyzhaozh/act
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py
- https://github.com/NVIDIA/Isaac-GR00T

## 체크포인트 호환성
Transformers 4.57.1에서 Qwen3VLModel에 conditional-generation checkpoint를 직접 로드하면
model.* prefix가 맞지 않아 weight 누락이 발생함을 작은 체크포인트로 재현했다.
따라서 Qwen3VLForConditionalGeneration으로 로드하고 missing/mismatched keys 검사 후
내부 .model만 보유/호출한다. LM head forward는 하지 않는다.
