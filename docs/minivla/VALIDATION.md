# 검증 결과 — 2026-10-06

이 작업 환경에는 NVIDIA GPU가 없다. 아래 검증과 실제 로봇 성능을 구분한다.

## 직접 실행한 것

- 19 tests passed, 0 skipped (실제 출력은 test-output.txt).
- deterministic/flow head: tensor shape, forward, masked loss, backward, optimizer step, sampling.
- padding 값 변경이 유효 action loss에 영향을 주지 않음; padded VL token 변경도 무시.
- 고정 synthetic batch loss 감소(overfit), VL feature까지 nonzero gradient.
- optional grounding UV query gradient.
- train-only 통계, normalization round-trip, episode-end padding.
- source group leakage 및 잘못된 timestamp rejection.
- HDF5 명시적 key/deg→rad/gripper percent→fraction 변환.
- checkpoint 저장/weights_only load/head 복원.
- 실제 Transformers 4.57.1 Qwen3VLModel을 작은 random configuration으로 생성하여
  RGB patch+text forward, final hidden states, cache off 확인.
- 실제 PEFT LoRA module을 붙인 tiny Qwen에서 action loss→LoRA gradient 확인.
- conditional-generation checkpoint를 저장하고 같은 클래스로 reload하여 weight prefix 보존 확인.
- 실제 Qwen3-VL-2B processor/tokenizer 파일을 다운로드하여 two-camera prompt 처리,
  image token count/좌표/camera id/token budget 검사. 2B model weight는 다운로드하지 않음.
- MuJoCo EGL renderer, explicit joint/actuator mapping, reset/step/RGB; 간단한 mechanical test fixture.
- official LeRobot 0.4.1 ACTPolicy: forward, CVAE loss, backward, eval chunk prediction.
  test에서는 pretrained ResNet 다운로드를 끄고 API/gradient만 검사함.
- Quickstart의 기본 2.98M head 셀 실제 CPU 실행: [1,16,6] 출력 및 optimizer step.
- 전체 Python compileall 통과.

## 아직 확인하지 못한 것

- 실제 Qwen3-VL-2B NF4 가중치 load/forward/backward, bitsandbytes GPU adapter dispatch.
- RTX 4060 Laptop 8GB의 peak VRAM, QLoRA memory fit, 실제 추론 latency.
- 실제 SO-101 demonstrations의 action 학습/수렴/성공률.
- 사용자의 SO-101 MJCF/mesh/camera/servo mapping 및 실제 pick-and-place.
- open-vocabulary / compositional OOD grounding, Frozen 대비 QLoRA 성능 차이.
- 실물 motor 제어 및 실시간 async control (이번 패키지의 구현 범위 밖).

CPU test에서 사용한 주요 환경: Python 3.12, torch 2.5.1+cpu, torchvision 0.20.1+cpu,
Transformers 4.57.1, PEFT 0.17.1. ACT import에 필요한 optional dependencies를 추가하여 검증했다.
이 CPU 환경은 LeRobot 0.4.1의 전체 권장 dependency 범위를 만족하는 clean deployment 검증은
아니다(특히 torchvision 버전). 실제 ACT 실행용 환경은 README대로 별도 환경에서 LeRobot
의 dependency resolver가 요구하는 호환 torch/torchvision 쌍을 설치해야 한다.

## 발견하여 수정한 오류

1. Transformers 4.57.1의 Qwen base class 직접 로드 시 model.* checkpoint key 누락.
   conditional-generation class로 로드 후 .model 추출 + missing keys 검사로 수정.
2. base 추출 시 PEFT 4bit adapter dispatch용 quantization flag 전파.
3. MuJoCo enum과 NumPy scalar의 tuple membership 비교 문제. int 변환으로 명시.
4. Flow instruction 비교에 다른 RNG seed가 섞이지 않도록 case별 seed 지원.

## 사용자의 장비에서 남은 검증 순서

1. Frozen profile → QLoRA profile (메모리/LoRA gradient).
2. 실제 데이터 import + train-only stats + 작은 subset overfit.
3. Frozen ID rollout → instruction-swap → QLoRA 동일 조건.
4. 별도 ACT 환경 baseline → OOD split별 평가.

profile의 검은 입력이나 random weight tests로 manipulation/semantic 성능을 주장하지 않는다.
