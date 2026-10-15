# MiniVLA 검증 — 2026-10-15

기존 MiniVLA를 Qwen3-VL-2B + Small Flow-DiT로 교체한 코드 기준이다.
실행 근거: `outputs/workflow_validation/minivla_flow_dit/`. 사용자 Scene/Calibration/사진/
Dataset/기존 checkpoint는 수정하지 않았다. 환경 dependency 설치/버전 변경 없음.

## 자동 검사

| 그룹 | 결과 | 로그 |
|---|---:|---|
| MiniVLA architecture/letterbox/실제 processor/tiny Qwen LoRA/API/data/checkpoint/ACT |20PASS|model_tests_final.log|
| 기존 UI/MuJoCo backend/integration/Source/Resume/data |67PASS|integration_tests_final.log|
| 별도 USD/Task/Workspace 계약 |13PASS|isaac_contract_tests.log|

합계100PASS. 엔진 import 충돌을 피하려고 USD와 MuJoCo 검사를 별도 프로세스로 실행했다.
실제 Qwen Processor는 로컬 cache 경로를 지정하여 생략하지 않았다. 256square/grid1×16×16/
image64 tokens와 XY·mask 정합을 확인했다. 실제 library의 tiny random28층 Qwen에서
layers15 output과 선택feature의 동일성, frozen no-grad, QLoRA8~15 action gradient를 확인했다.
Random tiny model은 pretrained2B/4bit 검증과 구분한다.

## RTX4060 Laptop8GB 실제 Qwen NF4

torch2.7.0+cu128, transformers4.57.1, bitsandbytes0.48.2, peft0.17.1.
로컬 Qwen revision89644892e4d85e24eaac8bacfd4f463576704203. CPU offload 없음.

| 항목 | Frozen | QLoRA |
|---|---:|---:|
| train peak allocated |1.627GiB|2.235GiB|
| train peak reserved |2.164GiB|3.328GiB|
| warm 추론 중앙값 |93.77ms|94.59ms|
| inference peak allocated |1.555GiB|2.146GiB|
| output |1×40×6|1×40×6|
| LoRA Action Loss gradient |해당 없음|PASS, q/v layers8~15|

ActionHead7,319,558params(adapter 포함). Forward/Backward/optimizer2회, inference5회
(첫 호출 제외4회 중앙값), B1·단일 합성 RGB·N111 조건이다. Frozen Qwen grad 없음.
학습 accumulation8 또는 긴 학습/동시 엔진 구동의 peak를 보장하지 않는다.
4/30=133.3ms replan budget을 이 단독 warm profile에서는 충족했다.
전체 camera/render/제어 I/O를 포함한 실시간 성능 인증은 아니다.

## 실제 Train / Checkpoint / Worker

기존 train.py CLI로 합성3frame train/val을2step 학습하고 best.pt/last.pt를 저장했다.
새 schema와 state_dict를 MiniVLA 추론 API로 복원하여40×6 출력/같은 noise 재현을 확인했다.
기존 MuJoCo Sim Eval로 해당 best.pt를4step 실행하고 Blue Task identity를 사용했다.
`mujoco_sim_eval/checkpoint_consumed.json` 참조. success=False이며 작업 성공으로 세지 않는다.
Isaac도 기존 evaluate 경로에서 같은 체크포인트/Blue identity로4step 실행·결과 저장을 완료했다.
단독 Resume2→3step 및 별도 Qwen QA generation도 통과했다. QA exact match는False로
기록됐고 Semantic 성공률로 세지 않는다. 자세한 로그/경위는 같은 폴더 REPORT.md 참조.

## 실패 기록과 남은 범위

초기 tiny API검사2건은 테스트 kwargs에 use_cache 중복으로 실패해 호출 옵션 병합 후 통과했다.
Resume fixture1건은 새 schema field가 없어 의도대로 거절돼 새 fixture를 수정했다.
Task검사는 pxr 없는 환경에서 collection 실패 후 별도 USD 환경으로13PASS.
Isaac 첫4step은 실행됐지만 fixture artifact ID 누락으로 결과 저장에 실패했다.
GPU 검증을 겹쳐 실행한 시도는 OOM이 발생했다. 로그를 보존하고 검증 프로세스만 종료했다.
패키지 변경/무음 offload/품질 저하로 숨기지 않았다. 후속 최종 결과는 REPORT.md 참조.

학습된 SO101 grasp/place 성공률, Red-only→Blue zero-shot 성공률, 실물 policy는 미검증이다.
원본 작업데이터로 충분히 학습한 뒤 독립 paired rollout이 필요하다.2step synthetic checkpoint와
instruction action 차이를 실제 Semantic Accuracy로 표시하지 않는다.
