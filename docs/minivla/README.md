# MiniVLA

Qwen3-VL-2B NF4 → 16번째 language block → token adapter → Small Flow-DiT 4blocks.
출력은40×6 canonical absolute joint targets, Euler8회,4action 실행 후 replan이다.
기존 패키지/Worker/UI를 그대로 사용한다. Deterministic/별도 UV head/구형 presets는 제거했다.

구조는 [DESIGN.md](DESIGN.md), 데이터는 [DATA_CONTRACT.md](DATA_CONTRACT.md),
검증은 [VALIDATION.md](VALIDATION.md), 평가법은 [EVALUATION.md](EVALUATION.md).
기존 Source→Train/Sim Eval UI 연결은 [MINIVLA_INTEGRATION.md](../real2sim/MINIVLA_INTEGRATION.md).

기존 conda 환경을 사용한다. 설치/버전 변경은 하지 않았다.

```bash
export PYTHONPATH="$PWD/source/soarm101_lab"
python -m soarm101_lab.learning.minivla.profile \
  --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/default.json \
  --out /tmp/minivla-profile.json
# 같은 config의 adaptation만 qlora로 비교
python -m soarm101_lab.learning.minivla.profile \
  --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/default.json \
  --adaptation qlora --out /tmp/minivla-qlora-profile.json
```

Dataset은 train/val source group 분리와 train-only normalization을 요구한다.
원본640×480은 유지하고 입력 때만256×256 letterbox를 적용한다. 저장 관절/목표를 재보정하지 않는다.

```bash
python -m soarm101_lab.learning.minivla.train --config /path/config.json \
  --manifest /path/manifest.json --stats /path/statistics.json --out /path/new-policy
# total steps를 늘린 같은 config로 optimizer/weights/step 복원
python -m soarm101_lab.learning.minivla.train --config /path/extended-config.json \
  --manifest /path/manifest.json --stats /path/statistics.json \
  --out /path/new-policy --resume /path/new-policy/last.pt
```

`best.pt/last.pt`는 새 architecture_schema를 포함한다. 구형 MiniVLA checkpoint와
head_type 설정은 지원하지 않는다. 원본 파일은 삭제/덮어쓰지 않는다. ACT/SmolVLA는 별도
기존 Workflow 모델로 유지된다.

실제 성공률은 충분한 학습과 rollout으로 확인해야 한다. 개발용2-step 합성 checkpoint는
배포·작업 수행용이 아니다. 현재 결과와 한계는 VALIDATION.md에 적었다.
