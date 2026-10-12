# SO-101 MiniVLA

첨부된 두 요구사항을 바탕으로 만든 **학습 가능한 소스 코드**다.
이미 학습된 가중치나 SO-101 성공률을 제공하는 패키지는 아니다.
Qwen3-VL-2B의 의미/시각 표현을 소수 데모로 학습하는 6D action chunk에 연결한다.

기본 모델: **Frozen Qwen NF4 → 공간 토큰 유지 → 256D, 2층 Transformer → 16×6 absolute targets**.
Deterministic head 2,976,264 parameters; flow head 3,109,640 parameters (Qwen 제외).
같은 구조에서 Frozen/QLoRA 비교. 별도 official LeRobot ACT 비교군도 포함한다.

## 포함 기능

- Qwen 4bit NF4 + BF16, frozen no_grad / language 마지막 8층 q,v LoRA r4.
- 마지막 hidden state만 추출. LM logits/전체 hidden states/KV cache 생성 안 함.
- image token별 xy·camera 정보 + token cross-attention. global mean pooling 없음.
- deterministic chunk 기본 + flow matching 옵션(Euler 8step).
- RGB/언어/관절 전처리, train-only 통계, episode/group split 검증, padding mask.
- episode-balanced sampling, accumulation, 차등 LR, early stopping, checkpoint/resume.
- optional target/destination UV supervision, instruction-swap probe.
- MuJoCo position-servo adapter, 4-step receding horizon, 관절/속도 제한.
- grounding/grasp/transport/place 분리 평가 + latency/메모리 측정.
- 직접 실행한 테스트 결과: [VALIDATION.md](VALIDATION.md).

## 1. 설치

프로젝트 루트에서 기존 conda 환경을 사용한다. 이번 통합은 패키지 설치나 버전 변경을 하지 않는다.
별도 `so101_minivla` 배포판은 제거했고 모델은 프로젝트의 `source` 안에 있다.
기존 `soarm101_lab` 설치가 없다면 아래 `PYTHONPATH`만 설정해 실행할 수 있다.

```bash
export PYTHONPATH="$PWD/source/soarm101_lab${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q tests/minivla
```

필요한 패키지는 torch, numpy, Pillow, Qwen 실행용 transformers/peft/accelerate/bitsandbytes,
시뮬레이션용 mujoco/scipy/psutil이다. 기존 설치 버전을 먼저 확인한다.
프로젝트 `setup.py`에 optional dependency 정보를 통합했지만 기존 conda에서 extras를
무조건 설치하거나 upgrade할 필요는 없다. 환경의 다른 패키지 버전을 바꾸지 않는다.
Qwen 및 processor는 최초 실행 시 Hugging Face에서 다운로드할 수 있다.
CPU의 tiny-model API 검사는 실제 2B/NF4 VRAM 검증과 구분한다.

## 2. 네 RTX 4060에서 먼저 확인

```bash
python -m soarm101_lab.learning.minivla.profile --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/frozen_deterministic.json
python -m soarm101_lab.learning.minivla.profile --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/qlora_deterministic.json
```

이 명령은 실제 2B NF4 backbone으로 forward → loss → backward → optimizer step → inference를
수행한다. train peak allocated/reserved, 추론 latency, LoRA gradient를 확인한다.
검은 dummy 이미지는 실행 검사 목적이며 성능 평가는 아니다.
**8GB 통과 여부는 이 장비에서 실측해야 한다.** QLoRA와 MuJoCo 동시 훈련은 기본 목표가 아니다.
시뮬레이터는 데모 생성 또는 rollout 때 실행하고 학습 중에는 종료한다.

## 3. 데이터 준비

[DATA_CONTRACT.md](DATA_CONTRACT.md)의 NPY episode format으로 준비한다.
현재 실제 데모 파일이 제공되지 않았으므로 기존 HDF5 key/단위/카메라 이름을 추측하지 않았다.
HDF5 파일이면 `source/soarm101_lab/soarm101_lab/learning/minivla/configs/hdf5_import.template.json`을 복사해 실제 경로와 key를 채운다.

```bash
# 현재 환경에 h5py가 설치되어 있어야 한다.
python scripts/real2sim/workflow.py import-hdf5 my_import_recipe.json data/so101
python -m soarm101_lab.learning.minivla.data data/so101/manifest.json data/so101/statistics.json --cameras side
```

이미 canonical NPY 형식이라면 import 명령은 생략한다.
train/val/test 각각 독립 episode/source group이 필요하다. 단일 train episode로 통계/형식은 검사할 수
있지만 학습 명령은 validation split도 요구한다. OOD는 checkpoint 선택에 사용하지 않는다.

## 4. Frozen부터 학습

```bash
python -m soarm101_lab.learning.minivla.train \
  --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/frozen_deterministic.json \
  --manifest data/so101/manifest.json \
  --stats data/so101/statistics.json \
  --out outputs/workflow/runs/minivla/frozen
```

기본 microbatch=1, accumulation=8, 5,000 optimizer updates.
`best.pt`, `last.pt`, `metrics.jsonl`, 실제 model revision/환경 버전을 기록한다.
중단 후에는 `--config outputs/workflow/runs/minivla/frozen/config.json`으로 저장된 revision을 고정하고
`--resume outputs/workflow/runs/minivla/frozen/last.pt`를 추가한다.
학습 config를 바꾼 이어학습은 조용히 허용하지 않는다. 별도 run을 만든다.
모델과 optimizer가 포함된 checkpoint는 본인이 만든 파일을 사용한다.

## 5. QLoRA와 Flow 비교

```bash
python -m soarm101_lab.learning.minivla.train \
  --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/qlora_deterministic.json \
  --manifest data/so101/manifest.json \
  --stats data/so101/statistics.json --out outputs/workflow/runs/minivla/qlora
```

Frozen/QLoRA는 같은 action head·dataset·seed로 처음부터 시작한다.
head만 비교할 때 `frozen_flow.json`을 사용한다. QLoRA 여부와 head 유형을 동시에 바꾸면
어떤 요소 때문에 성능이 변했는지 알 수 없다. Flow 4step은 8step baseline 이후 비교한다.

## 6. ACT 비교군

`soarm101_lab.learning.minivla.act_baseline`은 **LeRobot 0.4.1의 실제 ACTPolicy**를 사용한다.
pretrained ResNet18 + CVAE + 공식 기본 Transformer 구조. VLA imitation head를 ACT라고 부르지 않는다.
데이터, action normalization, episode split, sampling, horizon/replan은 동일하다.
ACT 입력 이미지는 geometry-preserving letterbox + ImageNet normalize를 사용한다.
stock ACT와의 차이는 padding loss를 valid element 수로 나누는 처리와 본 패키지의 학습 loop다.
언어 입력을 받지 않으므로 같은 장면/다른 instruction에서는 지시에 따라 구분할 수 없는 비교군이다.

별도 환경에서 맞는 CUDA torch/torchvision 설치 후:
```bash
python -m pip install 'lerobot==0.4.1'
python -m pip install -e '.[sim,test]'
python -m soarm101_lab.learning.minivla.act_baseline \
  --config source/soarm101_lab/soarm101_lab/learning/minivla/configs/frozen_deterministic.json \
  --manifest data/so101/manifest.json \
  --stats data/so101/statistics.json --out outputs/workflow/runs/minivla/act
```

ACT가 더 많은 학습을 필요로 하는 경우 공정한 validation tuning budget도 별도로 제공한다.
동일 step만으로 최적 성능이 같다고 가정하지 않는다.

## 7. MuJoCo rollout

**네 SO-101 MJCF/mesh 장면 파일은 첨부되지 않았다.** 본 패키지는 이를 새로 추측해 만들지 않는다.
`source/soarm101_lab/soarm101_lab/learning/minivla/configs/robot_mapping.template.json`을 복사해 실제 XML, actuator/joint/camera/site 이름,
물리 joint limit와 max_velocity, gripper 단위 변환을 채운다. `null`은 반드시 실측/모델값으로
바꿔야 한다. lower/upper/max_velocity는 canonical rad 또는 fraction, 초 단위다.
MuJoCo gripper qpos = canonical fraction × mj_per_canonical + mj_offset.
직접 joint transmission + gear=1 position actuator만 지원한다.

`source/soarm101_lab/soarm101_lab/learning/minivla/configs/cases.template.json`에는 동일 scene에서 빨강/파랑 선택을 비교할 두 case가 있다.
initial robot state, object freejoint pose, destination 중심/허용반경을 채운다.
장면 object body origin이 물체 중심과 다르면 거리/placement 판정을 맞춰 수정해야 한다.

```bash
MUJOCO_GL=egl python -m soarm101_lab.learning.minivla.rollout \
  --checkpoint outputs/workflow/runs/minivla/frozen/best.pt \
  --mapping my_robot_mapping.json \
  --cases my_cases.json --out evaluations/frozen
```

QLoRA도 checkpoint만 변경한다. ACT는 `--policy act --checkpoint outputs/workflow/runs/minivla/act/best.pt`.
각 episode JSON에 target selection, grasp, transport, place, failure, action smoothness,
limit violations, latency, peak GPU allocation 기록. summary.json은 split별 집계.
각 모델에 동일 cases와 seed를 사용한다. instruction-swap paired cases에는 동일한 `seed`를 지정한다. 초기 환경 seed와 stochastic flow seed도 기록된다.

**시간 모드:** inference 중 simulation을 멈추는 synchronous 평가다. 관측 시점이 변하지 않아
오래된 관측에 대한 action 적용을 피한다. 30Hz는 simulation time이며 실시간 보장이 아니다.
H16 예측 중 앞 4개만 실행하고 버린 뒤 새 RGB/state로 replan한다. inference latency가
4/30초를 넘으면 wall-time 실시간 제어 조건을 만족하지 못한다. asynchronous real-robot
제어는 별도 stale-result 폐기/hold controller가 필요하며 현재 구현하지 않았다.
Temporal ensemble은 gripper 전환 지연 가능성 때문에 기본 사용하지 않는다. 속도 제한만 적용한다.
이 패키지는 실물 serial motor 제어 코드를 포함하지 않는다.

## 8. 언어를 실제로 쓰는지 확인

```bash
python -m soarm101_lab.learning.minivla.probe --checkpoint outputs/workflow/runs/minivla/frozen/best.pt \
  --manifest data/so101/manifest.json --episode episode_002 --frame 0 \
  --instruction-a 'pick the red cube' \
  --instruction-b 'pick the blue cube' --out instruction_probe.json
```

동일 이미지/상태/flow noise에서 instruction만 바꾼다. 출력 차이가 있다는 것만으로
정확한 grounding이라고 판정하지 않는다. paired rollout에서 EEF가 맞는 대상으로 접근해야 한다.
UV head는 해당 supervision이 없다면 학습되지 않은 보조 출력이므로 위치 추정으로 신뢰하지 않는다.
banana를 알아보는 Qwen이라도 이 head가 적은 데모로 그 정보를 행동에 연결할지는 실험 대상이다.

## 코드 읽는 순서

`config.py` → `data.py` → `backbone.py` → `head.py` → `train.py` → `policy.py` → `rollout.py` → `simulation/mujoco_backend.py`.
`Quickstart.ipynb`는 단계별 shape/gradient smoke test와 실행 명령을 안내한다.
설계 선택·연구 출처·VRAM 추정은 [DESIGN.md](DESIGN.md).
# 프로젝트 MuJoCo 통합

전체 프로젝트에서 수집·학습·추론할 때는 [통합 실행 안내](../real2sim/MINIVLA_INTEGRATION.md)를
따른다. 기존 Real2Sim `Workspace Pipeline`의 `MuJoCo` backend와 `scripts/real2sim/workflow.py worker`가
같은 worker를 사용한다. 아래 standalone 모델 명령과 분리하여 workspace·좌표·dataset
provenance 및 실제 RGB VRAM gate를 검사한다.

## 책임별 코드 위치

모델·학습은 `source/soarm101_lab/soarm101_lab/learning/minivla/`에 있다.
MuJoCo 엔진/환경은 `simulation/mujoco_backend.py`, canonical recording·annotation·
datagen·HDF5 import는 `workflow/episodes.py`, 단계 실행은 `workflow/mujoco.py`,
계획과 공통 worker dispatch는 `workflow/operations.py`다. 이전 모델 폴더의
service/worker/runtime/teleop/episodes/import_hdf5/mujoco_env 모듈은 통합 후 삭제했다.
이를 재-export하는 호환 wrapper는 없다.

실행 진입점은 `python scripts/real2sim/workflow.py --help`에서 확인한다. 모델
config 4개는 Frozen/QLoRA × deterministic/flow의 서로 다른 실험 설정이므로 유지한다.
