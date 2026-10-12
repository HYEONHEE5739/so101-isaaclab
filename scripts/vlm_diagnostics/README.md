# 원본 SmolVLM2 단일 이미지 진단

전체 실행 코드는 [test_smolvlm.py](test_smolvlm.py)에 있습니다.
LeRobot dataset, SmolVLA action expert, fine-tuning, robot rollout은 사용하지 않습니다.
입력은 PNG/JPG 파일 하나이며 실제 Side/Wrist 카메라의 저장된 프레임을 사용하세요.

## 실행

프로젝트 루트에서:

```bash
conda activate lerobot-arena
python scripts/vlm_diagnostics/test_smolvlm.py --image "/절대/경로/real_side.png"
```

`--image` 바로 뒤를 본인의 이미지 경로로 바꾸면 됩니다. 상대경로는 실행한
터미널의 현재 디렉터리 기준입니다. 공백이 있으면 따옴표로 감싸세요.
Wrist 이미지는 별도 실행에서 `--image "/절대/경로/real_wrist.png"`로 지정합니다.
이미지를 별도 폴더에 복사하거나 UI에 등록할 필요는 없습니다.

현재 `lerobot-arena`에서 Transformers 4.57.1 / torch 2.7.0+cu128 API를 확인했습니다.
현재 lerobot-arena에는 필요한 패키지가 있으므로 아래 설치 명령을 실행할 필요가 없습니다.
기존 학습 환경에서 무조건 upgrade/reinstall하지 마세요.
별도의 새 CUDA PyTorch 환경에 패키지가 없는 경우에만:

```bash
python -m pip install 'transformers==4.57.1' pillow num2words
```

Flash Attention 추가 설치는 필요 없습니다. PyTorch SDPA를 사용합니다.
첫 실행은 Hugging Face에서 모델/processor 다운로드를 위해 인터넷과 저장공간이
필요합니다. 캐시가 완성된 이후에는 `--local-files-only`로 오프라인 실행할 수 있습니다.

## 동작

- 원본 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`를 한 번만 로드합니다.
- CUDA 자동 선택, 지원 시 BF16, 아니면 FP16; CPU는 FP32입니다.
- `eval()`, `requires_grad_(False)`, `inference_mode()`로 gradient를 만들지 않습니다.
- 질문마다 독립적인 chat template + 이미지 1장으로 처리합니다. 이전 답변이 다음
  질문에 들어가지 않으므로 첫 답변의 오류가 대화 이력으로 누적되지 않습니다.
- greedy decoding, beam 1, 최대 128 새 토큰; 입력 토큰 길이 뒤의 답변만 출력합니다.
- 8번째 bounding box는 **검증되지 않은 텍스트**입니다. 좌표 단위/정확성도
  보장되지 않으며 detection이나 로봇 이동 목표로 사용하지 않습니다.

출력 형태(실제 모델 답변을 미리 가정하지 않은 예시):

```text
Model: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
Device: cuda; dtype: torch.bfloat16
...
============================================================
Q1. Describe the objects visible in this robot workspace.
------------------------------------------------------------
A: <모델이 생성한 답변>
============================================================
...
Q8. Locate the red cube and return its bounding box as [x_min, y_min, x_max, y_max].
...
```

## 메모리 부족

먼저 Isaac Sim/학습 등 다른 GPU 작업을 종료하고 아래로 시도합니다.

```bash
python scripts/vlm_diagnostics/test_smolvlm.py \
  --image "/절대/경로/real_side.png" \
  --no-image-splitting --max-image-side 512 --max-new-tokens 64
```

이미지 splitting을 끄면 시각 토큰을 줄일 수 있지만 작은 물체 인식에 영향을
줄 수 있습니다. 해상도 축소만으로 processor의 고정 resize에 따른 토큰 수가
항상 줄지는 않습니다. 최후에는 `--device cpu`를 사용할 수 있으나 느립니다.
기본 실행에서는 원본 이미지 크기와 processor의 기본 전처리를 유지합니다.

이 테스트의 언어 답변 정확도는 VLA의 제어 성공률과 동일한 지표가 아닙니다.
색/개수/영상 기준 좌우/상대위치를 직접 확인해 기록하고, 실제 이미지가 바뀌어도
같은 답을 반복하는지 비교하세요. 잘못된 답변만으로 SmolVLA 실패 원인을 확정할
수는 없습니다.

공식 모델/API 참고:
https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# Qwen3-VL-2B 비교 진단

전체 코드: [test_qwen3vl.py](test_qwen3vl.py). 독립적인 질문 11개:
자유 묘사 → 블록/컵/그리퍼 인식 → 영상 내 공간 관계 → 지시 대상/목적지.
매 질문에서 이미지와 질문만 넣으며 이전 답변을 전달하지 않습니다.

현재 lerobot-arena의 Transformers 4.57.1에서 공식
`Qwen3VLForConditionalGeneration`, `AutoProcessor` import가 됩니다.
**기본 실행에 Transformers 재설치/업데이트는 필요 없습니다.**

```bash
conda activate lerobot-arena
python scripts/vlm_diagnostics/test_qwen3vl.py \
  --image "$HOME/Pictures/스크린샷/smolvla_test_image.png"
```

RTX 4060에서는 BF16을 우선 사용합니다. SDPA, batch 1, greedy decoding,
최대 256 새 토큰, 이미지 예산 262144 pixels로 설정했습니다. 이미지 예산은
processor resize 상한이며 작은 이미지의 디테일을 새로 만들어 주지 않습니다.
VRAM 8GB에서 성공을 보장하지는 않으므로 다른 학습/Isaac 프로세스는 종료하세요.
원본 2B 모델은 최초 다운로드가 필요합니다. 캐시 이후 `--local-files-only` 지원.

OOM 시 우선:

```bash
python scripts/vlm_diagnostics/test_qwen3vl.py \
  --image "$HOME/Pictures/스크린샷/smolvla_test_image.png" \
  --max-pixels 65536 --max-new-tokens 128
```

## 선택적 4-bit

현재 bitsandbytes는 미설치입니다. 기존 학습 환경을 바꾸지 않으려면 복제 환경에서
설치하세요(복제에 추가 디스크 공간이 필요합니다). 아래 명령은 자동 실행하지 않았습니다.

```bash
conda create --name qwen-vlm-test --clone lerobot-arena
conda activate qwen-vlm-test
python -m pip install --dry-run bitsandbytes
# 설치 계획을 확인한 후, 복제 환경에서만 설치
python -m pip install bitsandbytes
python scripts/vlm_diagnostics/test_qwen3vl.py \
  --image "$HOME/Pictures/스크린샷/smolvla_test_image.png" --load-in-4bit
```

NF4 + double quantization, BF16 compute를 사용합니다. 4-bit 모델에는 추가 `.to()`를
호출하지 않고 로드 시 GPU에 배치합니다. 양자화로 답변이 달라질 수 있으므로
BF16과 결과를 구분하세요. optimizer/학습/LeRobot 로봇 제어는 사용하지 않습니다.
다른 환경에서 Qwen3-VL import가 안 되면 별도 환경에서
`python -m pip install 'transformers==4.57.1' pillow accelerate`를 사용하세요.

출력은 `Q1. ... / A: ...` 형식이며 입력 prompt 토큰을 제외한 답변만 표시합니다.
공간 질문은 화면 기준 좌우/영상상 근접 관계로 안내합니다. 단일 이미지에서 실제
3D 거리나 로봇 제어 성공 여부를 검증하는 기능은 아닙니다.

검증 범위: 로컬 모델 클래스/API import, CLI 도움말 및 문법 확인.
Qwen 가중치가 로컬에 없어 실제 BF16/4-bit 생성은 아직 검증하지 않았습니다.
공식 사용법: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
