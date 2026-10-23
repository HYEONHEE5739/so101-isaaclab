# Native LeRobot Dataset / Training / Model 사용

현재 설치 환경: LeRobot **0.4.1**, Python3.11, 공식 entry point `lerobot-train` (`lerobot.scripts.lerobot_train:main`). Workflow는 같은 모듈을 선택한 Python의 별도 process에서 호출한다. 설치된 Trainer/optimizer/scheduler/Policy/Processor/Dataset 구현을 수정하거나 복제하지 않는다.

## 입력과 Policy 목록

Local Dataset은 `meta/info.json`이 있는 LeRobot v3.0 root를 선택한다. 외부 Hub Dataset은 **`hf://owner/dataset@revision`**으로 지정한다. `@revision`은 생략할 수 있으며 실제 실행은 commit SHA로 고정한 공식 HF snapshot cache를 사용한다. 메타데이터 준비 확인에서는 `meta/**`만 받고, Worker가 실제 Dataset payload와 공식 Reader sample을 확인한다. 같은 Repo의 다른 revision이 하나의 mutable Dataset cache root를 공유하지 않는다.

Workspace·Task Script는 일반 Training/Hub에 필요 없다. Source/Annotation/DataGen/Curation 결과는 기존 Conversion으로 LeRobot Dataset을 만든 다음 사용한다. 원본 Dataset을 다른 학습용 포맷으로 복사하거나 그 좌표/단위를 새로 추정하지 않는다. 외부 데이터에 SO-101 sidecar를 만들어 붙이지 않는다.

```sh
conda activate lerobot-arena
python scripts/real2sim/workflow.py lerobot-info
```

이 명령과 UI Policy 목록은 설치된 공식 factory를 읽는다. 현재 항목은 `act`, `diffusion`, `groot`, `pi0`, `pi05`, `reward_classifier`, `sac`, `smolvla`, `tdmpc`, `vqbet`다. 이는 **공식 factory 항목 목록**이며 모든 optional dependency·로봇·Dataset·학습 모드를 검증했다는 뜻이 아니다. 선택한 모델의 dependency/config/feature 오류는 공식 CLI 오류로 표시한다. 다른 모델로 대체하지 않는다. MiniVLA는 이 목록에 등록하지 않으며 기존 연구 모델/추론 경로를 유지한다.

## Training UI

1. **모델 학습**에서 Dataset Source를 Local 또는 Hugging Face로 선택한다. Hub ID는 `owner/dataset[@revision]`을 입력한다. **Dataset 가져오기**로 새 Local Path에 명시 다운로드하는 기존 기능도 유지한다.
2. Policy와 **Scratch / Fine-tuning / Resume**를 선택한다. 저장 이름, 출력 폴더, steps/batch/device 등을 설정한다. Fine-tuning의 `base_model`은 실행 설정에 local pretrained_model 또는 Hub ID를 입력한다. **Hugging Face Model 가져오기**는 commit에 고정한 native Model을 공식 cache에 받고 선택한다. 업로드 기능이 아니다.
3. **Native CLI 옵션**에 `--key=value`를 추가한다. **LeRobot CLI 명령 / Dataset Metadata 보기·복사**에서 선택 Dataset의 feature metadata·실행 argv·config를 확인한다. Hub base Model은 Worker에서 고정 snapshot으로 해석한 최종 명령을 `[LEROBOT CLI]` 로그와 `<출력이름>.command.json`에도 남긴다.
4. 시작/중지는 기존 QThread→Operations→독립 Worker/process group을 사용한다. 공식 로그의 Step/Loss/Checkpoint를 표시하고 **학습 Loss 그래프**는 현재 process.log의 공식 `step:... loss:...` 줄을 읽는다. 새로운 학습 metric을 계산하지 않는다.

Training 실행에는 새 출력 위치가 필요하다. Resume는 원래 출력 위치를 사용하며 기존 Checkpoint·Dataset·optimizer/RNG/step·선택적 scheduler 조건을 검증한다. 현재 학습이 소유한 출력 Lock만 내부 완료/Resume 검사에서 허용하고 외부 Reader의 쓰기 중 접근 거부는 유지한다.

## CLI 예제

`train_options.json`을 새 위치에 작성한다.

```json
{
  "architecture": "act",
  "training_mode": "scratch",
  "storage_name": "act_external",
  "device": "cpu",
  "steps": 10000,
  "batch_size": 8,
  "num_workers": 0,
  "save_freq": 1000,
  "native_args": ["--log_freq=10", "--policy.chunk_size=100"]
}
```

```sh
# Workspace 없이 공개 Dataset 사용. 준비 확인은 Training 전체 실행 검증이 아니다.
python scripts/real2sim/workflow.py ready --stage train --input hf://lerobot/pusht@v3.0 --options train_options.json
python scripts/real2sim/workflow.py run --stage train --input hf://lerobot/pusht@v3.0 --options train_options.json

# 같은 options와 로컬 Dataset도 지원한다.
python scripts/real2sim/workflow.py run --stage train --input /local/lerobot_dataset --options train_options.json
```

`native_args`는 문자열(shell quoting 규칙으로 분해) 또는 문자열 목록이다. shell을 실행하지 않으며 알 수 없는 공식 옵션은 LeRobot parser가 오류로 거부한다. optimizer/scheduler, Dataset image transforms, 모델별 옵션 등을 해당 설치 버전에서 지원하는 형식으로 지정할 수 있다. Dataset identity/root/revision/episodes, output/config/resume, Policy type/path/device/publication 및 Checkpoint 생성은 Workflow가 소유하므로 추가 CLI에서 중복 변경하면 거부한다. Train 전체를 쓰며 Episode subset은 Curation으로 별도 Dataset을 만든다. 실제 적용된 공식 `train_config.json`과 원래 argv를 보존한다.

### 세 가지 모드

| 모드 | 옵션 | 실제 실행 계약 |
|---|---|---|
| Scratch | `training_mode="scratch"` | 새 Policy/optimizer. `base_model`과 조합 불가. 모델 자체의 공식 Backbone 기본값은 해당 Config 책임 |
| Fine-tuning | `training_mode="finetune", base_model="/local/pretrained_model"` 또는 `"hf://owner/model@revision"` | 공식 `--policy.path`로 Config/weights를 읽고 새 Dataset으로 optimizer/step을 새로 시작. 선택 Policy type이 실제 Model type과 다르면 거부 |
| Resume | `training_mode="resume", resume_checkpoint="/output/checkpoints/000100", steps=...` | 원래 `train_config.json`·optimizer/RNG/step 및 설정된 scheduler 복원. 새 base_model/native_args와 조합 불가 |

`training_mode`가 없는 기존 호출은 이전 규칙을 유지한다(Resume 경로 우선, pretrained 지정 시 Fine-tuning, 기본 SmolVLA는 smolvla_base Fine-tuning, ACT는 Scratch). 기본 SmolVLA base도 Worker에서 commit에 고정한다. 명시 Model revision은 `model_revision` 또는 `hf://...@revision`을 사용하며 서로 다르면 오류다. Local Path에는 revision 의미가 없다.

현재 Offline Validation은 기존 ACT/SmolVLA만 지원한다. 다른 Policy의 학습이 가능하더라도 새로운 Validation objective까지 구현된 것은 아니다. Val을 지정한 경우 학습 후 독립 평가 실패는 Checkpoint를 삭제하거나 학습 성공을 되돌리지 않는다.

## Model 가져오기와 Sim Evaluation

```sh
python scripts/real2sim/workflow.py import-hf-model lerobot/smolvla_base --revision <tag-or-commit>
```

native config/weights·저장된 pre/postprocessor 및 각 state 파일을 유지한다. Model ID를 Local Path로 오인하지 않도록 Script에서는 다음처럼 명시한다.

```python
SCRIPT = {
    "stage": "sim_eval",
    "workspace": "/validated/workspace",
    "checkpoint": "hf://owner/so101_model@commit",
    # 기존 scene/options/evaluations 선언을 그대로 작성
}
```

일반 LeRobot Model 로딩과 SO-101 물리 평가 허가는 다르다. Sim/Real은 기존 Joint 순서·6D state/action·model-radian units·Robot/Mapping/Workspace provenance를 요구한다. 누락 정보를 다른 Model에서 빌리거나 보완하지 않는다. 공개 smolvla_base 및 일부 과거 사용자 Model은 native 로딩/추론이 가능해도 이 물리 실행 계약이 없어 거부된다.

현재 SO-101 Runner는 ACT/SmolVLA/Diffusion 경로를 받아들이며 다른 Policy는 명시 거부한다. Diffusion 실제 학습/배치 평가는 이번 미검증이다. Camera는 Runtime의 Side/Wrist640×480와 저장된 rename processor를 검사하고, SmolVLA의 공식 resize/missing-camera 동작을 유지한다. 추가 Observation feature나 처리 경로 없는 shape는 오류다. 모델의 Action horizon/queue/reset은 공식 Policy가 소유하고 Workflow는 유한 `[num_envs,6]` 출력과 기존 Guard를 검사한다.

Task별 Instruction은 `evaluate_sim`→`batch_actions`의 task batch→저장된 tokenizer/processor→실제 모델 입력으로 전달한다. 이번 실제 SmolVLA CPU2batch 검사에서 서로 다른 Instruction의 token이 실제 `select_action` 입력에서 달랐다. 이것은 언어 조건 전달 검사이며 Policy 작업 성공률 검사가 아니다.

## Hub 업로드 및 인증

기존 **Dataset 업로드 / Policy 업로드** Stage만 명시 실행 때 공식 HfApi로 게시한다. 학습 완료로 자동 공개하지 않으며 `push_to_hub=false`를 유지한다. Repo visibility·create_repo/branch/revision 설정과 기존 인증을 사용한다. Token은 HF 로그인/환경에서만 제공하며 UI 옵션·Config·로그에 넣지 않는다. 업로드의 실제 원격 변경은 이번 검증에서 실행하지 않았다.

## 내가 직접 확인할 것

1. Workspace 없이 Training에서 Local/Hub Dataset, Policy, 모드를 선택하고 명령 미리보기→시작→Loss/Checkpoint를 확인한다.
2. `learning.datasets`의 `installed_policies/inspect_dataset/resolve_model`, `learning.training.training_command/train`, 기존 `workflow.operations`를 읽는다. 공식 학습 코드는 설치된 `lerobot/scripts/lerobot_train.py`다.
3. [이번 검증 보고서](../../outputs/workflow_validation/native_integration/REPORT.md)에서 실제 학습/Resume/UI/Sim 결과와 미검증 Policy/Hub 업로드를 구분한다.
