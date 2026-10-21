"""Stage option examples and Korean field reference for the workflow editor.

Only options consumed by Operations and its stage runners are listed here.
UI-owned task/mode/source controls are intentionally not editable JSON examples.
"""
import json
from html import escape

# key: (JSON type, suggested/default value, explanation)
FIELDS = {
    "ik_failure_policy": ("IK Failure Policy", "strict", "Strict: 기존 동작. Hold: 미수렴/오차 초과 시 측정 자세 유지. Best Effort: 수렴 판정과 무관하게 유효한 candidate 적용. Hold/Best Effort는 직전 명령 대비 팔 관절당 0.5rad 초과 점프, 비유한값, 제한 위반 시 중단하며 clipping하지 않습니다."),
    "ik_solver": ("IK Solver", "scipy_ls", "SciPy Least Squares는 이전 명령, Pinocchio DLS는 측정 관절각에서 시작합니다. 수치 알고리즘은 그대로이며 미수렴 처리는 IK Failure Policy에서 선택합니다. Strict는 기존 SciPy 탈락/DLS hold를 유지합니다. 자동 solver 전환은 없습니다. 이 선택값이 실행 JSON에 저장됩니다."),
    'num_envs': ('환경 수', 1, '동시에 생성할 환경 수: 1, 2, 4, 8. Isaac은 벡터 환경, MuJoCo는 공유 모델과 독립 상태를 사용합니다. 목표 성공 수와 최대 시도 수는 모든 환경을 합산합니다.'),
    'validation_dataset':('문자열',None,'선택: 별도 Offline Validation Dataset Path. 과거 Partition 역할은 강제하지 않습니다.'),
    'identity_sim_for_validation':('true/false',False,'명시적 URDF 원형 검증: Scene 관절 잔차만 제외합니다. Hardware Precision은 유지합니다. 파생 단계는 원본 모델 계약을 상속합니다.'),
    'generation_seed': ('정수', None, '생략하면 실행마다 새 seed로 배치와 궤적 선택을 생성합니다. 결과 .generation.json의 seed를 지정하면 재현할 수 있습니다. Workspace seed와 같은 값은 허용하지 않습니다.'),
    'device': ('문자열', 'cuda:0', '실행 장치. 예: cuda:0. Isaac 작업에는 현재 NVIDIA GPU 설정을 유지하세요.'),
    'num_successful_demos': ('정수', 10, '만들 성공 데이터 수. 예: 10이면 성공 episode 10개까지 재시도합니다. 실패 시도는 포함하지 않으며 중지 버튼으로 중단할 수 있습니다.'),
    'repo_id': ('문자열', '', '필수. 예: my-account/so101-demo. 변환에서는 로컬 dataset 식별자이며 업로드는 별도 단계입니다.'),
    'architecture': ('문자열', 'smolvla', '학습 모델: smolvla 또는 act (MiniVLA 기존 추론만 지원). 엔진 선택과 독립적으로 고릅니다.'),
    'renderer': ('문자열', 'egl', 'MuJoCo RGB 렌더 경로: egl(GPU), software-egl(Mesa CPU), osmesa. Isaac은 기존 렌더러를 사용합니다.'),
    'include_failed': ('true/false', False, 'MuJoCo 변환에서 실패로 기록된 데모도 명시적으로 포함합니다. 기본값은 성공 데모만 포함합니다.'),
    'max_attempts': ('정수', 10000, '두 엔진 공통 전체 최대 시도 수. IK·계획 탈락과 실제 집기·배치 실패를 별도로 기록합니다. 목표 미달이면 완료 데이터와 실패 보고서를 보존합니다.'),
    'steps': ('정수', 10000, '총 목표 학습 step 수. Resume에서 40000 → 100000이면 추가 60000 step 학습합니다.'),
    'batch_size': ('정수', 8, '학습 batch 크기. GPU 메모리가 부족하면 줄이세요.'),
    'num_workers': ('정수', 0, '데이터 로더 worker 수. 0은 별도 worker 없이 실행합니다.'),
    'save_freq': ('정수', 1000, 'checkpoint 저장 간격(step).'),
    'dataset_repo_id': ('문자열', None, '선택. 생략하면 입력 dataset meta/info.json의 repo_id를 사용합니다.'),
    'base_model': ('문자열', None, '선택. 사용할 pretrained 모델의 경로 또는 repo ID. SmolVLA는 생략해도 lerobot/smolvla_base로 파인튜닝합니다. ACT는 지정한 경우에만 pretrained 모델을 사용합니다.'),
    'rename_map': ('객체', None, 'SmolVLA 기본: observation.images.side_cam → observation.images.camera1, observation.images.wrist_cam → observation.images.camera2. 다른 모델이면 입력 키에 맞춰 지정하세요.'),
    'policy_options': ('객체', {}, '선택. LeRobot 모델별 추가 config. type/device/push_to_hub/repo_id는 여기서 지정하지 않습니다.'),
    'episodes': ('정수', 1, '선택한 task마다 평가할 episode 수.'),
    'max_steps': ('정수', 300, 'episode당 최대 평가 step 수.'),
    'private': ('true/false', True, '업로드 repository 공개 여부. true는 비공개. 기존 repository의 설정과 일치해야 합니다.'),
    'create_repo': ('true/false', False, 'true면 새 Hugging Face repository를 생성합니다. 기존 repo에는 false를 사용하세요.'),
    'camera_devices': ('객체', {}, '추가 카메라 연결: Workspace/모델의 카메라 이름 → 실제 USB 장치 경로 또는 인덱스. 예: front → /dev/cam_side. 일반 Side/Wrist는 위 장치 경로만 설정하고 이 표는 비워두세요. 같은 이름을 표에도 입력하면 표의 경로를 우선합니다.'),
    'revision': ('문자열', 'main', '업로드할 Hugging Face branch 이름.'),
    'create_branch': ('true/false', False, '지정한 업로드 branch를 새로 만들지 여부.'),
    'port': ('문자열', '/dev/so101_follower', '실제 Follower serial 장치 경로. 저장된 연결 설정을 기본으로 채웁니다. 다른 로봇이면 직접 수정하세요.'),
    'follower_id': ('문자열', '', 'Follower 보정 파일을 찾는 ID. 저장된 연결 설정을 기본으로 채웁니다. 다른 로봇이면 해당 로봇의 보정 ID로 수정하세요.'),
    'calibration_dir': ('문자열', None, '선택. follower calibration 폴더. 생략하면 기존 기본 경로를 사용합니다.'),
    'side_device': ('문자열', '/dev/cam_side', '실제 Side 카메라 장치 경로.'),
    'wrist_device': ('문자열', '/dev/cam_wrist', '실제 Wrist 카메라 장치 경로.'),
    'fps': ('정수', 30, '실제 평가 루프 목표 주파수(Hz).'),
}
KEYS = {
    'source': ['identity_sim_for_validation','device', 'renderer'],
    'replay': ['device', 'renderer'],
    'annotate': ['num_envs', 'device', 'renderer'],
    'datagen': ['ik_solver', 'ik_failure_policy', 'num_successful_demos', 'num_envs', 'generation_seed', 'max_attempts', 'device', 'renderer'],
    'convert': ['repo_id', 'include_failed'],
    'train': ['architecture','validation_dataset','steps','batch_size','num_workers','save_freq','device','dataset_repo_id','base_model','rename_map','policy_options'],
    'validate': ['device','batch_size'],
    'sim_eval': ['episodes', 'num_envs', 'max_steps', 'device', 'renderer'],
    'real_eval': ['port', 'follower_id', 'side_device', 'wrist_device', 'camera_devices', 'episodes', 'max_steps', 'fps', 'device', 'calibration_dir'],
    'hf_dataset': ['repo_id', 'private', 'create_repo', 'revision', 'create_branch'],
    'hf_policy': ['repo_id', 'private', 'create_repo', 'revision', 'create_branch'],
}

# Only options consumed by each trainer belong in its execution form.
TRAIN_KEYS = {name:set(KEYS['train'])-{'architecture'} for name in ('act','smolvla')}


def template(stage):
    # Optional fields without a default are documented but not populated with fake paths.
    return {key: FIELDS[key][1] for key in KEYS[stage] if FIELDS[key][1] is not None}


def help_html(stage):
    rows = []
    for key in KEYS[stage]:
        kind, default, description = FIELDS[key]
        value = '생략 가능' if default is None else json.dumps(default, ensure_ascii=False)
        rows.append(f'<p><b>{key}</b> · {kind} · 기본/예시 <code>{escape(value)}</code><br>{escape(description)}</p>')
    owned = 'Task는 위 Task Selector, 실행 모드와 Python은 위 입력란에서 설정합니다.'
    if stage == 'train':
        owned += ' 학습 모델은 위 학습 모델 드롭다운에서 선택합니다. 이어서 학습은 checkpoint를 선택하세요. Resume에서는 총 steps/output 외 학습 설정은 checkpoint에서 복원하며 기본 출력은 원래 학습 폴더입니다. JSON의 architecture보다 드롭다운 선택이 우선합니다.'
    if stage == 'source':
        owned += ' Source 저장 이름은 필수이며 sources/<이름>/origin에 저장합니다. Leader port, Episodes, Episode Time, Reset Time도 위 전용 입력란에서 수정하세요.'
    if stage == 'annotate':
        owned += ' 자동/수동 방식은 위 Annotation 방식에서 선택하세요.'
    return '<p>아래 JSON의 <b>값</b>을 수정하세요. 문자열은 큰따옴표, 숫자는 따옴표 없이, 참/거짓은 true/false로 입력합니다. JSON에는 주석을 넣지 않습니다.</p><p>' + owned + '</p>' + ''.join(rows) + '<p>API key/token은 입력하지 마세요. 업로드 인증은 기존 로컬 로그인 설정을 사용합니다.</p>'
