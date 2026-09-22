"""Stage option examples and Korean field reference for the workflow editor.

Only options consumed by Operations and its stage runners are listed here.
UI-owned task/mode/source controls are intentionally not editable JSON examples.
"""
import json
from html import escape

# key: (JSON type, suggested/default value, explanation)
FIELDS = {
    'generation_seed': ('정수', None, '생략하면 실행마다 새 seed로 배치와 궤적 선택을 생성합니다. 결과 .generation.json의 seed를 지정하면 재현할 수 있습니다. Workspace seed와 같은 값은 허용하지 않습니다.'),
    'output': ('문자열', '', '선택. 비우면 고유 run 폴더에 자동 저장합니다. 직접 지정할 때는 아직 존재하지 않는 경로를 사용하세요.'),
    'device': ('문자열', 'cuda:0', '실행 장치. 예: cuda:0. Isaac 작업에는 현재 NVIDIA GPU 설정을 유지하세요.'),
    'workspace_id': ('문자열', 'workspace_002', '필수. 새 workspace 이름. 기존 ID/version을 덮어쓰지 않습니다.'),
    'version': ('정수', 1, 'Workspace 버전 번호.'),
    'task_file': ('문자열', '', '필수. 실제 존재하는 task JSON 파일 경로를 입력하세요.'),
    'num_successful_demos': ('정수', 10, '만들 성공 데이터 수. 예: 10이면 성공 episode 10개까지 재시도합니다. 실패 시도는 포함하지 않으며 중지 버튼으로 중단할 수 있습니다.'),
    'repo_id': ('문자열', '', '필수. 예: my-account/so101-demo. 변환에서는 로컬 dataset 식별자이며 업로드는 별도 단계입니다.'),
    'architecture': ('문자열', 'smolvla', '학습 모델: smolvla 또는 act.'),
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
    'revision': ('문자열', 'main', '업로드할 Hugging Face branch 이름.'),
    'create_branch': ('true/false', False, '지정한 업로드 branch를 새로 만들지 여부.'),
    'port': ('문자열', '', '필수. 실제 follower serial 장치 경로. 예: /dev/so101_follower.'),
    'follower_id': ('문자열', '', '필수. 실제 follower calibration에 사용한 ID.'),
    'calibration_dir': ('문자열', None, '선택. follower calibration 폴더. 생략하면 기존 기본 경로를 사용합니다.'),
    'side_device': ('문자열', '/dev/cam_side', '실제 Side 카메라 장치 경로.'),
    'wrist_device': ('문자열', '/dev/cam_wrist', '실제 Wrist 카메라 장치 경로.'),
    'fps': ('정수', 30, '실제 평가 루프 목표 주파수(Hz).'),
    'enable_motion': ('true/false', False, '실제 모터 동작 허용. 실제 평가를 의도할 때만 true로 변경하세요.'),
}
KEYS = {
    'accept': ['output'],
    'publish': ['workspace_id', 'version', 'task_file'],
    'source': ['device', 'output'],
    'replay': ['device', 'output'],
    'annotate': ['device', 'output'],
    'datagen': ['num_successful_demos', 'generation_seed', 'device', 'output'],
    'convert': ['repo_id', 'output'],
    'train': ['architecture', 'steps', 'batch_size', 'num_workers', 'save_freq', 'device', 'output', 'dataset_repo_id', 'base_model', 'rename_map', 'policy_options'],
    'sim_eval': ['episodes', 'max_steps', 'device', 'output'],
    'real_eval': ['port', 'follower_id', 'side_device', 'wrist_device', 'episodes', 'max_steps', 'fps', 'enable_motion', 'device', 'output', 'calibration_dir'],
    'hf_dataset': ['repo_id', 'private', 'create_repo', 'revision', 'create_branch', 'output'],
    'hf_policy': ['repo_id', 'private', 'create_repo', 'revision', 'create_branch', 'output'],
}


def template(stage):
    # Optional fields without a default are documented but not populated with fake paths.
    return {key: FIELDS[key][1] for key in KEYS[stage] if FIELDS[key][1] is not None}


def help_html(stage):
    rows = []
    for key in KEYS[stage]:
        kind, default, description = FIELDS[key]
        if key == 'output':
            example = ('/tmp/my_demo.hdf5' if stage in ('source', 'annotate', 'datagen') else
                       '/tmp/my_dataset' if stage == 'convert' else
                       '/tmp/my_training' if stage == 'train' else '/tmp/my_result.json')
            description += ' 경로 형식 예: ' + example + ' (직접 입력할 때만 사용).'
        value = '생략 가능' if default is None else json.dumps(default, ensure_ascii=False)
        rows.append(f'<p><b>{key}</b> · {kind} · 기본/예시 <code>{escape(value)}</code><br>{escape(description)}</p>')
    owned = 'Task는 위 Task Selector, 실행 모드와 Python은 위 입력란에서 설정합니다.'
    if stage == 'train':
        owned += ' 학습 모델은 위 학습 모델 드롭다운에서 선택합니다. 이어서 학습은 checkpoint를 선택하세요. Resume에서는 총 steps/output 외 학습 설정은 checkpoint에서 복원하며 결과는 새 출력 폴더에 저장합니다. JSON의 architecture보다 드롭다운 선택이 우선합니다.'
    if stage == 'source':
        owned += ' Leader port, Episodes, Episode Time, Reset Time도 위 전용 입력란에서 수정하세요.'
    if stage == 'annotate':
        owned += ' 자동/수동 방식은 위 Annotation 방식에서 선택하세요.'
    return '<p>아래 JSON의 <b>값</b>을 수정하세요. 문자열은 큰따옴표, 숫자는 따옴표 없이, 참/거짓은 true/false로 입력합니다. JSON에는 주석을 넣지 않습니다.</p><p>' + owned + '</p>' + ''.join(rows) + '<p>API key/token은 입력하지 마세요. 업로드 인증은 기존 로컬 로그인 설정을 사용합니다.</p>'
