"""Official LeRobot training orchestration; model algorithms remain upstream."""
from pathlib import Path
import json

def resume_checkpoint(path, artifact, options):
    """Resolve a full LeRobot checkpoint without modifying its saved state."""
    root = Path(path).expanduser().resolve()
    if root.name == 'train_config.json':
        root = root.parent
    if root.name == 'pretrained_model':
        root = root.parent
    model = root / 'pretrained_model'
    state = root / 'training_state'
    required = [model / 'train_config.json', model / 'config.json', model / 'model.safetensors']
    required += [state / name for name in (
        'training_step.json', 'optimizer_param_groups.json', 'optimizer_state.safetensors',
        'rng_state.safetensors', 'scheduler_state.json')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('불완전한 resume checkpoint: ' + ', '.join(missing))
    config = json.loads((model / 'train_config.json').read_text())
    step = int(json.loads((state / 'training_step.json').read_text())['step'])
    if Path(config['dataset']['root']).expanduser().resolve() != Path(artifact['path']).resolve():
        raise ValueError('Resume은 원래 학습에 사용한 dataset을 선택해야 합니다.')
    if config['policy']['type'] != options.get('architecture', 'smolvla'):
        raise ValueError('선택한 모델과 checkpoint 모델이 다릅니다.')
    if int(options.get('steps', config['steps'])) <= step:
        raise ValueError(f'총 목표 steps는 checkpoint step {step}보다 커야 합니다.')
    from ..workflow.policy import inspect_checkpoint
    from ..representation import require_compatible
    from ..so101_dataset_contract import JOINT_SPACE
    from .datasets import inspect_dataset
    record=inspect_checkpoint(model)
    selected=inspect_dataset(artifact['path'])
    require_compatible(record['coordinates'],selected['coordinates'],JOINT_SPACE)
    previous=record.get('provenance',{}).get('source_dataset_sha256')
    if previous:
        from ..utils.artifact_identity import fingerprint
        if fingerprint(Path(selected['path']))!=previous:raise ValueError('Resume dataset bytes changed')
    return model / 'train_config.json', config, step


def resume_output(path, architecture='smolvla'):
    """Resolve the source training directory for resume contract checks."""
    root = Path(path).expanduser().resolve()
    if architecture == 'minivla':raise ValueError('MiniVLA 학습 Resume는 비활성화되어 있습니다.')
    if root.name == 'train_config.json': root = root.parent
    if root.name == 'pretrained_model': root = root.parent
    config = json.loads((root / 'pretrained_model/train_config.json').read_text())
    if root.parent.name != 'checkpoints':
        raise ValueError('Resume checkpoint must be under the original output/checkpoints directory')
    output = root.parent.parent
    if config.get('output_dir') and Path(config['output_dir']).expanduser().resolve() != output:
        raise ValueError('Saved training output differs from checkpoint location')
    return output


def train(train_path, options, output):
    import subprocess
    import sys
    from ..so101_dataset_contract import write_contract, JOINT_SPACE
    from ..representation import describe
    if options.get('architecture') == 'minivla':
        raise ValueError('MiniVLA 신규 학습 및 학습 Resume는 비활성화되어 있습니다. 기존 추론은 지원합니다.')
    from .datasets import validate_datasets
    artifact,validation,evidence=validate_datasets(train_path,options.get('validation_dataset'))
    coordinates = artifact['coordinates']
    if coordinates['action_space'] != JOINT_SPACE:
        raise ValueError('Training requires joint-radian targets')
    from ..real2sim.storage import atomic
    dataset = Path(artifact['path'])
    info = json.loads((dataset / 'meta/info.json').read_text())
    config = {'dataset': {'repo_id': options.get('dataset_repo_id', info.get('repo_id', 'local/so101')),
                          'root': str(dataset)},
              'policy': {'type': options.get('architecture', 'smolvla'), 'push_to_hub': False,
                         'device': options.get('device', 'cuda')},
              'output_dir': str(output), 'steps': int(options.get('steps', 10000)),
              'batch_size': int(options.get('batch_size', 8)), 'num_workers': int(options.get('num_workers', 0)),
              'save_checkpoint': True, 'save_freq': int(options.get('save_freq', 1000)), 'wandb': {'enable': False}, 'eval_freq': 0}
    if config['policy']['type'] not in ('act', 'smolvla'):
        raise ValueError('Choose act or smolvla')
    overrides = options.get('policy_options', {})
    if not isinstance(overrides, dict) or any(k in overrides for k in ('type', 'push_to_hub', 'repo_id', 'device', 'pretrained_path')):
        raise ValueError('Invalid policy_options; workflow owns type/device/publication')
    config['policy'].update(overrides)
    base_model = options.get('base_model') or (
        'lerobot/smolvla_base' if config['policy']['type'] == 'smolvla' else None)
    camera_keys = [k for k in info.get('features', {}) if k.startswith('observation.images.')]
    default_map = {key:'observation.images.camera'+str(i+1) for i,key in enumerate(camera_keys)}
    config['rename_map'] = options.get('rename_map', default_map if config['policy']['type'] == 'smolvla' else {})
    if not isinstance(config['rename_map'], dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in config['rename_map'].items()):
        raise ValueError('rename_map must map observation key strings to strings')
    if base_model:
        config['policy']['pretrained_path'] = base_model
    cfg_path = Path(output).with_name(Path(output).name + '.train_config.json')
    atomic(cfg_path, config)
    argv = [sys.executable, '-m', 'lerobot.scripts.lerobot_train', '--config_path=' + str(cfg_path)]
    if base_model:
        # Load checkpoint configuration as well as weights; validate reloads it.
        argv.append('--policy.path=' + str(base_model))
        for key, value in config['policy'].items():
            if key not in ('type', 'pretrained_path'):
                argv.append('--policy.' + key + '=' + json.dumps(value))
    if options.get('resume_checkpoint'):
        saved_path, config, step = resume_checkpoint(options['resume_checkpoint'], artifact, options)
        config = {**config, 'output_dir': str(output), 'resume': True,
                  'steps': int(options.get('steps', config['steps']))}
        # Keep config_path at the original checkpoint: LeRobot derives optimizer/
        # scheduler and processor locations from its parent directories.
        argv = [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
                '--config_path=' + str(saved_path), '--resume=true',
                '--output_dir=' + str(output), '--steps=' + str(config['steps'])]
        atomic(cfg_path, config)
        print(f"[WORKFLOW] 이어서 학습: step {step} → {config['steps']} · {saved_path.parent}", flush=True)
    subprocess.run(argv, check=True)
    checkpoint = Path(output) / 'checkpoints/last/pretrained_model'
    if not (checkpoint / 'config.json').exists() or not list(checkpoint.glob('*.safetensors')):
        raise ValueError('Training did not produce a complete checkpoint')
    write_contract(checkpoint, coordinates)
    from ..utils.artifact_identity import fingerprint
    atomic(checkpoint / 'workflow_provenance.json', {'workspace': artifact.get('workspace'),
           'source_dataset': str(dataset), 'source_dataset_sha256':fingerprint(dataset), 'backend': options.get('backend', artifact.get('backend', 'isaac')), 'training': config, 'representation': describe(coordinates),
           'task_definitions': artifact.get('tasks') or options.get('tasks', []),
           'scene_configuration':artifact.get('provenance',{}).get('scene_configuration'),
           'scene_configurations':artifact.get('provenance',{}).get('scene_configurations',[]),
           'episode_review':artifact.get('provenance',{}).get('episode_review',{}),
           'episode_scene_configurations':artifact.get('provenance',{}).get('episode_scene_configurations',{})})


    if validation:
        # A separate process owns evaluation RNG/model memory; official trainer is finished.
        import uuid
        validation_output=Path(output)/('offline_validation_'+uuid.uuid4().hex[:8])
        result=subprocess.run([sys.executable,'-m','soarm101_lab.learning.validation',
            '--checkpoint',str(checkpoint),'--dataset',validation['path'],'--output',str(validation_output),
            '--device',options.get('device','cpu'),'--batch-size',str(options.get('validation_batch_size',1))])
        atomic(Path(output)/'offline_validation_result.json',dict(training_status='SUCCEEDED',
            validation_status='SUCCEEDED' if result.returncode==0 else 'FAILED',returncode=result.returncode,
            report=str(validation_output/'report.json')))
        if result.returncode:print('[VALIDATION] FAILED; completed Training checkpoint remains available',flush=True)
