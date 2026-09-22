"""Shared ACT/SmolVLA inference, preserving LeRobot saved normalization processors."""
import json
from pathlib import Path
from ..representation import require_policy


def load_policy(path, expected, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    require_policy(path, expected)
    cfg = PreTrainedConfig.from_pretrained(path)
    if cfg.type not in ('act', 'smolvla'):
        raise ValueError('Workflow supports ACT and SmolVLA policies')
    policy = get_policy_class(cfg.type).from_pretrained(path)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={'device_processor': {'device': device}})
    return policy, pre, post


def observation(q, side, wrist, task, device):
    import torch
    out = {'observation.state': torch.as_tensor(q, dtype=torch.float32, device=device), 'task': task}
    for key, image in [('side_cam', side), ('wrist_cam', wrist)]:
        value = torch.as_tensor(image, device=device)[..., :3].to(torch.float32) / 255.
        out['observation.images.' + key] = value.permute(2, 0, 1)
    return out



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
    return model / 'train_config.json', config, step


def train(artifact, options, output):
    import subprocess
    import sys
    from ..so101_dataset_contract import write_contract, JOINT_SPACE
    from ..representation import describe
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
    default_map = {'observation.images.side_cam': 'observation.images.camera1',
                   'observation.images.wrist_cam': 'observation.images.camera2'}
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
    atomic(checkpoint / 'workflow_provenance.json', {'workspace': artifact.get('workspace'),
           'source_dataset_artifact': artifact, 'training': config, 'representation': describe(coordinates),
           'task_definitions': artifact.get('tasks') or options.get('tasks', [])})


def import_checkpoint(path, registry):
    """Attach recorded training provenance to an intermediate workflow checkpoint."""
    from ..so101_dataset_contract import write_contract
    from ..real2sim.storage import atomic
    model = Path(path).expanduser().resolve()
    if (model / 'pretrained_model').is_dir():
        model = model / 'pretrained_model'
    for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json',
                 'policy_postprocessor.json', 'train_config.json'):
        if not (model / name).is_file():
            raise ValueError('Incomplete policy checkpoint: ' + name)
    if (model / 'workflow_provenance.json').exists():
        return registry.register('policy', model)
    request_path = next((p / 'request.json' for p in model.parents
                         if (p / 'request.json').is_file()), None)
    if request_path is None:
        raise ValueError('Workflow training provenance missing; cannot infer checkpoint workspace/coordinates')
    request = json.loads(request_path.read_text())
    if request.get('stage') != 'train':
        raise ValueError('Not a workflow training checkpoint')
    output = Path(request['output']).resolve()
    if not model.is_relative_to(output / 'checkpoints'):
        raise ValueError('Checkpoint is outside the recorded training output')
    artifact = request['input']
    registry.verify(artifact)
    config = json.loads((model / 'train_config.json').read_text())
    if Path(config['dataset']['root']).resolve() != Path(artifact['path']).resolve():
        raise ValueError('Checkpoint dataset differs from training provenance')
    from ..so101_dataset_contract import JOINT_SPACE
    if artifact['coordinates'].get('action_space') != JOINT_SPACE:
        raise ValueError('Checkpoint requires joint-radian training data')
    write_contract(model, artifact['coordinates'])
    atomic(model / 'workflow_provenance.json', {
        'workspace': artifact.get('workspace'), 'source_dataset_artifact': artifact,
        'training': config, 'task_definitions': artifact.get('tasks', []),
        'checkpoint_import': True})
    return registry.register('policy', model, parents=[artifact['id']])
