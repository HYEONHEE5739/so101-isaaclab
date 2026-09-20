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
              'save_freq': int(options.get('save_freq', 1000)), 'wandb': {'enable': False}, 'eval_freq': 0}
    if config['policy']['type'] not in ('act', 'smolvla'):
        raise ValueError('Choose act or smolvla')
    overrides = options.get('policy_options', {})
    if not isinstance(overrides, dict) or any(k in overrides for k in ('type', 'push_to_hub', 'repo_id', 'device')):
        raise ValueError('Invalid policy_options; workflow owns type/device/publication')
    config['policy'].update(overrides)
    if options.get('base_model'):
        config['policy']['pretrained_path'] = options['base_model']
    cfg_path = Path(output).with_name(Path(output).name + '.train_config.json')
    atomic(cfg_path, config)
    subprocess.run([sys.executable, '-m', 'lerobot.scripts.lerobot_train', '--config_path=' + str(cfg_path)], check=True)
    checkpoint = Path(output) / 'checkpoints/last/pretrained_model'
    if not (checkpoint / 'config.json').exists() or not list(checkpoint.glob('*.safetensors')):
        raise ValueError('Training did not produce a complete checkpoint')
    write_contract(checkpoint, coordinates)
    atomic(checkpoint / 'workflow_provenance.json', {'workspace': artifact.get('workspace'),
           'source_dataset_artifact': artifact, 'training': config, 'representation': describe(coordinates),
           'task_definitions': artifact.get('tasks') or options.get('tasks', [])})
