"""Official LeRobot training orchestration; model algorithms remain upstream."""
from pathlib import Path
import json

def validate_credentials(value):
    """Reject credential values before requests/configs/logs are persisted."""
    import re
    if isinstance(value,dict):
        for key,child in value.items():
            if key.lower() in ('token','hf_token','password','api_key','secret'):
                raise ValueError('Use existing environment/Hub login for credentials')
            validate_credentials(child)
    elif isinstance(value,(list,tuple)):
        for child in value:validate_credentials(child)
    elif isinstance(value,str) and (re.search(r'\bhf_[A-Za-z0-9]{20,}',value) or
            re.search(r'--(?:[\w]+\.)*(?:token|hf_token|password|api_key|secret)=',value)):
        raise ValueError('Do not put credentials in workflow options')


def native_arguments(options):
    """Forward upstream flags verbatim; reserve publication/identity/storage controls."""
    import shlex
    args=options.get('native_args',[])
    if isinstance(args,str):args=shlex.split(args)
    if not isinstance(args,list) or not all(isinstance(v,str) and v.startswith('--') and '=' in v for v in args):
        raise ValueError('native_args must contain --key=value arguments')
    reserved=('dataset.repo_id','dataset.root','dataset.revision','dataset.episodes',
              'output_dir','config_path','resume','policy.type','policy.path','policy.pretrained_path',
              'policy.push_to_hub','policy.repo_id','policy.device','save_checkpoint')
    seen=set()
    for arg in args:
        key=arg[2:].split('=',1)[0]
        if key in reserved:raise ValueError('Native option owned by workflow: '+key)
        if key in seen:raise ValueError('Duplicate native option: '+key)
        if any(v in ('token','hf_token','password','api_key','secret') for v in key.lower().split('.')):
            raise ValueError('Use existing environment/Hub login for credentials')
        seen.add(key)
    return args


def training_command(artifact, options, output, *, python=None, owned_lock=None):
    """Return official CLI argv and config without executing or writing files."""
    import sys
    validate_credentials(options)
    from .datasets import installed_policies
    architecture=options.get('architecture','smolvla')
    if architecture=='minivla':raise ValueError('MiniVLA 신규 학습/학습 Resume는 비활성화되어 있습니다.')
    if architecture not in installed_policies():raise ValueError('Policy not in installed LeRobot factory: '+architecture)
    mode=options.get('training_mode') or ('resume' if options.get('resume_checkpoint') else 'finetune' if options.get('base_model') or architecture=='smolvla' else 'scratch')
    if mode not in ('scratch','finetune','resume'):raise ValueError('Unknown training_mode')
    if mode=='resume' and not options.get('resume_checkpoint'):raise ValueError('Resume requires a complete local checkpoint')
    if mode!='resume' and options.get('resume_checkpoint'):raise ValueError('Resume checkpoint conflicts with training_mode')
    if mode=='scratch' and options.get('base_model'):raise ValueError('Scratch cannot use base_model')
    if mode=='resume' and options.get('base_model'):raise ValueError('Resume cannot use a fine-tuning base_model')
    info=artifact['format'];dataset=Path(artifact['path']);hub=artifact.get('hub')
    config={'dataset':{'repo_id':hub['repo_id'] if hub else options.get('dataset_repo_id',info.get('repo_id','local/so101')),'root':str(dataset)},
        'policy':{'type':architecture,'device':options.get('device','cuda'),'push_to_hub':False},
        'output_dir':str(output),'steps':int(options.get('steps',10000)),'batch_size':int(options.get('batch_size',8)),
        'num_workers':int(options.get('num_workers',0)),'save_checkpoint':True,'save_freq':int(options.get('save_freq',1000)),
        'wandb':{'enable':False},'eval_freq':0}
    if hub:
        if options.get('dataset_repo_id') not in (None,hub['repo_id']):raise ValueError('Dataset repo_id conflicts with the Hub input')
        config['dataset']['revision']=hub['revision']
    overrides=options.get('policy_options',{})
    if not isinstance(overrides,dict) or set(overrides)&{'type','push_to_hub','repo_id','device','pretrained_path'}:
        raise ValueError('Invalid policy_options; workflow owns type/device/publication')
    config['policy'].update(overrides)
    base=options.get('base_model') or ('lerobot/smolvla_base' if architecture=='smolvla' and mode=='finetune' else None)
    if mode=='finetune' and not base:raise ValueError('Fine-tuning requires base_model')
    if base and (Path(base)/'config.json').is_file():
        if json.loads((Path(base)/'config.json').read_text()).get('type')!=architecture:
            raise ValueError('Selected Policy type differs from pretrained model configuration')
    camera_keys=[k for k,v in info['features'].items() if k.startswith('observation.') and v.get('dtype') in ('video','image')]
    config['rename_map']=options.get('rename_map',{k:'observation.images.camera'+str(i+1) for i,k in enumerate(camera_keys)} if architecture=='smolvla' else {})
    if not isinstance(config['rename_map'],dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in config['rename_map'].items()):
        raise ValueError('rename_map must map observation key strings to strings')
    path=Path(output).with_name(Path(output).name+'.train_config.json')
    argv=[python or options.get('python') or sys.executable,'-m','lerobot.scripts.lerobot_train','--config_path='+str(path)]
    if base:
        config['policy']['pretrained_path']=str(base)
        argv.append('--policy.path='+str(base))
        argv.extend('--policy.'+k+'='+json.dumps(v) for k,v in config['policy'].items() if k not in ('type','pretrained_path'))
    if mode=='resume':
        saved,config,step=resume_checkpoint(options['resume_checkpoint'],artifact,options,owned_lock=owned_lock)
        config={**config,'resume':True,'output_dir':str(output),'steps':int(options.get('steps',config['steps']))}
        argv=[argv[0],'-m','lerobot.scripts.lerobot_train','--config_path='+str(saved),'--resume=true',
              '--output_dir='+str(output),'--steps='+str(config['steps']),'--policy.push_to_hub=false']
        if native_arguments(options):raise ValueError('Resume restores saved settings; native_args must be empty')
    argv+=native_arguments(options)
    return argv,config,path


def resume_checkpoint(path, artifact, options, *, owned_lock=None):
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
        'rng_state.safetensors')]
    if (model/'train_config.json').is_file() and json.loads((model/'train_config.json').read_text()).get('scheduler') is not None:
        required.append(state/'scheduler_state.json')
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
    from .datasets import inspect_policy
    from ..representation import require_compatible
    from ..so101_dataset_contract import JOINT_SPACE
    from .datasets import inspect_dataset
    record=inspect_policy(model,owned_lock=owned_lock)
    selected=inspect_dataset(artifact['path'])
    if record['coordinates'] is not None and selected['coordinates'] is not None:
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
    from ..so101_dataset_contract import write_contract
    from ..representation import describe
    if options.get('architecture') == 'minivla':
        raise ValueError('MiniVLA 신규 학습 및 학습 Resume는 비활성화되어 있습니다. 기존 추론은 지원합니다.')
    from .datasets import validate_datasets
    artifact,validation,evidence=validate_datasets(train_path,options.get('validation_dataset'))
    if evidence:
        for warning in evidence['warnings']:print('[VALIDATION WARNING] '+warning,flush=True)
    if (options.get('architecture','smolvla')=='smolvla' and not options.get('base_model')
            and not options.get('resume_checkpoint') and options.get('training_mode')!='scratch'):
        options={**options,'base_model':'hf://lerobot/smolvla_base'}
    if options.get('base_model') and not Path(options['base_model']).expanduser().exists():
        from .datasets import resolve_model
        folder,identity=resolve_model(options['base_model'],options.get('model_revision'))
        options={**options,'base_model':folder,'hub_model':identity}
    coordinates = artifact['coordinates']
    from ..real2sim.storage import atomic
    dataset = Path(artifact['path'])
    owned_lock=None
    candidate=Path(output).with_name('.'+Path(output).name+'.workflow-lock')
    if options.get('_run_directory') and candidate.is_file():
        if candidate.read_text()!=options['_run_directory']:raise ValueError('Training output lock belongs to another run')
        owned_lock=candidate
    argv,config,cfg_path=training_command(artifact,options,output,owned_lock=owned_lock)
    atomic(cfg_path,config)
    import shlex
    print('[LEROBOT CLI] '+shlex.join(argv),flush=True)
    atomic(Path(output).with_name(Path(output).name+'.command.json'),dict(argv=argv,config=config,hub_dataset=artifact.get('hub'),hub_model=options.get('hub_model')))
    import os
    trainer_env=os.environ.copy()
    if config['policy'].get('device') == 'cpu':
        trainer_env['ACCELERATE_USE_CPU']='true'
    subprocess.run(argv, check=True, env=trainer_env)
    checkpoint = Path(output) / 'checkpoints/last/pretrained_model'
    if not (checkpoint / 'config.json').exists() or not list(checkpoint.glob('*.safetensors')):
        raise ValueError('Training did not produce a complete checkpoint')
    if (checkpoint/'train_config.json').is_file():config=json.loads((checkpoint/'train_config.json').read_text())
    if coordinates is not None:write_contract(checkpoint, coordinates)
    from ..utils.artifact_identity import fingerprint
    atomic(checkpoint / 'workflow_provenance.json', {'workspace': artifact.get('workspace'),
           'source_dataset': str(dataset), 'hub_dataset':artifact.get('hub'), 'hub_model':options.get('hub_model'), 'source_dataset_sha256':fingerprint(dataset), 'backend': options.get('backend', artifact.get('backend')) if coordinates is not None else None, 'training': config, 'representation': describe(coordinates) if coordinates is not None else None,
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
