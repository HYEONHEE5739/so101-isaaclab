"""Offline official held-out objectives. Never updates or rewrites a checkpoint."""
import json
from pathlib import Path
from .datasets import inspect_dataset, leakage


def validate_inputs(checkpoint, dataset):
    from ..workflow.policy import inspect_checkpoint
    from ..representation import require_compatible
    from ..so101_dataset_contract import JOINT_SPACE
    model=inspect_checkpoint(checkpoint)
    root=Path(model['path'])
    config=json.loads((root/'config.json').read_text())
    if config.get('type') not in ('act','smolvla'):raise ValueError('Offline Validation supports ACT/SmolVLA only')
    data=inspect_dataset(dataset)
    require_compatible(data['coordinates'],model['coordinates'],JOINT_SPACE)
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):
        if not (root/name).is_file():raise ValueError('Saved processor missing: '+name)
        for step in json.loads((root/name).read_text())['steps']:
            if step.get('state_file') and not (root/step['state_file']).is_file():raise ValueError('Saved processor state missing: '+step['state_file'])
    processor=json.loads((root/'policy_preprocessor.json').read_text())
    rename={}
    for step in processor['steps']:
        if step.get('registry_name')=='rename_observations_processor':rename=step['config'].get('rename_map',{})
    features={rename.get(k,k):v for k,v in data['format']['features'].items()}
    required_cameras=set(rename.values())
    for key,feature in {**config['input_features'],**config['output_features']}.items():
        if key not in features:
            # Official SmolVLA accepts unused configured image slots; explicitly mapped cameras remain mandatory.
            if config['type']=='smolvla' and feature['type']=='VISUAL' and key not in required_cameras:continue
            raise ValueError('Checkpoint feature missing from Dataset: '+key)
        shape=features[key]['shape']
        if feature['type']=='VISUAL':shape=[shape[2],shape[0],shape[1]]
        if feature['type']=='VISUAL' and config['type']=='smolvla' and config.get('resize_imgs_with_padding'):
            if shape[0]!=3:raise ValueError('SmolVLA requires RGB input')
        elif list(feature['shape'])!=shape:raise ValueError('Checkpoint/Dataset feature shape differs: '+key)
    saved=root/'train_config.json'
    if saved.is_file():
        train_root=json.loads(saved.read_text()).get('dataset',{}).get('root')
        if train_root and Path(train_root).resolve()==Path(data['path']):raise ValueError('Train and Validation must be different Dataset Paths')
        info=Path(train_root or '')/'meta/info.json'
        if train_root and info.is_file() and json.loads(info.read_text())['fps']!=data['format']['fps']:
            raise ValueError('Checkpoint training FPS differs from Validation')
    return model,data


def evaluate_checkpoint(checkpoint, dataset, output, *, device='cpu', batch_size=1, seed=42, max_batches=None):
    """Sample-weighted official loss, official horizon/padding and saved normalization.

    ACT VAE requires train mode for its posterior/KL; this is a stochastic held-out
    training objective (including dropout), NOT an inference error or success rate.
    SmolVLA uses eval mode with seeded flow noise/time. No gradients or optimizer.
    """
    from ..real2sim.storage import atomic
    from ..utils.artifact_identity import fingerprint
    output=Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError(output)
    if batch_size<1 or (max_batches is not None and max_batches<1):raise ValueError('Positive batch limits required')
    # Reserve the report directory; failure reports remain distinguishable from success.
    if output.is_relative_to(Path(checkpoint).resolve()) or output.is_relative_to(Path(dataset).resolve()):
        raise ValueError('Validation output must be outside inputs')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',checkpoint=str(Path(checkpoint).resolve()),dataset=str(Path(dataset).resolve()),
                seed=seed,batch_size=batch_size,device=device,max_batches=max_batches)
    try:
        model,data=validate_inputs(checkpoint,dataset)
        import os
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import random
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class,make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        root=Path(model['path']); cfg=PreTrainedConfig.from_pretrained(root,local_files_only=True)
        cfg.device=device
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        policy=get_policy_class(cfg.type).from_pretrained(root,config=cfg,local_files_only=True).to(device)
        pre,post=make_pre_post_processors(cfg,pretrained_path=root,
            preprocessor_overrides={'device_processor':{'device':device}})
        meta=LeRobotDatasetMetadata(data['format'].get('repo_id','local/validation'),root=data['path'])
        ds=LeRobotDataset(meta.repo_id,root=data['path'],delta_timestamps=resolve_delta_timestamps(cfg,meta),download_videos=False)
        train_config=root/'train_config.json'
        evidence=dict(warnings=['Training Dataset identity unavailable; independence unverified'],independence_proven=False)
        if train_config.is_file():
            saved=json.loads(train_config.read_text()); train_path=saved.get('dataset',{}).get('root')
            if train_path and Path(train_path).is_dir():evidence=leakage(inspect_dataset(train_path),data)
        report.update(model_type=cfg.type,checkpoint_sha256=fingerprint(root),dataset_sha256=fingerprint(Path(data['path'])),
                      leakage=evidence,horizon=cfg.action_delta_indices,normalization='saved checkpoint processor; no validation statistics fitted',
                      metric=('act_vae_training_objective' if cfg.type=='act' and cfg.use_vae else 'act_l1_objective') if cfg.type=='act' else 'smolvla_flow_objective',
                      reduction='official masked/padded tensor mean per batch, weighted by batch sample count',
                      mode='train/no_grad (posterior and dropout)' if cfg.type=='act' and cfg.use_vae else 'eval/no_grad')
        report['metric_definition']=('Official masked L1 mean + kl_weight * posterior KL mean; padding remains in denominator' if cfg.type=='act' and cfg.use_vae else 'Official masked L1 tensor mean' if cfg.type=='act' else 'Official flow-matching noise/time MSE; official boundary/action padding and denominator')
        if cfg.type=='act':report['kl_weight']=cfg.kl_weight if cfg.use_vae else None
        policy.train(cfg.type=='act' and cfg.use_vae)
        totals={};samples=0;episodes=set()
        from itertools import islice
        loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
        batches=islice(loader,max_batches) if max_batches is not None else loader
        with torch.no_grad():
            for batch in batches:
                n=len(batch['action']);episodes.update(batch['episode_index'].tolist())
                batch=pre(batch)
                # Official forward owns masking, posterior KL/flow noise and its denominator.
                loss,metrics=policy.forward(batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite offline objective')
                values={'loss':float(loss),**{k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))}}
                for key,value in values.items():totals[key]=totals.get(key,0.)+value*n
                samples+=n
                print(f'[VALIDATION] samples={samples} loss={float(loss):.6g}',flush=True)
        if not samples:raise ValueError('No validation samples')
        report.update(status='SUCCEEDED',samples=samples,episodes=len(episodes),metrics={k:v/samples for k,v in totals.items()})
        atomic(output/'report.json',report)
        return report
    except BaseException as exc:
        report.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=f'{type(exc).__name__}: {exc}')
        atomic(output/'report.json',report)
        raise


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-batches',type=int)
    a=p.parse_args();evaluate_checkpoint(a.checkpoint,a.dataset,a.output,device=a.device,batch_size=a.batch_size,seed=a.seed,max_batches=a.max_batches)

if __name__=='__main__':main()
