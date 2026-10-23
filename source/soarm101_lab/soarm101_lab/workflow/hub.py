"""Explicit Hub publication using installed huggingface_hub; no import-time network."""
import json
import time
from pathlib import Path
from ..real2sim.storage import atomic


def publication_metadata(artifact, options):
    return {'schema': 'so101.hub-provenance/1', 'input': {'path':artifact['path']},
            'repository': {'repo_id': options['repo_id'], 'private': bool(options['private']),
                           'revision': options.get('revision', 'main')},
            'representation': artifact.get('representation'), 'workspace': artifact.get('workspace'),
            'created_at': time.time()}


def publish(artifact, options, output, api=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()  # HF cache/environment authentication; never store tokens in workflow configs
    kind = 'dataset' if artifact['type'] == 'dataset' else 'model'
    repo = options['repo_id']; revision = options.get('revision', 'main')
    if options.get('create_repo', False):
        api.create_repo(repo_id=repo, repo_type=kind, private=bool(options['private']), exist_ok=False)
    else:
        info = api.repo_info(repo_id=repo, repo_type=kind)
        if bool(info.private) != bool(options['private']):
            raise ValueError('Existing repository visibility differs from selected visibility')
    if options.get('create_branch', False):
        api.create_branch(repo_id=repo, repo_type=kind, branch=revision, exist_ok=False)
    folder = Path(artifact['path'])
    if kind == 'dataset' and artifact.get('format') == 'so101.canonical/1':
        folder = folder / 'lerobot'
        if not (folder / 'meta/info.json').is_file():
            raise ValueError('Convert the canonical dataset to LeRobot before upload')
    commit = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder), revision=revision,
                               ignore_patterns=['.cache/**', '.git/**', '.env*'], commit_message='Publish SO101 workflow artifact')
    commit = api.upload_file(repo_id=repo, repo_type=kind, revision=revision,
                             path_or_fileobj=json.dumps(publication_metadata(artifact, options), indent=2).encode(),
                             path_in_repo='hf_publication_provenance.json', parent_commit=commit.oid)
    result = {'repo_id': repo, 'repo_type': kind, 'revision': revision,
              'commit': commit.oid, 'url': str(commit.commit_url), 'status': 'SUCCEEDED',
              'local_path': str(folder), 'source_path': artifact['path']}
    atomic(output, result)
    return result


def import_dataset(repo_id, destination, revision=None):
    """Download into private staging, validate native data, then publish a new Path."""
    import shutil,tempfile,os
    from huggingface_hub import HfApi,snapshot_download
    from ..learning.datasets import inspect_dataset
    destination=Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    lock=destination.with_name('.'+destination.name+'.workflow-lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    staging=None
    try:
        os.close(fd)
        if destination.exists():raise FileExistsError(destination)
        info=HfApi().dataset_info(repo_id,revision=revision)
        staging=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.import-',dir=destination.parent))
        snapshot_download(repo_id,repo_type='dataset',revision=info.sha,local_dir=staging,
                          ignore_patterns=['.git/**','.env*','*.pt','*.safetensors'])
        inspect_dataset(staging)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset=LeRobotDataset(repo_id,root=staging,revision=info.sha,download_videos=False)
        dataset[0]  # Official schema, Task and first image/video sample.
        del dataset
        atomic(staging/'hf_import.json',dict(repo_id=repo_id,revision=info.sha))
        if destination.exists():raise FileExistsError(destination)
        staging.rename(destination);staging=None
        return str(destination)
    finally:
        if staging is not None:shutil.rmtree(staging)
        lock.unlink(missing_ok=True)
