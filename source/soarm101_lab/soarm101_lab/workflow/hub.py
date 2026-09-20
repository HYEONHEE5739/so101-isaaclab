"""Explicit Hub publication using installed huggingface_hub; no import-time network."""
import json
import time
from pathlib import Path
from ..real2sim.storage import atomic


def publication_metadata(artifact, options):
    return {'schema': 'so101.hub-provenance/1', 'artifact': artifact,
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
    commit = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder), revision=revision,
                               ignore_patterns=['.cache/**', '.git/**', '.env*'], commit_message='Publish SO101 workflow artifact')
    commit = api.upload_file(repo_id=repo, repo_type=kind, revision=revision,
                             path_or_fileobj=json.dumps(publication_metadata(artifact, options), indent=2).encode(),
                             path_in_repo='hf_publication_provenance.json', parent_commit=commit.oid)
    result = {'repo_id': repo, 'repo_type': kind, 'revision': revision,
              'commit': commit.oid, 'url': str(commit.commit_url), 'status': 'SUCCEEDED',
              'local_path': str(folder), 'source_artifact': artifact['id']}
    atomic(output, result)
    return result
