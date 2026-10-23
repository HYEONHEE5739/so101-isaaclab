"""Stable file/directory identities shared by Dataset and learning consumers.

Names and payload bytes both count. HDF5 identities include each referenced MP4
once; directory identities ignore .cache. No Registry, UI or engine is loaded.
"""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    p = Path(path).expanduser().resolve()
    files = [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file() and '.cache' not in f.relative_to(p).parts)
    if p.is_file() and p.suffix in ('.hdf5','.h5'):
        import h5py
        from .async_video_writer import external_video_path
        with h5py.File(p,'r') as h:
            videos=[]
            if 'data' in h:
                for name,g in h['data'].items():
                    if name.startswith('demo_'):
                        videos.extend(external_video_path(p,spec) for spec in json.loads(g.attrs.get('external_videos','{}')).values())
        files += sorted(set(videos))
    if not files:
        raise ValueError('Empty artifact')
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(p) if p.is_dir() else f.name if f==p else f.relative_to(p.parent)).encode())
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()
