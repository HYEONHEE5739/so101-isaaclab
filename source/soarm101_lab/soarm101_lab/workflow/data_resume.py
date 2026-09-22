"""Episode-boundary continuation: preserve inputs and combine completed segments."""
import json
import os
from pathlib import Path
from .artifacts import inspect_artifact


def validate(stage, path, workspace, tasks, source=None):
    item = inspect_artifact(stage, path)
    if item.get('workspace') != workspace or item.get('tasks') != tasks:
        raise ValueError('Resume dataset의 workspace/task가 현재 선택과 다릅니다.')
    if stage == 'datagen' and source:
        previous = item.get('env_args', {}).get('source_dataset')
        if not previous or Path(previous).resolve() != Path(source).resolve():
            raise ValueError('Datagen resume은 동일 annotated input을 사용해야 합니다.')
    return item['episode_count']


def combine(previous, segment):
    """Atomic replacement of the NEW segment only; previous HDF5 is read-only."""
    import h5py
    segment = Path(segment)
    temporary = segment.with_suffix('.merging.hdf5')
    try:
        with h5py.File(previous, 'r') as old, h5py.File(segment, 'r') as new, h5py.File(temporary, 'w') as out:
            for key, value in new.attrs.items():
                out.attrs[key] = value
            data = out.create_group('data')
            for key, value in new['data'].attrs.items():
                data.attrs[key] = value
            count = total = 0
            for src in (old, new):
                for name in sorted(src['data'], key=lambda n: int(n.split('_')[-1]) if n.startswith('demo_') else -1):
                    if not name.startswith('demo_'):
                        continue
                    episode = src['data'][name]
                    src.copy(episode, data, name=f'demo_{count}')
                    total += len(episode['actions'])
                    count += 1
            data.attrs['total'] = total
            data.attrs['resume_lineage'] = json.dumps({'previous': str(Path(previous).resolve()),
                                                       'new_segment': str(segment.resolve())})
            out.flush()
        os.replace(temporary, segment)
    finally:
        temporary.unlink(missing_ok=True)


def deferred_interrupt():
    """Defer SIGINT to a simulation boundary, never interrupt an HDF5 write."""
    import signal
    requested = False
    def request(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, request)
    def boundary():
        if requested:
            raise KeyboardInterrupt
    return boundary
