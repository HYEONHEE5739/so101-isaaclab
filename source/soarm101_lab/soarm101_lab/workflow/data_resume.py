"""Episode-boundary continuation: preserve inputs and combine completed segments."""
import json
import os
from pathlib import Path
from .artifacts import inspect_artifact


def validate(stage, path, workspace, tasks, source=None, scene_configuration=None, workspace_definition=None):
    item = inspect_artifact(stage, path)
    if stage=='source' and item.get('source_partition'):
        raise ValueError('Source Partition is locked; collect new demonstrations in the original unpartitioned Source dataset')
    if item.get('workspace') != workspace or item.get('tasks') != tasks:
        raise ValueError('Resume dataset의 workspace/task가 현재 선택과 다릅니다.')
    if scene_configuration:
        if workspace_definition:
            from .tasks import recorded_scene
            record = recorded_scene({**item,'path':str(path)},workspace_definition)
        else:
            record = item.get('env_args',{}).get('scene_configuration') or item.get('provenance',{}).get('scene_configuration')
        if not record or record['active_object_ids']!=scene_configuration['active_object_ids']:raise ValueError('Resume active object set differs')
    if stage == 'datagen' and source:
        previous = item.get('env_args', {}).get('source_dataset')
        if not previous or Path(previous).resolve() != Path(source).resolve():
            raise ValueError('Datagen resume은 동일 annotated input을 사용해야 합니다.')
    return item['episode_count']


def combine(previous, segment):
    """Atomic replacement of the NEW segment only; previous HDF5 is read-only."""
    import h5py
    from ..so101_dataset_contract import read_contract
    if read_contract(previous)!=read_contract(segment):
        raise ValueError('Resume calibration/robot/coordinate contract differs; refusing to mix episode models')
    segment = Path(segment)
    with h5py.File(segment,'r') as recorded:
        common=json.loads(recorded['data'].attrs.get('env_args','{}')).get('dataset_schema')=='so101.common-episode/2'
    if common:
        return _combine_common(previous,segment)
    temporary = segment.with_suffix('.merging.hdf5')
    try:
        with h5py.File(previous, 'r') as old, h5py.File(segment, 'r') as new, h5py.File(temporary, 'w') as out:
            partitions=[json.loads(h['data'].attrs.get('env_args','{}')).get('source_partition') for h in (old,new)]
            if partitions[0]!=partitions[1]:raise ValueError('Resume Source Partition differs; refusing Train/Validation mixture')
            scenes=[json.loads(h['data'].attrs.get('env_args','{}')).get('scene_configuration') for h in (old,new)]
            if any(scenes) and (not all(scenes) or scenes[0]['active_object_ids']!=scenes[1]['active_object_ids']):
                raise ValueError('Resume active object set differs or is unknown')
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
                    from ..utils.async_video_writer import copy_hdf5_videos
                    copy_hdf5_videos(episode,data[f'demo_{count}'],segment)
                    total += len(episode['actions'])
                    count += 1
            data.attrs['total'] = total
            data.attrs['resume_lineage'] = json.dumps({'previous': str(Path(previous).resolve()),
                                                       'new_segment': str(segment.resolve())})
            out.flush()
        os.replace(temporary, segment)
    finally:
        temporary.unlink(missing_ok=True)


def _combine_common(previous, segment):
    """Normalize legacy inputs before appending them to a new common recording."""
    import uuid
    from .dataset import DatasetWriter, EpisodeDataset, migrate
    from ..so101_dataset_contract import sidecar
    old=EpisodeDataset(previous);new=EpisodeDataset(segment)
    if old.args.get('source_partition')!=new.args.get('source_partition'):
        raise ValueError('Resume Source Partition differs; refusing Train/Validation mixture')
    scenes=[r.args.get('scene_configuration') for r in (old,new)]
    if any(scenes) and (not all(scenes) or scenes[0]['active_object_ids']!=scenes[1]['active_object_ids']):
        raise ValueError('Resume active object set differs or is unknown')
    temporary=segment.with_name(segment.stem+'.resume-'+uuid.uuid4().hex+'.hdf5')
    imported=segment.with_name(segment.stem+'.import-'+uuid.uuid4().hex+'.hdf5')
    writer=None
    try:
        if not old.common:old=EpisodeDataset(migrate(previous,imported))
        args={**new.args,'resume_lineage':{'previous':str(Path(previous).resolve()),'new_segment':str(segment.resolve())}}
        writer=DatasetWriter(temporary,args,new.contract)
        for reader in (old,new):
            from .episodes import source_catalog
            records=source_catalog(reader.path,allow_unknown=True)[0]
            for name in reader.names:
                values,attrs,videos=reader.episode(name)
                if not attrs.get('source_lineage') and name in records:attrs['source_lineage']=json.dumps(records[name])
                videos={key:{**spec,'file':str(reader.path.parent/spec['file'])} for key,spec in videos.items()}
                writer.commit(values,attrs,videos)
        writer.close();os.replace(temporary,segment)
    finally:
        if writer:writer.close(False)
        sidecar(temporary).unlink(missing_ok=True)
        # Only these newly created import intermediates belong to this operation.
        imported.unlink(missing_ok=True);sidecar(imported).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(imported.parent/(imported.stem+'.videos'),ignore_errors=True)


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


def compress(path):
    """Losslessly pack uncompressed arrays in a NEW completed HDF5, in bounded blocks."""
    import h5py
    import numpy as np
    import uuid
    path = Path(path)
    before = path.stat().st_size
    needs_compression = []
    references = []
    with h5py.File(path, 'r') as source:
        def inspect(name, item):
            if any(h5py.check_dtype(ref=item.attrs.get_id(k).dtype) for k in item.attrs): references.append(name)
            if isinstance(item, h5py.Dataset):
                if h5py.check_dtype(ref=item.dtype): references.append(name)
                if item.ndim and item.size and item.dtype.kind in 'biufc' and item.nbytes >= 65536 and not item.compression:
                    needs_compression.append(name)
        inspect('', source)
        source.visititems(inspect)
    # Object/region references and dimension-scale links require a dedicated adapter.
    if references: needs_compression = []
    if not needs_compression:
        return {'before_bytes': before, 'after_bytes': before, 'saved_bytes': 0}
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.compressing')
    seen = {}
    try:
        with h5py.File(path, 'r') as source, h5py.File(temporary, 'w') as destination:
            destination.attrs.update(source.attrs)
            def copy_group(old, new):
                for name in old:
                    link = old.get(name, getlink=True)
                    if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                        new[name] = link
                        continue
                    item = old[name]
                    address = h5py.h5o.get_info(item.id).addr
                    if address in seen:
                        new[name] = destination[seen[address]]
                        continue
                    if isinstance(item, h5py.Group):
                        copied = new.create_group(name)
                        seen[address] = copied.name
                        copied.attrs.update(item.attrs)
                        copy_group(item, copied)
                    elif item.name.lstrip('/') in needs_compression:
                        copied = new.create_dataset(name, shape=item.shape, dtype=item.dtype,
                                                    compression='gzip', compression_opts=1, shuffle=True)
                        rows = max(1, int(4*2**20 / max(1, np.prod(item.shape[1:])*item.dtype.itemsize)))
                        for start in range(0, item.shape[0], rows):
                            copied[start:start+rows] = item[start:start+rows]
                        copied.attrs.update(item.attrs)
                        seen[address] = copied.name
                    else:
                        old.copy(item, new, name=name)
                        seen[address] = new[name].name
            copy_group(source, destination)
            destination.flush()
        # Never increase storage simply to enable compression on incompressible noise.
        if temporary.stat().st_size < before:
            os.replace(temporary, path)
        after = path.stat().st_size
        return {'before_bytes': before, 'after_bytes': after, 'saved_bytes': before-after}
    finally:
        temporary.unlink(missing_ok=True)
