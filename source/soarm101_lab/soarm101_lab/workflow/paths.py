"""Read-only filesystem discovery. Candidates are paths, never validated artifacts.

No payload/JSON opens, persistence, model/engine imports or directory creation.
Discovery markers identify a format candidate; workers still validate its contents.
"""
import os
from pathlib import Path


INPUT_FOLDERS={'annotate':'sources','datagen':'annotations',
               'convert':'datagen','merge':'datasets','train':'datasets',
               'validate':'models','sim_eval':'models','real_eval':'models','hf_dataset':'datasets','hf_policy':'models'}
ROOT=Path(__file__).resolve().parents[4]


def input_directory(stage: str, root: Path, *, workspace_root=None) -> Path:
    if stage=='source':return Path(workspace_root) if workspace_root is not None else ROOT/'outputs/real2sim/environments'
    return Path(root)/INPUT_FOLDERS[stage]


def browse_directory(kind: str, current='', *, root=None) -> str:
    """Choose a dialog start directory without registration, latest IDs or writes."""
    if current:
        path=Path(current).expanduser()
        if path.exists():return str(path if path.is_dir() else path.parent)
    if kind in ('workspace','profile'):
        return str(ROOT/('outputs/real2sim/environments' if kind=='workspace' else 'outputs/real2sim'))
    bucket={'source':'sources','annotated':'annotations','datagen':'datagen','dataset':'datasets',
            'policy':'models','sim_eval':'evaluations','evaluation_dataset':'evaluations'}[kind]
    return str(Path(root) / bucket if root is not None else ROOT/'outputs/workflow'/bucket)


def _files(folder: Path) -> tuple[set, set]:
    """One scandir; do not follow arbitrary payload/directory symlinks."""
    files=set();dirs=set()
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name.startswith('.') or entry.is_symlink():continue
            if entry.is_dir(follow_symlinks=False):dirs.add(entry.name)
            elif entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_size>0:files.add(entry.name)
    return files,dirs


def _marker(path: Path, root: Path) -> bool:
    # Native LeRobot's `last` may be a link to a completed step in this root.
    try:return path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size>0
    except (OSError,RuntimeError):return False


def input_format(path) -> str | None:
    """Cheap explicit-picker probe. Does not certify JSON/HDF5/weight contents."""
    path=Path(path).expanduser().resolve()
    if path.is_file():
        if path.stat().st_size==0:return None
        if path.suffix.lower() in ('.hdf5','.h5'):return 'hdf5'
        if path.suffix=='.pt':return 'minivla-checkpoint'
        return None
    if not path.is_dir():return None
    files,dirs=_files(path)
    if {'config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'}<=files:return 'lerobot-policy'
    if 'config.json' in files and any(Path(n).suffix=='.pt' for n in files):return 'minivla-policy'
    if {'manifest.json','project_provenance.json'}<=files:return 'canonical'
    if 'meta' in dirs and 'data' in dirs and _marker(path/'meta/info.json',path):return 'lerobot'
    if 'manifest.json' in files:return 'workspace'
    if 'evaluation_datasets.json' in files:return 'evaluation'
    # Legacy evaluation bundles have no index. This is only a container clue;
    # their reader must check episode metadata in the analysis worker.
    if any(Path(n).suffix.lower() in ('.h5','.hdf5') for n in files):return 'hdf5-collection'
    if 'result' in dirs:
        children,_=_files(path/'result')
        if any(Path(n).suffix.lower() in ('.h5','.hdf5') for n in children):return 'hdf5-collection'
    return None


def discover_inputs(folder, *, category=None) -> list[dict]:
    """Find native/legacy candidates under one explicit root; no global index.

    Returns {path, format, category, name}; path is the resolved identity, name
    is relative display text. Source/train/val names carry no role. Known payload
    trees are pruned. Internal native `last` aliases are deduplicated by path;
    other links, locks and private staging trees are skipped. Missing roots yield
    an empty list; an unreadable directory raises OSError to the worker/UI.
    """
    if category not in (None,'dataset','policy','report','workspace'):raise ValueError('Unknown input category')
    root=Path(folder).expanduser().resolve()
    if not root.exists():return []
    if any(p.name and p.with_name('.'+p.name+'.workflow-lock').exists() for p in (root,*root.parents)):return []
    found={}
    def add(path,fmt,kind):
        if category is not None and category!=kind:return
        resolved=path.resolve()
        if not resolved.is_relative_to(root if root.is_dir() else root.parent):return
        if not _marker(path,root if root.is_dir() else root.parent) and path.is_file():return
        label=str(path.relative_to(root)) if path!=root else path.name
        found.setdefault(str(resolved),dict(path=str(resolved),format=fmt,category=kind,name=label))
    if root.is_file():
        fmt=input_format(root)
        if fmt:add(root,fmt,'policy' if fmt in ('minivla-checkpoint','lerobot-policy') else 'dataset')
        return list(found.values())
    def visit(base):
        if base.with_name('.'+base.name+'.workflow-lock').exists():return
        files,dirs=_files(base)
        if {'config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'}<=files:
            add(base,'lerobot-policy','policy');return
        mini_root='config.json' in files and any(Path(n).suffix=='.pt' for n in files)
        if mini_root:
            for name in sorted(files):
                if Path(name).suffix=='.pt':add(base/name,'minivla-checkpoint','policy')
        if not mini_root and {'manifest.json','project_provenance.json'}<=files:
            add(base,'canonical','dataset');return
        if 'manifest.json' in files and 'project_provenance.json' not in files:
            add(base,'workspace','workspace');return
        if 'meta' in dirs and 'data' in dirs and _marker(base/'meta/info.json',root):
            add(base,'lerobot','dataset');return
        if 'evaluation_datasets.json' in files:add(base,'evaluation','dataset')
        for name in sorted(files):
            p=base/name
            if p.suffix.lower() in ('.hdf5','.h5'):
                if not p.with_name('.'+p.name+'.workflow-lock').exists():add(p,'hdf5','dataset')
            elif p.suffix=='.pt' and 'config.json' not in files:add(p,'minivla-checkpoint','policy')
        # `last` is an intentional SDK alias; traverse only if it stays in root.
        if 'checkpoints' in dirs:
            last=base/'checkpoints/last/pretrained_model'
            if _marker(last/'config.json',root) and input_format(last)=='lerobot-policy':add(last,'lerobot-policy','policy')
        for name in sorted(dirs):
            if name=='__pycache__' or name.endswith('.videos'):continue
            if base==root and name in ('artifacts','runs'):continue
            if name=='training_state' and 'pretrained_model' in dirs:continue
            visit(base/name)
    visit(root)
    return sorted(found.values(),key=lambda v:(v['name'],v['path']))


OUTPUT_FOLDERS = {'source':'sources','annotate':'annotations','datagen':'datagen',
                  'convert':'datasets','merge':'datasets','train':'models','validate':'validations','sim_eval':'evaluations','real_eval':'evaluations',
                  'hf_dataset':'evaluations','hf_policy':'evaluations'}


def validate_storage_name(value):
    """A single portable leaf name; never interpret user names as paths."""
    if not isinstance(value,str) or not value or value!=value.strip() or len(value)>100 or len(value.encode('utf-8'))>200:
        raise ValueError('저장 이름은 1–100자(UTF-8 200바이트 이내)의 한글/영문/숫자/밑줄/하이픈으로 입력하세요.')
    if any(not (c.isalnum() or c in '_-') for c in value) or value.upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(10)),*(f'LPT{i}' for i in range(10))}:
        raise ValueError('저장 이름에 경로, 공백, 점 또는 금지 문자를 사용할 수 없습니다.')
    return value


def collection_at(path, root):
    """Source grouping is owned by Sources and their direct Annotations only."""
    try:parts=Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:return None
    if len(parts)<3 or parts[0] not in ('sources','annotations'):return None
    bucket,family,variant=parts[:3]
    try:validate_storage_name(variant)
    except ValueError:return None
    validate_storage_name(family)
    partition=parts[3] if len(parts)>3 and parts[3] in ('train','val') else None
    return {'source':family,'variant':variant,**({'partition':partition} if partition else {})}


def output_location(stage, input_path, options, root):
    """Return the existing worker's file/directory contract and its reserved boundary."""
    backend=options.get('backend','isaac')
    file_output=(stage=='source' and backend!='mujoco') or stage in ('real_eval','hf_dataset','hf_policy') or (stage=='sim_eval' and backend!='mujoco')
    supplied=options.get('storage_name')
    if supplied is not None:validate_storage_name(supplied)
    if options.get('output'):
        output=Path(options['output']).expanduser().resolve()
        if file_output and output.suffix.lower() not in (('.hdf5','.h5') if stage=='source' else ('.json',)):
            raise ValueError('이 단계의 출력 경로에는 '+('HDF5 파일 확장자가' if stage=='source' else 'JSON 파일 확장자가')+' 필요합니다.')
        names={output.stem if file_output else output.name,output.parent.name if file_output else output.name}
        origin=output.parent if file_output else output
        if stage=='source' and origin.name=='origin':names.add(origin.parent.name)
        if supplied and supplied not in names:
            raise ValueError('저장 이름과 지정 출력 경로의 이름이 다릅니다. 이름을 맞추거나 출력 경로 설정을 해제하세요.')
        return output,output
    if stage=='train' and options.get('resume_checkpoint') and options.get('architecture','smolvla')!='minivla':
        from ..learning.training import resume_output
        output=resume_output(options['resume_checkpoint'],options.get('architecture','smolvla'))
        if supplied and supplied!=output.name:
            raise ValueError('ACT/SmolVLA Resume은 기존 출력 위치를 사용합니다. 새 이름으로 저장하려면 같은 이름의 output 경로도 명시하세요.')
        return output,output
    leaf=validate_storage_name(supplied)
    if stage=='train' and options.get('architecture','smolvla') not in ('act','smolvla','minivla'):
        raise ValueError('Unknown policy architecture')
    folder=Path(root).resolve()/OUTPUT_FOLDERS[stage]/leaf
    if not folder.resolve().is_relative_to((Path(root).resolve()/OUTPUT_FOLDERS[stage])):raise ValueError('Output escapes storage directory')
    target=folder/'origin' if stage=='source' else folder
    output=target/('source.hdf5' if stage=='source' else 'result.json') if file_output else target
    return output,folder


def resumes_in_place(stage,options,output):
    if stage!='train' or not options.get('resume_checkpoint') or options.get('architecture','smolvla')=='minivla':return False
    from ..learning.training import resume_output
    return Path(output).resolve()==resume_output(options['resume_checkpoint'],options.get('architecture','smolvla')).resolve()






def source_collection(source, root):
    """Locate an existing managed Source family or a deterministic external namespace.

    This computes a destination only. Selecting or saving a subset does not
    register/copy the original; the existing Curation writer reads it directly.
    """
    import hashlib
    source=Path(source).resolve();root=Path(root).resolve()
    collection=collection_at(source,root)
    if collection and source.is_relative_to(root/'sources'):return root/'sources'/collection['source']
    label=source.parent.name if source.name in ('dataset.hdf5','source.hdf5','dataset') else source.stem
    label=''.join(c if c.isalnum() or c in '_-' else '_' for c in label)[:40] or 'source'
    return root/'sources'/(label+'_'+hashlib.sha256(str(source).encode()).hexdigest()[:8])
