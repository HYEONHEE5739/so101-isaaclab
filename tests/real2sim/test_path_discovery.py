"""Path discovery and bounded selected metadata, without registration or payload loading."""
import builtins
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from soarm101_lab.workflow.paths import discover_inputs,input_format,input_directory,browse_directory


def put(path,content=b'payload'):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content if isinstance(content,bytes) else json.dumps(content).encode())
    return path


def native(root,arch='smolvla'):
    for name in ('model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        put(root/name)
    put(root/'config.json',{'type':arch})
    put(root/'train_config.json',{'policy':{'type':arch},'steps':200})
    return root


def canonical(root):
    put(root/'project_provenance.json',{'backend':'mujoco'})
    put(root/'manifest.json',{'episodes':[{'path':'episode_0000'}],'control_hz':30})
    return root


@pytest.fixture
def store(tmp_path):
    root=tmp_path/'이름 공백'
    expected={}
    for p in ('sources/임의/source 이름/origin/a.hdf5','sources/임의/선별본/b.h5',
              'sources/임의/분할 이름/train/c.hdf5','sources/임의/다른 분할/val/d.hdf5',
              'annotations/train/not_a_role.hdf5','datagen/val/xyz.h5','evaluations/test/failure/dataset.hdf5'):
        path=put(root/p);expected[str(path.resolve())]='hdf5'
    path=canonical(root/'sources/data/legacy/dataset');expected[str(path.resolve())]='canonical'
    path=root/'datasets/자유 이름';put(path/'meta/info.json',{'fps':30,'total_episodes':45});put(path/'data/chunk/file.parquet');expected[str(path.resolve())]='lerobot'
    for arch in ('act','smolvla'):
        p=native(root/'models'/arch/'checkpoints/010000/pretrained_model',arch);expected[str(p.resolve())]='lerobot-policy'
    path=put(root/'models/green_train/last.pt');expected[str(path.resolve())]='minivla-checkpoint'
    put(path.parent/'config.json',{'steps':300,'batch_size':2})
    put(root/'evaluations/test/evaluation_datasets.json',{'schema':'so101.sim-evaluation/1','datasets':[]})
    expected[str((root/'evaluations/test').resolve())]='evaluation'
    return root,expected


def test_paths_without_registry_and_without_payload_opens(store,monkeypatch):
    root,expected=store
    def forbidden(*a,**kw):raise AssertionError('Discovery must not open payloads/JSON')
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'open',forbidden);patch.setattr(io,'open',forbidden)
        items=discover_inputs(root)
    assert {r['path']:r['format'] for r in items}==expected
    assert not (root/'artifacts').exists()
    assert all(set(r)=={'path','format','category','name'} for r in items)
    assert {r['path'] for r in discover_inputs(root,category='policy')}=={p for p,f in expected.items() if f in ('lerobot-policy','minivla-checkpoint')}


def test_discovery_prunes_native_payloads_but_does_not_interpret_user_names(store):
    root,expected=store
    for p in ('datasets/자유 이름/data/nested/accidental.hdf5','sources/data/legacy/dataset/episode_0000/accidental.h5'):
        put(root/p)
    assert {r['path']:r['format'] for r in discover_inputs(root)}==expected
    for name in ('data','meta','assets','scene','train','val','green'):
        p=put(root/'arbitrary'/name/'input.h5')
        assert str(p) in {r['path'] for r in discover_inputs(root/'arbitrary')}


def test_links_locks_partial_and_collisions(tmp_path):
    root=tmp_path/'store';one=native(root/'a/checkpoints/010000/pretrained_model')
    (one.parent.parent/'last').symlink_to('010000',target_is_directory=True)
    other=native(root/'b/checkpoints/010000/pretrained_model')
    external=native(tmp_path/'outside/pretrained_model')
    (root/'escape').symlink_to(external,target_is_directory=True)
    (root/'cycle').symlink_to(root,target_is_directory=True)
    put(root/'.private/input.h5');put(root/'temporary.hdf5.123.tmp');(root/'zero.h5').touch()
    incomplete=native(root/'incomplete');(incomplete/'policy_postprocessor.json').unlink()
    locked=put(root/'locked/origin/input.hdf5');put(root/'.locked.workflow-lock')
    items=discover_inputs(root,category='policy')
    assert {r['path'] for r in items}=={str(one),str(other)}
    assert len(items)==2 and len({r['name'] for r in items})==2
    assert discover_inputs(locked)==[]
    assert discover_inputs(external)[0]['path']==str(external) # explicit outside root is allowed
    assert not any(r['path'].endswith(('zero.h5','input.hdf5')) for r in discover_inputs(root))


def test_empty_missing_or_unknown_category(tmp_path):
    assert discover_inputs(tmp_path/'missing')==[]
    assert discover_inputs(tmp_path)==[]
    with pytest.raises(ValueError):discover_inputs(tmp_path,category='source')
    assert not (tmp_path/'missing').exists()
    assert input_directory('train',tmp_path)==tmp_path/'datasets'
    assert browse_directory('policy',root=tmp_path)==str(tmp_path/'models')
    assert not (tmp_path/'models').exists()


def test_selected_header_distinguishes_common_native_legacy_lerobot(store,tmp_path):
    import h5py
    from soarm101_lab.data.episodes import dataset_metadata
    for schema in ('so101.common-episode/2',None):
        path=tmp_path/(str(schema).replace('/','_')+'.h5')
        with h5py.File(path,'x') as h:
            h.create_group('data').attrs['env_args']=json.dumps({'dataset_schema':schema,'backend':'isaac'} if schema else {})
        assert dataset_metadata(path,'hdf5')['format']==(schema or 'isaac-hdf5')
    root,_=store
    assert dataset_metadata(root/'sources/data/legacy/dataset','canonical')['episodes']==1
    assert dataset_metadata(root/'datasets/자유 이름','lerobot')['episodes']==45
    corrupt=put(tmp_path/'corrupt.hdf5',b'not an HDF5')
    assert discover_inputs(corrupt)[0]['format']=='hdf5' # unverified clue, not valid data
    with pytest.raises(OSError):dataset_metadata(corrupt,'hdf5')


@pytest.mark.parametrize('arch',['act','smolvla'])
def test_checkpoint_metadata_is_not_resume_validation(tmp_path,arch):
    from soarm101_lab.workflow.policy import checkpoint_metadata
    from soarm101_lab.learning.training import resume_checkpoint
    p=native(tmp_path/'checkpoint/pretrained_model',arch)
    assert checkpoint_metadata(p)==dict(path=str(p),architecture=arch,steps=200)
    assert checkpoint_metadata(p.parent)['architecture']==arch
    with pytest.raises(ValueError,match='resume checkpoint'):resume_checkpoint(p,{'path':'data'},{'architecture':arch})
    put(p/'config.json',b'bad json')
    with pytest.raises(ValueError):checkpoint_metadata(p)
    put(p/'config.json',b' '* (1024*1024+1))
    with pytest.raises(ValueError,match='display limit'):checkpoint_metadata(p)


def test_mini_metadata_does_not_load_weights(tmp_path,monkeypatch):
    from soarm101_lab.workflow.policy import checkpoint_metadata
    p=put(tmp_path/'model/checkpoints/step_300.pt')
    put(tmp_path/'model/config.json',{'steps':300,'batch_size':8})
    assert checkpoint_metadata(p)['batch_size']==8
    assert checkpoint_metadata(p)['model_config']==str(tmp_path/'model/config.json')


def test_legacy_mini_checkpoint_beside_canonical_manifest(tmp_path):
    root=canonical(tmp_path/'hybrid')
    put(root/'config.json',{'steps':10});put(root/'best.pt')
    items=discover_inputs(root)
    assert {r['format'] for r in items}=={'minivla-checkpoint'}
    assert input_format(root)=='minivla-policy'


def test_discovery_import_and_cli_are_inert(tmp_path,store):
    root,_=store;script=Path(__file__).resolve().parents[2]/'scripts/real2sim/workflow.py'
    code="""
import runpy,sys
from soarm101_lab.workflow.paths import discover_inputs
assert not any(n.split('.')[0] in ('torch','h5py','PyQt6','isaaclab','omni','pxr','mujoco') for n in sys.modules)
sys.argv=[sys.argv[1],'discover','--path',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
assert not any(n.startswith('soarm101_lab.workflow.artifacts') for n in sys.modules)
assert not any(n.split('.')[0] in ('torch','h5py','PyQt6','isaaclab','omni','pxr','mujoco') for n in sys.modules)
"""
    result=subprocess.run([sys.executable,'-c',code,str(script),str(root)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)
