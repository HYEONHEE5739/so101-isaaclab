import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from soarm101_lab.real2sim import profile, references, bootstrap, scene

ROOT = Path(__file__).resolve().parents[2]
HYPOTHESIS = ROOT / 'docs/real2sim/bootstrap/reviewed_scene.json'


def test_reviewed_geometry_and_provenance():
    p = profile.load(HYPOTHESIS)
    assert p['robot_asset']['source'] == 'model_asset'
    assert p['workspace']['size_m'] == [.15, .15]
    for name in ['cube_red', 'cube_blue', 'cube_green']:
        o = p['objects'][name]
        assert o['dimensions_m'] == [.024]*3
        assert o['provenance']['dimensions_m'] == 'user_measured'
        assert np.all(np.abs(np.asarray(o['T_workspace'])[:2, 3]) + .012 < .075)
        assert o['provenance']['T_workspace'] == 'provisional'
    for name in ['cup_a', 'cup_b']:
        o = p['objects'][name]
        assert o['top_outer_diameter_m'] == .07 and o['height_m'] == .065
        assert o['bottom_outer_diameter_m'] is None
        assert o['provenance']['bottom_outer_diameter_m'] == 'unmeasured'
        with pytest.raises(ValueError, match='proxy'):
            profile.require_metric_geometry(p, 'object:' + name + '_visual')
    a,b = [np.asarray(p['objects'][n]['T_workspace'])[:3,3] for n in ['cup_a','cup_b']]
    assert np.linalg.norm(a-b) == pytest.approx(.09)
    for c in p['cameras'].values():
        assert not c['intrinsics_calibrated'] and not c['extrinsics_calibrated']
    assert profile.derived_measurements(p)['cup_a.top_inner_diameter_m']['value'] == pytest.approx(.066)


def test_real_only_revision_and_rejection(tmp_path):
    image = tmp_path / 'photo.png'
    Image.new('RGB', (32,32)).save(image)
    m = references.import_images(tmp_path / 'references', [image], 'side')
    p = profile.load(HYPOTHESIS)
    p['bootstrap']['reviewed_items'] = [{'import_id':m['import_id'], 'id':m['items'][0]['id'], 'view':'side', 'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}]
    h=tmp_path/'hypothesis.json';h.write_text(json.dumps(p))
    a=bootstrap.create(tmp_path,h); b=bootstrap.create(tmp_path,h)
    assert a != b and a.exists()
    q=profile.load(a)
    assert len(q['bootstrap']['inputs']) == 1
    assert q['bootstrap']['inputs'][0]['image_sha256']
    Image.new('RGB', (32,32), 'red').save(tmp_path/'references'/m['import_id']/m['items'][0]['stored_path'])
    with pytest.raises(ValueError, match='content changed'):
        bootstrap.create(tmp_path,h)
    p['bootstrap']['reviewed_items'][0]['view']='unknown'
    h.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='missing or its view'):
        bootstrap.create(tmp_path,h)


def test_deterministic_scene(tmp_path):
    from pxr import Usd
    p=profile.load(HYPOTHESIS)
    a=scene.build_usd(tmp_path/'a.usda',p);b=scene.build_usd(tmp_path/'b.usda',p)
    assert Path(a).read_text() == Path(b).read_text()
    stage=Usd.Stage.Open(a)
    for name in ['table','marker','cube_red','cube_blue','cube_green','cup_a_visual','cup_b_visual','background_roller']:
        assert stage.GetPrimAtPath('/Real2Sim/Workspace/'+name)
    assert not stage.GetPrimAtPath('/Real2Sim/Workspace/cup_a')
