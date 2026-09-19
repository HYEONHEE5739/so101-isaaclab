import json
from pathlib import Path
from PIL import Image
from soarm101_lab.real2sim import references


def photos(tmp_path):
    paths = [tmp_path / 'side.png', tmp_path / 'wrist.png']
    for p in paths:
        Image.new('RGB', (10, 10)).save(p)
    return paths


def test_batch_uuid_relative_and_reload(tmp_path):
    files = photos(tmp_path)
    root = tmp_path / 'references'
    manifest = references.import_images(root, files, 'overview')
    assert manifest['schema'] == references.SCHEMA
    assert manifest['imported_at']
    assert len(manifest['items']) == 2
    for i in manifest['items']:
        assert i['view'] == 'overview'
        assert i['role'] == 'real_reference'
        assert len(i['id']) == 32
        assert not Path(i['stored_path']).is_absolute()
        assert i['original_filename'] in ['side.png', 'wrist.png']
    records, warnings = references.discover(root)
    assert len(records) == 2 and not warnings
    references.import_images(root, files, 'unknown')
    assert len(references.discover(root)[0]) == 4


def test_legacy_untouched_and_incomplete_visible(tmp_path):
    folder = tmp_path / 'import_old'
    folder.mkdir()
    Image.new('RGB', (10, 10)).save(folder / 'abc.png')
    path = folder / 'sources.json'
    path.write_text(json.dumps({'source_paths': ['/old/sim_side.png'], 'frames': ['/old/import_old/abc.png']}))
    before = path.read_bytes()
    records, warnings = references.discover(tmp_path)
    assert records[0]['role'] == 'unclassified'
    assert records[0]['view'] == 'unknown'
    assert path.read_bytes() == before
    (tmp_path / 'import_incomplete').mkdir()
    assert references.discover(tmp_path)[1]


def test_ui_multiselect_and_restart(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from soarm101_lab.real2sim.ui import Window
    app = QApplication.instance() or QApplication([])
    files = photos(tmp_path)
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames', lambda *a: ([str(p) for p in files], ''))
    w = Window(tmp_path)
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ['1 · Reference', '2 · Real / Sim', '3 · Inspect']
    from PyQt6.QtWidgets import QPushButton
    live = w.tabs.widget(1)
    assert any(b.text() == 'Capture' for b in live.findChildren(QPushButton))
    assert live.isAncestorOf(w.capture_details)
    w.reference_view.setCurrentText('wrist')
    monkeypatch.setattr(w, 'work', lambda fn, done: done(fn(lambda s: None)))
    w.import_media()
    assert 'wrist: 2' in w.reference_summary.text()
    w.close()
    w2 = Window(tmp_path)
    assert len(w2.reference_files) == 2
    assert 'wrist: 2' in w2.reference_summary.text()
    assert list(w2.views) == ['real_side', 'sim_side', 'real_wrist', 'sim_wrist']
    w2.close()
    app.processEvents()
