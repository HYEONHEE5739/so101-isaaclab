"""Real-world reference inputs, independent of synchronized CaptureStore evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid
from PIL import Image
from .storage import atomic, new_dir

SCHEMA = 'so101.references/1'
VIEWS = ('side', 'wrist', 'overview', 'auxiliary', 'unknown')


def import_images(root, files, view='unknown', camera_id=None):
    if view not in VIEWS:
        raise ValueError('Unknown reference view')
    paths = [Path(p).resolve() for p in files]
    if not paths:
        raise ValueError('Select real reference images')
    # Validate the whole batch before publishing anything.
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    folder = new_dir(root, 'import')
    manifest = dict(schema=SCHEMA, import_id=folder.name,
                    imported_at=datetime.now(timezone.utc).isoformat(), items=[])
    for source in paths:
        identifier = uuid.uuid4().hex
        dest = folder / (identifier + source.suffix.lower())
        shutil.copy2(source, dest)
        manifest['items'].append(dict(
            id=identifier, stored_path=dest.name, original_filename=source.name,
            source_path=str(source), source_type='image_file', role='real_reference',
            view=view, camera_id=camera_id or None, classification_source='user_selected_batch',
        ))
    atomic(folder / 'sources.json', manifest)
    return manifest


def discover(root):
    """Return records and visible warnings. Never rewrite legacy data."""
    records, warnings = [], []
    for folder in sorted(Path(root).glob('import_*')):
        if not folder.is_dir():
            continue
        try:
            manifest = json.loads((folder / 'sources.json').read_text())
            if not isinstance(manifest, dict):
                raise ValueError('Manifest must be an object')
            if manifest.get('schema') == SCHEMA:
                if not manifest.get('import_id') or not manifest.get('imported_at'):
                    raise ValueError('Missing import identity/timestamp')
                items = manifest['items']
                if not isinstance(items, list):
                    raise ValueError('items must be a list')
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError('Image record must be an object')
                    for field in ('id', 'original_filename', 'source_path', 'source_type', 'camera_id', 'classification_source'):
                        if field not in item:
                            raise ValueError('Missing image field: ' + field)
                    relative = Path(item['stored_path'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('stored_path must stay relative to import directory')
                    if item['role'] != 'real_reference' or item['view'] not in VIEWS:
                        raise ValueError('Invalid role/view')
            elif 'schema' not in manifest and 'frames' in manifest:
                frames = manifest['frames']
                sources = manifest.get('source_paths', [])
                paired = len(sources) == len(frames) and all(
                    Path(s).suffix.lower() in ('.jpg', '.jpeg', '.png') for s in sources)
                items = [dict(id=Path(f).stem, stored_path=Path(f).name,
                              original_filename=Path(sources[i]).name if paired else None,
                              source_path=sources[i] if paired else None,
                              source_type='legacy_import', role='unclassified', view='unknown',
                              camera_id=None, classification_source='legacy_unverified')
                         for i, f in enumerate(frames)]
            else:
                raise ValueError('Unsupported reference manifest')
            batch = []
            for item in items:
                path = folder / item['stored_path']
                if not path.is_file():
                    raise ValueError('Missing image: ' + path.name)
                batch.append({**item, 'path': str(path), 'import_id': folder.name,
                              'imported_at': manifest.get('imported_at')})
            records.extend(batch)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f'{folder.name}: {exc}')
    return records, warnings
