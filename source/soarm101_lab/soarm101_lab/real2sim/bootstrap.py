"""Materialize an explicitly reviewed real-reference hypothesis as a new revision.

No automatic reconstruction: another environment requires a newly reviewed hypothesis.
"""
import argparse
import hashlib
from pathlib import Path
from .profile import load, digest
from .references import discover
from .storage import revision


def create(workspace, hypothesis):
    workspace = Path(workspace).resolve()
    p = load(hypothesis)
    records, warnings = discover(workspace / 'references')
    available = {(r['import_id'], r['id']): r for r in records
                 if r['role'] == 'real_reference' and r['view'] in ('side', 'wrist', 'overview', 'auxiliary')}
    reviewed = p.get('bootstrap', {}).get('reviewed_items', [])
    if not reviewed:
        raise ValueError('An explicitly reviewed real-reference hypothesis is required')
    inputs = []
    for item in reviewed:
        r = available.get((item['import_id'], item['id']))
        if r is None or r['view'] != item['view']:
            raise ValueError('Reviewed real reference is missing or its view changed: ' + item['id'])
        path = Path(r['path'])
        expected = item.get('image_sha256')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or expected != actual:
            raise ValueError('Reviewed image content changed or review hash missing: ' + item['id'])
        manifest = path.parent / 'sources.json'
        inputs.append({**item, 'image': str(path.relative_to(workspace)),
                       'manifest': str(manifest.relative_to(workspace)),
                       'image_sha256': actual,
                       'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
    p['bootstrap']['inputs'] = inputs
    p['revision'] = {'kind': 'bootstrap_provisional', 'parent': None,
                     'hypothesis_hash': digest(load(hypothesis))}
    return revision(workspace, p, {'stage': 'bootstrap', 'numerically_fitted': False,
                                  'render_verified': False, 'reference_warnings': warnings})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--hypothesis', required=True)
    args = parser.parse_args()
    print(create(args.workspace, args.hypothesis))


if __name__ == '__main__':
    main()
