#!/usr/bin/env python3
"""One explicit workflow operation, or independent artifact registration."""
from pathlib import Path
import sys
import json
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))
from soarm101_lab.workflow.artifacts import Registry, TYPES
from soarm101_lab.workflow.operations import Operations, STAGES
p = argparse.ArgumentParser()
p.add_argument('command', choices=['list', 'register', 'run', 'ready'])
p.add_argument('--type', choices=sorted(TYPES)); p.add_argument('--path')
p.add_argument('--stage', choices=sorted(STAGES)); p.add_argument('--artifact')
p.add_argument('--workspace', default=''); p.add_argument('--options', help='JSON options file')
p.add_argument('--registry')
a = p.parse_args(); registry = Registry(a.registry)
if a.command == 'list':
    result = registry.all()
elif a.command == 'register':
    result = registry.register(a.type, a.path)
else:
    opts = json.loads(Path(a.options).read_text()) if a.options else {}
    service = Operations(registry); artifact = registry.get(a.artifact)
    result = service.readiness(a.stage, artifact, a.workspace, opts) if a.command == 'ready' else service.run(a.stage, artifact, a.workspace, opts, print)
print(json.dumps(result, indent=2, ensure_ascii=False))
