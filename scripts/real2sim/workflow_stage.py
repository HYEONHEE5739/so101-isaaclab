#!/usr/bin/env python3
"""Subprocess entrypoint used by both workflow UI and CLI."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))
request = json.loads(Path(sys.argv[1]).read_text())
stage, artifact, options = request['stage'], request['input'], request['options']
output = Path(request['output'])
if stage == 'accept':
    import copy
    from soarm101_lab.real2sim.profile import load
    from soarm101_lab.real2sim.storage import atomic
    profile = copy.deepcopy(load(artifact['path']))
    profile['revision'] = {**profile.get('revision', {}), 'status': 'ACCEPTED',
                          'parent': artifact['path'], 'acceptance': 'explicit user workflow action'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic(output, profile)
elif stage in ('hf_dataset', 'hf_policy'):
    from soarm101_lab.workflow.hub import publish
    publish(artifact, options, output)
elif stage == 'train':
    from soarm101_lab.workflow.policy import train
    train(artifact, options, output)
elif stage in ('sim_eval', 'real_eval'):
    from soarm101_lab.workflow.evaluation import evaluate
    evaluate(request)
else:
    raise ValueError(stage)
