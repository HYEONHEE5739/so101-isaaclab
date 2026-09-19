#!/usr/bin/env python3
"""Publish an explicit task + accepted revision. Requires pxr (Isaac Python)."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/soarm101_lab'))
from soarm101_lab.real2sim.workspaces.package import publish,default_task
from soarm101_lab.real2sim.profile import load
p=argparse.ArgumentParser();p.add_argument('--revision');p.add_argument('--workspace-id',required=True);p.add_argument('--version',type=int,default=1);p.add_argument('--task');p.add_argument('--write-task-template')
a=p.parse_args();revision=a.revision or json.loads((ROOT/'outputs/real2sim/ui_settings.json').read_text())['profile']
if a.write_task_template:
 Path(a.write_task_template).write_text(json.dumps(default_task(load(revision)),indent=2));print(a.write_task_template)
else:
 if not a.task:p.error('--task is required; generate and review --write-task-template first')
 # Isaac exposes USD Python bindings after its application has started.
 app = None
 try:
  try:
   from pxr import Usd
  except ImportError:
   from isaaclab.app import AppLauncher
   app = AppLauncher(headless=True).app
  print(publish(revision,a.workspace_id,json.loads(Path(a.task).read_text()),a.version), flush=True)
 finally:
  if app is not None:
   app.close()
