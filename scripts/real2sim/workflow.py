#!/usr/bin/env python3
"""Workflow registry, engine workers and Builder workspace import in one CLI."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))


def main():
    from soarm101_lab.workflow.operations import Operations, STAGES, MUJOCO_STAGES, MODEL_ROOT, execute_request
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'register', 'run', 'ready'):
        sub = commands.add_parser(name)
        sub.add_argument('--type')
        sub.add_argument('--path')
        sub.add_argument('--stage', choices=sorted(STAGES))
        sub.add_argument('--artifact')
        sub.add_argument('--workspace', default='')
        sub.add_argument('--options', help='JSON options file')
        sub.add_argument('--ik-solver', choices=['scipy_ls','pinocchio_dls'], help='DataGen only; overrides options JSON')
        sub.add_argument('--ik-failure-policy',choices=['strict','hold','best_effort'],help='DataGen only; overrides options JSON')
        sub.add_argument('--registry')
        sub.add_argument('--summary', action='store_true')
        sub.add_argument('--storage-name',help='신규 결과물의 저장 이름')
        sub.add_argument('--display-name',help='등록 Artifact의 표시 이름')
    worker = commands.add_parser('worker', help='Execute a UI request or explicit simulation stage')
    worker.add_argument('--request', help='Immutable worker request JSON')
    worker.add_argument('--backend', choices=['common', 'mujoco', 'isaac'], default='mujoco')
    worker.add_argument('--stage', choices=sorted(MUJOCO_STAGES))
    worker.add_argument('--workspace', default='')
    worker.add_argument('--input', default='')
    worker.add_argument('--config', default=str(MODEL_ROOT / 'configs/default.json'))
    worker.add_argument('--checkpoint', default='')
    worker.add_argument('--renderer', choices=['egl', 'software-egl', 'osmesa', 'none'], default='egl')
    worker.add_argument('--options', default='{}', help='JSON options object')
    worker.add_argument('--ik-failure-policy',choices=['strict','hold','best_effort'])
    worker.add_argument('--output')
    worker.add_argument('--ik-solver', choices=['scipy_ls','pinocchio_dls'], help='Overrides --options; incompatible with immutable --request')
    importer = commands.add_parser('import-hdf5', help='Convert an explicit HDF5 recipe to canonical NPY episodes')
    importer.add_argument('recipe')
    importer.add_argument('output')
    migration=commands.add_parser('migrate',help='Import legacy episodes into common HDF5 + relative MP4 without modifying the source')
    migration.add_argument('source');migration.add_argument('output')
    builder = commands.add_parser('import-builder', help='Import a Builder export as an actionable workspace')
    builder.add_argument('source')
    builder.add_argument('--workspace-id')
    builder.add_argument('--version', type=int, default=1)
    builder.add_argument('--task', help='Optional existing workspace task JSON, including provisional physics')
    args = parser.parse_args()
    if args.command == 'import-builder':
        from soarm101_lab.real2sim.workspaces.builder import import_scene
        task = json.loads(Path(args.task).read_text()) if args.task else None
        destination = import_scene(args.source, args.workspace_id, args.version, task)
        from soarm101_lab.workflow.artifacts import Registry
        Registry().register('workspace', destination)
        print(destination, flush=True)
        return
    if args.command=='migrate':
        from soarm101_lab.workflow.dataset import migrate
        print(migrate(args.source,args.output));return
    if args.command == 'worker':
        if args.request:
            if args.ik_solver is not None or args.ik_failure_policy is not None:worker.error('Set ik_solver inside the immutable request options')
            request = json.loads(Path(args.request).read_text())
        else:
            if args.backend not in ('mujoco','isaac') or not args.stage or not args.output:
                worker.error('--request or simulation --stage and --output are required')
            if args.backend=='isaac' and args.stage not in ('replay','annotate','datagen'):
                worker.error('Explicit Isaac worker supports replay, annotate and datagen; other stages use --request')
            request = {'backend': args.backend, 'stage': args.stage, 'workspace': args.workspace,
                       'input': args.input, 'config': args.config, 'checkpoint': args.checkpoint,
                       'renderer': args.renderer, 'options': json.loads(args.options), 'output': args.output}
        if args.ik_solver is not None:
            if request['stage']!='datagen':worker.error('--ik-solver requires datagen')
            request['options']['ik_solver']=args.ik_solver
        if args.ik_failure_policy is not None:
            if request['stage']!='datagen':worker.error('--ik-failure-policy requires datagen')
            request['options']['ik_failure_policy']=args.ik_failure_policy
        execute_request(request)
        return
    if args.command == 'import-hdf5':
        from soarm101_lab.workflow.episodes import convert_hdf5
        print(convert_hdf5(args.recipe, args.output))
        return
    from soarm101_lab.workflow.artifacts import Registry, TYPES
    registry = Registry(args.registry)
    if args.command == 'list':
        result = registry.summary() if args.summary else registry.all()
    elif args.command == 'register':
        if args.type not in TYPES or not args.path:
            parser.error('--type must be a registered artifact type and --path is required')
        result = registry.register(args.type, args.path,display_name=args.display_name)
    else:
        options = json.loads(Path(args.options).read_text()) if args.options else {}
        if args.storage_name is not None:options['storage_name']=args.storage_name
        if args.ik_solver is not None:
            if args.stage!='datagen':parser.error('--ik-solver requires datagen')
            options['ik_solver']=args.ik_solver
        if args.ik_failure_policy is not None:
            if args.stage!='datagen':parser.error('--ik-failure-policy requires datagen')
            options['ik_failure_policy']=args.ik_failure_policy
        service = Operations(registry)
        artifact = registry.get(args.artifact)
        result = (service.readiness(args.stage, artifact, args.workspace, options)
                  if args.command == 'ready' else service.run(args.stage, artifact, args.workspace, options, print))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
