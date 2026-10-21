#!/usr/bin/env python3
"""Path-based workflow, engine workers and Builder workspace import in one CLI."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='discover':
        parser=argparse.ArgumentParser(description='Read-only native Dataset/Policy paths; candidates require execution validation')
        parser.add_argument('--path',default=str(ROOT/'outputs/workflow'))
        parser.add_argument('--category',choices=['dataset','policy','report','workspace'])
        parser.add_argument('--details',action='store_true',help='Read only the explicitly selected input header/config')
        args=parser.parse_args(sys.argv[2:])
        from soarm101_lab.workflow.paths import discover_inputs,input_format
        if args.details:
            fmt=input_format(args.path)
            if fmt in ('lerobot-policy','minivla-checkpoint','minivla-policy'):
                from soarm101_lab.workflow.policy import checkpoint_metadata,selected_minivla_checkpoint
                result=checkpoint_metadata(selected_minivla_checkpoint(args.path) if fmt=='minivla-policy' else args.path)
            elif fmt in ('hdf5','canonical','lerobot','evaluation'):
                from soarm101_lab.data.episodes import dataset_metadata
                result=dataset_metadata(args.path,fmt)
            else:parser.error('Select one native Dataset/Checkpoint path for --details')
        else:result=discover_inputs(args.path,category=args.category)
        print(json.dumps(result,indent=2,ensure_ascii=False));return
    from soarm101_lab.workflow.operations import Operations, STAGES, MUJOCO_STAGES, MODEL_ROOT, execute_request
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('discover', help='Read-only Dataset/Checkpoint paths; see discover --help')
    for name in ('run','ready'):
        sub = commands.add_parser(name)
        sub.add_argument('--stage', choices=sorted(STAGES),required=True)
        sub.add_argument('--input',help='Native Dataset/Model/Workspace Path',required=True)
        sub.add_argument('--root',help='Output and runs root; native Path discovery needs no registration')
        sub.add_argument('--workspace', default='')
        sub.add_argument('--options', help='JSON options file')
        sub.add_argument('--ik-solver', choices=['scipy_ls','pinocchio_dls'], help='DataGen only; overrides options JSON')
        sub.add_argument('--ik-failure-policy',choices=['strict','hold','best_effort'],help='DataGen only; overrides options JSON')
        sub.add_argument('--storage-name',help='신규 결과물의 저장 이름')
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
    options=json.loads(Path(args.options).read_text()) if args.options else {}
    if not isinstance(options,dict):parser.error('Options must be a JSON object')
    if args.storage_name is not None:options['storage_name']=args.storage_name
    for name in ('ik_solver','ik_failure_policy'):
        value=getattr(args,name)
        if value is not None:
            if args.stage!='datagen':parser.error('--'+name.replace('_','-')+' requires datagen')
            options[name]=value
    service=Operations(root=args.root or ROOT/'outputs/workflow')
    result=(service.readiness(args.stage,args.input,args.workspace,options) if args.command=='ready'
            else service.run(args.stage,args.input,args.workspace,options,lambda message:print(message,file=sys.stderr)))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
