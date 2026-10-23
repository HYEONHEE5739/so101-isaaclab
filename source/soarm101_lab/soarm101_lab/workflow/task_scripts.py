"""Literal execution inputs for planning; trusted functions execute only in Worker.

SCRIPT is an ordinary literal dict in a Python file, not a registry. Paths resolve
relative to that file. No engine, state or process is owned here.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path



def _read_python(path, label):
    """Bounded read/parse only. No imports or runtime objects during discovery."""
    path = Path(path).expanduser().resolve()
    if path.suffix != '.py': raise ValueError('Select a Python '+label.lower())
    with path.open('rb') as stream: source = stream.read(1024 * 1024 + 1)
    if len(source) > 1024 * 1024: raise ValueError(label+' exceeds 1 MiB')
    try: tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc: raise ValueError(label+' syntax: '+str(exc)) from exc
    return path, source, tree


def _execute_entry(identity, source, name):
    """Execute validated bytes in the explicit Worker, with sibling imports."""
    import sys
    namespace = {'__name__': name, '__file__': identity['path']}
    sys.path.insert(0, str(Path(identity['path']).parent))
    try: exec(compile(source, identity['path'], 'exec'), namespace)
    finally: sys.path.pop(0)
    return namespace


def _reset_terms(namespace, objects):
    resets={}
    for object_id,spec in objects.items():
        if spec.get('reset'):
            fn=namespace.get(spec['reset'])
            if not callable(fn):raise ValueError('Missing Reset term: '+spec['reset'])
            inspect.signature(fn).bind(object(),object_id,object())
            resets[object_id]=fn
    return resets


def read_followup(path):
    """Static Pick/Place contract; never import a provenance path or execute code."""
    path, source, tree = _read_python(path, 'Followup contract')
    declarations = [n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'FOLLOWUP' for t in n.targets)]
    if not declarations and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCRIPT' for t in n.targets) for n in tree.body):
        config,identity,source=_script_config(path,'source',source,tree)
        identity={k:v for k,v in identity.items() if k!='source_code_sha256'}
        declaration={k:copy.deepcopy(config[k]) for k in ('scene','annotation','mimic') if k in config}
        validate_processing(declaration)
        return dict(schema='so101.scene-terms/1',**declaration),identity,source
    if len(declarations) != 1: raise ValueError('Declare one literal FOLLOWUP dictionary')
    try: config = ast.literal_eval(declarations[0])
    except (ValueError, TypeError) as exc: raise ValueError('FOLLOWUP must be literal data') from exc
    if not isinstance(config, dict) or config.get('schema') != 'so101.pick-place-followup/1':
        raise ValueError('Only the Pick/Place followup contract is supported')
    if set(config) - {'schema', 'pick_object', 'target_object', 'condition', 'annotation', 'mimic'}:
        raise ValueError('Unsupported followup fields or Subtask configuration')
    for key in ('pick_object', 'target_object'):
        if not isinstance(config.get(key), str) or not config[key]: raise ValueError('Followup requires '+key)
    if config.get('annotation') is not None and (not isinstance(config['annotation'], str) or not config['annotation'].isidentifier()):
        raise ValueError('annotation must name a callable in this file')
    if config.get('mimic', 'pick_place/1') != 'pick_place/1':
        raise ValueError('Only the existing tool0/grasp/Pick→Place Mimic configuration is supported')
    return config, dict(path=str(path), sha256=hashlib.sha256(source).hexdigest(),
                       identity_scope='entry Python bytes; imported modules are not pinned'), source


def followup_context(reader, workspace, scene, options, *, stage=None):
    """Validate native binding/identity once. Returned data owns no runtime state."""
    from ..real2sim.workspaces.tasks import task_binding
    declaration = identity = None
    if options.get('followup_python'):
        declaration,identity,entry_bytes=read_followup(options['followup_python'])
        if declaration.get('schema')=='so101.scene-terms/1':
            if options.get('backend','mujoco')!='mujoco':raise ValueError('Scene Terms currently require MuJoCo')
            if options.get('followup_identity') is not None and options['followup_identity']!=identity:raise ValueError('Followup Python changed after planning')
            validate_processing(declaration,workspace,scene)
            if stage=='datagen' and not declaration.get('mimic'):raise ValueError('DataGen requires explicit Mimic subtasks/acceptance')
            if stage=='annotate' and not options.get('auto',True) and not declaration['annotation'].get('manual_event'):raise ValueError('Manual Annotation requires an explicit manual_event signal')
            result=dict(schema='so101.scene-terms/1',declaration=declaration,python=identity)
            previous=reader.args.get('task_followup')
            if previous is not None and previous!=result:raise ValueError('Use the same explicit Scene Terms script as the Annotated Dataset')
            recorded=reader.provenance.get('source_script') or reader.args.get('source_script') or {}
            if scene.get('scene_only') and any(s.get('reset') for s in declaration['scene']['objects'].values()):
                code_digest=source_code_digest(ast.parse(entry_bytes,filename=identity['path']))
                if not recorded.get('source_code_sha256') or recorded['source_code_sha256']!=code_digest:
                    raise ValueError('Reset code identity differs or is unavailable in recorded Source')
            if scene.get('scene_only') and (recorded.get('declaration') or {}).get('annotation') is not None:
                if recorded['declaration']!={k:v for k,v in declaration.items() if k!='schema'} or recorded.get('sha256')!=identity['sha256']:
                    raise ValueError('Scene/Reset/Terms script differs from recorded Source identity')
            return result
    if scene.get('scene_only'):raise ValueError('Scene-only recordings require an explicitly selected Annotation/Mimic Terms script')
    definition = scene['task_definition']
    pick, target = definition['pick_object_id'], definition['target_object_id']
    # Native metadata is authoritative; Python paths in it are never executable inputs.
    source = reader.provenance.get('source_script') or reader.args.get('source_script') or {}
    if not isinstance(source,dict):raise ValueError('Invalid recorded Source script identity')
    if source.get('followup_contracts') is not None:
        from ..real2sim.workspaces.tasks import pick_place_contracts
        if source['followup_contracts'] != pick_place_contracts(): raise ValueError('Unsupported recorded followup implementation contract')
    ref = source.get('source_reference')
    if ref is not None:
        if not isinstance(ref, dict): raise ValueError('Invalid recorded Source reference')
        if ref.get('pick_object') != pick or ref.get('target_object') != target:
            raise ValueError('Source reference differs from recorded Pick/Target')
        bound = task_binding(workspace, pick, target, ref.get('condition'))
        if bound['success_condition'] != definition['success_condition'] or ref.get('instruction') != definition['language_instruction']:
            raise ValueError('Source reference condition/instruction differs from recorded Task')
        if not isinstance(ref.get('id'), str) or not ref['id']: raise ValueError('Missing Source Task ID')
    config = None
    if options.get('followup_python'):
        if options.get('backend', 'mujoco') != 'mujoco':
            raise ValueError('Explicit followup Python currently requires MuJoCo')
        config = declaration
        if (config['pick_object'], config['target_object']) != (pick, target):
            raise ValueError('Followup Pick/Target differs from recorded Source')
        if task_binding(workspace, pick, target, config.get('condition'))['success_condition'] != definition['success_condition']:
            raise ValueError('Followup condition differs from recorded Source; record a new Source for a different condition')
        planned = options.get('followup_identity')
        if planned is not None and planned != identity: raise ValueError('Followup Python changed after planning')
    result = dict(schema='so101.pick-place-followup/1', source_reference=copy.deepcopy(ref),
                pick_object=pick, target_object=target, condition=copy.deepcopy(definition['success_condition']),
                python=identity, declaration=config, mimic='pick_place/1')
    previous = reader.args.get('task_followup')
    if previous is not None and previous != result:
        raise ValueError('Recorded followup metadata differs; use the same explicit followup Python as the Annotated Dataset')
    return result


def load_followup(contract, stage='annotate'):
    """Worker only. Execute exactly the statically validated entry bytes once/run."""
    if not contract['python']: return None
    config, identity, source = read_followup(contract['python']['path'])
    if identity != contract['python']: raise ValueError('Followup Python changed after planning')
    namespace = _execute_entry(identity, source, '__workflow_followup__')
    if config.get('schema')=='so101.scene-terms/1':
        declared={k:namespace['SCRIPT'][k] for k in ('scene','annotation','mimic') if k in namespace['SCRIPT']}
        if declared!={k:v for k,v in config.items() if k!='schema'}:raise ValueError('SCRIPT cannot be mutated at runtime')
        terms={}
        for event,spec in config['annotation']['events'].items():
            name=spec['factory'] if isinstance(spec,dict) else spec
            fn=namespace.get(name)
            if not callable(fn):raise ValueError('Missing Event term: '+name)
            inspect.signature(fn).bind() if isinstance(spec,dict) else inspect.signature(fn).bind(object())
            terms[event]=(fn,isinstance(spec,dict))
        hooks={}
        for key in ('accept','failure'):
            name=(config.get('mimic',{}) if stage=='datagen' else config['annotation']).get(key)
            if key=='accept' and not name:raise ValueError('Declare an explicit '+stage+' acceptance term')
            fn=namespace.get(name) if name else None
            if name:
                if not callable(fn):raise ValueError('Missing '+key+' term')
                inspect.signature(fn).bind(object())
            hooks[key]=fn
        resetters=_reset_terms(namespace,config['scene']['objects'])
        return dict(contract=contract,terms=terms,stage=stage,resetters=resetters,**hooks)
    if namespace.get('FOLLOWUP') != config: raise ValueError('FOLLOWUP cannot be mutated at runtime')
    fn = namespace.get(config['annotation']) if config.get('annotation') else None
    if config.get('annotation'):
        if not callable(fn): raise ValueError('Missing Annotation acceptance term')
        inspect.signature(fn).bind(object())
    return dict(contract=contract, acceptance=fn)


def read_script(path: str, stage: str) -> tuple[dict, dict, bytes]:
    path, source, tree = _read_python(path, 'Execution script')
    return _script_config(path, stage, source, tree)


def _script_config(path, stage, source, tree):
    """Stage semantics over one parsed entry; followups reuse the same bytes."""
    declarations = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCRIPT' for t in n.targets)]
    if len(declarations) != 1: raise ValueError('Declare exactly one literal SCRIPT dictionary')
    try: config = ast.literal_eval(declarations[0])
    except (ValueError, TypeError) as exc: raise ValueError('SCRIPT must be literal data; put runtime functions outside it') from exc
    if not isinstance(config, dict) or config.get('stage') != ('sim_eval' if stage == 'rollout' else stage):
        raise ValueError('Script stage differs from selected Source/Sim Evaluation stage')
    if config['stage'] not in ('source', 'sim_eval'): raise ValueError('Script supports Source/Sim Evaluation only')
    unexpected=set(config)-{'stage','workspace','checkpoint','options','bindings','evaluations','scene','source','annotation','mimic'}
    if unexpected:raise ValueError('Unsupported SCRIPT fields: '+str(sorted(unexpected)))
    opts = config.get('options')
    if not isinstance(opts, dict) or opts.get('backend') != 'mujoco': raise ValueError('Execution scripts currently support MuJoCo only')
    if opts.get('task_python') or opts.get('execution_script'): raise ValueError('Nested Task/script execution is not supported')
    if not config.get('workspace'): raise ValueError('Script requires an explicit Workspace')
    if not opts.get('storage_name') and not opts.get('output'): raise ValueError('Script requires a new storage_name or output')
    if 'source' in config and ('scene' not in config or config['stage']!='source'):
        raise ValueError('source reference requires a concise Source scene')
    if any(k.startswith('script_') for k in opts): raise ValueError('script_* options are internal Worker inputs')
    compact = 'scene' in config
    if compact:
        conflict = set(opts) & {'tasks','task_definition','active_object_ids','scene_configuration','scene_reset','task_scope','scene_objects'}
        if conflict or 'bindings' in config: raise ValueError('Legacy/New Script conflict: scene cannot coexist with legacy tasks/bindings/scene options')
        scenes = [config['scene']]
    else:
        scenes = opts.get('tasks')
        if not isinstance(scenes, list) or not scenes or any(not isinstance(s, dict) for s in scenes):
            raise ValueError('Specify explicit Scene definitions in SCRIPT.options.tasks; generic Scene construction is not yet supported')
        if 'active_object_ids' not in opts: raise ValueError('Script requires explicit active_object_ids')
        bindings=config.get('bindings')
        if not isinstance(bindings,dict):raise ValueError('Script requires explicit bindings dictionary')
        for field in ('OBJECT_IDS','CAMERA_IDS'):
            ids=bindings.get(field)
            if not isinstance(ids,(list,tuple)) or not ids or any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
                raise ValueError('Script requires unique explicit '+field)
    rows = config.get('evaluations', [])
    if config['stage'] == 'source':
        if opts.get('num_envs',1)!=1:raise ValueError('Source teleoperation supports one environment')
        if compact and not isinstance(config['scene'].get('objects'),dict):
            ref = config.get('source')
            if not isinstance(ref,dict) or set(ref)-{'id','instruction','pick_object','target_object','condition'}:
                raise ValueError('Source requires id/instruction/pick_object/target_object reference')
            from .paths import validate_storage_name
            validate_storage_name(ref.get('id',''))
            for key in ('instruction','pick_object','target_object'):
                if not isinstance(ref.get(key),str) or not ref[key].strip(): raise ValueError('Source requires '+key)
        elif compact:
            ref=config.get('source',{})
            if not isinstance(ref,dict) or set(ref)-{'id','instruction'}:raise ValueError('Scene Source metadata accepts optional id/instruction only; bindings belong to annotation')
            if any(not isinstance(v,str) for v in ref.values()):raise ValueError('Source labels must be strings')
            scene_objects(config['scene'])
        if len(scenes) != 1 or rows: raise ValueError('Source requires one Scene and no evaluation/success gate')
    else:
        if opts.get('scene_reset') is not None:raise ValueError('Evaluation reset belongs to each evaluations row')
        if not config.get('checkpoint'): raise ValueError('Evaluation script requires a checkpoint Path')
        if not isinstance(rows, list) or not rows: raise ValueError('Declare at least one evaluation')
        ids = []
        for row in rows:
            if not isinstance(row, dict): raise ValueError('Each evaluation must be a dictionary')
            extra=set(row)-{'id','instruction','scene','success','contract','episodes','reset','objects','cameras','setup','observe'}
            if extra:raise ValueError('Unsupported evaluation fields: '+str(sorted(extra)))
            from .paths import validate_storage_name
            ids.append(validate_storage_name(row.get('id', '')))
            if compact:
                if 'scene' in row and not isinstance(row['scene'],dict): raise ValueError('Evaluation scene override must be an object dictionary')
            elif type(row.get('scene')) is not int or not 0 <= row['scene'] < len(scenes): raise ValueError('Evaluation scene index is invalid')
            if type(row.get('episodes')) is not int or row['episodes'] < 1: raise ValueError('Positive per-task episodes required')
            if not isinstance(row.get('instruction'), str) or not row['instruction'].strip(): raise ValueError('Evaluation instruction required')
            if not isinstance(row.get('success'), str) or not row['success'].isidentifier(): raise ValueError('Success must name a function in this Python file')
            if row.get('contract', 'task.success/1') not in ('task.success/1', 'so101.placement/1'): raise ValueError('Unsupported evaluation contract')
        if len(set(ids)) != len(ids): raise ValueError('Duplicate evaluation Task IDs')
    identity = dict(path=str(path), sha256=hashlib.sha256(source).hexdigest(), stage=config['stage'],
                    identity_scope='entry Python bytes; imported modules are not pinned')
    if config['stage']=='source' and isinstance(config.get('scene',{}).get('objects'),dict):
        # Resume may change counts/output/resume Path, but not imported code or reset implementation.
        identity['source_code_sha256']=source_code_digest(tree)
    return config, identity, source


def script_inputs(path: str, stage: str, python: str | None = None) -> tuple[str, str, dict]:
    """Static planning inputs; script configuration takes precedence over UI defaults."""
    config, identity, source = read_script(path, stage)
    root = Path(identity['path']).parent
    def absolute(value):
        value = Path(value).expanduser()
        return str((root / value).resolve() if not value.is_absolute() else value.resolve())
    workspace = absolute(config['workspace'])
    from ..learning.datasets import hub_reference
    input_path = workspace if stage == 'source' else (config['checkpoint'] if hub_reference(config['checkpoint']) else absolute(config['checkpoint']))
    opts = copy.deepcopy(config['options'])
    for key in ('output', 'resume_data', 'calibration_file_name'):
        if opts.get(key) and (key != 'calibration_file_name' or '/' in opts[key]): opts[key] = absolute(opts[key])
    if python: opts['python'] = python
    opts.update(execution_script=identity['path'], script_identity=identity,
                script_evaluations=config.get('evaluations',[]),script_bindings=config.get('bindings',{}))
    if 'scene' in config:
        opts['script_config'] = config
        opts['script_placements'] = _literal_placements(source)
    return input_path, workspace, opts


def load_execution_script(request: dict) -> None:
    """Worker-only execution after reservation; refuse changed source before hardware/GPU."""
    opts = request['options']; path = opts['execution_script']
    config, identity, source = read_script(path, request['stage'])
    if identity != opts.get('script_identity'): raise ValueError('Execution script changed after planning; select/run it again')
    if 'scene' in config and 'script_scenes' not in opts:
        from ..real2sim.workspaces.package import load
        opts = resolve_script(load(request['workspace']), opts)
        request['options'] = opts
    namespace = _execute_entry(identity, source, '__workflow_script__')
    if namespace.get('SCRIPT') != config: raise ValueError('Runtime mutation of literal SCRIPT is not allowed')
    if opts.get('scene_only'):
        request['_scene_resets']=_reset_terms(namespace,config['scene']['objects'])
        request['_source_declaration']={k:copy.deepcopy(config[k]) for k in ('scene','annotation','mimic') if k in config}
    rows = []
    for row in opts.get('script_evaluations', []):
        fn = namespace.get(row['success'])
        if not callable(fn): raise ValueError('Missing callable success term: ' + row['success'])
        inspect.signature(fn).bind(object())
        if row.get('placement_binding') is not None:
            from functools import partial
            from ..task_terms.placement import placement
            if not isinstance(fn,partial) or fn.func is not placement or fn.args or fn.keywords != row['placement_binding']:
                raise ValueError('Placement callable differs from its static declaration')
        hooks = {}
        for key in ('setup', 'observe'):
            if row.get(key):
                hook = namespace.get(row[key])
                if not callable(hook): raise ValueError('Missing hook: ' + key)
                inspect.signature(hook).bind(object()); hooks[key] = hook
        rows.append({**row, 'success_fn': fn, **hooks})
    if namespace.get('FOLLOWUP_CONTRACTS') is not None:
        from ..real2sim.workspaces.tasks import pick_place_contracts
        if namespace['FOLLOWUP_CONTRACTS'] != pick_place_contracts():
            raise ValueError('Unsupported Annotation/Mimic followup contracts')
        request['_followup_contracts'] = namespace['FOLLOWUP_CONTRACTS']
    request['_script_evaluations'] = rows
    request['_script_bindings'] = opts['script_bindings']
    if config.get('source'): request['_source_reference'] = config['source']
    if not request['_script_bindings'].get('OBJECT_IDS') or not request['_script_bindings'].get('CAMERA_IDS'):
        raise ValueError('Script requires explicit Object/Camera bindings')


def validate_script_scenes(workspace: dict, options: dict) -> None:
    """Preflight declared bindings/reset against the already loaded Workspace."""
    from ..real2sim.workspaces.tasks import configured, validate_task_scene
    bindings = options['script_bindings']
    if options.get('scene_only'):
        from ..simulation.task_context import validate_scene_objects
        validate_scene_objects(workspace,options['script_scene']['objects'])
        return
    entries = options.get('script_evaluations') or [{'scene': 0, 'reset': options.get('scene_reset')}]
    for entry in entries:
        index = entry['scene']
        ids = options.get('script_scenes', [None]*len(options['tasks']))[index]
        active = options['active_object_ids'] if ids is None else [n for n in workspace['task']['dynamic'] if n in ids]
        scene = configured(workspace, options['tasks'][index], active, 'mujoco', reset=entry.get('reset'), scene_objects=ids)
        selected_bindings = {**bindings, 'OBJECT_IDS':ids} if ids is not None else bindings
        validate_task_scene(selected_bindings, scene)
        for field in ('objects', 'cameras'):
            if field in entry:
                ids = entry[field]
                if not isinstance(ids, (list, tuple)) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
                    raise ValueError('Evaluation requires unique explicit ' + field)
        validate_task_scene({**selected_bindings, 'OBJECT_IDS': entry.get('objects', selected_bindings['OBJECT_IDS']),
                             'CAMERA_IDS': entry.get('cameras', bindings['CAMERA_IDS'])}, scene)


def _literal_placements(source):
    """Recognize imported placement partials as data, never evaluate Python in UI.

    Arbitrary functions remain Worker-only. They may declare scene.diagnostics to
    bind the existing Recorder's grasp reference separately from their success.
    """
    tree = ast.parse(source)
    imports = {alias.asname or alias.name: (node.module, alias.name)
               for node in tree.body if isinstance(node,ast.ImportFrom) for alias in node.names}
    result = {}
    for node in tree.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name): continue
        call = node.value
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or imports.get(call.func.id)!=('functools','partial'): continue
        if len(call.args)!=1 or not isinstance(call.args[0],ast.Name): continue
        if imports.get(call.args[0].id)!=('soarm101_lab.task_terms.placement','placement'): continue
        try: binding = {kw.arg:ast.literal_eval(kw.value) for kw in call.keywords}
        except (ValueError,TypeError): raise ValueError('Placement bindings must be literal Object/Target/condition values')
        if set(binding)-{'object_id','target_id','condition'} or not all(isinstance(binding.get(k),str) for k in ('object_id','target_id')):
            raise ValueError('Placement requires literal object_id and target_id')
        result[node.targets[0].id] = binding
    return result


def resolve_script(workspace, options):
    """Compile concise declarations against one validated Metadata input.

    Produces existing worker contracts, not a second Metadata store. Task IDs and
    success functions do not choose Scene membership. Legacy input is unchanged.
    """
    if 'script_config' not in options or 'script_scenes' in options: return options
    from ..real2sim.workspaces.tasks import scene_selection, task_binding, configured
    cfg=options['script_config']; opts=copy.deepcopy(options)
    if cfg['stage']=='source' and isinstance(cfg['scene'].get('objects'),dict):
        names=scene_objects(cfg['scene'])
        from ..simulation.task_context import validate_scene_objects
        validate_scene_objects(workspace,cfg['scene']['objects'])
        active=scene_selection(workspace,{'objects':names})
        selected=configured(workspace,active_object_ids=active,backend='mujoco',scene_objects=names,
                            scene_only=True,object_configuration=cfg['scene']['objects'])
        opts.update(scene_only=True,script_scene=cfg['scene'],tasks=[],task_definition=None,
                    script_scenes=[names],scene_objects=names,active_object_ids=active,
                    scene_configuration=selected['scene_configuration'],instruction=cfg.get('source',{}).get('instruction',''),
                    script_bindings=dict(OBJECT_IDS=names,CAMERA_IDS=list(workspace['profile']['cameras']),ROBOT_ASSET=workspace['profile'].get('robot_asset')))
        return opts
    entries=cfg.get('evaluations') or [cfg['source']]
    definitions=[]; scenes=[]; rows=[]
    for index,row in enumerate(entries):
        spec=row.get('scene',cfg['scene'])
        active=scene_selection(workspace,spec)
        if cfg['stage']=='source':
            binding=dict(object_id=row['pick_object'],target_id=row['target_object'],condition=row.get('condition'))
            reset=spec.get('reset')
        else:
            binding=opts['script_placements'].get(row['success'])
            reset=row.get('reset',spec.get('reset'))
            if binding is None:
                binding=spec.get('diagnostics')
                if binding is None:
                    raise ValueError('Custom success requires scene.diagnostics with object_id/target_id for existing grasp/recording diagnostics; no Task is inferred')
        if reset is not None and (not isinstance(reset,dict) or not reset):
            raise ValueError('Reset must be a nonempty supported Workspace reset dictionary')
        if not isinstance(binding,dict) or set(binding)-{'object_id','target_id','condition'} or not all(isinstance(binding.get(k),str) for k in ('object_id','target_id')):
            raise ValueError('Invalid Scene diagnostic placement reference')
        definition=task_binding(workspace,**binding)
        configured(workspace,definition,active,'mujoco',reset=reset,scene_objects=spec['objects'])
        if cfg['stage']=='source': definition['language_instruction']=row['instruction']
        definitions.append(definition);scenes.append(spec['objects'])
        if cfg['stage']=='sim_eval':
            entry={**row,'scene':index,'reset':reset,'objects':row.get('objects',spec['objects'])}
            if row['success'] in opts['script_placements']:
                entry['placement_binding']=opts['script_placements'][row['success']]
                entry.setdefault('contract','so101.placement/1')
            rows.append(entry)
    opts.update(tasks=definitions,task_definition=definitions[0],script_scenes=scenes,
                active_object_ids=scene_selection(workspace,{'objects':scenes[0]}),script_evaluations=rows,
                script_bindings=dict(OBJECT_IDS=scenes[0],CAMERA_IDS=list(workspace['profile']['cameras']),
                    ROBOT_ASSET=workspace['profile'].get('robot_asset'),ROBOT_ASSET_SHA256=workspace.get('manifest',{}).get('robot_asset_sha256')))
    if cfg['stage']=='source':opts['scene_reset']=cfg['scene'].get('reset')
    return opts


def scene_objects(scene):
    """Validate Scene-driven declarations as data; never infer manipulation roles."""
    if not isinstance(scene,dict) or set(scene)-{'objects'}:raise ValueError('Scene-driven Source uses objects; per-object pose/reset are explicit')
    objects=scene.get('objects')
    if not isinstance(objects,dict) or not objects:raise ValueError('Declare a nonempty scene.objects mapping')
    for name,spec in objects.items():
        if not isinstance(name,str) or not name or not isinstance(spec,dict) or set(spec)-{'pose','reset'}:raise ValueError('Invalid Scene object specification')
        if 'pose' in spec and 'reset' in spec:raise ValueError('Choose pose OR reset for each object')
        if 'reset' in spec and (not isinstance(spec['reset'],str) or not spec['reset'].isidentifier()):raise ValueError('Reset must name a callable in this Script')
    return list(objects)


def validate_processing(config, workspace=None, scene=None):
    """Static supported Event/Signal/tool0 contract. No callable execution here."""
    objects=scene_objects(config['scene']);annotation=config.get('annotation')
    if not isinstance(annotation,dict) or set(annotation)-{'diagnostics','events','signals','final_events','manual_event','accept','failure'}:
        raise ValueError('Declare annotation events/accept and optional signals/diagnostics')
    events=annotation.get('events')
    if not isinstance(events,dict) or not events:raise ValueError('Annotation requires named Events')
    for name,spec in events.items():
        if not isinstance(name,str) or not name.isidentifier():raise ValueError('Event names must be identifiers')
        fn=spec.get('factory') if isinstance(spec,dict) and set(spec)=={'factory'} else spec
        if not isinstance(fn,str) or not fn.isidentifier():raise ValueError('Event must name a function or {factory: function}')
    for key in ('accept','failure'):
        name=annotation.get(key)
        if key=='accept' and not name:raise ValueError('Annotation acceptance is required')
        if name is not None and (not isinstance(name,str) or not name.isidentifier()):raise ValueError('Invalid '+key+' function name')
    for key in ('signals','final_events'):
        values=annotation.get(key,[])
        if not isinstance(values,list) or len(set(values))!=len(values) or set(values)-set(events):raise ValueError('Unknown/duplicate '+key)
    if annotation.get('manual_event') is not None and annotation['manual_event'] not in annotation.get('signals',[]):
        raise ValueError('Manual Event must be one of the declared signals')
    diagnostics=annotation.get('diagnostics')
    if diagnostics is not None:
        if not isinstance(diagnostics,dict) or set(diagnostics)-{'object_id','target_id','condition'}:raise ValueError('Invalid diagnostic binding')
        refs={diagnostics.get('object_id')}|({diagnostics['target_id']} if diagnostics.get('target_id') else set())
        if not refs<=set(objects):raise ValueError('Diagnostic Object/Target absent from Scene')
        if not diagnostics.get('target_id') and diagnostics.get('condition') is not None:raise ValueError('Placement condition requires a diagnostic target')
        if workspace:
            from ..real2sim.workspaces.tasks import task_binding
            if diagnostics['object_id'] not in workspace['task']['dynamic']:raise ValueError('StableGrasp diagnostics require a dynamic Object')
            bound=task_binding(workspace,**diagnostics) if diagnostics.get('target_id') else None
            definition=(scene or {}).get('task_definition')
            if definition is not None and (bound is None or any(bound[k]!=definition[k] for k in ('pick_object_id','target_object_id','success_condition'))):
                raise ValueError('Diagnostic binding differs from recorded Source')
    mimic=config.get('mimic')
    if mimic is not None:
        if not isinstance(mimic,dict) or set(mimic)-{'recipe','subtasks','accept','failure'}:raise ValueError('Unsupported Mimic fields')
        for key in ('accept','failure'):
            fn=mimic.get(key)
            if key=='accept' and not fn:raise ValueError('Mimic acceptance must be explicit')
            if fn is not None and (not isinstance(fn,str) or not fn.isidentifier()):raise ValueError('Invalid Mimic '+key)
        if mimic.get('recipe') is not None:
            if mimic['recipe']!='pick_place/1' or 'subtasks' in mimic or not diagnostics or not diagnostics.get('target_id'):raise ValueError('Unsupported/ambiguous Mimic recipe')
            if 'grasp' not in annotation.get('signals',[]):raise ValueError('Existing Pick/Place recipe requires grasp signal')
        else:
            validate_subtasks(mimic.get('subtasks'),objects,annotation.get('signals',[]))
    if scene:
        expected=scene.get('object_ids') or scene['active_object_ids']+list(workspace['task']['targets'])
        if set(objects)!=set(expected):raise ValueError('Terms Scene differs from recorded physical objects')
        if scene.get('scene_only') and config['scene']['objects']!=scene.get('objects'):
            raise ValueError('Terms Pose/Reset configuration differs from recorded Scene')
        if not scene.get('scene_only') and any(config['scene']['objects'].values()):
            raise ValueError('Legacy Source must keep its recorded native Reset; Scene overrides require a new Source')
    return config


def validate_subtasks(subtasks, objects, signals):
    """The portable reference supports tool0 sequential segments, final end=None."""
    if not isinstance(subtasks,list) or not subtasks:raise ValueError('Declare nonempty Mimic subtasks')
    names=[];ends=[]
    for i,spec in enumerate(subtasks):
        if not isinstance(spec,dict) or set(spec)-{'name','object','eef','end_event','offset','selection','nn_k','interpolation','fixed'}:
            raise ValueError('Unsupported Subtask fields')
        if not isinstance(spec.get('name'),str) or not spec['name']:raise ValueError('Subtask name required')
        names.append(spec['name'])
        if spec.get('eef')!='tool0':raise ValueError('Only existing tool0 action/EEF mapping is supported')
        obj=spec.get('object')
        if obj is not None and obj not in objects:raise ValueError('Subtask Object absent from Scene')
        end=spec.get('end_event')
        if i==len(subtasks)-1:
            if end is not None:raise ValueError('Reference final Subtask must end at Episode end (end_event=None)')
        elif end not in signals:raise ValueError('Subtask boundary requires a recorded Annotation signal')
        else:ends.append(end)
        strategy=spec.get('selection','nearest_neighbor_object' if obj else 'random')
        if strategy not in ('random','nearest_neighbor_object') or (obj is None and strategy!='random'):raise ValueError('Unsupported Object/selection combination')
        offsets=spec.get('offset',[0,0])
        if not isinstance(offsets,(list,tuple)) or len(offsets)!=2 or any(type(v)is not int for v in offsets) or not 0<=offsets[0]<=offsets[1]:raise ValueError('Subtask offset requires ordered nonnegative integers')
        if i==len(subtasks)-1 and list(offsets)!=[0,0]:raise ValueError('Final Subtask offset must be zero')
        for key,default,minimum in (('nn_k',3,1),('interpolation',5,0),('fixed',0,0)):
            v=spec.get(key,default)
            if type(v)is not int or v<minimum:raise ValueError('Invalid Subtask '+key)
    if len(set(names))!=len(names) or len(set(ends))!=len(ends):raise ValueError('Duplicate Subtask names/boundaries')



def source_code_digest(tree):
    """Source Resume ignores run counts/paths in SCRIPT, and pins all other code."""
    code=ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCRIPT' for t in n.targets))],type_ignores=[])
    return hashlib.sha256(ast.dump(code,include_attributes=False).encode()).hexdigest()
