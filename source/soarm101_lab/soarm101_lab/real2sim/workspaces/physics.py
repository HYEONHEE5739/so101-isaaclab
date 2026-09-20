"""Semantic physics compiler. Defaults are templates, never image measurements."""
import copy

ROLES = {'ROBOT', 'STATIC_SUPPORT', 'STATIC_GEOMETRY', 'DYNAMIC_MANIPULAND', 'FIXED_TARGET', 'VISUAL_ONLY'}


def compile_physics(profile, task):
    material = {k: task['physics'][k] for k in ('static_friction', 'dynamic_friction', 'restitution')}
    provenance_note = task['physics'].get('provenance', 'default_template')
    sources = {'user_measured', 'model_asset', 'derived', 'numerically_fitted', 'default_template', 'provisional', 'unmeasured'}
    provenance = provenance_note if provenance_note in sources else 'provisional'
    objects = {'robot': {'role': 'ROBOT', 'source': 'model_asset', 'physics': 'existing articulation'},
               'table': {'role': 'STATIC_SUPPORT'}}
    for name in profile['objects']:
        if name in task['dynamic']:
            role = 'DYNAMIC_MANIPULAND'
        elif name in task['targets'].values():
            continue  # one logical object, visual owned by target below
        else:
            role = 'VISUAL_ONLY'
        objects[name] = {'role': role}
    for name, visual in task['targets'].items():
        objects[name] = {'role': 'FIXED_TARGET', 'visual': visual}
    objects['wrist_mount'] = {'role': 'VISUAL_ONLY'}
    # Explicit opt-in collision, not a guess from an object's appearance/name.
    for name, role in task.get('roles', {}).items():
        if name not in objects or role not in ROLES:
            raise ValueError('Unknown physics object/role')
        if objects[name]['role'] in ('ROBOT', 'DYNAMIC_MANIPULAND', 'FIXED_TARGET') and role != objects[name]['role']:
            raise ValueError('Physics role conflicts with task classification')
        if role in ('ROBOT', 'DYNAMIC_MANIPULAND', 'FIXED_TARGET') and role != objects[name]['role']:
            raise ValueError('Declare dynamic/target roles in task first')
        objects[name]['role'] = role
    for name, item in objects.items():
        role = item['role']
        if role == 'ROBOT':
            continue
        item.update(collision=role != 'VISUAL_ONLY', rigid_body=role in ('DYNAMIC_MANIPULAND', 'FIXED_TARGET'),
                    kinematic=role == 'FIXED_TARGET', resettable=role == 'DYNAMIC_MANIPULAND',
                    material=copy.deepcopy(material) if role != 'VISUAL_ONLY' else None,
                    material_provenance=provenance, provenance_note=provenance_note,
                    mass_kg=task['physics']['mass_kg'] if role == 'DYNAMIC_MANIPULAND' else None,
                    mass_provenance=provenance if role == 'DYNAMIC_MANIPULAND' else 'not_applicable',
                    inertia='derived_by_PhysX_from_collision_and_mass' if role == 'DYNAMIC_MANIPULAND' else 'not_applicable',
                    collision_approximation='open_segmented_walls_and_bottom' if role == 'FIXED_TARGET' else 'profile_primitive')
    return {'schema': 'so101.workspace-physics/1', 'objects': objects}


def bind_material(stage, prim, material):
    from pxr import UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, '/Real2Sim/PhysicsMaterial')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(material['static_friction'])
    api.CreateDynamicFrictionAttr(material['dynamic_friction'])
    api.CreateRestitutionAttr(material['restitution'])
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')
