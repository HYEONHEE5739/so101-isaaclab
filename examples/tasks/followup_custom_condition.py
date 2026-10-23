"""Record this exact condition in Source first; no rewriting existing Source.

The custom term is a pure final acceptance refinement. StableGrasp, Placement,
manual marks, grasp signal and native Mimic's two subtasks remain mandatory.
"""
from soarm101_lab.task_terms.annotation import grasp_boundary

FOLLOWUP = {
    'schema': 'so101.pick-place-followup/1',
    'pick_object': 'cube_red',
    'target_object': 'cup_a',
    'condition': {'relation': 'contained_and_settled', 'max_speed_m_s': 0.02},
    'annotation': 'accept',
    'mimic': 'pick_place/1',
}


def accept(ctx):
    # Runtime/recording query only. Adapter owns tracker update/reset and frames.
    return grasp_boundary(ctx) is not None
