"""Optional explicit contract for Source → Annotation → Mimic; no Source gate.

Omit --followup-python to reuse the same native contract from recorded metadata.
Use this file for BOTH Annotation and DataGen when explicitly selected.
"""
FOLLOWUP = {
    'schema': 'so101.pick-place-followup/1',
    'pick_object': 'cube_red',
    'target_object': 'cup_a',
    'mimic': 'pick_place/1',
}
