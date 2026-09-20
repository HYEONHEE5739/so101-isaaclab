"""Single presentation contract shared by UI and runtime diagnostics."""
SIM_STAGES = {'source', 'replay', 'annotate', 'datagen', 'sim_eval'}
INSPECTION_STAGES = {'annotate', 'datagen', 'sim_eval'}


def controls(stage, manual=False):
    if stage == 'source': return {'save', 'discard', 'quit'}
    if stage == 'annotate': return {'resume', 'pause'} | ({'mark', 'skip'} if manual else set())
    if stage in ('replay', 'datagen', 'sim_eval'): return {'resume', 'pause', 'step'}
    if stage == 'real_eval': return {'resume', 'pause'}
    return set()


def feeds(stage):
    if stage in SIM_STAGES:
        return ('sim_overview', 'sim_side', 'sim_wrist') if stage in INSPECTION_STAGES else ('sim_side', 'sim_wrist')
    return ('real_side', 'real_wrist') if stage == 'real_eval' else ()
