"""Queries over the existing Adapter tracker/recording; never advance temporal state."""


def grasp_boundary(ctx):
    """First pre-step recorded grasp signal; manual marks remain Annotation-owned."""
    if ctx.adapter is None: raise ValueError('Annotation boundary requires an episode Adapter')
    return next((i for i, row in enumerate(ctx.adapter.buffer.rows) if row['grasp']), None)


def place_boundary(ctx):
    """First measured pre-step place signal (final post-step mapping belongs to Writer)."""
    if ctx.adapter is None: raise ValueError('Annotation boundary requires an episode Adapter')
    return next((i for i, row in enumerate(ctx.adapter.buffer.rows) if row['place']), None)


def native_acceptance(ctx):
    """Original Placement AND latched StableGrasp, with the SAME Adapter tracker."""
    if ctx.adapter is None or ctx.adapter.tracker is None or ctx.adapter.place_target is None:raise ValueError('Declare native diagnostic Object/Target binding')
    return ctx.adapter.native_success()


def placed(ctx):
    if ctx.adapter is None or ctx.adapter.place_target is None:raise ValueError('Declare diagnostic Object/Target binding')
    if hasattr(ctx,'_native_placement'):return ctx._native_placement
    return bool(ctx.adapter.place_success())


def event_boundary(ctx, event):
    if ctx.adapter is None:raise ValueError('Event boundary requires an episode Adapter')
    return ctx.adapter.event_boundaries.get(event)
