"""Read the episode's existing Adapter tracker. Never advance or duplicate it."""


def grasp_completed(ctx) -> bool:
    """Latched StableGrasp; the owning Adapter updates/resets this state."""
    if ctx.adapter is None or ctx.adapter.tracker is None: raise ValueError('Grasp diagnostics require the existing episode Adapter')
    return bool(ctx.adapter.tracker.completed)


def failure_category(ctx) -> str:
    """Existing drop/release/failure classification without changing temporal state."""
    if ctx.adapter is None: raise ValueError('Failure diagnostics require the existing episode Adapter')
    return ctx.adapter.native_failure_category()


def released(ctx):
    """Existing command-based release evidence, without a second tracker."""
    if ctx.adapter is None:raise ValueError('Release diagnostics require an Adapter')
    return bool(ctx.adapter.release_commanded)


def dropped(ctx):
    """Existing contact-loss/drift duration evidence; never unlatches StableGrasp."""
    if ctx.adapter is None:raise ValueError('Drop diagnostics require an Adapter')
    return ctx.adapter.drop_step is not None


def drop_failure(ctx):
    return 'drop_failure' if dropped(ctx) else None
