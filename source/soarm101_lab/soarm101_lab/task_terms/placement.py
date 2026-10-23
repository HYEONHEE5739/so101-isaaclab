"""Use the existing MuJoCo placement mathematics with explicit object bindings."""


def placement(ctx, *, object_id: str, target_id: str, condition: dict | None = None) -> bool:
    """Instantaneous placement only; not StableGrasp or Annotation acceptance."""
    return bool(ctx.placement(object_id, target_id, condition)['success'])
