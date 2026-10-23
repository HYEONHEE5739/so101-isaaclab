"""Contact queries over the same Backend filter used by the owning tracker."""


def finger_contact(ctx, *, object_id, bilateral=True):
    ctx.require(objects=(object_id,))
    left,right=ctx.backend.finger_contacts(object_id)
    return bool(left and right) if bilateral else bool(left or right)
