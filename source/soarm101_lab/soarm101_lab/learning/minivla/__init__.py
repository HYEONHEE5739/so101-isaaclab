"""SO-101 MiniVLA model, training and inference; heavy imports are lazy."""

__all__ = ['MiniVLAConfig', 'ActionHead']


def __getattr__(name):
    if name == 'MiniVLAConfig':
        from .config import MiniVLAConfig
        return MiniVLAConfig
    if name == 'ActionHead':
        from .head import ActionHead
        return ActionHead
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
