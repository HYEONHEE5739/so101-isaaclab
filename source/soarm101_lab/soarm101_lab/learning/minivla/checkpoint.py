"""MiniVLA checkpoint architecture and adapter state; independent of training/data preparation."""
from .config import ARCHITECTURE_SCHEMA


def adapter_state(backbone):
    if backbone.cfg.adaptation=='frozen': return None
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu() for k,v in get_peft_model_state_dict(backbone.model).items()}

def restore_adapter(backbone, state):
    if state is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(backbone.model,state)

def require_architecture(checkpoint):
    if checkpoint.get('architecture_schema') != ARCHITECTURE_SCHEMA:
        raise ValueError('Legacy/incompatible MiniVLA checkpoint: expected '+ARCHITECTURE_SCHEMA+'; retraining is required')
