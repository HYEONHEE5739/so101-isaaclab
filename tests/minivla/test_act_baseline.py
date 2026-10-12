import numpy as np
import pytest
import torch
pytest.importorskip('lerobot.policies.act.modeling_act')
from soarm101_lab.learning.minivla.act_baseline import build,prepare,objective
from soarm101_lab.learning.minivla.config import MiniVLAConfig

def test_official_act_forward_backward_and_eval():
    c=MiniVLAConfig(horizon=4,replan_steps=2,image_height=64,image_width=64)
    model=build(c,pretrained=False)
    batch={'images':[[np.zeros((64,64,3),np.uint8)]], 'state':torch.zeros(1,6),
        'actions':torch.zeros(1,4,6),'mask':torch.tensor([[True,True,False,False]])}
    b=prepare(batch,c,'cpu'); loss,_=objective(model,b,c); assert torch.isfinite(loss)
    loss.backward(); assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        result=model.predict_action_chunk(b); val,_=objective(model,b,c)
    assert result.shape==(1,4,6) and torch.isfinite(val)
