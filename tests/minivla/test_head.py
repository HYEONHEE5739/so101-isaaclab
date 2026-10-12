import pytest
import torch
from soarm101_lab.learning.minivla import MiniVLAConfig, ActionHead

torch.set_num_threads(2)

def make(kind='deterministic'):
    c=MiniVLAConfig(head_type=kind,vl_dim=32,dim=32,heads=4,ffn=64,horizon=4,replan_steps=2,dropout=0)
    h=ActionHead(c)
    f=(torch.randn(2,7,32,requires_grad=True),torch.tensor([[1]*5+[0]*2,[1]*7],dtype=torch.bool),torch.rand(2,7,2),torch.zeros(2,7,dtype=torch.long))
    s=torch.randn(2,6); a=torch.randn(2,4,6); m=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    return h,f,s,a,m

@pytest.mark.parametrize('kind',['deterministic','flow'])
def test_train_and_infer(kind):
    h,f,s,a,m=make(kind); opt=torch.optim.AdamW(h.parameters(),lr=.001)
    before=h.out[-1].weight.detach().clone()
    loss,_=h.loss(f,s,a,m); assert torch.isfinite(loss)
    loss.backward(); assert f[0].grad is not None and f[0].grad.abs().sum()>0
    opt.step(); assert not torch.equal(before,h.out[-1].weight)
    h.eval(); result=h.sample(f,s); assert result.shape==(2,4,6) and torch.isfinite(result).all()

@pytest.mark.parametrize('kind',['deterministic','flow'])
def test_padding_cannot_change_valid_loss(kind):
    h,f,s,a,m=make(kind)
    torch.manual_seed(11); x=h.loss(f,s,a,m)[0]
    altered=a.clone(); altered[~m]=100000
    torch.manual_seed(11); y=h.loss(f,s,altered,m)[0]
    torch.testing.assert_close(x,y)

def test_context_padding_and_language_path():
    h,f,s,a,m=make(); h.eval()
    original=h.sample(f,s); changed=f[0].detach().clone(); changed[~f[1]]=100000
    torch.testing.assert_close(original,h.sample((changed,*f[1:]),s))
    changed=f[0].detach().clone(); changed[:,0]+=torch.randn_like(changed[:,0])*5
    assert not torch.allclose(original,h.sample((changed,*f[1:]),s))

def test_mask_denominator_and_grounding():
    h,f,s,a,m=make()
    assert h.weighted_loss(torch.ones_like(a),m).item()==1
    loss,metrics=h.loss(f,s,a,m,torch.rand(2,2,2),torch.ones(2,2,dtype=torch.bool))
    loss.backward(); assert 'grounding_loss' in metrics and h.ground_queries.grad.abs().sum()>0

def test_overfit_small_fixed_batch():
    torch.manual_seed(4); h,f,s,a,m=make(); f=(f[0].detach(),*f[1:])
    opt=torch.optim.AdamW(h.parameters(),lr=.003)
    initial=h.loss(f,s,a,m)[0].item()
    for _ in range(160):
        opt.zero_grad(); loss,_=h.loss(f,s,a,m); loss.backward(); opt.step()
    assert h.loss(f,s,a,m)[0].item()<initial*.25
