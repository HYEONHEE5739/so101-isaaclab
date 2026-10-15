import pytest
import torch
from soarm101_lab.learning.minivla import MiniVLAConfig, ActionHead
from soarm101_lab.learning.minivla.head import AdaLNZero, FlowDiTBlock

torch.set_num_threads(2)


def make():
    c=MiniVLAConfig(vl_dim=32,dim=32,heads=4,ffn=64,horizon=4,replan_steps=2,dropout=0)
    h=ActionHead(c)
    f=(torch.randn(2,7,32,requires_grad=True),torch.tensor([[1]*5+[0]*2,[1]*7],dtype=torch.bool),torch.rand(2,7,2),torch.zeros(2,7,dtype=torch.long))
    return h,f,torch.randn(2,6),torch.randn(2,4,6),torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)


def open_gates(h):
    # A first optimizer step learns gates; zero-initialized branches cannot pass
    # language/state gradients until those gates open.
    with torch.no_grad():
        for block in h.blocks:
            block.adaln.modulation[-1].bias.fill_(.1)


def test_default_architecture_and_adaln_zero():
    c=MiniVLAConfig();h=ActionHead(c)
    assert (c.vl_dim,c.dim,c.layers,c.horizon,c.flow_steps,c.replan_steps)==(2048,256,4,40,8,4)
    assert len(h.blocks)==4 and all(isinstance(b,FlowDiTBlock) for b in h.blocks)
    assert h.position.shape==(1,40,256)
    for b in h.blocks:
        assert b.adaln.modulation[-1].weight.shape==(9*256,256)
        assert torch.count_nonzero(b.adaln.modulation[-1].weight)==0
        assert all(torch.count_nonzero(v)==0 for v in b.adaln(torch.randn(2,256)))
    shape=[]
    hook=h.blocks[0].register_forward_pre_hook(lambda _m,args:shape.append(tuple(args[0].shape)))
    f=(torch.randn(1,70,2048),torch.ones(1,70,dtype=torch.bool),torch.zeros(1,70,2),torch.zeros(1,70,dtype=torch.long))
    y=h.sample(f,torch.zeros(1,6));hook.remove()
    assert y.shape==(1,40,6) and shape==[(1,41,256)]*8
    assert not any('ground' in k for k in h.state_dict())


def test_adaln_and_upstream_gradient_after_first_step():
    h,f,s,a,m=make();opt=torch.optim.AdamW(h.parameters(),lr=.001)
    loss,_=h.loss(f,s,a,m);loss.backward()
    assert h.blocks[0].adaln.modulation[-1].weight.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    loss,_=h.loss(f,s,a,m);loss.backward()
    assert f[0].grad is not None and f[0].grad.abs().sum()>0
    assert h.state[0].weight.grad.abs().sum()>0
    assert h.time.mlp[-1].weight.grad.abs().sum()>0
    assert h.blocks[0].self_attn.in_proj_weight.grad.abs().sum()>0


def test_padding_cannot_change_loss_or_valid_outputs():
    h,f,s,a,m=make();open_gates(h)
    torch.manual_seed(11);x=h.loss(f,s,a,m)[0]
    altered=a.clone();altered[~m]=float('nan')
    torch.manual_seed(11);y=h.loss(f,s,altered,m)[0]
    torch.testing.assert_close(x,y)
    h.eval();noise=torch.randn_like(a);ctx,cm=h.context(f)
    x=h(ctx,cm,s,noise,torch.ones(2),m)
    noise[~m]=float('nan');y=h(ctx,cm,s,noise,torch.ones(2),m)
    torch.testing.assert_close(x[m],y[m])


def test_context_padding_and_language_path():
    h,f,s,a,m=make();open_gates(h);h.eval();noise=torch.randn_like(a)
    original=h.sample(f,s,noise);changed=f[0].detach().clone();changed[~f[1]]=float('nan')
    torch.testing.assert_close(original,h.sample((changed,*f[1:]),s,noise))
    changed=f[0].detach().clone();changed[:,0]+=torch.randn_like(changed[:,0])*5
    assert not torch.allclose(original,h.sample((changed,*f[1:]),s,noise))


def test_euler_uses_one_context_and_eight_flow_times(monkeypatch):
    h,f,s,a,m=make();calls=[];context_calls=[]
    original=h.context
    monkeypatch.setattr(h,'context',lambda f:(context_calls.append(1) or original(f)))
    def velocity(ctx,cm,state,noisy,time,action_mask=None):
        calls.append(time[0].item());return torch.ones_like(noisy)*2
    monkeypatch.setattr(h,'forward',velocity)
    zero=torch.zeros_like(a);torch.testing.assert_close(h.sample(f,s,zero),zero+2)
    assert calls==[k/8 for k in range(8)] and len(context_calls)==1


def test_weighted_masked_mse():
    h,_,_,a,m=make();assert h.weighted_loss(torch.ones_like(a),m).item()==1
    with pytest.raises(ValueError):h.weighted_loss(a,torch.zeros_like(m))
