"""Real Transformers/PEFT classes at random tiny size. Not pretrained/4-bit verification."""
import pytest
import torch
transformers=pytest.importorskip('transformers')
peft=pytest.importorskip('peft')
from transformers import Qwen3VLConfig, Qwen3VLModel
from peft import LoraConfig, get_peft_model
from soarm101_lab.learning.minivla import MiniVLAConfig, ActionHead

def tiny_qwen():
    cfg=Qwen3VLConfig(
        text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=28,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=512,
            rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
            patch_size=4,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,
            num_position_embeddings=16,deepstack_visual_indexes=[0]),
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    return Qwen3VLModel(cfg)

@pytest.mark.parametrize('adapt',['frozen','qlora'])
def test_actual_base_model_multimodal_and_adapter_gradient(adapt):
    torch.manual_seed(3); base=tiny_qwen(); base.requires_grad_(False)
    if adapt=='qlora':
        base.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        base=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=[f'language_model.layers.{i}.self_attn.{p}' for i in range(8,16) for p in ('q_proj','v_proj')]))
    c=MiniVLAConfig(vl_dim=32,dim=32,heads=4,ffn=64,horizon=4,replan_steps=2,dropout=0)
    head=ActionHead(c)
    ids=torch.tensor([[1,62,60,60,60,60,63,2]])
    kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(16,96),
        image_grid_thw=torch.tensor([[1,4,4]]),use_cache=False,output_hidden_states=False,return_dict=True)
    from soarm101_lab.learning.minivla.backbone import QwenBackbone
    backbone=QwenBackbone.__new__(QwenBackbone);torch.nn.Module.__init__(backbone)
    backbone.cfg=MiniVLAConfig(adaptation=adapt,vl_dim=32,dim=32,ffn=64)
    backbone.model=base; backbone.training=True
    captured=[]
    stack=base.get_base_model() if adapt=='qlora' else base
    hook=stack.language_model.layers[15].register_forward_hook(lambda m,a,o:captured.append(o))
    tokens=backbone.extract(kwargs);hook.remove()
    assert tokens.shape==(1,8,32) and tokens is captured[0]
    # AdaLN-Zero initially blocks upstream gradients; emulate learned open gates.
    from test_head import open_gates
    open_gates(head)
    f=(tokens,torch.ones(1,8,dtype=torch.bool),torch.zeros(1,8,2),torch.zeros(1,8,dtype=torch.long))
    loss,_=head.loss(f,torch.zeros(1,6),torch.ones(1,4,6),torch.ones(1,4,dtype=torch.bool))
    loss.backward()
    grads=[p.grad for n,p in base.named_parameters() if 'lora_' in n]
    if adapt=='qlora':
        assert any(g is not None and g.abs().sum()>0 for g in grads)
        assert all(8<=int(n.split('layers.')[1].split('.')[0])<=15 for n,p in base.named_parameters() if p.requires_grad)
    else: assert all(p.grad is None for p in base.parameters())

def test_published_checkpoint_prefix_load(tmp_path):
    from transformers import Qwen3VLForConditionalGeneration
    source=Qwen3VLForConditionalGeneration(tiny_qwen().config)
    source.save_pretrained(tmp_path)
    loaded, info=Qwen3VLForConditionalGeneration.from_pretrained(tmp_path,output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys']
    base=loaded.model
    torch.testing.assert_close(base.language_model.embed_tokens.weight,source.model.language_model.embed_tokens.weight)
