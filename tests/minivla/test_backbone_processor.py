"""Contract tests for actual Qwen processor if local downloaded files are supplied."""
import os
import pytest
import numpy as np
import torch
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.learning.minivla.backbone import QwenBackbone, letterbox

@pytest.mark.parametrize('batch_size',[1,8])
def test_real_processor_camera_grid_and_budget(batch_size):
    path=os.environ.get('QWEN_PROCESSOR_PATH')
    if not path: pytest.skip('Set QWEN_PROCESSOR_PATH to local Qwen processor directory')
    from transformers import AutoProcessor
    b=QwenBackbone.__new__(QwenBackbone); torch.nn.Module.__init__(b)
    b.cfg=MiniVLAConfig(); b.device_name='cpu'; b.merge=2; b.image_token_id=151655
    b.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=65536,max_pixels=65536)
    images=[[np.zeros((480,640,3),np.uint8)]]*batch_size
    instructions=['pick the red cube'+' carefully'*(i%3) for i in range(batch_size)]
    inputs,xy,camera=b.prepare(images,instructions)
    assert (camera==1).sum()==64*batch_size and (camera==2).sum()==0
    assert inputs['image_grid_thw'].tolist()==[[1,16,16]]*batch_size
    assert b.last_preprocessing['processor_hw']==[[256,256]]*batch_size
    assert ((xy[camera>0]>0)&(xy[camera>0]<1)).all()
    assert inputs['input_ids'].shape==camera.shape
    assert inputs['input_ids'].shape==inputs['attention_mask'].shape and inputs['input_ids'].shape[0]==batch_size
    b.cfg.max_tokens=2
    with pytest.raises(ValueError,match='budget'): b.prepare([[np.zeros((480,640,3),np.uint8)]],['pick red'])


def test_letterbox_exact_padding_and_no_color_shift():
    source=np.full((480,640,3),[255,0,0],dtype=np.uint8);copy=source.copy()
    image=np.asarray(letterbox(source))
    assert image.shape==(256,256,3)
    assert not image[:32].any() and not image[224:].any()
    assert np.all(image[32:224]==[255,0,0])
    np.testing.assert_array_equal(source,copy)
