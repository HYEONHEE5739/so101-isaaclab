"""Contract tests for actual Qwen processor if local downloaded files are supplied."""
import os
import pytest
import numpy as np
import torch
from soarm101_lab.learning.minivla.config import MiniVLAConfig
from soarm101_lab.learning.minivla.backbone import QwenBackbone

def test_real_processor_camera_grid_and_budget():
    path=os.environ.get('QWEN_PROCESSOR_PATH')
    if not path: pytest.skip('Set QWEN_PROCESSOR_PATH to local Qwen processor directory')
    from transformers import AutoProcessor
    b=QwenBackbone.__new__(QwenBackbone); torch.nn.Module.__init__(b)
    b.cfg=MiniVLAConfig(cameras=('side','wrist')); b.device_name='cpu'; b.merge=2; b.image_token_id=151655
    b.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=4096,max_pixels=81920)
    inputs,xy,camera=b.prepare([[np.zeros((480,640,3),np.uint8)]*2],['pick the red cube'])
    assert (camera==1).sum()>0 and (camera==2).sum()>0
    assert ((xy[camera>0]>0)&(xy[camera>0]<1)).all()
    assert inputs['input_ids'].shape==camera.shape
    b.cfg.max_tokens=2
    with pytest.raises(ValueError,match='budget'): b.prepare([[np.zeros((480,640,3),np.uint8)]*2],['pick red'])
