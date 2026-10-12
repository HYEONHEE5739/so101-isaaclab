"""Qwen multimodal base forward, without LM logits or all-layer hidden states."""
from contextlib import nullcontext
import re
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn

class QwenBackbone(nn.Module):
    def __init__(self, cfg, device='cuda:0', adapter_path=None):
        super().__init__()
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available(): raise RuntimeError('Real NF4 Qwen integration requires an NVIDIA CUDA GPU')
        self.cfg=cfg; self.device_name=device
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('This profile requires BF16; use a supported GPU')
        quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.processor=AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision,
            min_pixels=64*64, max_pixels=cfg.image_height*cfg.image_width)
        # Load the published checkpoint class to preserve model.* key mapping.
        # Loading Qwen3VLModel directly in Transformers 4.57.1 silently misses those keys.
        loaded, loading_info=Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_id, revision=cfg.revision,
            quantization_config=quant, dtype=torch.bfloat16,
            device_map={'':device}, attn_implementation='sdpa', output_loading_info=True)
        if loading_info['missing_keys'] or loading_info.get('mismatched_keys'):
            raise RuntimeError('Backbone checkpoint missing/mismatched weights: '+str(loading_info))
        self.model=loaded.model  # no LM-head forward, no vocabulary logits
        # PEFT uses these flags to dispatch Linear4bit adapters after extracting the base.
        self.model.is_loaded_in_4bit=getattr(loaded,'is_loaded_in_4bit',False)
        self.model.is_loaded_in_8bit=getattr(loaded,'is_loaded_in_8bit',False)
        self.model.is_quantized=getattr(loaded,'is_quantized',False)
        del loaded
        self.resolved_revision=getattr(self.model.config,'_commit_hash',None)
        if self.model.config.text_config.hidden_size!=cfg.vl_dim: raise ValueError('Backbone hidden dimension mismatch')
        self.image_token_id=self.model.config.image_token_id
        self.merge=self.model.config.vision_config.spatial_merge_size
        self.model.config.use_cache=False
        self.model.requires_grad_(False)
        self.targets=[]
        if cfg.adaptation=='qlora':
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
            self.model=prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            # Checkpoint only the language stack; vision remains frozen.
            self.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            layers=self.model.config.text_config.num_hidden_layers
            self.targets=[name for name,_ in self.model.named_modules()
                if (match:=re.search(r'language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$',name))
                and int(match.group(1))>=layers-cfg.lora_last_layers]
            if len(self.targets)!=2*min(layers,cfg.lora_last_layers): raise RuntimeError('Unexpected Qwen LoRA target names')
            if adapter_path:
                self.model=PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
            else:
                self.model=get_peft_model(self.model,LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_alpha,
                    target_modules=self.targets,lora_dropout=cfg.lora_dropout,bias='none'))
            if not any('bnb' in type(module).__module__ and hasattr(module,'lora_A') for module in self.model.modules()):
                raise RuntimeError('Expected bitsandbytes LoRA dispatch; check quantization flags')
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self,'model'):
            if self.cfg.adaptation=='frozen': self.model.eval()
            else: self.model.get_base_model().visual.eval()
        return self

    def prepare(self, image_batches, instructions, augment=False):
        texts=[]; flat_images=[]
        if len(image_batches)!=len(instructions): raise ValueError('Image/text batch mismatch')
        for images, instruction in zip(image_batches,instructions):
            instruction=' '.join(instruction.split())
            if not instruction: raise ValueError('Empty instruction')
            if len(self.processor.tokenizer.encode(instruction,add_special_tokens=False))>self.cfg.max_instruction_tokens:
                raise ValueError('Instruction too long; refusing silent truncation')
            if len(images)!=len(self.cfg.cameras): raise ValueError('Camera count/order mismatch')
            content=[{'type':'text','text':'Robot task: '+instruction}]
            for camera, image in zip(self.cfg.cameras, images):
                arr=np.asarray(image)
                if arr.dtype!=np.uint8 or arr.ndim!=3 or arr.shape[-1]!=3: raise ValueError('Expected RGB uint8 HWC')
                im=Image.fromarray(arr).convert('RGB')
                im.thumbnail((self.cfg.image_width,self.cfg.image_height),Image.Resampling.LANCZOS)
                if augment:
                    im=ImageEnhance.Brightness(im).enhance(float(np.random.uniform(.9,1.1)))
                    im=ImageEnhance.Contrast(im).enhance(float(np.random.uniform(.9,1.1)))
                # No crop, flip, hue shift: preserve geometry/color semantics.
                flat_images.append(im)
                content.extend([{'type':'text','text':camera+' camera:'},{'type':'image'}])
            content.append({'type':'text','text':'Select the target and destination for this task: '+instruction})
            texts.append(self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
        inputs=self.processor(text=texts,images=flat_images,padding=True,return_tensors='pt')
        if inputs.input_ids.shape[1]>self.cfg.max_tokens: raise ValueError('VL token budget exceeded; lower resolution/camera count')
        ids=inputs.input_ids
        xy=torch.zeros(*ids.shape,2); cams=torch.zeros_like(ids)
        grids=iter(inputs.image_grid_thw.tolist())
        for b in range(ids.shape[0]):
            positions=(ids[b]==self.image_token_id).nonzero().flatten(); offset=0
            for cam in range(len(self.cfg.cameras)):
                t,h,w=next(grids); h//=self.merge; w//=self.merge
                if t!=1: raise ValueError('Only still images supported')
                count=h*w
                pos=positions[offset:offset+count]
                if len(pos)!=count: raise ValueError('Processor/image-grid token mismatch')
                y,x=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
                xy[b,pos]=torch.stack([x.flatten(),y.flatten()],-1)
                cams[b,pos]=cam+1; offset+=count
            if offset!=len(positions): raise ValueError('Unexpected image tokens')
        inputs={k:v.to(self.device_name) for k,v in inputs.items()}
        if 'pixel_values' in inputs: inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
        return inputs,xy.to(self.device_name),cams.to(self.device_name)

    def forward(self, image_batches, instructions, augment=False):
        inputs,xy,camera=self.prepare(image_batches,instructions,augment)
        context=torch.no_grad() if self.cfg.adaptation=='frozen' else nullcontext()
        with context:
            out=self.model(**inputs,use_cache=False,output_hidden_states=False,return_dict=True)
        tokens=out.last_hidden_state
        if self.training and self.cfg.adaptation=='qlora' and not tokens.requires_grad:
            raise RuntimeError('QLoRA features detached from action loss')
        return tokens,inputs['attention_mask'].bool(),xy,camera
