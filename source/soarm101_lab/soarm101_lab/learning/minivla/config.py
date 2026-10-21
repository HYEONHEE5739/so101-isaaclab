from dataclasses import dataclass, asdict
import json
from pathlib import Path

ARCHITECTURE_SCHEMA = 'so101.minivla.flow-dit/1'

MODEL_ROOT = Path(__file__).resolve().parent

@dataclass
class MiniVLAConfig:
    model_id: str = 'Qwen/Qwen3-VL-2B-Instruct'
    revision: str = 'main'
    adaptation: str = 'frozen'
    architecture_schema: str = ARCHITECTURE_SCHEMA
    feature_layer: int = 16
    vl_dim: int = 2048
    dim: int = 256
    layers: int = 4
    heads: int = 4
    ffn: int = 1024
    dropout: float = 0.1
    horizon: int = 40
    action_dim: int = 6
    cameras: tuple = ('side',)
    image_height: int = 256
    image_width: int = 256
    max_tokens: int = 512
    max_instruction_tokens: int = 96
    flow_steps: int = 8
    gripper_weight: float = 1.0
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_first_layer: int = 8
    lora_dropout: float = 0.05
    head_lr: float = 1e-4
    lora_lr: float = 1e-5
    weight_decay: float = 0.01
    accumulation: int = 1
    batch_size: int = 8
    steps: int = 5000
    save_freq: int = 1000
    val_every: int = 100
    early_stopping: bool = False
    patience: int = 10
    seed: int = 42
    replan_steps: int = 4

    def __post_init__(self):
        self.cameras = tuple(self.cameras)
        if type(self.save_freq) is not int or self.save_freq<1:raise ValueError("Positive integer save_freq required")
        if type(self.early_stopping) is not bool:raise ValueError('early_stopping must be boolean')
        for key in ('steps','val_every','patience','batch_size','accumulation'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f'Positive integer {key} required')
        assert self.adaptation in ('frozen', 'qlora')
        if self.architecture_schema != ARCHITECTURE_SCHEMA:
            raise ValueError('Unsupported MiniVLA architecture; retrain with Flow-DiT')
        if self.feature_layer != 16 or self.layers != 4 or self.flow_steps != 8:
            raise ValueError('MiniVLA requires decoder block 16, four DiT blocks, eight Euler steps')
        if not 0 <= self.lora_first_layer < self.feature_layer:
            raise ValueError('LoRA must reach the selected feature layer')
        assert self.action_dim == 6 and self.dim % self.heads == 0 and self.dim % 2 == 0
        assert 1 <= self.replan_steps <= self.horizon and self.flow_steps > 0
        assert self.layers > 0 and self.accumulation > 0 and self.steps > 0
        assert 0 < len(self.cameras) <= 4 and len(set(self.cameras)) == len(self.cameras)
        assert self.gripper_weight > 0

    def validate_architecture(self):
        expected = dict(model_id='Qwen/Qwen3-VL-2B-Instruct', feature_layer=16, vl_dim=2048,
            dim=256, layers=4, heads=4, ffn=1024, dropout=.1, horizon=40, action_dim=6,
            cameras=('side',), image_height=256, image_width=256, flow_steps=8, replan_steps=4)
        different = [key for key,value in expected.items() if getattr(self,key) != value]
        if different: raise ValueError('Fixed MiniVLA Flow-DiT architecture differs: '+', '.join(different))

    def save(self, path):
        with open(path, 'w') as f: json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f: return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, values):
        values=dict(values)
        # Configs from the original trainer omitted batch_size and used accumulation 8.
        if 'batch_size' not in values:
            values['batch_size']=1
            values.setdefault('accumulation',8)
        return cls(**values)
