from dataclasses import dataclass, asdict
import json
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent

@dataclass
class MiniVLAConfig:
    model_id: str = 'Qwen/Qwen3-VL-2B-Instruct'
    revision: str = 'main'
    adaptation: str = 'frozen'
    head_type: str = 'deterministic'
    vl_dim: int = 2048
    dim: int = 256
    layers: int = 2
    heads: int = 4
    ffn: int = 1024
    dropout: float = 0.1
    horizon: int = 16
    action_dim: int = 6
    cameras: tuple = ('side',)
    image_height: int = 256
    image_width: int = 320
    max_tokens: int = 512
    max_instruction_tokens: int = 96
    flow_steps: int = 8
    gripper_weight: float = 1.0
    grounding_weight: float = 0.1
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_last_layers: int = 8
    lora_dropout: float = 0.05
    head_lr: float = 1e-4
    lora_lr: float = 1e-5
    weight_decay: float = 0.01
    accumulation: int = 8
    steps: int = 5000
    val_every: int = 100
    patience: int = 10
    seed: int = 42
    replan_steps: int = 4

    def __post_init__(self):
        self.cameras = tuple(self.cameras)
        assert self.adaptation in ('frozen', 'qlora')
        assert self.head_type in ('deterministic', 'flow')
        assert self.action_dim == 6 and self.dim % self.heads == 0 and self.dim % 2 == 0
        assert 1 <= self.replan_steps <= self.horizon and self.flow_steps > 0
        assert self.layers > 0 and self.accumulation > 0 and self.steps > 0
        assert 0 < len(self.cameras) <= 4 and len(set(self.cameras)) == len(self.cameras)
        assert self.gripper_weight > 0 and self.grounding_weight >= 0

    def save(self, path):
        with open(path, 'w') as f: json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f: return cls(**json.load(f))
