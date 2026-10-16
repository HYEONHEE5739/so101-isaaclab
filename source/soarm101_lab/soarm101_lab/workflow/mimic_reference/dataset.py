"""Reference pool I/O backed by the common loader, without importing Isaac."""
from dataclasses import dataclass
import torch
from ..dataset import EpisodeDataset


@dataclass
class EpisodeData:
    data:dict


class HDF5DatasetFileHandler:
    def open(self,path):self.reader=EpisodeDataset(path)
    def get_episode_names(self):return self.reader.mimic_names
    def load_episode(self,name,device):
        values,_,_=self.reader.episode(name)
        def tensor(value):
            if isinstance(value,dict):return {k:tensor(v) for k,v in value.items()}
            result=torch.as_tensor(value,device=device)
            return result.float() if result.dtype.is_floating_point else result
        # The reference runtime uses float32 poses. Stored float64 telemetry remains lossless.
        needed={'actions':values['actions'],'obs':{'datagen_info':values['obs']['datagen_info']}}
        return EpisodeData(tensor(needed))
