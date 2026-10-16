"""Portable construction for the reference configuration, not a simulator mock."""
import copy
from dataclasses import dataclass,field
from typing import Any,Protocol


def configclass(cls):
    def __init__(self,**kwargs):
        annotations={k:v for base in reversed(cls.__mro__) for k,v in getattr(base,'__annotations__',{}).items()}
        unknown=set(kwargs)-set(annotations)
        if unknown:raise TypeError('Unknown Mimic configuration fields: '+str(unknown))
        for name in annotations:setattr(self,name,copy.deepcopy(kwargs.get(name,getattr(cls,name,None))))
    cls.__init__=__init__
    return cls


class ManagerBasedRLMimicEnv(Protocol):
    """Reference generator's environment protocol; implemented by backend adapters."""
    cfg:Any
    device:str


@dataclass
class TerminationTermCfg:
    func:Any
    params:dict=field(default_factory=dict)
