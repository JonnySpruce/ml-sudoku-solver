from dataclasses import dataclass
from torch.nn import Module

@dataclass
class NNModel:
    path: str
    model: Module
