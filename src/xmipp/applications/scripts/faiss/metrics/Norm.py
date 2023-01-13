from typing import Sequence
import torch

class Norm:
    def __init__(self, shape: Sequence[int], ord: int):
        self._dim = list(range(-len(shape), 0))
        self._ord = ord
    
    def __call__(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(diff, ord=self._ord, dim=self._dim)
        