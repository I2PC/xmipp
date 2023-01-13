from typing import Iterable
import numpy as np
import torch

from ..read import read_data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths: Iterable[str]):
        self._paths = paths
        
    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, index) -> torch.Tensor:
        path = self._paths[index]
        return torch.tensor(read_data(path))