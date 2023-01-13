from typing import Optional, Sequence
import torch

from dct import dct_ii_basis, project_nd
from Transformer2D import Transformer2D

class DctTransformer2D(Transformer2D):
    DIMS = (-1, -2) # Last two dimensions
    
    def __init__(self, dim: int) -> None:
        self._bases = (dct_ii_basis(dim), )*len(self.DIMS)
    
    def __call__(   self, 
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # To avoid warnings
        if out is not None:
            out.resize_(0)
            
        return project_nd(input, dims=self.DIMS, bases=self._bases, out=out)