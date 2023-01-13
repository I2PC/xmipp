from typing import Optional
import torch

from Transformer2D import Transformer2D

class FourierTransformer2D(Transformer2D):
    def __call__(   self, 
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # To avoid warnings
        if out is not None:
            out.resize_(0)
            
        return torch.fft.rfft2(input, out=out)