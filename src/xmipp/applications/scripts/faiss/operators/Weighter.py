from typing import Optional
import torch

from SpectraFlattener import SpectraFlattener

class Weighter:
    def __init__(self,
                 weights: torch.Tensor,
                 flattener: SpectraFlattener,
                 device: Optional[torch.device] = None):
        
        self._weights = flattener(weights[:,:flattener._mask.shape[1]].to(device))
    
    def __call__(   self,
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None ):
        torch.mul(input, self._weights, out=out)
        return out
    