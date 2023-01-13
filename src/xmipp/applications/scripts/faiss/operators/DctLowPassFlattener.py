from typing import Optional, Sequence
import torch

from SpectraFlattener import SpectraFlattener

class DctLowPassFlattener(SpectraFlattener):
    def __init__(   self, 
                    dim: int, 
                    cutoff: float, 
                    device: Optional[torch.device] = None ):
        SpectraFlattener.__init__(
            self, 
            self._compute_mask(dim, cutoff), 
            device=device
        )
    
    def _compute_mask(  self, 
                        dim: int, 
                        cutoff: float ) -> torch.Tensor:
        
        # Compute the frequency grid
        freq_x = torch.linspace(start=0, end=0.5, steps=dim)
        freq_y = freq_x[...,None]
        freq2 = freq_x**2 + freq_y**2
        
        # Compute the mask
        cutoff2 = cutoff ** 2
        mask = freq2.less_equal(cutoff2)
        return mask
