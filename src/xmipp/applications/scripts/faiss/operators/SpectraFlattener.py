from typing import Optional
import torch

class SpectraFlattener:
    def __init__(   self, 
                    mask: torch.Tensor,
                    device: Optional[torch.device]):
        self._mask = mask
        self._length = int(torch.count_nonzero(mask))
    
    def __call__(   self,
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # To avoid warnings
        if out is not None:
            out.resize_(0)
        
        out = torch.masked_select(input, self.get_mask(), out=out)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        return out
    
    def get_mask(self) -> torch.BoolTensor:
        return self._mask
    
    def get_length(self) -> int:
        return self._length
