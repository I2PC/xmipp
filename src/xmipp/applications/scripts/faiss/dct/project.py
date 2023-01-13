from typing import Optional, Iterable
import torch

def project(x: torch.Tensor, 
            dim: int, 
            basis: torch.Tensor, 
            out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    if out is x:
        raise Exception('Aliasing between x and out is not supported')
    
    def t(x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim, -1)
    
    # Transpose the input to have dim
    # on the last axis
    x = t(x)
    if out is not None:
        out = t(out)
    
    # Perform the projection
    out = torch.matmul(basis, x, out=out)
    
    # Undo the transposition
    out = t(out) # Return dim to its place

    return out

def project_nd(x: torch.Tensor, 
               dims: Iterable[int], 
               bases: Iterable[torch.Tensor],
               out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    temp = None
    for i, (dim, basis) in enumerate(zip(dims, bases)):
        if i == 0:
            # First iteration
            out = project(x, dim, basis, out=out)
        else:
            assert(out is not None)
            if temp is None:
                temp = out.clone()
            else:
                temp[...] = out
            
            out = project(temp, dim, basis, out=out)
            
    return out