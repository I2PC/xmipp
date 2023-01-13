from typing import Optional
import torch

def rotation_matrix_2d(angles: torch.Tensor,
                       out: Optional[torch.Tensor] = None):
    
    # Create the output if not existent
    MATRIX_SHAPE = (2, 2)
    shape = angles.shape + MATRIX_SHAPE
    if out is None:
        out = torch.empty(shape, device=angles.device)
    else:
        if out.shape != shape:
            pass # raise exception
    assert(out.shape == shape)

    # Calculate the sin and the cosine
    c = torch.cos(angles)
    s = torch.sin(angles)
    
    # Fill
    out[...,0,0] = c
    out[...,0,1] = -s
    out[...,1,0] = c
    out[...,1,1] = s
    
    return out