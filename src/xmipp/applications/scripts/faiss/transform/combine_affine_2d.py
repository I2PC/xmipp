from typing import Optional
import torch

def combine_affine_2d(rotation_matrices_2d: torch.Tensor,
                      shifts: torch.Tensor,
                      out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # Create the output if not existent
    MATRIX_SHAPE = (3, 3)
    shape = torch.broadcast_shapes(
        rotation_matrices_2d.shape[:-2], 
        shifts.shape[:-1]
    ) + MATRIX_SHAPE
    device = rotation_matrices_2d.device
    if out is None:
        out = torch.empty(shape, device=device)
    else:
        if out.shape != shape:
            pass # raise exception
    assert(out.shape == shape)
    
    # Write the rotation matrix
    out[...,0:2,0:2] = rotation_matrices_2d
    
    # Write the shift
    out[...,0:2,2] = shifts
    
    # Write the last row
    out[...,2,:] = [0, 0, 1]
        
    return out
    
    