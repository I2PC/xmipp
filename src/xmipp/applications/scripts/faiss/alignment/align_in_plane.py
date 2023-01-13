from typing import Tuple, NamedTuple, Sequence, Optional
import torch
import torchvision
import math

DISTANCE_SLICE = 0
ANGLE_SLICE = 1
SHIFTX_SLICE = 2
SHIFTY_SLICE = 3
N_SLICES = 4

def align_in_plane(ref: torch.Tensor,
                   exps: torch.Tensor,
                   angles: torch.Tensor,
                   shifts: torch.Tensor,
                   metric,
                   interpolation: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR ) -> torch.Tensor:
    
    n_rotations = angles.shape[0]
    n_shifts = shifts.shape[0]
    n_exp = exps.shape[0]
    
    result = torch.empty((N_SLICES, n_exp))
    result[DISTANCE_SLICE] = torch.inf
    for rot_index in range(n_rotations):
        angle = float(angles[rot_index])
        
        for shift_index in range(n_shifts):
            shift = shifts[shift_index]
            
            # Compute the residual image after transform
            residual = torchvision.transforms.functional.affine(
                img=exps,
                angle=angle,
                translate=tuple(shift),
                interpolation=interpolation,
                scale=1.0,
                shear=90.0
            )
            residual -= ref
            
            # Compute the distances
            distances = metric(residual)
            
            # Select the values to be updated
            update_mask = distances < result[DISTANCE_SLICE]
            
            # Update values
            result[DISTANCE_SLICE, update_mask] = distances[update_mask]
            result[ANGLE_SLICE, update_mask] = angle
            result[SHIFTX_SLICE, update_mask] = shift[0]
            result[SHIFTY_SLICE, update_mask] = shift[1]

    return result