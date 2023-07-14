# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from typing import Optional
import torch
import kornia
import math

"""
TODO: Compute only half of the sinogram and for the other half use a 
reflection of it.
"""

def compute_sinogram_2d(images: torch.Tensor,
                        n_angles: int,
                        interpolation: str = 'bilinear',
                        out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    if len(images.shape) != 3:
        raise RuntimeError('Input images must have [B, H, W] shape')
    
    # Compute the angle grid
    angles = torch.linspace(0.0, 360.0, n_angles+1)[:-1]
    
    # Create a gallery of rotated images
    transformed = kornia.geometry.transform.rotate(
        images.expand(len(angles), -1, -1, -1),
        angle=angles,
        mode=interpolation,
        padding_mode='zeros'
    )
    
    # Project in the X direction. TODO maybe in Y?
    out = torch.sum(transformed, dim=-1, out=out)
    
    # Swap the axes so that batch dimension comes first
    out = torch.transpose(out, 0, 1)
    
    assert(out.shape[0] == images.shape[0])
    assert(out.shape[1] == n_angles)
    return out

def angle_from_line_2d(lines: torch.Tensor,
                       out: Optional[torch.Tensor] = None):
    if lines.shape[-1] != 2:
        raise RuntimeError('lines2d parameter must have 2 elements in the last dim')

    # According to the prior sinogram, 0deg is (0, 1)
    return torch.atan2(lines[...,0], lines[...,1], out=out)

def index_from_line_2d(lines: torch.Tensor,
                       n_angles: float,
                       out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # TODO check the sign
    out = angle_from_line_2d(lines, out=out)
    out *= n_angles / 2*torch.pi
    return out

def _extract_projection_2d_nearest(sinogram: torch.Tensor,
                                   indices: torch.Tensor,
                                   out: Optional[torch.Tensor] = None) -> torch.Tensor:
    indices = torch.round(indices).to(torch.int32)
    indices %= len(sinogram)
    assert(torch.all(torch.abs(indices) < len(sinogram)))
    return torch.index_select(sinogram, dim=0, index=indices, out=out)

def _extract_projection_2d_linear(sinogram: torch.Tensor,
                                  indices: float,
                                  out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # TODO optimize this function
    prev_indices = torch.floor(indices)
    prev_indices %= len(sinogram)
    assert(torch.all(torch.abs(prev_indices) < len(sinogram)))
    next_indices = (prev_indices + 1)
    next_indices %= len(sinogram)
    assert(torch.all(torch.abs(next_indices) < len(sinogram)))
    frac_indices = indices - prev_indices
    assert(torch.all(frac_indices >= 0.0))
    assert(torch.all(frac_indices < 1.0))

    return torch.lerp(sinogram[prev_indices], sinogram[next_indices], frac_indices, out=out)



def extract_projection_2d(sinogram: torch.Tensor,
                          indices: torch.Tensor,
                          interpolation: str = 'linear',
                          out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    if interpolation == 'nearest':
        return _extract_projection_2d_nearest(sinogram=sinogram, indices=indices, out=out)   
    elif interpolation == 'linear':
        return _extract_projection_2d_linear(sinogram=sinogram, indices=indices, out=out)   
    else:
        raise RuntimeError('Interpolation must be nearest or linear')
