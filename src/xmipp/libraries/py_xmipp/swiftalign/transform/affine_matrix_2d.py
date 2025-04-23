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

from typing import Optional, Union
import torch

from .rotation_matrix_2d import rotation_matrix_2d

def _apply_shifts(m23: torch.Tensor,
                  shifts: torch.Tensor,
                  centre: torch.Tensor,
                  shift_first: bool = False ) -> torch.Tensor:
    if shift_first:
        pre_shift = shifts - centre
        post_shift = centre
    else:
        pre_shift = -centre
        post_shift = shifts + centre
    
    # Apply the shifts
    torch.matmul(
        m23[...,:2,:2], 
        pre_shift[...,None], 
        out=m23[...,:,2,None]
    )
    m23[...,:,2] += post_shift

def affine_matrix_2d(angles: torch.Tensor,
                     shifts: torch.Tensor,
                     centre: torch.Tensor,
                     mirror: Union[torch.Tensor, bool] = False,
                     shift_first: bool = False,
                     out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    if isinstance(angles, tuple):
        batch_shape = torch.broadcast_shapes(angles[0].shape, shifts.shape[:-1])
    else:
        batch_shape = torch.broadcast_shapes(angles.shape, shifts.shape[:-1])

    if shifts.shape[-1] != 2:
        raise RuntimeError('shifts has not the expected size')

    out = torch.empty(
        batch_shape + (2, 3), 
        dtype=shifts.dtype, 
        device=shifts.device, 
        out=out
    )

    # Compute the rotation matrix
    rotation_matrix_2d(
        angles=angles, 
        out=out[...,:2,:2]
    )
    
    # Flip if necessary
    if isinstance(mirror, bool):
        if mirror:
            out[...,0,:] = -out[...,0,:]
    else:
        out[mirror,:2,1] = -out[mirror,:2,1]
    
    _apply_shifts(
        m23=out,
        shifts=shifts,
        centre=centre,
        shift_first=shift_first
    )

    return out

def make_affine_matrix_2d(m22: torch.Tensor,
                          shifts: torch.Tensor,
                          centre: torch.Tensor,
                          shift_first: bool = False,
                          out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    batch_shape = m22.shape[:-2]

    if m22.shape != batch_shape + (2, 2):
        raise RuntimeError('angles has not the expected size')

    if shifts.shape != batch_shape + (2, ):
        raise RuntimeError('shifts has not the expected size')

    out = torch.empty(batch_shape + (2, 3), out=out)
    
    out[...,:2,:2] = m22
    _apply_shifts(
        m23=out,
        shifts=shifts,
        centre=centre,
        shift_first=shift_first
    )
    
    return out