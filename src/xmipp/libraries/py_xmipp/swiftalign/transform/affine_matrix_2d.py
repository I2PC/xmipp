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

from .rotation_matrix_2d import rotation_matrix_2d

def affine_matrix_2d(angles: torch.Tensor,
                     shifts: torch.Tensor,
                     centre: torch.Tensor,
                     shift_first: bool = False,
                     out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    batch_size = len(angles)

    if angles.shape != (batch_size, ):
        raise RuntimeError('angles has not the expected size')

    if shifts.shape != (batch_size, 2):
        raise RuntimeError('shifts has not the expected size')

    out = torch.empty((batch_size, 2, 3), out=out)

    # Compute the rotation matrix
    rotation_matrices = out[:,:2,:2]
    rotation_matrices = rotation_matrix_2d(
        angles=angles, 
        out=rotation_matrices
    )
    
    # Determine the shifts
    if shift_first:
        pre_shift = shifts - centre
        post_shift = centre
    else:
        pre_shift = -centre
        post_shift = shifts + centre
    
    # Apply the shifts
    out[:,:,2,None] = torch.matmul(
        rotation_matrices, 
        pre_shift[...,None], 
        out=out[:,:,2,None]
    )
    out[:,:,2] += post_shift

    return out
