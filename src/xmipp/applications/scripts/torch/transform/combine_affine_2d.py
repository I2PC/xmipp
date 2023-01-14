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

def combine_affine_2d(rotation_matrices_2d: torch.Tensor,
                      shifts: torch.Tensor,
                      out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # Create the output if not existent
    MATRIX_SHAPE = (2, 3)
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
    
    return out
    
    