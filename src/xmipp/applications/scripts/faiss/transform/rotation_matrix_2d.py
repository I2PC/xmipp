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