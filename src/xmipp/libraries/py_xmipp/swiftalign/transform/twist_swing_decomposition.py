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

from .quaternion_arithmetic import quaternion_product, quaternion_conj

def twist_decomposition(quaternions: torch.Tensor,
                        direction: Union[torch.Tensor, int],
                        assume_normalized: bool = True,
                        normalize_output: bool = True,
                        out: Optional[torch.Tensor] = None) -> torch.Tensor:

    if not assume_normalized:
        raise NotImplementedError()
    
    # Create the output
    out = torch.empty(quaternions.shape, dtype=quaternions.dtype, device=quaternions.device, out=out)
    
    if isinstance(direction, int):
        # Only copy the selected axis to te output
        for i in range(3):
            out[...,i+1] = quaternions[...,i+1] if i == direction else 0
        
    else:
        # Compute the dot product of the direction and quaternion xyz
        # and store it on the w component of the output
        torch.matmul(direction[...,None,:], quaternions[...,1:4,None], out=out[...,0,None])
    
        # Compute the xyz components by scaling the direction
        # with the dot product. This is equivalent to the 
        # projection of q.xyz onto the direction
        torch.mul(direction, out[...,0,None], out=out[1:4])
    
    # Overwrite the w component with the w component of the
    # input quaternions
    out[...,0] = quaternions[...,0]
    
    # Normalize
    if normalize_output:
        out /= torch.norm(out, dim=-1, keepdim=True)
    
    return out

def swing_decomposition(quaternions: torch.Tensor,
                        twists: torch.Tensor,
                        assume_normalized: bool = True,
                        out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    if not assume_normalized:
        raise NotImplementedError()
                        
    return quaternion_product(
        quaternions,
        quaternion_conj(twists),
        out=out
    )
    