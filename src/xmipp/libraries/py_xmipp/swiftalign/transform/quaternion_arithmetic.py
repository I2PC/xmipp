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

def quaternion_conj(q: torch.Tensor,
                    out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    return torch.cat((q[...,0,None], -q[...,1:4]), dim=-1, out=out)

def quaternion_product(q: torch.Tensor,
                       r: torch.Tensor,
                       out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    if q.shape[-1] != 4:
        raise RuntimeError('q0 must have 4 dimensions in the last dimension')
    if r.shape[-1] != 4:
        raise RuntimeError('q1 must have 4 dimensions in the last dimension')
    
    # Create the output
    output_shape = torch.broadcast_shapes(q.shape, r.shape)
    out = torch.empty(output_shape, dtype=q.dtype, device=q.device, out=out)
    
    # Alias components of the input
    q0 = q[...,0]
    q1 = q[...,1]
    q2 = q[...,2]
    q3 = q[...,3]
    r0 = r[...,0]
    r1 = r[...,1]
    r2 = r[...,2]
    r3 = r[...,3]
    
    out[...,0] = q0*r0 - q1*r1 - q2*r2 - q3*r3
    out[...,1] = q0*r1 + q1*r0 + q2*r3 - q3*r2
    out[...,2] = q0*r2 - q1*r3 + q2*r0 + q3*r1
    out[...,3] = q0*r3 + q1*r2 - q2*r1 + q3*r0
    
    return out
