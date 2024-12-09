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

def euler_to_matrix(rot: torch.Tensor,
                    tilt: torch.Tensor,
                    psi: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    # Create the output
    batch_shape = torch.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype
    device = rot.device
    out = torch.empty(result_shape, dtype=dtype, device=device, out=out)

    ai = -psi
    aj = -tilt
    ak = -rot 

    # Obtain sin and cos of the half angles
    ci = torch.cos(ai)
    si = torch.sin(ai)
    cj = torch.cos(aj)
    sj = torch.sin(aj)
    ck = torch.cos(ak)
    sk = torch.sin(ak)
    
    # Obtain the combinations
    ci_ck = ci * ck
    ci_sk = ci * sk
    si_ck = si * ck
    si_sk = si * sk

    # Build the matrix
    out[...,0,0] = cj * ci_ck - si_sk
    out[...,0,1] = cj * si_ck + ci_sk
    out[...,0,2] = -sj * ck
    out[...,1,0] = -cj * ci_sk - si_ck
    out[...,1,1] = -cj * si_sk + ci_ck
    out[...,1,2] = sj * sk
    out[...,2,0] = sj * ci
    out[...,2,1] = sj * si
    out[...,2,2] = cj

    return out