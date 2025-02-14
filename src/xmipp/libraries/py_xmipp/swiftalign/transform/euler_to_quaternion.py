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

def euler_to_quaternion(rot: torch.Tensor,
                        tilt: torch.Tensor,
                        psi: torch.Tensor,
                        out: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    # Create the output
    batch_shape = torch.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (4, )
    dtype = rot.dtype
    device = rot.device
    out = torch.empty(result_shape, dtype=dtype, device=device, out=out)
    
    # Use halves
    ai = psi / 2
    aj = -tilt / 2
    ak = rot / 2
    
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

    # Compute the quaternion values
    # WXYZ
    out[...,0] = +cj * (ci_ck - si_sk)
    out[...,1] = +sj * (ci_sk - si_ck)
    out[...,2] = -sj * (ci_ck + si_sk)
    out[...,3] = +cj * (ci_sk + si_ck)
    
    return out