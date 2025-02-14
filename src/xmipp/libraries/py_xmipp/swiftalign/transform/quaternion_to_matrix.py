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

import math
import torch

def quaternion_to_matrix(q: torch.Tensor,
                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    # Create the output
    batch_shape = q.shape[:-1]
    result_shape = batch_shape + (3, 3)
    dtype = q.dtype
    device = q.device
    out = torch.empty(result_shape, dtype=dtype, device=device, out=out)
    
    # Multiply the quaternion by sqrt2 to avoid multiplying pairs by 2
    q = q * math.sqrt(2)
    
    # Alias the components
    qw = q[...,0]
    qx = q[...,1]
    qy = q[...,2]
    qz = q[...,3]

    # Obtain pairwise products of the quaternion elements
    qx2 = qx*qx
    qy2 = qy*qy
    qz2 = qz*qz
    qxqy = qx*qy
    qxqz = qx*qz
    qxqw = qx*qw
    qyqz = qy*qz
    qyqw = qy*qw
    qzqw = qz*qw

    # Ensemble the matrix
    out[...,0,0] = 1 - qy2 - qz2
    out[...,0,1] = qxqy - qzqw
    out[...,0,2] = qxqz + qyqw
    out[...,1,0] = qxqy + qzqw
    out[...,1,1] = 1 - qx2 - qz2
    out[...,1,2] = qyqz - qxqw
    out[...,2,0] = qxqz - qyqw
    out[...,2,1] = qyqz + qxqw
    out[...,2,2] = 1 - qx2 - qy2
    
    return out