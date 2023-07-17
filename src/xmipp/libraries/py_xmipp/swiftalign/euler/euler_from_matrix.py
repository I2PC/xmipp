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

def euler_from_matrix(matrices: torch.Tensor,
                      out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Allocate the output
    out = torch.empty(matrices.shape[:-2] + (3, ), dtype=matrices.dtype, device=matrices.device, out=out)

    sy = torch.norm(matrices[...,2,0:2], dim=-1)
    torch.atan2(matrices[...,2,1], matrices[...,2,0], out=out[...,0])
    torch.atan2(sy, matrices[...,2,2], out=out[...,1])
    torch.atan2(matrices[...,1,2], -matrices[...,0,2], out=out[...,2])
    
    out *= -1
    return out